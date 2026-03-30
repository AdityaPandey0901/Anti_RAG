"""
Tool Forge — runtime tool creation and management for the pi-agent.

Allows the agent to dynamically create, list, and delete Python tools
at runtime.  Every action is logged to an append-only audit file for
full observability.

Design:
  • create_tool() — the agent supplies name, description, parameter schema,
    and raw Python source code.  The code is compiled and exec'd into an
    isolated namespace, then registered so Gemini can call it on the very
    next turn.
  • list_custom_tools() — returns metadata for all agent-created tools.
  • delete_tool() — removes a tool from the registry.
  • get_dynamic_declarations() — returns Gemini FunctionDeclaration objects
    for all currently registered dynamic tools.
  • get_dynamic_functions() — returns a {name: callable} dict for dispatch.
"""

from __future__ import annotations

import datetime
import json
import textwrap
import traceback
from pathlib import Path
from typing import Any, Callable

from google.genai import types

from config import OUTPUT_DIR

# ── Audit log ────────────────────────────────────────────────────────────────
AUDIT_LOG_PATH = OUTPUT_DIR / "tool_audit.jsonl"


def _audit(event: str, payload: dict):
    """Append a single JSON line to the audit log."""
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "event": event,
        **payload,
    }
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Internal registry ────────────────────────────────────────────────────────

# Each entry: {
#   "name": str,
#   "description": str,
#   "parameters": dict (JSON-Schema-like),
#   "code": str,
#   "callable": <function>,
#   "created_at": str,
# }
_dynamic_tools: dict[str, dict] = {}


# ── Schema helpers ───────────────────────────────────────────────────────────

_TYPE_MAP = {
    "string":  "STRING",
    "integer": "INTEGER",
    "number":  "NUMBER",
    "boolean": "BOOLEAN",
    "array":   "ARRAY",
    "object":  "OBJECT",
}


def _json_schema_to_gemini(schema: dict) -> types.Schema:
    """
    Convert a simplified JSON-schema-like dict into a Gemini types.Schema.

    Supports: type, description, properties, required, items.
    """
    gtype = _TYPE_MAP.get(schema.get("type", "string"), "STRING")
    kwargs: dict[str, Any] = {"type": gtype}

    if "description" in schema:
        kwargs["description"] = schema["description"]

    if "properties" in schema:
        kwargs["properties"] = {
            k: _json_schema_to_gemini(v)
            for k, v in schema["properties"].items()
        }

    if "required" in schema:
        kwargs["required"] = schema["required"]

    if "items" in schema:
        kwargs["items"] = _json_schema_to_gemini(schema["items"])

    return types.Schema(**kwargs)


# ── Public API ───────────────────────────────────────────────────────────────

def create_tool(
    name: str,
    description: str,
    code: str,
    parameters: dict | None = None,
) -> str:
    """
    Dynamically create a new tool the agent can call on subsequent turns.

    Args:
        name:        Tool name (must be a valid Python identifier and not
                     collide with built-in tools).
        description: Human-readable description for Gemini's schema.
        code:        Python source code defining a function with the same name
                     as *name*.  May import standard-library and any packages
                     already installed in the venv.  The function must accept
                     keyword arguments matching the parameter schema and return
                     a string result.
        parameters:  Optional JSON-Schema-style dict describing parameters.
                     If None, the tool takes no arguments.

    Returns:
        A status string.
    """
    # ── Guard: reserved names ────────────────────────────────────────────
    from tools import TOOL_FUNCTIONS as _builtins
    reserved = set(_builtins.keys()) | {
        "create_tool", "list_custom_tools", "delete_tool",
    }
    if name in reserved:
        msg = f"ERROR: '{name}' is a reserved tool name."
        _audit("create_tool_rejected", {"name": name, "reason": "reserved name"})
        return msg

    if not name.isidentifier():
        msg = f"ERROR: '{name}' is not a valid Python identifier."
        _audit("create_tool_rejected", {"name": name, "reason": "invalid identifier"})
        return msg

    # ── Compile & exec ───────────────────────────────────────────────────
    namespace: dict[str, Any] = {}
    try:
        compiled = compile(code, f"<dynamic_tool:{name}>", "exec")
        exec(compiled, namespace)
    except Exception:
        tb = traceback.format_exc()
        _audit("create_tool_error", {"name": name, "code": code, "error": tb})
        return f"ERROR compiling tool '{name}':\n{tb}"

    func = namespace.get(name)
    if not callable(func):
        _audit("create_tool_error", {
            "name": name, "code": code,
            "error": f"No callable named '{name}' found after exec."
        })
        return (
            f"ERROR: After executing the code, no callable named '{name}' was found. "
            f"Make sure your code defines `def {name}(...):`"
        )

    # ── Register ─────────────────────────────────────────────────────────
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    _dynamic_tools[name] = {
        "name": name,
        "description": description,
        "parameters": parameters or {},
        "code": code,
        "callable": func,
        "created_at": ts,
    }

    _audit("create_tool_ok", {
        "name": name,
        "description": description,
        "parameters": parameters or {},
        "code": code,
    })

    return (
        f"Tool '{name}' created successfully.  "
        f"It will be available for calling on the next turn."
    )


def list_custom_tools() -> str:
    """Return a JSON summary of all dynamically created tools."""
    if not _dynamic_tools:
        return "No custom tools have been created yet."

    summary = []
    for t in _dynamic_tools.values():
        summary.append({
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"],
            "created_at": t["created_at"],
            "code_preview": t["code"][:200] + ("..." if len(t["code"]) > 200 else ""),
        })

    _audit("list_custom_tools", {"count": len(summary)})
    return json.dumps(summary, indent=2, ensure_ascii=False)


def delete_tool(name: str) -> str:
    """Remove a dynamically created tool."""
    if name not in _dynamic_tools:
        return f"ERROR: No custom tool named '{name}' exists."

    del _dynamic_tools[name]
    _audit("delete_tool", {"name": name})
    return f"Tool '{name}' deleted.  It will no longer be available."


# ── Helpers for the agent loop ───────────────────────────────────────────────

def get_dynamic_declarations() -> list[types.FunctionDeclaration]:
    """
    Build Gemini FunctionDeclaration objects for every registered dynamic tool.
    Called by the agent loop on each turn to produce an up-to-date tool list.
    """
    decls = []
    for t in _dynamic_tools.values():
        params_schema = t["parameters"]
        if params_schema and params_schema.get("properties"):
            gemini_params = _json_schema_to_gemini(params_schema)
        else:
            gemini_params = types.Schema(type="OBJECT", properties={})

        decls.append(
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=gemini_params,
            )
        )
    return decls


def get_dynamic_functions() -> dict[str, Callable]:
    """
    Return a {name: callable} dict for all dynamic tools.
    Used by the dispatch function in the agent loop.
    """
    return {name: t["callable"] for name, t in _dynamic_tools.items()}


def get_tool_audit_log(last_n: int = 50) -> str:
    """Return the last N entries from the audit log as formatted text."""
    if not AUDIT_LOG_PATH.exists():
        return "Audit log is empty."
    lines = AUDIT_LOG_PATH.read_text().strip().split("\n")
    tail = lines[-last_n:]
    return "\n".join(tail)
