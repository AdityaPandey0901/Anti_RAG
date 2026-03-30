"""
Tool Forge — runtime tool creation and management for the LangChain agent.

Allows the agent to dynamically create, list, and delete Python tools
at runtime.  Every action is logged to an append-only audit file.
Dynamic tools are registered as LangChain StructuredTool instances
so they integrate seamlessly with the agent's tool-calling loop.
"""

from __future__ import annotations

import datetime
import json
import traceback
from pathlib import Path
from typing import Any, Callable

from langchain_core.tools import tool, StructuredTool

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
# name -> {"name", "description", "parameters", "code", "callable", "created_at", "lc_tool"}
_dynamic_tools: dict[str, dict] = {}


# ── Reserved names ───────────────────────────────────────────────────────────
_RESERVED = {
    "summarise_documents", "plan_questions", "deep_research",
    "deep_research_all_unanswered", "check_pipeline_state",
    "get_question_detail", "get_final_report",
    "create_tool", "list_custom_tools", "delete_tool", "get_tool_audit_log",
}


# ═══════════════════════════════════════════════════════════════════════════════
# META-TOOLS (decorated with @tool for LangChain)
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def create_tool(name: str, description: str, code: str,
                parameters: str = "{}") -> str:
    """Dynamically create a brand-new tool that can be called on subsequent turns.
    Write Python source code that defines a function with the given name.
    The function must accept keyword arguments and return a string.

    Args:
        name: Tool name — must be a valid Python identifier.
        description: Human-readable description of what the tool does.
        code: Python source defining `def <name>(...): ...`. May import stdlib and installed packages.
        parameters: JSON string describing parameters in JSON-Schema style. Use '{}' if no arguments.
    """
    if name in _RESERVED:
        msg = f"ERROR: '{name}' is a reserved tool name."
        _audit("create_tool_rejected", {"name": name, "reason": "reserved name"})
        return msg

    if not name.isidentifier():
        msg = f"ERROR: '{name}' is not a valid Python identifier."
        _audit("create_tool_rejected", {"name": name, "reason": "invalid identifier"})
        return msg

    # Parse parameters JSON
    try:
        params = json.loads(parameters) if parameters else {}
    except json.JSONDecodeError as e:
        return f"ERROR: Could not parse parameters JSON: {e}"

    # Compile & exec
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

    # Wrap as a LangChain StructuredTool
    lc_tool = StructuredTool.from_function(
        func=func,
        name=name,
        description=description,
    )

    ts = datetime.datetime.utcnow().isoformat() + "Z"
    _dynamic_tools[name] = {
        "name": name,
        "description": description,
        "parameters": params,
        "code": code,
        "callable": func,
        "lc_tool": lc_tool,
        "created_at": ts,
    }

    _audit("create_tool_ok", {
        "name": name,
        "description": description,
        "parameters": params,
        "code": code,
    })

    return (
        f"Tool '{name}' created successfully. "
        f"It will be available for calling on the next turn."
    )


@tool
def list_custom_tools() -> str:
    """List all dynamically created tools with their descriptions and code previews."""
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


@tool
def delete_tool(name: str) -> str:
    """Delete a dynamically created tool by name."""
    if name not in _dynamic_tools:
        return f"ERROR: No custom tool named '{name}' exists."

    del _dynamic_tools[name]
    _audit("delete_tool", {"name": name})
    return f"Tool '{name}' deleted. It will no longer be available."


@tool
def get_tool_audit_log(last_n: int = 50) -> str:
    """Return the last N entries from the tool creation audit log.
    Use this to review what dynamic tools have been created, modified, or deleted."""
    if not AUDIT_LOG_PATH.exists():
        return "Audit log is empty."
    lines = AUDIT_LOG_PATH.read_text().strip().split("\n")
    tail = lines[-last_n:]
    return "\n".join(tail)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS FOR AGENT LOOP
# ═══════════════════════════════════════════════════════════════════════════════

META_TOOLS = [create_tool, list_custom_tools, delete_tool, get_tool_audit_log]


def get_dynamic_lc_tools() -> list:
    """Return a list of LangChain StructuredTool instances for all dynamic tools."""
    return [t["lc_tool"] for t in _dynamic_tools.values()]
