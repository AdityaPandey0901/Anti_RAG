"""
path_cache.py — Execution path cache for the LangChain agent.

Records the complete execution path (ordered tool calls + final answer) for
every unique user_message.  On a cache hit the entire agent loop is bypassed
and the stored answer is returned in microseconds.

Design
------
* Key   : xxhash64(normalised user_message)  — normalisation strips whitespace
          and lowercases so minor phrasing differences collapse to the same key.
* Value : {tool_calls: [...], final_answer: str, ...} persisted as JSON in
          SQLite.  The tool call log is stored for auditability and debugging
          (it is never re-executed — the answer is returned directly).
* The table also records a hit_count so you can see how often each path is
  reused, and the original message text so you can inspect/invalidate entries.

Schema
------
execution_paths (
    msg_hash     TEXT PRIMARY KEY,
    user_message TEXT NOT NULL,
    tool_calls   TEXT NOT NULL,    -- JSON array of tool call records
    final_answer TEXT NOT NULL,
    created_at   REAL NOT NULL,
    hit_count    INTEGER NOT NULL DEFAULT 0
)
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

import xxhash

from config import OUTPUT_DIR

PATH_CACHE_DB = OUTPUT_DIR / "cache" / "execution_paths.db"


def _ensure_db_dir() -> None:
    PATH_CACHE_DB.parent.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MESSAGE PARSING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_tool_calls(messages: list) -> list[dict]:
    """
    Parse a LangGraph message list into an ordered tool-call log.

    LangGraph represents a tool interaction as two messages:
      1. AIMessage  — contains .tool_calls  (list of {id, name, args})
      2. ToolMessage — contains .tool_call_id and .content (the result)

    We join them by tool_call_id to produce structured records:
      {name, args, result_preview, turn_index}

    Args:
        messages: The full message list from agent.invoke result["messages"].

    Returns:
        Ordered list of tool-call records (one per tool invocation).
    """
    pending: dict[str, dict] = {}   # tool_call_id → {name, args, turn_index}
    calls: list[dict] = []
    turn = 0

    for msg in messages:
        msg_type = getattr(msg, "type", None) or type(msg).__name__.lower()

        # AIMessage → collect pending tool call requests
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id   = tc.get("id") or tc.get("tool_call_id", "")
                tc_name = tc.get("name", "")
                tc_args = tc.get("args", {})
                pending[tc_id] = {
                    "name":       tc_name,
                    "args":       tc_args,
                    "turn_index": turn,
                }
            turn += 1

        # ToolMessage → match with pending request, finalise record
        elif hasattr(msg, "tool_call_id") and msg.tool_call_id:
            tc_id  = msg.tool_call_id
            result = str(getattr(msg, "content", "") or "")
            info   = pending.pop(tc_id, None)
            if info:
                calls.append({
                    "name":           info["name"],
                    "args":           info["args"],
                    "result_preview": result[:500] + ("…" if len(result) > 500 else ""),
                    "turn_index":     info["turn_index"],
                })

    return calls


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION PATH CACHE
# ═══════════════════════════════════════════════════════════════════════════════

class ExecutionPathCache:
    """
    Disk-persistent cache that maps user_message → execution path + answer.

    On a cache hit the agent loop is skipped entirely.  The stored final_answer
    is returned directly; the tool_calls log is preserved for debugging but is
    never re-executed (all underlying results are already in GeminiResponseCache
    and BlobCache from Parts 1 & 2, so a replay would be instant anyway).

    Message normalisation
    ---------------------
    Input is stripped and lower-cased before hashing so that:
      "Run the pipeline"  ==  "run the pipeline"  ==  "  Run the pipeline  "
    If you want case-sensitive matching, remove the .lower() call.
    """

    def __init__(self, db_path: Path = PATH_CACHE_DB) -> None:
        _ensure_db_dir()
        self._db_path = str(db_path)
        self._hits   = 0
        self._misses = 0
        self._init_db()

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_paths (
                    msg_hash     TEXT PRIMARY KEY,
                    user_message TEXT NOT NULL,
                    tool_calls   TEXT NOT NULL,
                    final_answer TEXT NOT NULL,
                    created_at   REAL NOT NULL,
                    hit_count    INTEGER NOT NULL DEFAULT 0
                )
            """)

    # ── Key ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _hash(user_message: str) -> str:
        normalised = user_message.strip().lower()
        return xxhash.xxh64(normalised.encode("utf-8", errors="replace")).hexdigest()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_answer(self, user_message: str) -> Optional[str]:
        """
        Return the cached final answer for this message, or None on a miss.

        Also increments hit_count and prints the stored tool-call path for
        visibility so you can confirm the right path is being replayed.
        """
        key = self._hash(user_message)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT final_answer, tool_calls, hit_count "
                "FROM execution_paths WHERE msg_hash=?",
                (key,),
            ).fetchone()
        if row is None:
            self._misses += 1
            return None

        final_answer, tool_calls_json, hit_count = row

        # Increment hit counter
        with self._conn() as conn:
            conn.execute(
                "UPDATE execution_paths SET hit_count=? WHERE msg_hash=?",
                (hit_count + 1, key),
            )

        self._hits += 1
        return final_answer

    def get_path(self, user_message: str) -> Optional[dict]:
        """
        Return the full cached record (answer + tool call log), or None.
        """
        key = self._hash(user_message)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT user_message, tool_calls, final_answer, created_at, hit_count "
                "FROM execution_paths WHERE msg_hash=?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "user_message": row[0],
            "tool_calls":   json.loads(row[1]),
            "final_answer": row[2],
            "created_at":   row[3],
            "hit_count":    row[4],
        }

    def record(
        self,
        user_message: str,
        tool_calls:   list[dict],
        final_answer: str,
    ) -> None:
        """
        Persist an execution path after a successful agent run.

        If a record for this message already exists it is overwritten with the
        fresh result (the agent may have produced a better answer on a re-run
        with newer documents).
        """
        key = self._hash(user_message)
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO execution_paths
                   (msg_hash, user_message, tool_calls, final_answer, created_at, hit_count)
                   VALUES (?, ?, ?, ?, ?, 0)""",
                (
                    key,
                    user_message,
                    json.dumps(tool_calls, ensure_ascii=False),
                    final_answer,
                    time.time(),
                ),
            )

    def invalidate(self, user_message: str) -> bool:
        """Remove the cached path for a specific message.  Returns True if found."""
        key = self._hash(user_message)
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM execution_paths WHERE msg_hash=?", (key,)
            )
        return cursor.rowcount > 0

    def clear(self) -> None:
        """Remove all cached paths."""
        with self._conn() as conn:
            conn.execute("DELETE FROM execution_paths")

    def list_all(self) -> list[dict]:
        """Return a summary of all cached paths (for inspection / debugging)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT msg_hash, user_message, created_at, hit_count, "
                "       json_array_length(tool_calls) as n_calls "
                "FROM execution_paths ORDER BY created_at DESC"
            ).fetchall()
        return [
            {
                "msg_hash":     r[0],
                "user_message": r[1][:80] + ("…" if len(r[1]) > 80 else ""),
                "created_at":   r[2],
                "hit_count":    r[3],
                "tool_call_count": r[4],
            }
            for r in rows
        ]

    def stats(self) -> str:
        with self._conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM execution_paths"
            ).fetchone()[0]
            total_hits = conn.execute(
                "SELECT COALESCE(SUM(hit_count), 0) FROM execution_paths"
            ).fetchone()[0]
        return (
            f"ExecutionPathCache: {total} paths stored, "
            f"{total_hits} total replays (all-time), "
            f"{self._hits} hits / {self._misses} misses this session"
        )


# ── Module-level singleton ────────────────────────────────────────────────────
path_cache = ExecutionPathCache()
