"""
cache.py — Three-layer caching infrastructure for the LangChain agent.

Layer 1 — LangChainSQLiteCache
    Exact-match cache for every ChatGoogleGenerativeAI call made by the
    LangGraph agent loop.  Activated once at startup via init_langchain_cache().
    Key: xxhash64(prompt) + xxhash64(llm_string).

Layer 2 — GeminiResponseCache
    Hash-keyed SQLite cache for every direct google.genai.generate_content()
    call made inside tools.py (summarise, plan, query, evaluate, refine).
    Key: xxhash64 over arbitrary content parts (strings, bytes, dicts).

Layer 3 — BlobCache
    Disk cache for GCS blob downloads.  Files are stored as raw bytes under
    output/cache/blobs/<sha256_key>.bin.  The GCS object generation number
    is included in the key so a re-uploaded file always produces a cache miss.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import xxhash
from langchain_core.caches import BaseCache
from langchain_core.globals import set_llm_cache
from langchain_core.load import dumps as lc_dumps, loads as lc_loads
from langchain_core.outputs import Generation

from config import OUTPUT_DIR


# ── Cache directory layout ────────────────────────────────────────────────────
CACHE_DIR         = OUTPUT_DIR / "cache"
LANGCHAIN_DB_PATH = CACHE_DIR / "langchain_llm.db"
GEMINI_DB_PATH    = CACHE_DIR / "gemini_responses.db"
BLOB_CACHE_DIR    = CACHE_DIR / "blobs"


def _ensure_cache_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    BLOB_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — LangChain LLM cache
# ═══════════════════════════════════════════════════════════════════════════════

class LangChainSQLiteCache(BaseCache):
    """
    SQLite-backed exact-match cache for LangChain LLM calls.

    Every ChatGoogleGenerativeAI call (agent turns, tool-choice reasoning,
    etc.) is intercepted.  If the exact (prompt, llm_string) pair was seen
    before, the stored Generation list is returned without hitting the API.

    WAL mode is enabled so concurrent reads during parallel tool execution
    never block writes.

    Schema
    ------
    llm_cache(prompt_hash TEXT, llm_hash TEXT, value TEXT, created_at REAL)
    """

    def __init__(self, db_path: Path = LANGCHAIN_DB_PATH) -> None:
        _ensure_cache_dirs()
        self._db_path = str(db_path)
        self._hits = 0
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
                CREATE TABLE IF NOT EXISTS llm_cache (
                    prompt_hash TEXT NOT NULL,
                    llm_hash    TEXT NOT NULL,
                    value       TEXT NOT NULL,
                    created_at  REAL NOT NULL,
                    PRIMARY KEY (prompt_hash, llm_hash)
                )
            """)

    # ── Hashing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _hash(s: str) -> str:
        return xxhash.xxh64(s.encode("utf-8", errors="replace")).hexdigest()

    # ── BaseCache interface ───────────────────────────────────────────────────

    def lookup(self, prompt: str, llm_string: str) -> Optional[Sequence[Generation]]:
        ph = self._hash(prompt)
        lh = self._hash(llm_string)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM llm_cache WHERE prompt_hash=? AND llm_hash=?",
                (ph, lh),
            ).fetchone()
        if row is None:
            self._misses += 1
            return None
        try:
            result = lc_loads(row[0])
            self._hits += 1
            return result
        except Exception:
            self._misses += 1
            return None

    def update(
        self, prompt: str, llm_string: str, return_val: Sequence[Generation]
    ) -> None:
        ph = self._hash(prompt)
        lh = self._hash(llm_string)
        try:
            serialised = lc_dumps(return_val)
        except Exception:
            return
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO llm_cache
                   (prompt_hash, llm_hash, value, created_at)
                   VALUES (?, ?, ?, ?)""",
                (ph, lh, serialised, time.time()),
            )

    def clear(self, **kwargs: Any) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM llm_cache")

    def stats(self) -> str:
        with self._conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM llm_cache"
            ).fetchone()[0]
        return (
            f"LangChainSQLiteCache: {total} entries stored, "
            f"{self._hits} hits / {self._misses} misses this session"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — Gemini response cache (for direct google.genai calls)
# ═══════════════════════════════════════════════════════════════════════════════

class GeminiResponseCache:
    """
    SQLite-backed cache for direct google.genai generate_content() calls.

    Every helper in tools.py that calls _get_gemini().models.generate_content()
    should route through this cache.  The caller computes a key with
    GeminiResponseCache.make_key(*parts) and wraps the call:

        key = gemini_cache.make_key(doc_name, blob_bytes, question)
        cached = gemini_cache.get(key)
        if cached is not None:
            return cached
        response = gemini.models.generate_content(...)
        text = response.text.strip()
        gemini_cache.set(key, text)
        return text

    make_key() accepts any mix of strings, bytes, and JSON-serialisable objects
    and folds them into a single xxhash64 digest.

    Schema
    ------
    gemini_cache(cache_key TEXT PRIMARY KEY, response TEXT, created_at REAL)
    """

    def __init__(self, db_path: Path = GEMINI_DB_PATH) -> None:
        _ensure_cache_dirs()
        self._db_path = str(db_path)
        self._hits = 0
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
                CREATE TABLE IF NOT EXISTS gemini_cache (
                    cache_key  TEXT PRIMARY KEY,
                    response   TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)

    # ── Key construction ──────────────────────────────────────────────────────

    @staticmethod
    def make_key(*parts: Any) -> str:
        """
        Hash arbitrary parts into a single cache key.

        Accepts strings, bytes, and any JSON-serialisable value.
        Order matters — make_key("a", "b") != make_key("b", "a").
        """
        h = xxhash.xxh64()
        for part in parts:
            if isinstance(part, bytes):
                h.update(part)
            elif isinstance(part, str):
                h.update(part.encode("utf-8", errors="replace"))
            else:
                h.update(
                    json.dumps(part, sort_keys=True, default=str).encode()
                )
            # Separator so make_key("ab", "c") != make_key("a", "bc")
            h.update(b"\x00")
        return h.hexdigest()

    # ── Get / set ─────────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT response FROM gemini_cache WHERE cache_key=?", (key,)
            ).fetchone()
        if row:
            self._hits += 1
            return row[0]
        self._misses += 1
        return None

    def set(self, key: str, response_text: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO gemini_cache
                   (cache_key, response, created_at)
                   VALUES (?, ?, ?)""",
                (key, response_text, time.time()),
            )

    def delete(self, key: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM gemini_cache WHERE cache_key=?", (key,)
            )

    def stats(self) -> str:
        with self._conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM gemini_cache"
            ).fetchone()[0]
        return (
            f"GeminiResponseCache: {total} entries stored, "
            f"{self._hits} hits / {self._misses} misses this session"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — GCS blob download cache
# ═══════════════════════════════════════════════════════════════════════════════

class BlobCache:
    """
    Disk cache for GCS blob downloads.

    Files are written as raw bytes to BLOB_CACHE_DIR/<key>.bin where key is
    sha256(bucket + "/" + name + "@" + generation).

    Including the GCS generation number (a monotonically-increasing integer
    that changes on every overwrite) in the key guarantees that a re-uploaded
    document always produces a cache miss — no stale bytes are ever returned.

    Usage
    -----
        cached = blob_cache.get(blob.bucket.name, blob.name, blob.generation)
        if cached is not None:
            return cached
        data = blob.download_as_bytes()
        blob_cache.set(blob.bucket.name, blob.name, blob.generation, data)
        return data
    """

    def __init__(self, cache_dir: Path = BLOB_CACHE_DIR) -> None:
        _ensure_cache_dirs()
        self._dir = cache_dir
        self._hits = 0
        self._misses = 0

    def _key(self, bucket: str, name: str, generation: Any) -> str:
        raw = f"{bucket}/{name}@{generation}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.bin"

    def get(self, bucket: str, name: str, generation: Any) -> Optional[bytes]:
        p = self._path(self._key(bucket, name, generation))
        if p.exists():
            self._hits += 1
            return p.read_bytes()
        self._misses += 1
        return None

    def set(self, bucket: str, name: str, generation: Any, data: bytes) -> None:
        p = self._path(self._key(bucket, name, generation))
        p.write_bytes(data)

    def size(self) -> int:
        return len(list(self._dir.glob("*.bin")))

    def stats(self) -> str:
        return (
            f"BlobCache: {self.size()} blobs on disk, "
            f"{self._hits} hits / {self._misses} misses this session"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETONS  (import these from other modules)
# ═══════════════════════════════════════════════════════════════════════════════

gemini_cache = GeminiResponseCache()
blob_cache   = BlobCache()


def init_langchain_cache() -> None:
    """
    Activate the LangChain LLM cache.

    Must be called once before any ChatGoogleGenerativeAI instance is created.
    Subsequent calls are no-ops (set_llm_cache replaces the previous cache).
    Called from agent.py at module import time.
    """
    set_llm_cache(LangChainSQLiteCache())


def print_cache_stats() -> None:
    """Print a summary of all cache layers to stdout."""
    lc = LangChainSQLiteCache()
    gc = gemini_cache
    bc = blob_cache
    print(lc.stats())
    print(gc.stats())
    print(bc.stats())
