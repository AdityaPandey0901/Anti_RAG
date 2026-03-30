"""
State management for the LangChain agent.

Identical logic to the pi-agent: tracks pipeline phases and exposes
read/write helpers for question plans and metadata store.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from config import (
    OUTPUT_DIR, PARTS_DIR, PLANS_PATH, CSV_PATH, DB_PATH,
)


# ── Ensure directories exist ────────────────────────────────────────────────

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Metadata store ──────────────────────────────────────────────────────────

def metadata_exists() -> bool:
    return CSV_PATH.exists() or DB_PATH.exists()


def load_metadata() -> list[dict]:
    """Return the metadata store as a list of dicts."""
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        return df[["document_name", "brief_summary", "extraction_method"]].to_dict(orient="records")
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT document_name, brief_summary, extraction_method FROM metadata_store", conn
        )
        conn.close()
        return df.to_dict(orient="records")
    return []


# ── Question plans ──────────────────────────────────────────────────────────

def plans_exist() -> bool:
    if PLANS_PATH.exists():
        return True
    if PARTS_DIR.exists() and any(PARTS_DIR.glob("q_*.json")):
        return True
    return False


def load_plans() -> list[dict]:
    if PLANS_PATH.exists():
        return json.loads(PLANS_PATH.read_text())
    parts = sorted(PARTS_DIR.glob("q_*.json"))
    return [json.loads(p.read_text()) for p in parts]


def save_plans(plans: list[dict]):
    PLANS_PATH.write_text(json.dumps(plans, indent=2, ensure_ascii=False))
    for i, p in enumerate(plans):
        _part_path(i).write_text(json.dumps(p, indent=2, ensure_ascii=False))


def save_single_plan(idx: int, entry: dict):
    _part_path(idx).write_text(json.dumps(entry, indent=2, ensure_ascii=False))
    if PLANS_PATH.exists():
        plans = json.loads(PLANS_PATH.read_text())
        if idx < len(plans):
            plans[idx] = entry
            PLANS_PATH.write_text(json.dumps(plans, indent=2, ensure_ascii=False))


def _part_path(idx: int) -> Path:
    return PARTS_DIR / f"q_{idx + 1:03d}.json"


# ── High-level state snapshot ───────────────────────────────────────────────

def get_pipeline_state() -> dict:
    """Return a concise dict describing the current state of the pipeline."""
    ensure_dirs()
    result = {
        "metadata_ready": metadata_exists(),
        "metadata_count": 0,
        "plans_ready": plans_exist(),
        "total_questions": 0,
        "answered": 0,
        "unanswered": 0,
        "unanswered_indices": [],
    }

    if result["metadata_ready"]:
        result["metadata_count"] = len(load_metadata())

    if result["plans_ready"]:
        plans = load_plans()
        result["total_questions"] = len(plans)
        for i, p in enumerate(plans):
            if p.get("answer_found"):
                result["answered"] += 1
            else:
                result["unanswered"] += 1
                result["unanswered_indices"].append(i)

    return result
