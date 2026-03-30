"""
Centralised configuration for the LangChain agent.

Reads from the project-root .env and exposes typed constants.
Also configures LangSmith tracing for full observability.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # Lucio_Challenge_End_State/
ENV_PATH     = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# ── LangSmith Observability ──────────────────────────────────────────────────
os.environ["LANGSMITH_API_KEY"]  = "REDACTED_LANGSMITH_KEY"
os.environ["LANGSMITH_TRACING"]  = "true"
os.environ["LANGSMITH_PROJECT"]  = "pi-agent-langchain"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

# ── GCP credentials ──────────────────────────────────────────────────────────
CREDENTIALS_PATH        = os.getenv("CREDENTIALS_PATH", "")
VERTEX_CREDENTIALS_PATH = os.getenv("VERTEX_CREDENTIALS_PATH", "")
VERTEX_PROJECT           = os.getenv("VERTEX_PROJECT", "cciscrape")
VERTEX_LOCATION          = os.getenv("VERTEX_LOCATION", "us-central1")

# Resolve credential paths relative to project root
_cred_path = Path(VERTEX_CREDENTIALS_PATH)
if not _cred_path.is_absolute():
    _cred_path = PROJECT_ROOT / _cred_path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_cred_path)

# ── GCS document bucket / prefix ─────────────────────────────────────────────
DATA_PATH   = os.getenv("DATA_PATH", "")
_dp         = DATA_PATH.split("/", 1)
BUCKET_NAME = _dp[0]
PREFIX      = (_dp[1] + "/") if len(_dp) > 1 else ""

# ── GCS questions file ───────────────────────────────────────────────────────
QUESTIONS_PATH = os.getenv("QUESTIONS_PATH", "")
_qp            = QUESTIONS_PATH.split("/", 1)
Q_BUCKET       = _qp[0]
Q_BLOB         = _qp[1] if len(_qp) > 1 else ""

# ── Gemini ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME     = "gemini-2.5-flash"

# ── Agent output directory ───────────────────────────────────────────────────
AGENT_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR  = AGENT_DIR / "output"
PARTS_DIR   = OUTPUT_DIR / "question_parts"
PLANS_PATH  = OUTPUT_DIR / "question_plans.json"
CSV_PATH    = OUTPUT_DIR / "metadata_store.csv"
DB_PATH     = OUTPUT_DIR / "metadata_store.db"

# ── Concurrency ──────────────────────────────────────────────────────────────
MAX_WORKERS = 10
MAX_ROUNDS  = 5

# ── File types ───────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls"}
NATIVE_MIME = {".pdf": "application/pdf"}
