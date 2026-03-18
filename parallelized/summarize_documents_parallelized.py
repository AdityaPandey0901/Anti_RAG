from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
summarize_documents_parallelized.py

Parallelized version of summarize_documents.py.

- Downloads ALL blobs from GCS concurrently (ThreadPool).
- Extracts text from each blob concurrently.
- Sends up to 10 concurrent Gemini summarization calls.
- Coalesces every result into the metadata_store (SQLite + CSV) at the very end.
"""

import os
import io
import re
import json
import sqlite3
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
import openpyxl
import pandas as pd
import pytesseract
from PIL import Image
from docx import Document as DocxDocument
from dotenv import load_dotenv
from google.cloud import storage
from google import genai

# ── Configuration ────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

CREDENTIALS_PATH        = os.getenv("CREDENTIALS_PATH")
DATA_PATH               = os.getenv("DATA_PATH")          # e.g. "vidhi_core/Lucio_Test"
VERTEX_CREDENTIALS_PATH = os.getenv("VERTEX_CREDENTIALS_PATH")
VERTEX_PROJECT          = os.getenv("VERTEX_PROJECT", "cciscrape")
VERTEX_LOCATION         = os.getenv("VERTEX_LOCATION", "us-central1")

# Resolve Vertex AI service-account JSON relative to project root
_project_root = Path(__file__).resolve().parent.parent
_vertex_cred  = Path(VERTEX_CREDENTIALS_PATH)
if not _vertex_cred.is_absolute():
    _vertex_cred = _project_root / _vertex_cred
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_vertex_cred)

# Parse bucket / prefix from DATA_PATH
_parts      = DATA_PATH.split("/", 1)
BUCKET_NAME = _parts[0]
PREFIX      = _parts[1] + "/" if len(_parts) > 1 else ""

DB_PATH     = Path(__file__).parent / "metadata_store.db"
MODEL_NAME  = "gemini-2.5-flash"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls"}
MAX_CONCURRENT_GEMINI = 10   # concurrency cap for Gemini calls


# ── Database helpers ─────────────────────────────────────────────────────────

def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metadata_store (
            document_name      TEXT PRIMARY KEY,
            brief_summary      TEXT,
            extraction_method  TEXT,
            inference_chunk    TEXT
        )
    """)
    conn.commit()
    return conn


def upsert_row(conn, doc_name, summary, extraction_method, chunk_text):
    conn.execute("""
        INSERT INTO metadata_store (document_name, brief_summary, extraction_method, inference_chunk)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(document_name) DO UPDATE SET
            brief_summary     = excluded.brief_summary,
            extraction_method = excluded.extraction_method,
            inference_chunk   = excluded.inference_chunk
    """, (doc_name, summary, extraction_method, chunk_text))
    conn.commit()


# ── Extraction helpers ───────────────────────────────────────────────────────

def extract_pdf_text(blob_bytes: bytes) -> tuple[str, str]:
    """Return (text, extraction_method) for a PDF. First 3 pages only."""
    doc = fitz.open(stream=blob_bytes, filetype="pdf")
    pages = min(3, len(doc))

    raw_texts = [doc[i].get_text() for i in range(pages)]
    raw = "\n\n".join(raw_texts).strip()
    if len(raw) > 50:
        doc.close()
        return raw, "pdf_raw_text"

    # Fall back to OCR
    ocr_parts = []
    for i in range(pages):
        pix = doc[i].get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        ocr_parts.append(pytesseract.image_to_string(img))
    doc.close()
    return "\n\n".join(ocr_parts).strip(), "pdf_ocr"


def extract_docx_text(blob_bytes: bytes) -> tuple[str, str]:
    doc = DocxDocument(io.BytesIO(blob_bytes))
    pages: list[list[str]] = [[]]
    for para in doc.paragraphs:
        xml = para._element.xml
        if 'w:br' in xml and 'type="page"' in xml:
            pages.append([])
            if len(pages) > 3:
                break
        pages[-1].append(para.text)

    selected = pages[:3]
    text = "\n\n--- page break ---\n\n".join(
        "\n".join(p) for p in selected
    ).strip()

    if len(pages) == 1 and len(text) > 3000:
        text = text[:3000] + "\n... [truncated]"

    return text, "docx_raw_text"


def extract_excel_text(blob_bytes: bytes, ext: str) -> tuple[str, str]:
    xls = pd.ExcelFile(io.BytesIO(blob_bytes), engine="openpyxl" if ext == ".xlsx" else None)
    parts = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        parts.append(f"=== Sheet: {sheet} ===\n{df.to_string(index=False, max_rows=200)}")
    text = "\n\n".join(parts).strip()
    if len(text) > 30000:
        text = text[:30000] + "\n... [truncated]"
    return text, "excel_full_read"


# ── Gemini helper ────────────────────────────────────────────────────────────

def summarize_with_gemini(client, doc_name: str, chunk: str) -> str:
    prompt = (
        f"You are looking at the first few pages (or full contents for spreadsheets) "
        f"of a larger document named \"{doc_name}\".\n\n"
        f"--- DOCUMENT CONTENT ---\n{chunk}\n--- END ---\n\n"
        f"Based on this excerpt, guess what the full document is probably about and "
        f"provide a brief summary (2-4 sentences)."
    )
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    return response.text.strip()


# ── Public helper (same interface as original) ──────────────────────────────

def get_metadata_store() -> list[dict]:
    """Return metadata_store as a list of dicts."""
    csv_path = Path(__file__).parent / "Plan_Docs/metadata_store.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df[["document_name", "brief_summary", "extraction_method"]].to_dict(orient="records")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT document_name, brief_summary, extraction_method FROM metadata_store", conn
    )
    conn.close()
    return df.to_dict(orient="records")


# ── Parallel pipeline helpers ────────────────────────────────────────────────

def _download_blob(blob) -> tuple[str, bytes | None, str | None]:
    """Download a single blob. Returns (doc_name, blob_bytes, ext) or (doc_name, None, None) to skip."""
    name = blob.name
    if name.endswith("/"):
        return (name, None, None)
    ext = Path(name).suffix.lower()
    doc_name = Path(name).name
    if ext not in SUPPORTED_EXTENSIONS:
        return (doc_name, None, None)
    try:
        blob_bytes = blob.download_as_bytes()
        return (doc_name, blob_bytes, ext)
    except Exception as e:
        print(f"  DOWNLOAD ERROR ({doc_name}): {e}")
        return (doc_name, None, None)


def _extract_text(doc_name: str, blob_bytes: bytes, ext: str) -> tuple[str, str, str, str]:
    """Extract text from bytes. Returns (doc_name, chunk, method, error_or_empty)."""
    try:
        if ext == ".pdf":
            chunk, method = extract_pdf_text(blob_bytes)
        elif ext == ".docx":
            chunk, method = extract_docx_text(blob_bytes)
        elif ext in (".xlsx", ".xls"):
            chunk, method = extract_excel_text(blob_bytes, ext)
        else:
            return (doc_name, "", "unsupported", "unsupported")
        return (doc_name, chunk, method, "")
    except Exception as e:
        return (doc_name, "", "error", str(e))


def _summarize_one(gemini_client, doc_name: str, chunk: str, method: str) -> dict:
    """Call Gemini for a single document. Returns a result dict."""
    if not chunk:
        return {
            "doc_name": doc_name,
            "summary": "[no content extracted]",
            "method": method,
            "chunk": "",
        }
    try:
        summary = summarize_with_gemini(gemini_client, doc_name, chunk)
    except Exception as e:
        summary = f"[gemini failed: {e}]"
    return {
        "doc_name": doc_name,
        "summary": summary,
        "method": method,
        "chunk": chunk,
    }


# ── Main pipeline (fully parallelized) ──────────────────────────────────────

def main():
    # ── 1. List blobs ────────────────────────────────────────────────────────
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=PREFIX))
    print(f"Found {len(blobs)} objects under gs://{BUCKET_NAME}/{PREFIX}")

    # Check which documents are already in the DB so we can skip them
    conn = init_db()
    existing_docs = {
        row[0]
        for row in conn.execute("SELECT document_name FROM metadata_store").fetchall()
    }

    # ── 2. Parallel download ─────────────────────────────────────────────────
    print("Downloading blobs in parallel...")
    downloaded: list[tuple[str, bytes, str]] = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_download_blob, b): b for b in blobs}
        for fut in as_completed(futures):
            doc_name, blob_bytes, ext = fut.result()
            if blob_bytes is None:
                continue
            if doc_name in existing_docs:
                print(f"  CACHED: {doc_name}")
                continue
            downloaded.append((doc_name, blob_bytes, ext))

    print(f"  {len(downloaded)} new document(s) to process.")
    if not downloaded:
        print("Nothing new to process — exiting.")
        _finalize(conn)
        return

    # ── 3. Parallel extraction ───────────────────────────────────────────────
    print("Extracting text in parallel...")
    extracted: list[tuple[str, str, str]] = []  # (doc_name, chunk, method)
    extraction_errors: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            pool.submit(_extract_text, dn, bb, ex): dn
            for dn, bb, ex in downloaded
        }
        for fut in as_completed(futures):
            doc_name, chunk, method, error = fut.result()
            if error:
                extraction_errors.append((doc_name, error))
            else:
                extracted.append((doc_name, chunk, method))

    print(f"  Extracted: {len(extracted)}, Errors: {len(extraction_errors)}")

    # ── 4. Parallel Gemini summarization (capped at MAX_CONCURRENT_GEMINI) ──
    print(f"Summarizing with Gemini via Vertex AI ({MAX_CONCURRENT_GEMINI} concurrent calls)...")
    gemini_client = genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_GEMINI) as pool:
        futures = {
            pool.submit(_summarize_one, gemini_client, dn, ch, mt): dn
            for dn, ch, mt in extracted
        }
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            print(f"  OK: {result['doc_name']} ({result['method']})")

    # ── 5. Coalesce: write ALL results to DB at once ─────────────────────────
    print("Writing results to metadata store...")

    # Extraction errors first
    for doc_name, error in extraction_errors:
        upsert_row(conn, doc_name, f"[extraction failed: {error}]", "error", "")

    # Successful results
    for r in results:
        upsert_row(conn, r["doc_name"], r["summary"], r["method"], r["chunk"])

    _finalize(conn)


def _finalize(conn):
    """Print the final table and save CSV."""
    df = pd.read_sql_query(
        "SELECT document_name, brief_summary, extraction_method FROM metadata_store", conn
    )
    conn.close()

    print("\n" + "=" * 80)
    print("METADATA STORE")
    print("=" * 80)
    print(df.to_string(index=False))

    csv_path = Path(__file__).parent / "Plan_Docs/metadata_store.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")


if __name__ == "__main__":
    main()
