import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
Summarize documents from a GCP bucket using Gemini 2.5 Flash.

For each file in DATA_PATH:
  - PDF:   extract raw text from first 3 pages; fall back to OCR if empty
  - DOCX:  extract raw text from first 3 pages (approximated by paragraphs)
  - XLSX:  extract entire file contents
Send the inference chunk to Gemini for a brief summary, then persist
results in a local SQLite table `metadata_store`.
"""

import os
import io
import re
import json
import sqlite3
import tempfile
from pathlib import Path

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
load_dotenv()

CREDENTIALS_PATH = os.getenv("CREDENTIALS_PATH")
DATA_PATH        = os.getenv("DATA_PATH")          # e.g. "vidhi_core/Lucio_Test"
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

# Parse bucket / prefix from DATA_PATH
_parts      = DATA_PATH.split("/", 1)
BUCKET_NAME = _parts[0]
PREFIX      = _parts[1] + "/" if len(_parts) > 1 else ""

DB_PATH     = Path(__file__).parent / "metadata_store.db"
MODEL_NAME  = "gemini-2.5-flash"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls"}

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

    # Try raw text extraction first
    raw_texts = [doc[i].get_text() for i in range(pages)]
    raw = "\n\n".join(raw_texts).strip()
    if len(raw) > 50:  # meaningful text found
        doc.close()
        return raw, "pdf_raw_text"

    # Fall back to OCR via rendering + pytesseract
    ocr_parts = []
    for i in range(pages):
        pix = doc[i].get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        ocr_parts.append(pytesseract.image_to_string(img))
    doc.close()
    return "\n\n".join(ocr_parts).strip(), "pdf_ocr"


def extract_docx_text(blob_bytes: bytes) -> tuple[str, str]:
    """Return (text, method) for a DOCX. Approximates first 3 'pages' via page breaks."""
    doc = DocxDocument(io.BytesIO(blob_bytes))

    # Collect paragraphs, splitting on page breaks
    pages: list[list[str]] = [[]]
    for para in doc.paragraphs:
        # Check for hard page break in the paragraph's XML
        xml = para._element.xml
        if 'w:br' in xml and 'type="page"' in xml:
            pages.append([])
            if len(pages) > 3:
                break
        pages[-1].append(para.text)

    # Take first 3 pages
    selected = pages[:3]
    text = "\n\n--- page break ---\n\n".join(
        "\n".join(p) for p in selected
    ).strip()

    # If no page breaks found, just take first ~3000 chars as approximation
    if len(pages) == 1 and len(text) > 3000:
        text = text[:3000] + "\n... [truncated]"

    return text, "docx_raw_text"


def extract_excel_text(blob_bytes: bytes, ext: str) -> tuple[str, str]:
    """Return (text, method) for an Excel file. Reads entire file."""
    # pandas can read from bytes via BytesIO
    xls = pd.ExcelFile(io.BytesIO(blob_bytes), engine="openpyxl" if ext == ".xlsx" else None)
    parts = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        parts.append(f"=== Sheet: {sheet} ===\n{df.to_string(index=False, max_rows=200)}")
    text = "\n\n".join(parts).strip()
    # Truncate if extremely large to stay within Gemini context
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


# ── Public helper ───────────────────────────────────────────────────────────

def get_metadata_store() -> list[dict]:
    """Return metadata_store as a list of dicts (document_name, brief_summary, extraction_method).
    Prefers the CSV if it already exists; falls back to SQLite."""
    csv_path = Path(__file__).parent / "Plan_Docs/metadata_store.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df[["document_name", "brief_summary", "extraction_method"]].to_dict(orient="records")
    # Fall back to SQLite
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT document_name, brief_summary, extraction_method FROM metadata_store", conn
    )
    conn.close()
    return df.to_dict(orient="records")


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    # Init
    conn = init_db()
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(BUCKET_NAME)
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    blobs = list(bucket.list_blobs(prefix=PREFIX))
    print(f"Found {len(blobs)} objects under gs://{BUCKET_NAME}/{PREFIX}")

    for blob in blobs:
        name = blob.name
        # Skip "folders" (zero-byte directory markers)
        if name.endswith("/"):
            continue

        ext = Path(name).suffix.lower()
        doc_name = Path(name).name  # just the filename

        if ext not in SUPPORTED_EXTENSIONS:
            print(f"  SKIP (unsupported): {doc_name}")
            continue

        # Check if already processed
        existing = conn.execute(
            "SELECT 1 FROM metadata_store WHERE document_name = ?", (doc_name,)
        ).fetchone()
        if existing:
            print(f"  CACHED: {doc_name}")
            continue

        print(f"  Processing: {doc_name} ...", end=" ", flush=True)

        # Download
        blob_bytes = blob.download_as_bytes()

        # Extract
        try:
            if ext == ".pdf":
                chunk, method = extract_pdf_text(blob_bytes)
            elif ext == ".docx":
                chunk, method = extract_docx_text(blob_bytes)
            elif ext in (".xlsx", ".xls"):
                chunk, method = extract_excel_text(blob_bytes, ext)
            else:
                continue
        except Exception as e:
            print(f"EXTRACTION ERROR: {e}")
            upsert_row(conn, doc_name, f"[extraction failed: {e}]", "error", "")
            continue

        if not chunk:
            print("EMPTY")
            upsert_row(conn, doc_name, "[no content extracted]", method, "")
            continue

        # Summarize via Gemini
        try:
            summary = summarize_with_gemini(gemini_client, doc_name, chunk)
        except Exception as e:
            print(f"GEMINI ERROR: {e}")
            upsert_row(conn, doc_name, f"[gemini failed: {e}]", method, chunk)
            continue

        upsert_row(conn, doc_name, summary, method, chunk)
        print(f"OK ({method})")

    # Print final table
    df = pd.read_sql_query(
        "SELECT document_name, brief_summary, extraction_method FROM metadata_store", conn
    )
    conn.close()

    print("\n" + "=" * 80)
    print("METADATA STORE")
    print("=" * 80)
    print(df.to_string(index=False))

    # Save to CSV
    csv_path = Path(__file__).parent / "Plan_Docs/metadata_store.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")


if __name__ == "__main__":
    main()
