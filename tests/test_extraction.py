"""
Characterization tests for the byte-level extraction functions shared by
every implementation in this repo (root, parallelized/, pi-agent/,
lang-chain-agent/ all define copies of these three functions).

These are pure functions — no GCS, no Gemini, no network — so they run
against the small fixtures in tests/fixtures/.
"""

from pathlib import Path

import pytest

from summarize_documents import extract_docx_text, extract_excel_text, extract_pdf_text

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def pdf_bytes() -> bytes:
    return (FIXTURES_DIR / "sample.pdf").read_bytes()


@pytest.fixture
def docx_bytes() -> bytes:
    return (FIXTURES_DIR / "sample.docx").read_bytes()


@pytest.fixture
def xlsx_bytes() -> bytes:
    return (FIXTURES_DIR / "sample.xlsx").read_bytes()


def test_extract_pdf_text_finds_raw_text(pdf_bytes):
    text, method = extract_pdf_text(pdf_bytes)
    assert method == "pdf_raw_text"
    assert "PDF_FIXTURE_MARKER" in text


def test_extract_docx_text_finds_paragraphs(docx_bytes):
    text, method = extract_docx_text(docx_bytes)
    assert method == "docx_raw_text"
    assert "DOCX_FIXTURE_MARKER" in text
    assert "second paragraph" in text


def test_extract_excel_text_reads_all_sheets(xlsx_bytes):
    text, method = extract_excel_text(xlsx_bytes, ".xlsx")
    assert method == "excel_full_read"
    assert "XLSX_FIXTURE_MARKER" in text
    assert "Sheet1" in text
