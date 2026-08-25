"""
Tests for core/sources.py — local-mode paths only, no GCS/network.
"""

from pathlib import Path

from core.sources import (
    GCSDocumentSource,
    LocalDocumentSource,
    get_document_source,
    load_questions,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_local_document_source_lists_supported_files():
    source = LocalDocumentSource(FIXTURES_DIR)
    docs = source.list_documents()

    assert set(docs) == {"sample.pdf", "sample.docx", "sample.xlsx"}
    for name, ref in docs.items():
        assert ref.name == name
        assert ref.download_as_bytes() == (FIXTURES_DIR / name).read_bytes()


def test_local_document_source_ignores_unsupported_files():
    source = LocalDocumentSource(FIXTURES_DIR)
    docs = source.list_documents()
    assert "questions.csv" not in docs
    assert "make_fixtures.py" not in docs


def test_get_document_source_picks_local_for_existing_dir():
    source = get_document_source(str(FIXTURES_DIR))
    assert isinstance(source, LocalDocumentSource)


def test_get_document_source_falls_back_to_gcs_for_nonexistent_path():
    source = get_document_source("some-bucket/some/prefix")
    assert isinstance(source, GCSDocumentSource)
    assert source.bucket_name == "some-bucket"
    assert source.prefix == "some/prefix"


def test_get_document_source_strips_gs_scheme():
    source = get_document_source("gs://some-bucket/some/prefix")
    assert isinstance(source, GCSDocumentSource)
    assert source.bucket_name == "some-bucket"
    assert source.prefix == "some/prefix"


def test_load_questions_from_local_csv():
    questions = load_questions(str(FIXTURES_DIR / "questions.csv"))
    assert questions == [
        "What was the quarterly revenue?",
        "When was the merger agreement signed?",
        "How many documents are in the set?",
    ]


def test_load_questions_from_local_xlsx_first_column_fallback(tmp_path):
    import pandas as pd

    xlsx_path = tmp_path / "no_question_header.xlsx"
    pd.DataFrame({"Prompt": ["Q1?", "Q2?"]}).to_excel(xlsx_path, index=False)

    questions = load_questions(str(xlsx_path))
    assert questions == ["Q1?", "Q2?"]
