"""
Source-agnostic document and question loading, shared by every
implementation in this repo (root, parallelized/, pi-agent/,
lang-chain-agent/).

Every implementation's GCS-specific code boils down to two things:
  1. building a {filename: blob} map, where `blob` exposes
     .name and .download_as_bytes()
  2. downloading a questions spreadsheet and picking the question column

`get_document_source()` / `load_questions()` resolve either to a local
path or the existing GCS convention (`bucket/prefix`, optionally
`gs://bucket/prefix`), so the rest of each implementation — which only
ever calls `blob.download_as_bytes()` — doesn't need to change: a
LocalDocRef duck-types the same interface a real GCS blob already has.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Protocol

import pandas as pd

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls"}


class DocRef(Protocol):
    """What every call site actually relies on — satisfied by both
    LocalDocRef and a real google.cloud.storage.Blob."""

    name: str

    def download_as_bytes(self) -> bytes: ...


class LocalDocRef:
    def __init__(self, path: Path):
        self._path = path
        self.name = path.name

    def download_as_bytes(self) -> bytes:
        return self._path.read_bytes()


class LocalDocumentSource:
    """Documents from a local directory, searched recursively."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)

    def list_documents(self) -> dict[str, LocalDocRef]:
        if not self.folder.is_dir():
            raise FileNotFoundError(f"Not a directory: {self.folder}")
        result: dict[str, LocalDocRef] = {}
        for path in sorted(self.folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                result[path.name] = LocalDocRef(path)
        return result


class GCSDocumentSource:
    """Documents from a GCS bucket/prefix — the pre-existing behaviour,
    lifted out of each implementation's own _build_gcs_blob_map()."""

    def __init__(self, bucket_name: str, prefix: str = ""):
        self.bucket_name = bucket_name
        self.prefix = prefix

    def list_documents(self) -> dict[str, object]:
        from google.cloud import storage  # lazy: local-only users need no GCP deps

        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        result: dict[str, object] = {}
        for blob in bucket.list_blobs(prefix=self.prefix):
            if not blob.name.endswith("/"):
                result[Path(blob.name).name] = blob
        return result


def get_document_source(data_path: str) -> LocalDocumentSource | GCSDocumentSource:
    """
    Resolve DATA_PATH into a document source.

    A path that exists locally as a directory wins; otherwise it's treated
    as the existing GCS convention ("bucket/prefix", optionally prefixed
    with "gs://").
    """
    local = Path(data_path)
    if local.is_dir():
        return LocalDocumentSource(local)

    uri = data_path[len("gs://"):] if data_path.startswith("gs://") else data_path
    bucket, _, prefix = uri.partition("/")
    return GCSDocumentSource(bucket, prefix)


def _pick_question_column(df: pd.DataFrame) -> str:
    """Prefer a column literally named 'question(s)', else the first column
    — the existing rule from plan_questions.py's download_questions()."""
    return next(
        (c for c in df.columns if str(c).strip().lower() in ("question", "questions")),
        df.columns[0],
    )


def load_questions(path_or_uri: str) -> list[str]:
    """
    Load a flat list of question strings from a local .csv/.xlsx/.xls file,
    or (if no such local file exists) the existing GCS convention
    ("bucket/prefix/file.xlsx", optionally "gs://...").
    """
    local_path = Path(path_or_uri)
    ext = local_path.suffix.lower()

    if local_path.exists():
        if ext == ".csv":
            df = pd.read_csv(local_path)
        elif ext in (".xlsx", ".xls"):
            xls = pd.ExcelFile(local_path, engine="openpyxl" if ext == ".xlsx" else None)
            df = xls.parse(xls.sheet_names[0])
        else:
            raise ValueError(f"Unsupported local questions file: {path_or_uri}")
    else:
        from google.cloud import storage  # lazy: local-only users need no GCP deps

        uri = path_or_uri[len("gs://"):] if path_or_uri.startswith("gs://") else path_or_uri
        bucket_name, _, blob_name = uri.partition("/")
        client = storage.Client()
        data = client.bucket(bucket_name).blob(blob_name).download_as_bytes()
        xls = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
        df = xls.parse(xls.sheet_names[0])

    col = _pick_question_column(df)
    return df[col].dropna().astype(str).str.strip().tolist()
