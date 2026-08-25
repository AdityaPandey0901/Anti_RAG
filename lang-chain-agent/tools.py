"""
Tools for the LangChain agent.

Each function is decorated with @tool so LangChain automatically generates
the schema and makes it available to the agent.  The heavy lifting is
identical to the pi-agent implementation.
"""

from __future__ import annotations

import io
import json
import sqlite3
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from google import genai
from google.genai import types as genai_types
from langchain_core.tools import tool

# ── Extraction helpers ───────────────────────────────────────────────────────
import fitz          # PyMuPDF
import pytesseract
from PIL import Image
from docx import Document as DocxDocument

from config import (
    DATA_PATH, QUESTIONS_PATH,
    VERTEX_PROJECT, VERTEX_LOCATION,
    MODEL_NAME, MAX_WORKERS,
    SUPPORTED_EXTENSIONS, NATIVE_MIME,
    OUTPUT_DIR, CSV_PATH, DB_PATH,
)
import state as st
from cache import gemini_cache, blob_cache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `core`
from core.sources import get_document_source, load_questions


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED CLIENTS (lazy-initialised)
# ═══════════════════════════════════════════════════════════════════════════════

_gemini: genai.Client | None = None
_blob_map: dict[str, object] | None = None


def _get_gemini() -> genai.Client:
    global _gemini
    if _gemini is None:
        _gemini = genai.Client(
            vertexai=True,
            project=VERTEX_PROJECT,
            location=VERTEX_LOCATION,
        )
    return _gemini


def _get_blob_map() -> dict[str, object]:
    global _blob_map
    if _blob_map is None:
        _blob_map = get_document_source(DATA_PATH).list_documents()
    return _blob_map


def _cached_download(blob) -> bytes:
    """Download a document's bytes, serving from the local disk cache when
    possible. Only real GCS blobs (which expose .bucket/.generation) are
    disk-cached — local files are already on disk, so there's nothing to
    cache for them; just read directly."""
    bucket = getattr(blob, "bucket", None)
    if bucket is None:
        return blob.download_as_bytes()

    bucket_name = bucket.name
    generation  = getattr(blob, "generation", None)
    cached = blob_cache.get(bucket_name, blob.name, generation)
    if cached is not None:
        return cached
    data = blob.download_as_bytes()
    blob_cache.set(bucket_name, blob.name, generation, data)
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pdf_text(blob_bytes: bytes) -> tuple[str, str]:
    doc = fitz.open(stream=blob_bytes, filetype="pdf")
    pages = min(3, len(doc))
    raw_texts = [doc[i].get_text() for i in range(pages)]
    raw = "\n\n".join(raw_texts).strip()
    if len(raw) > 50:
        doc.close()
        return raw, "pdf_raw_text"
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
    text = "\n\n--- page break ---\n\n".join("\n".join(p) for p in selected).strip()
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


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS (Gemini calls for summarise / plan / research)
# ═══════════════════════════════════════════════════════════════════════════════

def _init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
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


def _upsert_row(conn, doc_name, summary, method, chunk):
    conn.execute("""
        INSERT INTO metadata_store (document_name, brief_summary, extraction_method, inference_chunk)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(document_name) DO UPDATE SET
            brief_summary     = excluded.brief_summary,
            extraction_method = excluded.extraction_method,
            inference_chunk   = excluded.inference_chunk
    """, (doc_name, summary, method, chunk))
    conn.commit()


def _summarize_with_gemini(client, doc_name: str, chunk: str) -> str:
    key = gemini_cache.make_key("summarize", doc_name, chunk)
    cached = gemini_cache.get(key)
    if cached is not None:
        return cached
    prompt = (
        f'You are looking at the first few pages (or full contents for spreadsheets) '
        f'of a larger document named "{doc_name}".\n\n'
        f"--- DOCUMENT CONTENT ---\n{chunk}\n--- END ---\n\n"
        f"Based on this excerpt, guess what the full document is probably about and "
        f"provide a brief summary (2-4 sentences)."
    )
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    text = response.text.strip()
    gemini_cache.set(key, text)
    return text


def _download_questions() -> list[str]:
    return load_questions(QUESTIONS_PATH)


def _build_plan_prompt(questions: list[str], metadata: list[dict]) -> str:
    metadata_text = json.dumps(metadata, indent=2)
    questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    return f"""You are a research analyst. Below is a metadata table describing a set of documents,
followed by a list of research questions.

=== DOCUMENT METADATA ===
{metadata_text}
=== END METADATA ===

=== QUESTIONS ===
{questions_text}
=== END QUESTIONS ===

For each question, produce a JSON research plan. Return ONLY a valid JSON array (no markdown fences, no extra commentary).
Each element must have exactly these three keys:

  "question"     : the full question text (string)
  "answer_found" : if the answer can be determined directly from the document summaries above
                   (e.g. counting documents of a certain type, identifying what a document is about,
                   anything where reading the summaries is sufficient), set this to the concise
                   answer as a string. Otherwise set to null.
  "plan"         : if "answer_found" is null, a list of single-key objects where the key is the
                   document name and the value is a string describing exactly what information to
                   extract from that document to help answer the question.
                   Only include documents that are genuinely relevant.
                   If "answer_found" is not null, set "plan" to [].

CRITICAL — Evidential completeness:
  Before finalising a plan, think through what a correct answer would need to assert.
  For every factual sub-claim the answer would rely on (a number, a threshold, a name,
  a date, a monetary value, a legal condition), ask: is there a document in the library
  that could confirm or refute that claim directly?
  If yes, that document MUST appear in the plan with an instruction to verify that
  specific sub-claim — even if it seems tangential to the main question.
  Do NOT rely on assumptions or general knowledge for any sub-claim that a corpus
  document could resolve.

Return a JSON array, one object per question, in the same order as the questions above.
No extra keys. No markdown. No explanation outside the JSON."""


def _query_document(doc_name: str, blob_bytes: bytes,
                    question: str, what_to_find: str) -> str:
    ext = Path(doc_name).suffix.lower()
    mime = NATIVE_MIME.get(ext)
    prompt = (
        f"You are a precise legal and financial research assistant.\n\n"
        f"RESEARCH QUESTION: \"{question}\"\n\n"
        f"From this document (\"{doc_name}\"), I need: \"{what_to_find}\"\n\n"
        f"Instructions:\n"
        f"1. Search the document carefully for the requested information.\n"
        f"2. Quote the EXACT text passages that contain this information.\n"
        f"3. For each quoted passage, include a citation: [{doc_name}, Page <N>].\n"
        f"4. After listing all relevant passages, write a concise ATTEMPTED ANSWER "
        f"to the research question using ONLY the information found in this document. "
        f"If the document does not contain enough information to answer, state that explicitly.\n"
        f"5. If no relevant information is present at all, respond with: "
        f"NOT FOUND: <brief reason>.\n\n"
        f"Use this exact format:\n\n"
        f"EXTRACTED PASSAGES:\n"
        f"[{doc_name}, Page <N>] \"exact quoted text...\"\n"
        f"... (repeat for each relevant passage)\n\n"
        f"ATTEMPTED ANSWER:\n"
        f"<your answer to the research question based solely on the above passages, "
        f"or a statement that this document is insufficient to answer it>"
    )
    key = gemini_cache.make_key("query_doc", doc_name, blob_bytes, question, what_to_find)
    cached = gemini_cache.get(key)
    if cached is not None:
        return cached
    gemini = _get_gemini()
    if mime:
        part = genai_types.Part.from_bytes(data=blob_bytes, mime_type=mime)
        contents = [part, prompt]
    else:
        if ext == ".docx":
            text, _ = extract_docx_text(blob_bytes)
        elif ext in (".xlsx", ".xls"):
            text, _ = extract_excel_text(blob_bytes, ext)
        else:
            return f"NOT FOUND: unsupported file type {ext}"
        contents = [f"DOCUMENT CONTENTS ({doc_name}):\n{text}\n\nQUESTION:\n{prompt}"]
    response = gemini.models.generate_content(model=MODEL_NAME, contents=contents)
    result = response.text.strip()
    gemini_cache.set(key, result)
    return result


def _evaluate_answer(question: str, answer_found: str | None,
                     data_found: list[str]) -> dict:
    key = gemini_cache.make_key("evaluate", question, answer_found or "", data_found)
    cached = gemini_cache.get(key)
    if cached is not None:
        return json.loads(cached)

    sections = []
    if answer_found:
        sections.append(f"=== CURRENT ANSWER ===\n{answer_found}\n=== END CURRENT ANSWER ===")
    if data_found:
        joined = "\n\n---\n\n".join(data_found)
        sections.append(f"=== EXTRACTED DATA FROM DOCUMENTS ===\n{joined}\n=== END EXTRACTED DATA ===")
    context_block = "\n\n".join(sections) if sections else "(none)"

    prompt = (
        f"You are a research quality-control assistant.\n\n"
        f"QUESTION: \"{question}\"\n\n"
        f"Below is what has been gathered so far:\n\n"
        f"{context_block}\n\n"
        f"Does the information above adequately and completely answer the question?\n"
        f"Rules:\n"
        f"- 'Adequate' means a reader could act on this answer without needing to look further.\n"
        f"- If there is genuinely no relevant information above, answer is NOT sufficient.\n"
        f"- EVIDENTIAL COMPLETENESS CHECK: Identify every factual sub-claim the answer "
        f"makes (numbers, thresholds, values, names, dates, legal conditions). "
        f"If any such claim uses hedging language (likely, probably, approximately, may, "
        f"might, unclear, assumed) AND that claim is pivotal to the conclusion, the answer "
        f"is NOT sufficient — even if the overall logic is sound.\n"
        f"- Respond with ONLY valid JSON (no markdown fences), exactly this shape:\n"
        f'{{"sufficient": true/false, "reason": "...", "synthesised_answer": "..."}}\n'
        f"- If sufficient=true, write a clean, concise answer in 'synthesised_answer' using only the information above.\n"
        f"- If sufficient=false, set 'synthesised_answer' to null and explain why in 'reason'."
    )
    gemini = _get_gemini()
    response = gemini.models.generate_content(model=MODEL_NAME, contents=prompt)
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0].strip()
    result = json.loads(raw)
    gemini_cache.set(key, json.dumps(result))
    return result


def _refine_plan(question: str, existing_plan: list[dict],
                 data_found: list[str], metadata: list[dict]) -> list[dict]:
    key = gemini_cache.make_key("refine", question, existing_plan, data_found, metadata)
    cached = gemini_cache.get(key)
    if cached is not None:
        return json.loads(cached)

    metadata_text = json.dumps(metadata, indent=2)
    plan_text = json.dumps(existing_plan, indent=2)
    data_text = "\n\n---\n\n".join(data_found) if data_found else "(none yet)"

    prompt = (
        f"You are a research analyst refining a document research plan.\n\n"
        f"QUESTION: \"{question}\"\n\n"
        f"=== DOCUMENTS ALREADY QUERIED AND THEIR RESULTS ===\n"
        f"Plan executed so far:\n{plan_text}\n\n"
        f"Data extracted so far:\n{data_text}\n"
        f"=== END ===\n\n"
        f"=== FULL DOCUMENT LIBRARY (metadata) ===\n{metadata_text}\n=== END ===\n\n"
        f"The data found so far is NOT sufficient to answer the question.\n"
        f"Your task: produce a revised research plan that fills the gaps.\n\n"
        f"Return ONLY a valid JSON array (no markdown fences) of plan items:\n"
        f"[{{\"<document_name>\": \"what specifically to look for\"}}, ...]\n"
        f"List only the NEW items to execute — do not re-list items already covered above."
    )
    gemini = _get_gemini()
    response = gemini.models.generate_content(model=MODEL_NAME, contents=prompt)
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0].strip()
    result = json.loads(raw)
    gemini_cache.set(key, json.dumps(result))
    return result


def _answer_directly_from_doc(doc_name: str, blob_bytes: bytes, question: str) -> str:
    ext = Path(doc_name).suffix.lower()
    mime = NATIVE_MIME.get(ext)
    prompt = (
        f"You are a precise legal and financial research assistant.\n\n"
        f"Please answer the following question as completely and accurately as possible "
        f"using the content of this document (\"{doc_name}\"):\n\n"
        f"QUESTION: \"{question}\"\n\n"
        f"Instructions:\n"
        f"1. Search the entire document for any information relevant to the question.\n"
        f"2. Provide a direct, concise answer.\n"
        f"3. Support your answer with exact quoted passages and citations "
        f"([{doc_name}, Page <N>]).\n"
        f"4. If the document does not contain the answer, state clearly: "
        f"NOT FOUND IN THIS DOCUMENT: <reason>."
    )
    key = gemini_cache.make_key("answer_direct", doc_name, blob_bytes, question)
    cached = gemini_cache.get(key)
    if cached is not None:
        return cached
    gemini = _get_gemini()
    if mime:
        part = genai_types.Part.from_bytes(data=blob_bytes, mime_type=mime)
        contents = [part, prompt]
    else:
        if ext == ".docx":
            text, _ = extract_docx_text(blob_bytes)
        elif ext in (".xlsx", ".xls"):
            text, _ = extract_excel_text(blob_bytes, ext)
        else:
            return f"NOT FOUND: unsupported file type {ext}"
        contents = [f"DOCUMENT CONTENTS ({doc_name}):\n{text}\n\nQUESTION:\n{prompt}"]
    response = gemini.models.generate_content(model=MODEL_NAME, contents=contents)
    result = response.text.strip()
    gemini_cache.set(key, result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# LANGCHAIN TOOLS (decorated with @tool)
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def summarise_documents() -> str:
    """Ingest every supported document (PDF, DOCX, XLSX) from the GCS bucket,
    extract text, and build the metadata store with Gemini-generated summaries.
    Idempotent — skips documents already processed. Must run before plan_questions."""
    st.ensure_dirs()
    conn = _init_db()
    gemini = _get_gemini()

    docs = _get_blob_map()

    existing = {
        r[0] for r in conn.execute("SELECT document_name FROM metadata_store").fetchall()
    }

    processed = 0
    skipped = 0
    errors = 0

    for doc_name, blob in docs.items():
        ext = Path(doc_name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        if doc_name in existing:
            skipped += 1
            continue

        try:
            blob_bytes = _cached_download(blob)
            if ext == ".pdf":
                chunk, method = extract_pdf_text(blob_bytes)
            elif ext == ".docx":
                chunk, method = extract_docx_text(blob_bytes)
            elif ext in (".xlsx", ".xls"):
                chunk, method = extract_excel_text(blob_bytes, ext)
            else:
                continue

            if not chunk:
                _upsert_row(conn, doc_name, "[no content extracted]", method, "")
                continue

            summary = _summarize_with_gemini(gemini, doc_name, chunk)
            _upsert_row(conn, doc_name, summary, method, chunk)
            processed += 1
        except Exception as e:
            _upsert_row(conn, doc_name, f"[error: {e}]", "error", "")
            errors += 1

    # Export CSV
    df = pd.read_sql_query(
        "SELECT document_name, brief_summary, extraction_method FROM metadata_store", conn
    )
    conn.close()
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)

    total = len(df)
    return (
        f"Metadata store ready. {total} document(s) total — "
        f"{processed} newly processed, {skipped} cached, {errors} error(s)."
    )


@tool
def plan_questions() -> str:
    """Download the questions spreadsheet from GCS, load the metadata store,
    and call Gemini to produce a structured JSON research plan for every question.
    Some questions may be answered directly from summaries.
    Requires metadata store to be ready first."""
    st.ensure_dirs()
    metadata = st.load_metadata()
    if not metadata:
        return "ERROR: Metadata store is empty. Run summarise_documents first."

    questions = _download_questions()
    prompt = _build_plan_prompt(questions, metadata)

    plan_key = gemini_cache.make_key("plan_questions", prompt)
    plan_cached = gemini_cache.get(plan_key)
    if plan_cached is not None:
        raw = plan_cached
    else:
        gemini = _get_gemini()
        response = gemini.models.generate_content(model=MODEL_NAME, contents=prompt)
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0].strip()
        gemini_cache.set(plan_key, raw)

    plans = json.loads(raw)
    st.save_plans(plans)

    answered = sum(1 for p in plans if p.get("answer_found"))
    planned = len(plans) - answered
    return (
        f"Question planning complete. {len(plans)} question(s) total — "
        f"{answered} answered from summaries, {planned} need deep research."
    )


@tool
def deep_research(question_number: int) -> str:
    """Perform one round of deep research for a single question (1-indexed).
    Downloads the plan's referenced documents from GCS, queries each via Gemini,
    evaluates answer sufficiency, and refines the plan if needed.
    Call this for specific questions that need attention."""
    plans = st.load_plans()
    idx = question_number - 1
    if idx < 0 or idx >= len(plans):
        return f"ERROR: question_number {question_number} out of range (1–{len(plans)})."

    entry = plans[idx]
    question = entry.get("question", "")
    answer_found = entry.get("answer_found")
    plan = entry.get("plan", [])
    data_found = entry.get("data_found", [])
    blob_map = _get_blob_map()
    metadata = st.load_metadata()

    # Already answered — verify
    if answer_found:
        verdict = _evaluate_answer(question, answer_found, data_found)
        if verdict["sufficient"]:
            return f"Q{question_number} already answered and verified sufficient."
        answer_found = None
        entry["answer_found"] = None

    # Check existing data_found
    if data_found:
        verdict = _evaluate_answer(question, None, data_found)
        if verdict["sufficient"]:
            entry["answer_found"] = verdict["synthesised_answer"]
            st.save_single_plan(idx, entry)
            return f"Q{question_number} answered via synthesis from existing data."

        refined = _refine_plan(question, plan, data_found, metadata)
        if refined:
            plan = plan + refined
            entry["plan"] = plan

    # Generate plan if missing
    if not plan:
        prompt = _build_plan_prompt([question], metadata)
        _single_plan_key = gemini_cache.make_key("plan_questions", prompt)
        _cached_raw = gemini_cache.get(_single_plan_key)
        if _cached_raw is not None:
            raw = _cached_raw
        else:
            gemini = _get_gemini()
            response = gemini.models.generate_content(model=MODEL_NAME, contents=prompt)
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                raw = raw.rsplit("```", 1)[0].strip()
            gemini_cache.set(_single_plan_key, raw)
        new_plans = json.loads(raw)
        entry.update(new_plans[0])
        plan = entry.get("plan", [])

    # Execute plan — parallel doc queries
    start = len(data_found)
    remaining = list(enumerate(plan))[start:]

    if remaining:
        def _query_one(item):
            i, plan_item = item
            doc_name = next(iter(plan_item))
            what = plan_item[doc_name]
            blob = blob_map.get(doc_name)
            if blob is None:
                return (i, f"NOT FOUND IN GCS: {doc_name}")
            try:
                bb = _cached_download(blob)
                return (i, _query_document(doc_name, bb, question, what))
            except Exception as e:
                return (i, f"ERROR: {e}")

        results = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = {pool.submit(_query_one, item): item for item in remaining}
            for fut in as_completed(futs):
                i, result = fut.result()
                results[i] = result

        for i, _ in remaining:
            data_found.append(results[i])
        entry["data_found"] = data_found

    # Post-execution synthesis
    verdict = _evaluate_answer(question, None, data_found)
    if verdict["sufficient"]:
        entry["answer_found"] = verdict["synthesised_answer"]
        st.save_single_plan(idx, entry)
        return f"Q{question_number} answered after executing plan ({len(plan)} doc(s) queried)."

    # Direct fallback
    refined = _refine_plan(question, plan, data_found, metadata)
    if refined:
        entry["plan"] = plan + refined
        st.save_single_plan(idx, entry)
        return (
            f"Q{question_number} not yet answered — plan refined with "
            f"{len(refined)} new item(s) for next round."
        )

    direct_answers = []
    for plan_item in plan:
        doc_name = next(iter(plan_item))
        blob = blob_map.get(doc_name)
        if blob is None:
            direct_answers.append(f"NOT IN BUCKET: {doc_name}")
            continue
        try:
            bb = _cached_download(blob)
            ans = _answer_directly_from_doc(doc_name, bb, question)
            direct_answers.append(f"[Direct — {doc_name}]\n{ans}")
        except Exception as e:
            direct_answers.append(f"ERROR ({doc_name}): {e}")

    combined = _evaluate_answer(question, None, direct_answers)
    if combined["sufficient"]:
        entry["answer_found"] = combined["synthesised_answer"]
    else:
        best = next(
            (a for a in direct_answers if not a.startswith("NOT") and not a.startswith("ERROR")),
            direct_answers[0] if direct_answers else "No answer could be determined.",
        )
        entry["answer_found"] = best
    entry["data_found"] = data_found + direct_answers
    st.save_single_plan(idx, entry)
    return f"Q{question_number} answered via direct document fallback."


@tool
def deep_research_all_unanswered() -> str:
    """Run deep_research for every unanswered question in parallel.
    Use this to process an entire batch of unanswered questions at once."""
    plans = st.load_plans()
    unanswered = [i for i, p in enumerate(plans) if not p.get("answer_found")]
    if not unanswered:
        return "All questions are already answered."

    results = []

    def _run(idx):
        try:
            return (idx, deep_research.invoke({"question_number": idx + 1}))
        except Exception as e:
            return (idx, f"Q{idx+1} ERROR: {e}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(_run, i): i for i in unanswered}
        for fut in as_completed(futs):
            idx, msg = fut.result()
            results.append(msg)

    plans = st.load_plans()
    st.save_plans(plans)
    still_unanswered = sum(1 for p in plans if not p.get("answer_found"))

    header = (
        f"Deep research batch complete. "
        f"{len(plans) - still_unanswered}/{len(plans)} answered, "
        f"{still_unanswered} remaining."
    )
    return header + "\n" + "\n".join(results)


@tool
def check_pipeline_state() -> str:
    """Return a summary of the current pipeline state: whether the metadata store
    and question plans exist, how many questions are answered vs. unanswered,
    and which question numbers still need work. Use this to decide what to do next."""
    s = st.get_pipeline_state()
    lines = []
    lines.append(f"Metadata store ready: {s['metadata_ready']} ({s['metadata_count']} documents)")
    lines.append(f"Question plans ready: {s['plans_ready']}")
    if s["plans_ready"]:
        lines.append(f"Total questions: {s['total_questions']}")
        lines.append(f"Answered: {s['answered']}")
        lines.append(f"Unanswered: {s['unanswered']}")
        if s["unanswered_indices"]:
            nums = [str(i + 1) for i in s["unanswered_indices"]]
            lines.append(f"Unanswered question numbers: {', '.join(nums)}")
    return "\n".join(lines)


@tool
def get_question_detail(question_number: int) -> str:
    """Return the full current state for a single question: its text,
    answer_found (if any), plan items, and count of data_found entries."""
    plans = st.load_plans()
    idx = question_number - 1
    if idx < 0 or idx >= len(plans):
        return f"ERROR: question_number {question_number} out of range."
    entry = plans[idx]
    lines = [
        f"Question {question_number}: {entry.get('question', '')}",
        f"Answer: {entry.get('answer_found') or '(none)'}",
        f"Plan items: {len(entry.get('plan', []))}",
        f"Data found entries: {len(entry.get('data_found', []))}",
    ]
    if entry.get("plan"):
        lines.append("Plan:")
        for pi in entry["plan"]:
            doc = next(iter(pi))
            lines.append(f"  - {doc}: {pi[doc][:80]}...")
    return "\n".join(lines)


@tool
def get_final_report() -> str:
    """Return a formatted report of every question and its current answer.
    Use at the end to present results."""
    plans = st.load_plans()
    if not plans:
        return "No question plans found."
    lines = []
    for i, p in enumerate(plans, 1):
        q = p.get("question", "")
        a = p.get("answer_found") or "(no answer yet)"
        lines.append(f"[{i}] {q}")
        lines.append(f"    → {a}")
        lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL REGISTRY — all LangChain tools
# ═══════════════════════════════════════════════════════════════════════════════

PIPELINE_TOOLS = [
    summarise_documents,
    plan_questions,
    deep_research,
    deep_research_all_unanswered,
    check_pipeline_state,
    get_question_detail,
    get_final_report,
]
