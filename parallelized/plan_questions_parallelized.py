from __future__ import annotations
import time

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
plan_questions_parallelized.py

Parallelized version of plan_questions.py.

Key differences from the sequential version:
  1. Each question's sub-JSON is written to its own file under
     Plan_Docs/question_parts/  (e.g. q_001.json, q_002.json).
  2. All Gemini calls (planning, deep-pull doc queries, evaluation) are
     dispatched in batches of up to 10 concurrent calls.
  3. At the end of all deep-pull iterations, every per-question file is
     coalesced into one large Plan_Docs/question_plans.json.
"""

import io
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage
from google import genai
from google.genai import types

# Import extraction helpers from the parallelized summarize module
# Support both direct execution (python parallelized/plan_questions_parallelized.py)
# and package import (import parallelized.plan_questions_parallelized).
try:
    from summarize_documents_parallelized import (
        get_metadata_store,
        extract_pdf_text,
        extract_docx_text,
        extract_excel_text,
        BUCKET_NAME,
        PREFIX,
    )
except ModuleNotFoundError:
    from parallelized.summarize_documents_parallelized import (
        get_metadata_store,
        extract_pdf_text,
        extract_docx_text,
        extract_excel_text,
        BUCKET_NAME,
        PREFIX,
    )

# ── Config ───────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

CREDENTIALS_PATH        = os.getenv("CREDENTIALS_PATH")
QUESTIONS_PATH          = os.getenv("QUESTIONS_PATH")
VERTEX_CREDENTIALS_PATH = os.getenv("VERTEX_CREDENTIALS_PATH")
VERTEX_PROJECT          = os.getenv("VERTEX_PROJECT", "cciscrape")
VERTEX_LOCATION         = os.getenv("VERTEX_LOCATION", "us-central1")

# Resolve Vertex AI service-account JSON relative to project root
_project_root = Path(__file__).resolve().parent.parent
_vertex_cred  = Path(VERTEX_CREDENTIALS_PATH)
if not _vertex_cred.is_absolute():
    _vertex_cred = _project_root / _vertex_cred
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_vertex_cred)

_q_parts  = QUESTIONS_PATH.split("/", 1)
Q_BUCKET  = _q_parts[0]
Q_BLOB    = _q_parts[1]

MODEL_NAME = "gemini-2.5-flash"

# Output paths
PARTS_DIR  = Path(__file__).parent / "Plan_Docs/question_parts"
OUT_PATH   = Path(__file__).parent / "Plan_Docs/question_plans.json"

MAX_WORKERS = 10  # concurrent Gemini calls

# MIME types Gemini can ingest natively as inline bytes
_NATIVE_MIME = {
    ".pdf": "application/pdf",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ensure_dirs():
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def _part_path(idx: int) -> Path:
    """Path for the per-question JSON shard (0-indexed internally, but named 1-based)."""
    return PARTS_DIR / f"q_{idx + 1:03d}.json"


def _write_part(idx: int, entry: dict):
    """Persist a single question entry to its own file."""
    _part_path(idx).write_text(json.dumps(entry, indent=2, ensure_ascii=False))


def _read_part(idx: int) -> dict:
    p = _part_path(idx)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _coalesce_parts(total: int) -> list[dict]:
    """Read every per-question file and combine into one ordered list."""
    plans = []
    for i in range(total):
        p = _part_path(i)
        if p.exists():
            plans.append(json.loads(p.read_text()))
        else:
            plans.append({})
    return plans


def _write_coalesced(plans: list[dict]):
    OUT_PATH.write_text(json.dumps(plans, indent=2, ensure_ascii=False))


# ── GCS / Question download ─────────────────────────────────────────────────

def download_questions() -> list[str]:
    client = storage.Client()
    bucket = client.bucket(Q_BUCKET)
    blob   = bucket.blob(Q_BLOB)
    data   = blob.download_as_bytes()

    xls = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
    df  = xls.parse(xls.sheet_names[0])

    col = next(
        (c for c in df.columns if str(c).strip().lower() in ("question", "questions")),
        df.columns[0],
    )
    questions = df[col].dropna().astype(str).str.strip().tolist()
    print(f"  Loaded {len(questions)} questions from column '{col}'.")
    return questions


def _build_gcs_blob_map(gcs_client) -> dict[str, object]:
    bucket = gcs_client.bucket(BUCKET_NAME)
    blobs  = bucket.list_blobs(prefix=PREFIX)
    result = {}
    for b in blobs:
        if not b.name.endswith("/"):
            result[Path(b.name).name] = b
    return result


# ── Gemini planning call (batch-friendly) ────────────────────────────────────

def build_plan_with_gemini(questions: list[str], metadata: list[dict]) -> list[dict]:
    """Single Gemini call → returns the parsed plan list (same as original)."""
    client = genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)

    metadata_text  = json.dumps(metadata, indent=2)
    questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

    prompt = f"""You are a research analyst. Below is a metadata table describing a set of documents,
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

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0].strip()

    return json.loads(raw)


# ── Document querying (parallelizable) ───────────────────────────────────────

def _query_document(gemini_client, doc_name: str, blob_bytes: bytes,
                    original_question: str, what_to_find: str) -> str:
    ext = Path(doc_name).suffix.lower()
    mime = _NATIVE_MIME.get(ext)

    prompt = (
        f"You are a precise legal and financial research assistant.\n\n"
        f"RESEARCH QUESTION: \"{original_question}\"\n\n"
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

    if mime:
        part = types.Part.from_bytes(data=blob_bytes, mime_type=mime)
        contents = [part, prompt]
    else:
        if ext == ".docx":
            text, _ = extract_docx_text(blob_bytes)
        elif ext in (".xlsx", ".xls"):
            text, _ = extract_excel_text(blob_bytes, ext)
        else:
            return f"NOT FOUND: unsupported file type {ext}"
        contents = [
            f"DOCUMENT CONTENTS ({doc_name}):\n{text}\n\nQUESTION:\n{prompt}"
        ]

    response = gemini_client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
    )
    return response.text.strip()


def _answer_directly_from_doc(gemini_client, doc_name: str,
                              blob_bytes: bytes, question: str) -> str:
    ext  = Path(doc_name).suffix.lower()
    mime = _NATIVE_MIME.get(ext)

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

    if mime:
        part     = types.Part.from_bytes(data=blob_bytes, mime_type=mime)
        contents = [part, prompt]
    else:
        if ext == ".docx":
            text, _ = extract_docx_text(blob_bytes)
        elif ext in (".xlsx", ".xls"):
            text, _ = extract_excel_text(blob_bytes, ext)
        else:
            return f"NOT FOUND: unsupported file type {ext}"
        contents = [f"DOCUMENT CONTENTS ({doc_name}):\n{text}\n\nQUESTION:\n{prompt}"]

    response = gemini_client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
    )
    return response.text.strip()


# ── Evaluation / refinement (parallelizable) ────────────────────────────────

def _evaluate_answer(gemini_client, question: str,
                     answer_found: str | None,
                     data_found: list[str]) -> dict:
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
        f"is NOT sufficient — even if the overall logic is sound. "
        f"The 'reason' must name the specific unverified claim.\n"
        f"- Respond with ONLY valid JSON (no markdown fences), exactly this shape:\n"
        f'{{"sufficient": true/false, "reason": "...", "synthesised_answer": "..."}}\n'
        f"- If sufficient=true, write a clean, concise answer in 'synthesised_answer' "
        f"using only the information above.\n"
        f"- If sufficient=false, set 'synthesised_answer' to null and explain why in 'reason'."
    )

    response = gemini_client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0].strip()
    return json.loads(raw)


def _refine_plan(gemini_client, question: str, existing_plan: list[dict],
                 data_found: list[str], metadata: list[dict]) -> list[dict]:
    already_queried = [next(iter(p)) for p in existing_plan]
    metadata_text   = json.dumps(metadata, indent=2)
    plan_text       = json.dumps(existing_plan, indent=2)
    data_text       = "\n\n---\n\n".join(data_found) if data_found else "(none yet)"

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
        f"Step 1 — Identify unverified sub-claims:\n"
        f"  Read through the data extracted so far. Find every factual assertion that "
        f"uses hedging language (likely, probably, approximately, may, might, assumed, "
        f"unclear) or that is stated without a direct document citation. "
        f"List these as the claims that MUST be verified before the question can be closed.\n\n"
        f"Step 2 — Map each unverified claim to a document:\n"
        f"  For each unverified claim, check the document library above. "
        f"If any document could plausibly contain the definitive value or fact, "
        f"add it to the plan with an instruction to find that specific piece of evidence.\n\n"
        f"Step 3 — Add any other missing coverage:\n"
        f"  You MAY also add entries for documents that haven't been queried yet "
        f"if they are relevant, or re-add already-queried documents with a DIFFERENT "
        f"search target. Do NOT repeat search targets that already failed.\n\n"
        f"Return ONLY a valid JSON array (no markdown fences) of plan items:\n"
        f"[{{\"<document_name>\": \"what specifically to look for\"}}, ...]\n"
        f"List only the NEW items to execute — do not re-list items already covered above."
    )

    response = gemini_client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0].strip()
    return json.loads(raw)


# ── Parallel deep-pull for a single question ─────────────────────────────────

def _deep_pull_one(idx: int, entry: dict, gcs_blob_map: dict,
                   gemini_client, metadata: list[dict]) -> dict:
    """
    Execute one full deep-pull iteration for a single question (by index).
    Writes intermediate progress to the per-question file.
    Returns the updated entry dict.
    """
    question     = entry.get("question", "")
    answer_found = entry.get("answer_found")
    plan         = entry.get("plan", [])
    data_found   = entry.get("data_found", [])
    q_label      = f"Q{idx+1}"

    # ── Already answered — verify adequacy ──────────────────────────────────
    if answer_found:
        print(f"  {q_label}: answer_found present — checking adequacy...")
        verdict = _evaluate_answer(gemini_client, question, answer_found, data_found)
        if verdict["sufficient"]:
            print(f"    {q_label} answer is sufficient — skipping.")
            return entry
        else:
            print(f"    {q_label} answer insufficient: {verdict['reason']}")
            answer_found = None

    # ── Check existing data_found ───────────────────────────────────────────
    if data_found:
        print(f"  {q_label}: checking existing data_found ({len(data_found)} item(s))...")
        verdict = _evaluate_answer(gemini_client, question, None, data_found)
        if verdict["sufficient"]:
            entry["answer_found"] = verdict["synthesised_answer"]
            _write_part(idx, entry)
            print(f"    {q_label} data_found sufficient.")
            return entry

        # Retry synthesis
        retry_verdict = _evaluate_answer(gemini_client, question, None, data_found)
        if retry_verdict["sufficient"]:
            entry["answer_found"] = retry_verdict["synthesised_answer"]
            _write_part(idx, entry)
            print(f"    {q_label} retry synthesis succeeded.")
            return entry

        # Refine plan
        print(f"    {q_label} refining plan...")
        refined_items = _refine_plan(gemini_client, question, plan, data_found, metadata)
        if refined_items:
            plan = plan + refined_items
            entry["plan"] = plan
            _write_part(idx, entry)

    # ── No plan yet — generate one ──────────────────────────────────────────
    if not plan:
        print(f"  {q_label}: generating plan...")
        new_plans = build_plan_with_gemini([question], metadata)
        entry.update(new_plans[0])
        plan = entry.get("plan", [])
        _write_part(idx, entry)
        print(f"    {q_label} plan generated ({len(plan)} source(s)).")

    # ── Execute plan — parallel document queries ────────────────────────────
    start_idx = len(data_found)
    remaining_items = list(enumerate(plan))[start_idx:]

    if remaining_items:
        print(f"  {q_label}: querying {len(remaining_items)} document(s) in parallel...")

        def _query_one_plan_item(item_idx_and_plan):
            i, plan_item = item_idx_and_plan
            doc_name     = next(iter(plan_item))
            what_to_find = plan_item[doc_name]
            blob = gcs_blob_map.get(doc_name)
            if blob is None:
                return (i, f"NOT FOUND IN GCS: {doc_name}")
            try:
                blob_bytes = blob.download_as_bytes()
                result = _query_document(
                    gemini_client, doc_name, blob_bytes, question, what_to_find
                )
                return (i, result)
            except Exception as e:
                return (i, f"ERROR: {e}")

        # Dispatch queries in parallel (capped at MAX_WORKERS)
        results_by_idx = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(_query_one_plan_item, item): item
                for item in remaining_items
            }
            for fut in as_completed(futures):
                i, result = fut.result()
                results_by_idx[i] = result

        # Merge results in order
        for i, _ in remaining_items:
            data_found.append(results_by_idx[i])

        entry["data_found"] = data_found
        _write_part(idx, entry)

    # ── Post-execution synthesis ────────────────────────────────────────────
    final_verdict = _evaluate_answer(gemini_client, question, None, data_found)
    if final_verdict["sufficient"]:
        entry["answer_found"] = final_verdict["synthesised_answer"]
        _write_part(idx, entry)
        print(f"    {q_label} post-execution synthesis succeeded.")
        return entry

    # Refine and possibly fallback
    print(f"    {q_label} synthesis insufficient — checking plan refinement...")
    refined_items = _refine_plan(gemini_client, question, plan, data_found, metadata)

    if refined_items:
        print(f"    {q_label}: {len(refined_items)} new plan item(s) — deferring to next round.")
        entry["plan"] = plan + refined_items
        _write_part(idx, entry)
    else:
        # Direct fallback — parallel
        print(f"    {q_label}: plan unchanged — direct document fallback (parallel)...")
        direct_answers = []

        def _direct_one(plan_item):
            doc_name = next(iter(plan_item))
            blob     = gcs_blob_map.get(doc_name)
            if blob is None:
                return f"NOT IN BUCKET: {doc_name}"
            try:
                blob_bytes = blob.download_as_bytes()
                return f"[Direct — {doc_name}]\n" + _answer_directly_from_doc(
                    gemini_client, doc_name, blob_bytes, question
                )
            except Exception as e:
                return f"ERROR ({doc_name}): {e}"

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            direct_answers = list(pool.map(_direct_one, plan))

        combined_verdict = _evaluate_answer(gemini_client, question, None, direct_answers)
        if combined_verdict["sufficient"]:
            entry["answer_found"] = combined_verdict["synthesised_answer"]
        else:
            best = next(
                (a for a in direct_answers if not a.startswith("NOT") and not a.startswith("ERROR")),
                direct_answers[0] if direct_answers else "No answer could be determined."
            )
            entry["answer_found"] = best

        entry["data_found"] = data_found + direct_answers
        _write_part(idx, entry)

    return entry


# ── Orchestrators ────────────────────────────────────────────────────────────

def main():
    """Phase 1: Download questions, build plans, write each question to its own file."""
    _ensure_dirs()

    print("Loading metadata store...")
    metadata = get_metadata_store()
    print(f"  {len(metadata)} documents in metadata store.")

    print("Downloading questions from GCS...")
    questions = download_questions()

    print(f"Sending {len(questions)} questions to Gemini for planning...")
    plans = build_plan_with_gemini(questions, metadata)

    # Write each question to its own file
    for i, p in enumerate(plans):
        _write_part(i, p)

    # Also write coalesced file
    _write_coalesced(plans)

    print(f"\nWrote {len(plans)} question plans → individual files + {OUT_PATH}\n")

    print(f"{'#':<4} {'STATUS':<12} QUESTION")
    print("-" * 80)
    for i, p in enumerate(plans, 1):
        q  = p.get("question", "")[:65]
        af = p.get("answer_found")
        if af:
            status = "ANSWERED"
        else:
            n = len(p.get("plan", []))
            status = f"PLAN({n})"
        print(f"{i:<4} {status:<12} {q}")


def run_deep_dive():
    """
    Phase 2: Parallel deep-pull across all questions.

    For each round, dispatches up to MAX_WORKERS concurrent question-level
    deep-pulls.  Each question's Gemini doc queries also run in parallel
    (capped at MAX_WORKERS), so total Gemini concurrency can spike but is
    bounded by the thread pool within each question.
    """
    _ensure_dirs()

    # Load all per-question files (or fall back to coalesced file)
    if OUT_PATH.exists():
        plans = json.loads(OUT_PATH.read_text())
    else:
        raise FileNotFoundError(f"No question plans found at {OUT_PATH}. Run main() first.")

    total = len(plans)
    # Ensure per-question files exist
    for i, p in enumerate(plans):
        if not _part_path(i).exists():
            _write_part(i, p)

    print(f"Starting parallel deep dive on {total} question(s)...\n")

    gemini_client = genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)
    gcs_client    = storage.Client()
    print("Building GCS blob index...")
    gcs_blob_map  = _build_gcs_blob_map(gcs_client)
    print(f"  {len(gcs_blob_map)} documents indexed.\n")

    metadata = get_metadata_store()

    max_rounds = 5
    for round_num in range(1, max_rounds + 1):
        # Re-read per-question files to pick up mutations
        plans = [_read_part(i) for i in range(total)]

        unanswered = [i for i, p in enumerate(plans) if not p.get("answer_found")]
        if not unanswered:
            print(f"✓ All {total} questions answered after round {round_num - 1}.")
            break

        print(f"{'='*60}")
        print(f"  ROUND {round_num}/{max_rounds}  —  {len(unanswered)} unanswered question(s)")
        print(f"{'='*60}\n")

        # Dispatch deep-pulls in parallel (up to MAX_WORKERS questions at a time)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(
                    _deep_pull_one, i, plans[i], gcs_blob_map, gemini_client, metadata
                ): i
                for i in unanswered
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    updated_entry = fut.result()
                    plans[i] = updated_entry
                    _write_part(i, updated_entry)
                    status = "ANSWERED" if updated_entry.get("answer_found") else "PENDING"
                    print(f"  Q{i+1} → {status}")
                except Exception as e:
                    print(f"  Q{i+1} → ERROR: {e}")

        still_unanswered = [i for i, p in enumerate(plans) if not p.get("answer_found")]
        answered_count   = total - len(still_unanswered)
        print(f"\nAfter round {round_num}: {answered_count}/{total} answered.\n")

        if not still_unanswered:
            print("✓ All questions answered!")
            break
    else:
        remaining = [plans[i].get("question", "") for i in range(total) if not plans[i].get("answer_found")]
        print(f"\n⚠  Max rounds ({max_rounds}) reached. {len(remaining)} question(s) still unanswered:")
        for q in remaining:
            print(f"   - {q}")

    # ── Coalesce all per-question files into one large JSON ──────────────────
    print("\nCoalescing per-question files into final output...")
    final_plans = _coalesce_parts(total)
    _write_coalesced(final_plans)
    print(f"Wrote coalesced output → {OUT_PATH}")

    # Print state
    get_state()


def get_state():
    """Print each question and its current answer."""
    if OUT_PATH.exists():
        plans = json.loads(OUT_PATH.read_text())
    else:
        # Try coalescing from parts
        parts = sorted(PARTS_DIR.glob("q_*.json")) if PARTS_DIR.exists() else []
        plans = [json.loads(p.read_text()) for p in parts]

    print(f"\n{'#':<4} {'QUESTION':<60} ANSWER")
    print("=" * 120)
    for i, p in enumerate(plans, 1):
        question = p.get("question", "")
        answer   = p.get("answer_found") or "(no answer yet)"
        print(f"\n[{i}] {question}")
        print(f"    → {answer}")
        print("-" * 120)
    print()


def run_all(max_rounds: int = 5):
    start=time.time()
    """
    Full end-to-end orchestration (parallelized):

    1. Build (or load) the metadata store via summarize_documents_parallelized.
    2. Build (or load) the question plan (each question → own file).
    3. Loop up to max_rounds times, running parallel deep-pulls for unanswered Qs.
    4. Coalesce all per-question files into one JSON at the end.
    """
    import summarize_documents_parallelized as sd


    _ensure_dirs()

    # ── Step 1: Metadata store ───────────────────────────────────────────────
    csv_path = Path(__file__).parent / "Plan_Docs/metadata_store.csv"
    db_path  = sd.DB_PATH

    if csv_path.exists() or db_path.exists():
        print("✓ Metadata store already exists — loading.")
        metadata = get_metadata_store()
        print(f"  {len(metadata)} documents loaded.\n")
    else:
        print("▶ Metadata store not found — running summarize_documents_parallelized...\n")
        sd.main()
        metadata = get_metadata_store()
        print(f"\n✓ Metadata store built: {len(metadata)} documents.\n")

    # ── Step 2: Question plan (per-question files) ───────────────────────────
    parts_exist = PARTS_DIR.exists() and any(PARTS_DIR.glob("q_*.json"))
    if parts_exist or OUT_PATH.exists():
        print("✓ Question plan already exists — loading.")
        if OUT_PATH.exists():
            plans = json.loads(OUT_PATH.read_text())
        else:
            plans = _coalesce_parts(len(list(PARTS_DIR.glob("q_*.json"))))
        # Ensure per-question files exist
        for i, p in enumerate(plans):
            if not _part_path(i).exists():
                _write_part(i, p)
        print(f"  {len(plans)} questions loaded.\n")
    else:
        print("▶ Question plan not found — downloading questions and planning...\n")
        questions = download_questions()
        plans = build_plan_with_gemini(questions, metadata)
        for i, p in enumerate(plans):
            _write_part(i, p)
        _write_coalesced(plans)
        print(f"\n✓ Question plan written: {len(plans)} questions.\n")

    total = len(plans)

    # ── Step 3: Parallel deep-dive loop ──────────────────────────────────────
    gemini_client = genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)
    gcs_client    = storage.Client()
    print("Building GCS blob index...")
    gcs_blob_map  = _build_gcs_blob_map(gcs_client)
    print(f"  {len(gcs_blob_map)} documents indexed.\n")

    for round_num in range(1, max_rounds + 1):
        plans = [_read_part(i) for i in range(total)]

        unanswered = [i for i, p in enumerate(plans) if not p.get("answer_found")]
        if not unanswered:
            print(f"✓ All {total} questions answered — stopping after round {round_num - 1}.")
            break

        print(f"{'='*60}")
        print(f"  ROUND {round_num}/{max_rounds}  —  {len(unanswered)} unanswered question(s)")
        print(f"{'='*60}\n")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(
                    _deep_pull_one, i, plans[i], gcs_blob_map, gemini_client, metadata
                ): i
                for i in unanswered
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    updated_entry = fut.result()
                    plans[i] = updated_entry
                    _write_part(i, updated_entry)
                    status = "ANSWERED" if updated_entry.get("answer_found") else "PENDING"
                    print(f"  Q{i+1} → {status}")
                except Exception as e:
                    print(f"  Q{i+1} → ERROR: {e}")

        still_unanswered = [i for i, p in enumerate(plans) if not p.get("answer_found")]
        answered_count   = total - len(still_unanswered)
        print(f"\nAfter round {round_num}: {answered_count}/{total} answered.\n")

        if not still_unanswered:
            print("✓ All questions answered!")
            break
    else:
        remaining = [plans[i].get("question", "") for i in range(total) if not plans[i].get("answer_found")]
        print(f"\n⚠  Max rounds ({max_rounds}) reached. {len(remaining)} question(s) still unanswered:")
        for q in remaining:
            print(f"   - {q}")

    # ── Coalesce all per-question files ──────────────────────────────────────
    print("\nCoalescing per-question files into final output...")
    final_plans = _coalesce_parts(total)
    _write_coalesced(final_plans)
    print(f"Wrote coalesced output → {OUT_PATH}")
    endtime=time.time()
    print()
    get_state()
    print(f"\nTotal execution time: {endtime - start:.2f} seconds, {max_rounds} loops")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--runAll" in sys.argv:
        run_all()
    elif "--deep" in sys.argv:
        run_deep_dive()
    elif "--getState" in sys.argv:
        get_state()
    else:
        main()
