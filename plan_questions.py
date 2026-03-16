import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
plan_questions.py

Downloads the questions spreadsheet from GCS (QUESTIONS_PATH), loads the
metadata store built by summarize_documents.py, then makes a single Gemini
call that produces a research plan for every question.

Output is written to question_plans.json:
[
  {
    "question": "...",
    "answer_found": "<answer string or null>",
    "plan": [
      {"Document Name": "what to look for inside it"},
      ...
    ]
  },
  ...
]
"""

import io
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage
from google import genai
from google.genai import types

from summarize_documents import (
    get_metadata_store,
    extract_pdf_text,
    extract_docx_text,
    extract_excel_text,
    BUCKET_NAME,
    PREFIX,
)

# ── Config ───────────────────────────────────────────────────────────────────
load_dotenv()

CREDENTIALS_PATH = os.getenv("CREDENTIALS_PATH")
QUESTIONS_PATH   = os.getenv("QUESTIONS_PATH")   # e.g. "vidhi_core/test_questions/file.xlsx"
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

_q_parts  = QUESTIONS_PATH.split("/", 1)
Q_BUCKET  = _q_parts[0]
Q_BLOB    = _q_parts[1]

MODEL_NAME = "gemini-2.5-flash"   # change to gemini-3.0-flash once available in your project
OUT_PATH   = Path(__file__).parent / "Plan_Docs/question_plans.json"


# ── Helpers ──────────────────────────────────────────────────────────────────

def download_questions() -> list[str]:
    """Download the questions Excel from GCS and return a flat list of question strings."""
    client = storage.Client()
    bucket = client.bucket(Q_BUCKET)
    blob   = bucket.blob(Q_BLOB)
    data   = blob.download_as_bytes()

    xls = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
    df  = xls.parse(xls.sheet_names[0])

    # Prefer a column literally named "question(s)", otherwise take the first column
    col = next(
        (c for c in df.columns if str(c).strip().lower() in ("question", "questions")),
        df.columns[0],
    )
    questions = df[col].dropna().astype(str).str.strip().tolist()
    print(f"  Loaded {len(questions)} questions from column '{col}'.")
    return questions


def build_plan_with_gemini(questions: list[str], metadata: list[dict]) -> list[dict]:
    """Single Gemini call → returns the parsed plan list."""
    client = genai.Client(api_key=GEMINI_API_KEY)

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

Return a JSON array, one object per question, in the same order as the questions above.
No extra keys. No markdown. No explanation outside the JSON."""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    raw = response.text.strip()
    # Strip markdown code fences if the model adds them despite instructions
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0].strip()

    return json.loads(raw)


# ── Deep research helpers ────────────────────────────────────────────────────

# MIME types Gemini can ingest natively as inline bytes
_NATIVE_MIME = {
    ".pdf": "application/pdf",
}


def _build_gcs_blob_map(gcs_client) -> dict[str, object]:
    """Return {filename: blob} for every file under DATA_PATH in GCS."""
    bucket = gcs_client.bucket(BUCKET_NAME)
    blobs  = bucket.list_blobs(prefix=PREFIX)
    result = {}
    for b in blobs:
        if not b.name.endswith("/"):
            result[Path(b.name).name] = b
    return result


def _query_document(gemini_client, doc_name: str, blob_bytes: bytes,
                    original_question: str, what_to_find: str) -> str:
    """Upload one document to Gemini and ask for targeted extraction with citations."""
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
        # Native file upload (PDF) — Gemini reads full document
        part = types.Part.from_bytes(data=blob_bytes, mime_type=mime)
        contents = [part, prompt]
    else:
        # Non-native format — extract text first and embed as plain text
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


def _evaluate_answer(gemini_client, question: str,
                     answer_found: str,
                     data_found: list[str]) -> dict:
    """
    Ask Gemini whether the existing answer_found / data_found adequately answers
    the question.

    Returns a dict:
      {
        "sufficient": bool,
        "reason": str,              # why it is or isn't sufficient
        "synthesised_answer": str   # populated only when sufficient=True
      }
    """
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
    """
    Given that the current data_found is insufficient, ask Gemini to produce a
    refined research plan — amending what to look for in existing docs and/or
    adding new documents from the metadata store.

    Returns ONLY the new/amended plan items that have not yet been queried
    (i.e. not in existing_plan), so they can be appended and executed.
    """
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
        f"Your task: produce a revised research plan that fills the gaps.\n"
        f"You MAY:\n"
        f"  - Add entries for documents that haven't been queried yet.\n"
        f"  - Re-add already-queried documents if you need to look for DIFFERENT "
        f"information than before (use a clearly different search target).\n"
        f"Do NOT simply repeat the same search targets that already failed.\n\n"
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


def goDeep(question_number: int,
           plans: list[dict]=None,
           gcs_blob_map: dict=None ,
           gemini_client=None) -> str:
    """
    Perform deep research for a single question (1-indexed).

    Behaviour:
      - answer_found already set → return JSON as-is (answer is sufficient)
      - no answer, no plan      → generate a plan via Gemini, store it, return JSON
      - no answer, has plan     → query each document in the plan, collect data_found,
                                  return updated JSON string

    Returns the JSON string for the question's entry (updated in-place inside `plans`).
    """
    idx = question_number - 1

    # ── Load state ──────────────────────────────────────────────────────────
    if plans is None:
        plans = json.loads(OUT_PATH.read_text())

    if idx < 0 or idx >= len(plans):
        raise IndexError(f"Question number {question_number} out of range (1–{len(plans)}).")

    entry = plans[idx]

    # ── Lazy-init shared objects ─────────────────────────────────────────────
    if gemini_client is None:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    gcs_client = storage.Client()
    if gcs_blob_map is None:
        print("    Building GCS blob index...")
        gcs_blob_map = _build_gcs_blob_map(gcs_client)

    question     = entry.get("question", "")
    answer_found = entry.get("answer_found")
    plan         = entry.get("plan", [])
    data_found   = entry.get("data_found", [])

    # ── Case 1: answer already found — verify it looks sufficient ───────────
    if answer_found:
        print(f"  Q{question_number}: answer_found present — checking adequacy...")
        verdict = _evaluate_answer(gemini_client, question, answer_found, data_found)
        if verdict["sufficient"]:
            print(f"    Answer is sufficient — skipping.")
            return json.dumps(entry, indent=2, ensure_ascii=False)
        else:
            print(f"    Answer deemed insufficient: {verdict['reason']}")
            answer_found = None   # fall through to data_found / plan checks

    # ── Case 1b: no (or rejected) answer — check existing data_found ────────
    if data_found:
        print(f"  Q{question_number}: checking existing data_found ({len(data_found)} item(s))...")
        verdict = _evaluate_answer(gemini_client, question, None, data_found)
        if verdict["sufficient"]:
            synthesised = verdict["synthesised_answer"]
            print(f"    data_found is sufficient — synthesising answer.")
            entry["answer_found"] = synthesised
            plans[idx] = entry
            OUT_PATH.write_text(json.dumps(plans, indent=2, ensure_ascii=False))
            return json.dumps(entry, indent=2, ensure_ascii=False)

        # ── data_found exists but is insufficient — retry synthesis harder ──
        print(f"    data_found insufficient ({verdict['reason']}) — retrying synthesis...")
        retry_verdict = _evaluate_answer(gemini_client, question, None, data_found)
        if retry_verdict["sufficient"]:
            print(f"    Retry synthesis succeeded.")
            entry["answer_found"] = retry_verdict["synthesised_answer"]
            plans[idx] = entry
            OUT_PATH.write_text(json.dumps(plans, indent=2, ensure_ascii=False))
            return json.dumps(entry, indent=2, ensure_ascii=False)

        # ── Still insufficient — refine the plan ────────────────────────────
        print(f"    Retry synthesis also insufficient — refining plan...")
        metadata = get_metadata_store()
        refined_items = _refine_plan(
            gemini_client, question, plan, data_found, metadata
        )
        if refined_items:
            print(f"    Plan refined: {len(refined_items)} new item(s) added.")
            plan = plan + refined_items
            entry["plan"] = plan
            plans[idx] = entry
            OUT_PATH.write_text(json.dumps(plans, indent=2, ensure_ascii=False))
        else:
            print(f"    Refinement returned no new items — proceeding with existing plan.")

    # ── Case 2: no plan yet — generate one ──────────────────────────────────
    if not plan:
        print(f"  Q{question_number}: No plan found — generating...")
        metadata = get_metadata_store()
        new_plans = build_plan_with_gemini([question], metadata)
        entry.update(new_plans[0])
        plan = entry.get("plan", [])
        plans[idx] = entry
        OUT_PATH.write_text(json.dumps(plans, indent=2, ensure_ascii=False))
        print(f"    Plan generated ({len(plan)} source(s)).")
        # Fall through to execute the plan immediately

    # ── Case 3: has plan — query each document ───────────────────────────────
    # data_found already loaded above; resume from where we left off
    start_idx = len(data_found)

    for i, plan_item in enumerate(plan):
        if i < start_idx:
            continue  # already done in a previous run

        doc_name    = next(iter(plan_item))
        what_to_find = plan_item[doc_name]

        print(f"    [{i+1}/{len(plan)}] Querying: {doc_name} ...", end=" ", flush=True)

        blob = gcs_blob_map.get(doc_name)
        if blob is None:
            result = f"NOT FOUND IN GCS: {doc_name}"
            print("NOT IN BUCKET")
        else:
            try:
                blob_bytes = blob.download_as_bytes()
                result = _query_document(
                    gemini_client, doc_name, blob_bytes, question, what_to_find
                )
                print("OK")
            except Exception as e:
                result = f"ERROR: {e}"
                print(f"ERROR: {e}")

        data_found.append(result)
        # Write progress after every document so interruptions don't lose work
        entry["data_found"] = data_found
        plans[idx] = entry
        OUT_PATH.write_text(json.dumps(plans, indent=2, ensure_ascii=False))

    return json.dumps(entry, indent=2, ensure_ascii=False)


def run_deep_dive():
    """
    Orchestrate goDeep for every question in question_plans.json, sequentially.
    Persists updates to the file after each question completes.
    """
    plans = json.loads(OUT_PATH.read_text())
    print(f"Starting deep dive on {len(plans)} question(s)...\n")

    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    gcs_client    = storage.Client()
    print("Building GCS blob index...")
    gcs_blob_map  = _build_gcs_blob_map(gcs_client)
    print(f"  {len(gcs_blob_map)} documents indexed.\n")

    for i, _ in enumerate(plans):
        q_num = i + 1
        print(f"── Question {q_num}: {plans[i].get('question', '')[:70]}")
        result_json = goDeep(
            question_number=q_num,
            plans=plans,
            gcs_blob_map=gcs_blob_map,
            gemini_client=gemini_client,
        )
        # plans is modified in-place by goDeep; reload the serialised result
        plans[i] = json.loads(result_json)
        OUT_PATH.write_text(json.dumps(plans, indent=2, ensure_ascii=False))
        print()

    print(f"Deep dive complete. Results written to {OUT_PATH}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading metadata store...")
    metadata = get_metadata_store()
    print(f"  {len(metadata)} documents in metadata store.")

    print("Downloading questions from GCS...")
    questions = download_questions()

    print(f"Sending {len(questions)} questions to Gemini for planning (single call)...")
    plans = build_plan_with_gemini(questions, metadata)

    OUT_PATH.write_text(json.dumps(plans, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(plans)} question plans → {OUT_PATH}\n")

    # Console summary
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


def get_state():
    """Print each question and its current answer from question_plans.json."""
    plans = json.loads(OUT_PATH.read_text())
    print(f"{'#':<4} {'QUESTION':<60} ANSWER")
    print("=" * 120)
    for i, p in enumerate(plans, 1):
        question = p.get("question", "")
        answer   = p.get("answer_found") or "(no answer yet)"
        # Print question header
        print(f"\n[{i}] {question}")
        print(f"    → {answer}")
        print("-" * 120)
    print()


def run_all(max_rounds: int = 5):
    """
    Full end-to-end orchestration:

    1. Build (or load) the metadata store via summarize_documents.
    2. Build (or load) the question plan by downloading questions and calling Gemini.
    3. Loop up to `max_rounds` times, calling goDeep for every unanswered question.
       Stops early once every question has an answer_found.
    """
    import summarize_documents as sd

    # ── Step 1: Metadata store ───────────────────────────────────────────────
    csv_path = Path(__file__).parent / "metadata_store.csv"
    db_path  = sd.DB_PATH

    if csv_path.exists() or db_path.exists():
        print("✓ Metadata store already exists — loading.")
        metadata = get_metadata_store()
        print(f"  {len(metadata)} documents loaded.\n")
    else:
        print("▶ Metadata store not found — running summarize_documents...\n")
        sd.main()
        metadata = get_metadata_store()
        print(f"\n✓ Metadata store built: {len(metadata)} documents.\n")

    # ── Step 2: Question plan ────────────────────────────────────────────────
    if OUT_PATH.exists():
        print("✓ Question plan already exists — loading.")
        plans = json.loads(OUT_PATH.read_text())
        print(f"  {len(plans)} questions loaded.\n")
    else:
        print("▶ Question plan not found — downloading questions and planning...\n")
        questions = download_questions()
        plans = build_plan_with_gemini(questions, metadata)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(plans, indent=2, ensure_ascii=False))
        print(f"\n✓ Question plan written: {len(plans)} questions.\n")

    # ── Step 3: Deep-dive loop ───────────────────────────────────────────────
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    gcs_client    = storage.Client()
    print("Building GCS blob index...")
    gcs_blob_map  = _build_gcs_blob_map(gcs_client)
    print(f"  {len(gcs_blob_map)} documents indexed.\n")

    for round_num in range(1, max_rounds + 1):
        # Re-read plans from disk so each round picks up any mutations
        plans = json.loads(OUT_PATH.read_text())

        unanswered = [i for i, p in enumerate(plans) if not p.get("answer_found")]
        if not unanswered:
            print(f"✓ All {len(plans)} questions answered — stopping after round {round_num - 1}.")
            break

        print(f"{'='*60}")
        print(f"  ROUND {round_num}/{max_rounds}  —  {len(unanswered)} unanswered question(s)")
        print(f"{'='*60}\n")

        for i in unanswered:
            q_num = i + 1
            print(f"── Q{q_num}: {plans[i].get('question', '')[:70]}")
            result_json = goDeep(
                question_number=q_num,
                plans=plans,
                gcs_blob_map=gcs_blob_map,
                gemini_client=gemini_client,
            )
            plans[i] = json.loads(result_json)
            OUT_PATH.write_text(json.dumps(plans, indent=2, ensure_ascii=False))
            print()

        # Check again after the round
        still_unanswered = [i for i, p in enumerate(plans) if not p.get("answer_found")]
        answered_count   = len(plans) - len(still_unanswered)
        print(f"\nAfter round {round_num}: {answered_count}/{len(plans)} answered.\n")

        if not still_unanswered:
            print("✓ All questions answered!")
            break
    else:
        remaining = [p.get("question", "") for p in plans if not p.get("answer_found")]
        print(f"\n⚠  Max rounds ({max_rounds}) reached. {len(remaining)} question(s) still unanswered:")
        for q in remaining:
            print(f"   - {q}")

    # ── Final state ──────────────────────────────────────────────────────────
    print()
    get_state()



if __name__ == "__main__":
    import sys
    if "--runAll" in sys.argv:
        run_all()
    elif "--deep" in sys.argv:
        run_deep_dive()
    elif "--getState" in sys.argv:
        get_state()
    else:
        main()
