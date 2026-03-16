# Document Intelligence Pipeline

An agentic, multi-pass research system that ingests a document library from Google Cloud Storage, builds semantic summaries, then autonomously plans and executes targeted deep-dives to answer complex research questions — looping until every question is resolved or a maximum iteration budget is exhausted.

---

## Overview

Most RAG systems treat retrieval as a single-shot lookup. This pipeline treats answering a question as a **research process**: it first understands what it has, forms a plan, executes that plan against the raw documents, evaluates whether the results are good enough, and iterates if not.

```
┌─────────────────────────────────────────────────────────────────┐
│                        run_all()                                │
│                                                                 │
│  1. Summarise Documents        →   metadata_store (CSV + DB)   │
│  2. Download Questions                                          │
│  3. Build Question Plans       →   question_plans.json         │
│  4. Loop (≤5 rounds):                                           │
│       for each unanswered question → goDeep()                  │
│  5. Print final state                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture

### Stage 1 — Document Summarisation (`summarize_documents.py`)

For each file in the GCS bucket under `DATA_PATH`:

| File Type | Extraction Strategy |
|-----------|-------------------|
| **PDF** | Raw text extraction (PyMuPDF). Falls back to OCR (pytesseract) if text layer is empty or sparse. First 3 pages only. |
| **DOCX** | Paragraph extraction split on hard page breaks. First 3 pages only. |
| **XLSX / XLS** | Full file — all sheets, up to 30,000 characters. |

Each extracted chunk is sent to **Gemini 2.5 Flash** with a prompt asking it to guess what the full document is about. Results are stored in:
- `metadata_store.db` (SQLite, for fast lookups)
- `metadata_store.csv` (human-readable export)

Already-processed documents are skipped on re-runs, making the stage idempotent and incremental.

---

### Stage 2 — Question Planning (`plan_questions.py → build_plan_with_gemini`)

Questions are downloaded from a GCS Excel file (`QUESTIONS_PATH`). All questions and the full metadata table are sent to Gemini in a **single call**.

The model reasons over the summaries and produces a structured JSON plan:

```json
[
  {
    "question": "What metrics helped CCI determine anticompetitiveness?",
    "answer_found": null,
    "plan": [
      { "CCI Combination Guide.pdf": "Identify assessment criteria and thresholds..." },
      { "1022 - Air India _ Tata SIA.pdf": "Review competitive analysis framework..." }
    ]
  },
  {
    "question": "How many SCOTUS cases are in the set?",
    "answer_found": "There are 5 SCOTUS cases: Bell Atlantic v. Twombly ...",
    "plan": []
  }
]
```

Questions answerable purely from the summaries (e.g. counting documents, identifying document types) are resolved immediately with `answer_found` set — no document access needed.

---

### Stage 3 — Deep Research (`goDeep`)

`goDeep(question_number)` is the core agent. For a given question it follows this decision tree:

```
answer_found present?
    │
    ├─ YES → Evaluate sufficiency via Gemini
    │            │
    │            ├─ Sufficient  → Return ✓
    │            └─ Insufficient → clear answer, fall through
    │
    └─ NO
         │
         data_found present?
         │
         ├─ YES → Evaluate sufficiency
         │            │
         │            ├─ Sufficient  → Synthesise answer, return ✓
         │            │
         │            └─ Insufficient → Retry synthesis
         │                                │
         │                                ├─ Now sufficient → return ✓
         │                                │
         │                                └─ Still insufficient
         │                                     → _refine_plan():
         │                                       add new docs / new search
         │                                       targets to the plan
         │
         plan present?
         │
         ├─ NO  → build_plan_with_gemini() → write plan, fall through
         │
         └─ YES → Execute each plan item sequentially:
                    for each document:
                      upload to Gemini (native PDF bytes or extracted text)
                      prompt: extract relevant passages + attempt partial answer
                      append result to data_found
                      persist to question_plans.json
```

#### Document Query Prompt

Each per-document Gemini call returns two clearly delineated sections:

```
EXTRACTED PASSAGES:
[DocumentName.pdf, Page 4] "exact quoted text..."

ATTEMPTED ANSWER:
<Gemini's best answer to the original question using only this document>
```

This means every entry in `data_found` is self-contained — it carries both evidence and a partial answer attempt, which the subsequent evaluation step can immediately reason over.

---

### Stage 4 — Iterative Orchestration (`run_all`)

```
Round 1: goDeep on all unanswered questions
Round 2: goDeep on questions still unanswered after Round 1
...
Round N: stop when all answered OR max_rounds reached (default: 5)
```

Each round only processes questions without a confirmed `answer_found`. The loop terminates early as soon as all questions are resolved. After each individual document query, state is persisted to disk — crashes and interruptions are safe to resume from.

---

### Plan Refinement (`_refine_plan`)

When `data_found` exists but is still insufficient after two synthesis attempts, the refinement agent is invoked. It receives:

- The original question
- The plan that was already executed
- All data extracted so far
- The full document metadata library

It returns **only new plan items** — either documents not yet queried, or the same documents with meaningfully different search targets. These are appended to the existing plan and executed in the same `goDeep` pass.

---

## File Structure

```
.
├── summarize_documents.py   # Stage 1: GCS ingestion, extraction, summarisation
├── plan_questions.py        # Stages 2–4: planning, goDeep, orchestration
├── Plan_Docs/
│   └── question_plans.json  # Live research state (questions, plans, data, answers)
├── metadata_store.csv       # Human-readable document summaries
├── metadata_store.db        # SQLite store (fast lookups)
└── .env                     # Credentials and paths (not committed)
```

---

## Setup

### Prerequisites

```bash
pip install google-cloud-storage google-genai pymupdf python-docx \
            openpyxl pandas pytesseract pillow python-dotenv
```

Tesseract OCR must also be installed at the system level (e.g. `brew install tesseract`).

### Environment Variables (`.env`)

```
CREDENTIALS_PATH=path/to/gcp-service-account.json
DATA_PATH=your-bucket/path/to/documents
QUESTIONS_PATH=your-bucket/path/to/questions.xlsx
GEMINI_API_KEY=your-gemini-api-key
```

---

## Usage

| Command | Description |
|---------|-------------|
| `python plan_questions.py --runAll` | **Full pipeline** — summarise, plan, deep-dive, iterate |
| `python plan_questions.py --deep` | Run deep-dive only (assumes plan already exists) |
| `python plan_questions.py` | Build/refresh the question plan only |
| `python plan_questions.py --getState` | Print current Q&A state from `question_plans.json` |
| `python summarize_documents.py` | Rebuild the metadata store only |

---

## Design Principles

**Incremental & resumable** — every stage checks for existing state before recomputing. A crash mid-run loses at most one document query.

**Separation of evidence and synthesis** — raw extracted passages and per-document answer attempts are stored separately from the final synthesised answer, making it easy to audit how an answer was reached.

**Single planning call, targeted execution** — the planning stage uses one Gemini call across all questions and all metadata, allowing cross-document reasoning about which sources are relevant. Execution is then narrowly scoped to only the documents the planner identified.

**Self-correcting** — when data is insufficient, the system doesn't just fail. It attempts re-synthesis, and if that still fails, it asks the model to revise the research plan before retrying.
