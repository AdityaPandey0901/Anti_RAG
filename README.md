# Document Intelligence Pipeline

An agentic, multi-pass research system that ingests a document library from Google Cloud Storage, builds semantic summaries, then autonomously plans and executes targeted deep-dives to answer complex research questions — looping until every question is resolved or a maximum iteration budget is exhausted.

This repo contains **four parallel implementations of the same idea**, built up in stages of sophistication. They share the same core logic (summarise → plan → deep-research → iterate) but differ in how that logic is orchestrated:

| Folder | Orchestration style | Concurrency | Status |
|--------|---------------------|-------------|--------|
| [`/`](.) (root) | Hard-coded Python control flow (`run_all()`) | Sequential | Reference implementation |
| [`parallelized/`](parallelized/) | Same hard-coded control flow | Concurrent (ThreadPoolExecutor, batches of 10) | Performance variant |
| [`pi-agent/`](pi-agent/) | Gemini native function-calling — the model decides what to call next | Sequential/parallel tool calls | Agentic (single-framework) |
| [`lang-chain-agent/`](lang-chain-agent/) | LangGraph ReAct agent, LangChain tools, LangSmith tracing, **dynamic tool creation at runtime** | Sequential/parallel tool calls | Agentic (framework + observability) |

Each one is runnable on its own. See its section below for setup and usage.

---

## Repo Map

```
.
├── run_agent.py                   # Unified entrypoint: pick an engine, point it at docs+questions
├── core/
│   └── sources.py                 # Source-agnostic document/question loading (local dir/CSV or GCS),
│                                   #   shared by all four implementations below
├── summarize_documents.py         # Stage 1 (root): ingestion, extraction, summarisation
├── plan_questions.py              # Stages 2–4 (root): planning, deep-research, orchestration
├── decrypt_and_store.py           # One-off utility: pulls a password-protected zip from GCS,
│                                   #   decrypts it, re-uploads contents unzipped
├── tests/                         # pytest suite (extraction, source loading, run_agent dispatch)
│   └── fixtures/                  # small generated PDF/DOCX/XLSX + questions.csv used by tests
│                                   #   and as a --docs/--questions demo for run_agent.py
├── Plan_Docs/                     # Generated: live research state for the root pipeline
│   ├── metadata_store.csv
│   └── question_plans.json
├── metadata_store.db              # Generated: SQLite mirror of the metadata store
│
├── parallelized/                  # Concurrent rewrite of the root pipeline
│   ├── summarize_documents_parallelized.py
│   ├── plan_questions_parallelized.py
│   └── Plan_Docs/                 # Generated: per-question files coalesced at the end
│
├── pi-agent/                      # Gemini function-calling agent (see pi-agent/README.md)
│   ├── agent.py                   # Function-calling loop + tool declarations
│   ├── tools.py                   # Tool implementations wrapping the pipeline
│   ├── tool_forge.py              # Lets the agent write and register new tools at runtime
│   ├── state.py, config.py, run.py
│   └── output/                    # Generated
│
├── lang-chain-agent/              # LangGraph/LangChain agent
│   ├── agent.py                   # LangGraph ReAct loop, Gemini API, LangSmith tracing
│   ├── tools.py                   # LangChain @tool-decorated pipeline tools
│   ├── tool_forge.py              # Dynamic tool creation (LangChain-flavoured)
│   ├── cache.py, path_cache.py    # LLM response caching / call-path memoisation
│   ├── state.py, config.py, run.py
│   └── output/                    # Generated
│
└── CTF_Stuff/                     # NOT part of the document-intelligence system — see note below
```

> **`CTF_Stuff/`** holds scripts and captured responses from an unrelated side-challenge (a timed trivia/prompt-injection puzzle against a third-party site). It currently contains a live bearer token tied to a personal account and should not be published as-is — see the [Making This Public](#making-this-public) section.

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

## Architecture (root pipeline)

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
    "question": "...",
    "answer_found": null,
    "plan": [
      { "doc_a.pdf": "Identify assessment criteria and thresholds..." },
      { "doc_b.pdf": "Review competitive analysis framework..." }
    ]
  },
  {
    "question": "How many documents are in the set?",
    "answer_found": "There are 5 documents: ...",
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

## `parallelized/` — concurrent rewrite

Same four stages as the root pipeline, restructured for throughput:

- `summarize_documents_parallelized.py` downloads and extracts all blobs concurrently (ThreadPoolExecutor), then dispatches up to 10 concurrent Gemini summarisation calls, coalescing into the metadata store at the end.
- `plan_questions_parallelized.py` writes each question's state to its own file (`Plan_Docs/question_parts/q_00N.json`) so concurrent workers never contend on one JSON file, dispatches planning/deep-pull/evaluation Gemini calls in batches of up to 10, then coalesces everything into one `question_plans.json` at the end.

Useful when the document set or question count is large enough that the sequential root pipeline's wall-clock time matters.

---

## `pi-agent/` — Gemini function-calling agent

Replaces the hard-coded `run_all()` loop with a Gemini agent that decides which pipeline stage to run next via native function calling, and can create brand-new tools at runtime (`tool_forge.py`). Uses Vertex AI credentials. Full details, tool list, and usage in [`pi-agent/README.md`](pi-agent/README.md).

## `lang-chain-agent/` — LangGraph agent

Same agentic idea as `pi-agent`, rebuilt on LangGraph's ReAct pattern with LangChain `@tool`-decorated functions, full LangSmith tracing, an LLM response cache (`cache.py`), and call-path memoisation (`path_cache.py`). Uses the Gemini API directly (no Vertex service account required).

---

## `run_agent.py` — unified entrypoint

Pick an engine, point it at a document set and a question list — each either
a local path or the existing GCS convention — and run it, without needing to
know each implementation's individual CLI:

```bash
python run_agent.py --agent {sequential|parallel|pi-agent|langchain} \
    --docs <local-folder-or-gs://bucket/prefix> \
    --questions <path.csv-or-gs://bucket/questions.xlsx> \
    [--ask "..."] [--state] [--report] [--quiet]   # pi-agent/langchain only
```

`--docs`/`--questions` override `DATA_PATH`/`QUESTIONS_PATH` from `.env` for
that run only; omit them to fall back to whatever's already in `.env`. Under
the hood this just resolves those two values and dispatches to the chosen
engine's own entrypoint (`plan_questions.py --runAll`,
`parallelized/plan_questions_parallelized.py --runAll`, `pi-agent/run.py`,
`lang-chain-agent/run.py`) as a subprocess — each implementation's own CLI
still works standalone exactly as documented below.

## Setup

### Prerequisites

```bash
pip install -r requirements.txt
```

Tesseract OCR must also be installed at the system level (e.g. `brew install tesseract`).

### Environment Variables (`.env`, gitignored)

```
CREDENTIALS_PATH=path/to/gcp-service-account.json
DATA_PATH=your-bucket/path/to/documents
QUESTIONS_PATH=your-bucket/path/to/questions.xlsx
GEMINI_API_KEY=your-gemini-api-key

# Only needed for pi-agent (Vertex AI)
VERTEX_CREDENTIALS_PATH=path/to/vertex-service-account.json
VERTEX_PROJECT=your-gcp-project
VERTEX_LOCATION=us-central1

# Only needed for lang-chain-agent (tracing, optional)
LANGSMITH_API_KEY=your-langsmith-api-key
```

Every implementation reads its own copy of these values from `.env` at the repo root — none of it is checked into git (see [Making This Public](#making-this-public)).

---

## Usage

| Command | Description |
|---------|-------------|
| `python plan_questions.py --runAll` | Root pipeline, full run — summarise, plan, deep-dive, iterate |
| `python plan_questions.py --deep` | Root pipeline, deep-dive only (assumes plan already exists) |
| `python plan_questions.py` | Root pipeline, build/refresh the question plan only |
| `python plan_questions.py --getState` | Print current Q&A state from `question_plans.json` |
| `python summarize_documents.py` | Rebuild the metadata store only |
| `python parallelized/plan_questions_parallelized.py --runAll` | Concurrent full run |
| `python pi-agent/run.py` | Gemini function-calling agent, full autonomous run |
| `python pi-agent/run.py --ask "..."` | Gemini agent, natural-language instruction |
| `python lang-chain-agent/run.py` | LangGraph agent, full autonomous run |
| `python run_agent.py --agent sequential --docs <path> --questions <path>` | Unified entrypoint — see below |

---

## Design Principles

**Incremental & resumable** — every stage checks for existing state before recomputing. A crash mid-run loses at most one document query.

**Separation of evidence and synthesis** — raw extracted passages and per-document answer attempts are stored separately from the final synthesised answer, making it easy to audit how an answer was reached.

**Single planning call, targeted execution** — the planning stage uses one Gemini call across all questions and all metadata, allowing cross-document reasoning about which sources are relevant. Execution is then narrowly scoped to only the documents the planner identified.

**Self-correcting** — when data is insufficient, the system doesn't just fail. It attempts re-synthesis, and if that still fails, it asks the model to revise the research plan before retrying.

---

## Making This Public

Before flipping this repo to public, be aware of what's currently in it:

- `.env` and a GCP service-account key (`cciscrape-*.json`) were committed in this repo's **first commit** and later removed from tracking — but they still exist in git history. Removing them from the working tree (done) does not remove them from history. **Rotate the Gemini/Vertex/LangSmith API keys and the GCP service-account key before making this repo public**, regardless of whether history is rewritten.
- `CTF_Stuff/` contains a live bearer token/cookie (a signed JWT embedding a real name and email) hardcoded in two tracked scripts, plus a large captured response dump. It's unrelated to the document-intelligence system and has been added to `.gitignore` rather than published as-is.
- The various `metadata_store.db` / `.csv` / `question_plans.json` files are generated **outputs of running the pipeline against a real document set** — they may contain summaries/excerpts of the underlying documents. They've been gitignored since they're regenerable and their contents aren't yours to decide to publish unilaterally.

See the accompanying cleanup for exactly what changed.
