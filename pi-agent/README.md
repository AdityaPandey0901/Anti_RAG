# pi-agent — Pipeline Intelligence Agent

A Gemini function-calling agent that **autonomously orchestrates** the full document research pipeline: ingesting documents, planning research, executing deep-dives, and iterating until every question is answered.

## How it differs from `plan_questions.py`

| Aspect | `plan_questions.py` | `pi-agent` |
|--------|-------------------|------------|
| **Control flow** | Hard-coded `run_all()` loop | Gemini decides what to call next via function calling |
| **Adaptability** | Fixed sequence: summarise → plan → deep-dive × N | Agent inspects state and chooses tools dynamically |
| **Interaction** | CLI flags only | Natural-language instructions (`--ask "..."`) |
| **Extensibility** | Add code to `plan_questions.py` | Add a tool function + declaration — agent discovers it |

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Gemini (Model)                     │
│                                                      │
│  System prompt: "You are a Pipeline Intelligence     │
│  agent. Use these tools to run the research          │
│  pipeline…"                                          │
│                                                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │  Function calling loop (agent.py)               │ │
│  │                                                 │ │
│  │  Turn 1: check_pipeline_state() → state dict    │ │
│  │  Turn 2: summarise_documents()  → status        │ │
│  │  Turn 3: plan_questions()       → status        │ │
│  │  Turn 4: deep_research_all_unanswered() → ...   │ │
│  │  Turn 5: check_pipeline_state() → all answered? │ │
│  │  Turn 6: get_final_report()     → results       │ │
│  │  Turn 7: "Here are the results…" (text)  → DONE │ │
│  └─────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

Each "turn" is one Gemini request → response cycle. The model chooses which tool(s) to call; `agent.py` executes them and feeds results back.

## File Structure

```
pi-agent/
├── __init__.py     # Package marker
├── config.py       # Env vars, paths, constants (reads ../.env)
├── state.py        # State management — metadata store, question plans
├── tools.py        # Tool implementations wrapping the pipeline functions
├── agent.py        # Gemini function-calling loop + tool declarations
├── run.py          # CLI entry point
├── README.md       # This file
└── output/         # Generated at runtime
    ├── metadata_store.db
    ├── metadata_store.csv
    ├── question_plans.json
    └── question_parts/
        ├── q_001.json
        ├── q_002.json
        └── ...
```

## Available Tools

| Tool | Description |
|------|-------------|
| `check_pipeline_state` | Reports metadata/plan readiness, answered/unanswered counts |
| `summarise_documents` | Ingests GCS documents, extracts text, generates summaries |
| `plan_questions` | Downloads questions from GCS, creates Gemini research plans |
| `deep_research(question_number)` | Deep-dives a single question with document queries |
| `deep_research_all_unanswered` | Batch deep-dive for all unanswered questions (parallel) |
| `get_question_detail(question_number)` | Shows a single question's full state |
| `get_final_report` | Formatted report of all questions and answers |

## Usage

```bash
# Activate the same venv as the parent project
source ../venv/bin/activate

# Full autonomous pipeline run (default)
python run.py

# Check current state
python run.py --state

# Print final report
python run.py --report

# Custom instruction
python run.py --ask "Only deep-research question 3 and report the result"

# Quiet mode (just the final answer)
python run.py --quiet
```

## Environment

Uses the same `.env` as the parent project (auto-loaded from `../`). Required vars:

```
CREDENTIALS_PATH=...
DATA_PATH=vidhi_core/Lucio_Test
QUESTIONS_PATH=vidhi_core/test_questions/Revised_Testing_Set_Questions.xlsx
GEMINI_API_KEY=...
VERTEX_CREDENTIALS_PATH=cciscrape-1d50a9ea47ef.json
VERTEX_PROJECT=cciscrape
VERTEX_LOCATION=us-central1
```

## Dependencies

Same as the parent project — no new packages required:

```
google-cloud-storage google-genai pymupdf python-docx
openpyxl pandas pytesseract pillow python-dotenv
```
