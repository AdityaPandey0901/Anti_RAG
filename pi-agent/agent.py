"""
Core agent loop for the pi-agent.

Uses Gemini's automatic function calling to let the model autonomously
decide which pipeline tools to invoke and in what order. The agent
inspects current state, plans, executes, and loops until it decides
the job is done.
"""

from __future__ import annotations

import json
import time
from typing import Any

from google import genai
from google.genai import types

from config import MODEL_NAME, VERTEX_PROJECT, VERTEX_LOCATION, MAX_ROUNDS
from tools import TOOL_FUNCTIONS

# ═══════════════════════════════════════════════════════════════════════════════
# TOOL DECLARATIONS (Gemini function-calling schema)
# ═══════════════════════════════════════════════════════════════════════════════

_TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="summarise_documents",
        description=(
            "Ingest every supported document (PDF, DOCX, XLSX) from the GCS bucket, "
            "extract text, and build the metadata store with Gemini-generated summaries. "
            "Idempotent — skips documents already processed. Must run before plan_questions."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    ),
    types.FunctionDeclaration(
        name="plan_questions",
        description=(
            "Download the questions spreadsheet from GCS, load the metadata store, "
            "and call Gemini to produce a structured JSON research plan for every question. "
            "Some questions may be answered directly from summaries. "
            "Requires metadata store to be ready first."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    ),
    types.FunctionDeclaration(
        name="deep_research",
        description=(
            "Perform one round of deep research for a single question (1-indexed). "
            "Downloads the plan's referenced documents from GCS, queries each via Gemini, "
            "evaluates answer sufficiency, and refines the plan if needed. "
            "Call this for specific questions that need attention."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "question_number": types.Schema(
                    type="INTEGER",
                    description="1-based index of the question to research.",
                ),
            },
            required=["question_number"],
        ),
    ),
    types.FunctionDeclaration(
        name="deep_research_all_unanswered",
        description=(
            "Run deep_research in parallel for ALL unanswered questions. "
            "Use this to process an entire batch of unanswered questions at once."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    ),
    types.FunctionDeclaration(
        name="check_pipeline_state",
        description=(
            "Return a summary of the current pipeline state: whether the metadata store "
            "and question plans exist, how many questions are answered vs. unanswered, "
            "and which question numbers still need work. Use this to decide what to do next."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    ),
    types.FunctionDeclaration(
        name="get_question_detail",
        description=(
            "Return the full current state for a single question: text, answer, "
            "plan items, and count of data_found entries."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "question_number": types.Schema(
                    type="INTEGER",
                    description="1-based index of the question.",
                ),
            },
            required=["question_number"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_final_report",
        description=(
            "Return a formatted report of every question and its current answer. "
            "Use at the end to present results."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    ),
]

GEMINI_TOOLS = [types.Tool(function_declarations=_TOOL_DECLARATIONS)]


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = f"""You are a Pipeline Intelligence agent that answers research questions
about a document library stored in Google Cloud Storage.

You have these tools:
  1. check_pipeline_state — see what has been done so far
  2. summarise_documents — build the metadata store (must run first)
  3. plan_questions — download questions and create research plans
  4. deep_research — deep-dive a single question (by number)
  5. deep_research_all_unanswered — batch deep-dive all unanswered questions
  6. get_question_detail — inspect one question's current state
  7. get_final_report — print all questions and answers

Your workflow:
  Step 1: Call check_pipeline_state to see what exists.
  Step 2: If metadata is not ready, call summarise_documents.
  Step 3: If plans are not ready, call plan_questions.
  Step 4: If there are unanswered questions, call deep_research_all_unanswered.
  Step 5: Call check_pipeline_state again. If questions remain unanswered,
          repeat Step 4 (up to {MAX_ROUNDS} total rounds).
  Step 6: Call get_final_report and present the results.

Always check state before and after major actions. Be methodical."""


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCTION CALL HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

def _dispatch_tool_call(fn_call: types.FunctionCall) -> Any:
    """Execute a tool function call and return the result."""
    name = fn_call.name
    args = dict(fn_call.args) if fn_call.args else {}
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return f"ERROR: Unknown tool '{name}'."
    try:
        result = func(**args)
        return result
    except Exception as e:
        return f"ERROR executing {name}: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_agent(user_message: str = "Run the full research pipeline end-to-end.",
              max_turns: int = 30,
              verbose: bool = True) -> str:
    """
    Launch the pi-agent. The agent will autonomously invoke tools via Gemini
    function calling until it decides the pipeline is complete or max_turns
    is exhausted.

    Args:
        user_message: The initial instruction for the agent.
        max_turns: Maximum number of model turns (safety cap).
        verbose: Print progress to stdout.

    Returns:
        The agent's final text response.
    """
    start = time.time()

    client = genai.Client(
        vertexai=True,
        project=VERTEX_PROJECT,
        location=VERTEX_LOCATION,
    )

    # Build the initial conversation
    contents: list[types.Content] = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        ),
    ]

    generate_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=GEMINI_TOOLS,
        temperature=0.0,
    )

    final_text = ""

    for turn in range(1, max_turns + 1):
        if verbose:
            print(f"\n{'─'*60}")
            print(f"  AGENT TURN {turn}/{max_turns}")
            print(f"{'─'*60}")

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=generate_config,
        )

        # Collect the model's response parts
        model_parts = response.candidates[0].content.parts

        # Add the model's response to the conversation
        contents.append(
            types.Content(role="model", parts=model_parts)
        )

        # Check if the model made any function calls
        fn_calls = [p.function_call for p in model_parts if p.function_call]

        if not fn_calls:
            # Model produced a text response with no tool calls — we're done
            final_text = "".join(
                p.text for p in model_parts if p.text
            )
            if verbose:
                print(f"\n  Agent response:\n{final_text}")
            break

        # Execute each function call and build responses
        fn_response_parts = []
        for fc in fn_calls:
            if verbose:
                args_str = json.dumps(dict(fc.args)) if fc.args else "{}"
                print(f"  → Calling: {fc.name}({args_str})")

            result = _dispatch_tool_call(fc)

            if verbose:
                # Truncate long results for display
                display = str(result)
                if len(display) > 500:
                    display = display[:500] + "..."
                print(f"    Result: {display}")

            fn_response_parts.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": str(result)},
                )
            )

        # Add all function responses as a single user turn
        contents.append(
            types.Content(role="user", parts=fn_response_parts)
        )
    else:
        final_text = "(Agent reached maximum turns without completing.)"
        if verbose:
            print(f"\n  {final_text}")

    elapsed = time.time() - start
    if verbose:
        print(f"\n{'═'*60}")
        print(f"  Agent finished in {elapsed:.1f}s ({turn} turn(s))")
        print(f"{'═'*60}")

    return final_text
