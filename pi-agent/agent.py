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
from tool_forge import get_dynamic_declarations, get_dynamic_functions

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
    # ── META-TOOLS: dynamic tool creation ────────────────────────────────────
    types.FunctionDeclaration(
        name="create_tool",
        description=(
            "Dynamically create a brand-new tool that you can call on subsequent turns. "
            "You write the Python source code that implements the tool. "
            "Use this when the existing tools are insufficient for a sub-task "
            "(e.g. you need a web search, a custom parser, a calculator, etc.)."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "name": types.Schema(
                    type="STRING",
                    description="Tool name — must be a valid Python identifier (e.g. 'search_web').",
                ),
                "description": types.Schema(
                    type="STRING",
                    description="Human-readable description of what the tool does.",
                ),
                "code": types.Schema(
                    type="STRING",
                    description=(
                        "Python source code defining a function with the same name as 'name'. "
                        "The function must accept keyword arguments matching 'parameters' "
                        "and return a string. May import stdlib and installed packages."
                    ),
                ),
                "parameters": types.Schema(
                    type="STRING",
                    description=(
                        "A JSON string describing the tool's parameters in JSON-Schema style. "
                        'Example: {"type":"object","properties":{"query":{"type":"string","description":"search query"}},"required":["query"]}. '
                        'Use "{}" if the tool takes no arguments.'
                    ),
                ),
            },
            required=["name", "description", "code"],
        ),
    ),
    types.FunctionDeclaration(
        name="list_custom_tools",
        description=(
            "List all dynamically created tools with their descriptions and code previews."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    ),
    types.FunctionDeclaration(
        name="delete_tool",
        description="Delete a dynamically created tool by name.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "name": types.Schema(
                    type="STRING",
                    description="Name of the custom tool to delete.",
                ),
            },
            required=["name"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_tool_audit_log",
        description=(
            "Return the last N entries from the tool creation audit log. "
            "Use this to review what dynamic tools have been created, modified, or deleted."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "last_n": types.Schema(
                    type="INTEGER",
                    description="Number of recent audit entries to return (default 50).",
                ),
            },
        ),
    ),
]


def _build_gemini_tools() -> list[types.Tool]:
    """
    Build the Gemini tool list from static declarations + any dynamic tools
    created at runtime.  Called on EVERY turn so newly created tools appear.
    """
    dynamic_decls = get_dynamic_declarations()
    all_decls = _TOOL_DECLARATIONS + dynamic_decls
    return [types.Tool(function_declarations=all_decls)]


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = f"""You are a Pipeline Intelligence agent that answers research questions
about a document library stored in Google Cloud Storage.

You have these built-in tools:
  1. check_pipeline_state — see what has been done so far
  2. summarise_documents — build the metadata store (must run first)
  3. plan_questions — download questions and create research plans
  4. deep_research — deep-dive a single question (by number)
  5. deep_research_all_unanswered — batch deep-dive all unanswered questions
  6. get_question_detail — inspect one question's current state
  7. get_final_report — print all questions and answers

You also have TOOL CREATION capabilities:
  8. create_tool — dynamically create a brand-new tool by writing Python code.
     The new tool becomes callable on the NEXT turn.
  9. list_custom_tools — see what dynamic tools you've created so far
 10. delete_tool — remove a dynamic tool
 11. get_tool_audit_log — review the audit trail of all tool creation/deletion

═══ WHEN TO CREATE A TOOL ═══

You SHOULD use create_tool when you detect any of these situations:

  A. CALCULATION NEEDED — A question requires arithmetic, date math, financial
     computations (NPV, amortization, percentages), or any numeric derivation
     from extracted data.  Do NOT do math in your head — create a calculator
     tool and run it so the answer is verifiably correct.

  B. EXTERNAL DATA NEEDED — A question references a fact not present in any
     document (e.g. a current market rate, a public statute, a conversion
     factor).  Create a tool that fetches or hardcodes the needed reference.

  C. CROSS-QUESTION AGGREGATION — A question asks you to count, compare, or
     aggregate information across multiple documents or previous answers
     (e.g. "How many documents mention X?", "Which contracts exceed $1M?").
     Create a tool that scans the data_found or metadata programmatically.

  D. UNSUPPORTED FORMAT — You encounter a file type the pipeline can't
     currently extract (e.g. .csv, .txt, .json, images, .msg).  Create a
     parser tool for that format.

  E. VALIDATION / DOUBLE-CHECK — After producing an answer that involves
     numbers, dates, or legal thresholds, create a verification tool to
     independently confirm the answer from raw data.  Prefer computed
     verification over re-reading the same passages.

  F. REPEATED PATTERN — If you find yourself wishing you could do the same
     non-trivial operation for multiple questions (e.g. regex extraction,
     currency conversion, date parsing), create a reusable tool once and
     call it many times.

  If in doubt, lean toward creating a tool — it is cheap, audited, and
  makes your answer more trustworthy than mental arithmetic or guessing.

═══ HOW TO CREATE A TOOL ═══

  - The Python function name MUST match the 'name' argument.
  - The function MUST return a string.
  - You may import standard library modules and installed packages.
  - Provide a meaningful 'description' so you can remember what it does.
  - After creating a tool, call it on the NEXT turn.

═══ MAIN WORKFLOW ═══

  Step 1: Call check_pipeline_state to see what exists.
  Step 2: If metadata is not ready, call summarise_documents.
  Step 3: If plans are not ready, call plan_questions.
  Step 4: If there are unanswered questions, call deep_research_all_unanswered.
  Step 5: Call check_pipeline_state again. Review unanswered questions.
          For each unanswered question, check if any of the trigger conditions
          (A–F above) apply.  If so, create the needed tool(s) and use them
          before retrying.  Repeat up to {MAX_ROUNDS} total rounds.
  Step 6: Call get_final_report and present the results.

Always check state before and after major actions. Be methodical."""


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCTION CALL HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

def _dispatch_tool_call(fn_call: types.FunctionCall) -> Any:
    """Execute a tool function call and return the result.

    Checks the static TOOL_FUNCTIONS first, then any dynamically
    created tools from the forge.
    """
    name = fn_call.name
    args = dict(fn_call.args) if fn_call.args else {}

    # Look up in static tools first, then dynamic tools
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        dynamic = get_dynamic_functions()
        func = dynamic.get(name)
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

    final_text = ""

    for turn in range(1, max_turns + 1):
        if verbose:
            print(f"\n{'─'*60}")
            print(f"  AGENT TURN {turn}/{max_turns}")
            print(f"{'─'*60}")

        # Rebuild tool list every turn so dynamically created tools appear
        current_tools = _build_gemini_tools()
        generate_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=current_tools,
            temperature=0.0,
        )

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
