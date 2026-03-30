"""
Core agent loop for the LangChain agent.

Uses LangGraph's ReAct agent with ChatGoogleGenerativeAI (Gemini API) as the
LLM, LangChain tools for the pipeline, and LangSmith for full observability.

Key differences from the pi-agent:
  • Tools are LangChain @tool-decorated functions (auto-schema).
  • The agent loop is managed by LangGraph (ReAct pattern).
  • Every step is traced to LangSmith automatically.
  • Dynamic tools are injected each turn by rebuilding the agent.
  • Uses the Gemini API directly (no Vertex AI service account needed).
"""

from __future__ import annotations

import time
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent

from config import MODEL_NAME, GEMINI_API_KEY, MAX_ROUNDS
from tools import PIPELINE_TOOLS
from tool_forge import META_TOOLS, get_dynamic_lc_tools


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
# AGENT CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def _build_llm() -> ChatGoogleGenerativeAI:
    """Create the Gemini LLM via the Google Generative AI API (API key)."""
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=0.0,
        max_retries=3,
    )


def _get_all_tools() -> list:
    """Combine static pipeline tools + meta-tools + any dynamic tools."""
    return PIPELINE_TOOLS + META_TOOLS + get_dynamic_lc_tools()


def _build_agent(llm: ChatGoogleGenerativeAI):
    """Build a LangChain agent with all currently available tools."""
    all_tools = _get_all_tools()
    return create_agent(
        model=llm,s
        tools=all_tools,
        system_prompt=SYSTEM_PROMPT,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_agent(
    user_message: str = "Run the full research pipeline end-to-end.",
    max_turns: int = 30,
    verbose: bool = True,
) -> str:
    """
    Launch the LangChain agent. The agent will autonomously invoke tools
    until it decides the pipeline is complete or max_turns is exhausted.

    All calls are traced to LangSmith via the environment variables set
    in config.py.

    Args:
        user_message: The initial instruction for the agent.
        max_turns: Maximum number of agent steps (safety cap).
        verbose: Print progress to stdout.

    Returns:
        The agent's final text response.
    """
    start = time.time()

    llm = _build_llm()

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  LangChain PI-Agent starting")
        print(f"  Model: {MODEL_NAME} (Vertex AI)")
        print(f"  LangSmith tracing: ENABLED")
        print(f"  Max turns: {max_turns}")
        print(f"{'═'*60}\n")

    # We use a multi-turn approach: rebuild the agent each "epoch"
    # to pick up dynamically created tools.
    messages = [HumanMessage(content=user_message)]
    final_text = ""
    turns_used = 0

    # Run in epochs. Each epoch creates a fresh agent (picking up new tools)
    # and streams through until the agent stops calling tools or hits the cap.
    epoch = 0
    while turns_used < max_turns:
        epoch += 1
        agent = _build_agent(llm)

        if verbose:
            print(f"\n{'─'*60}")
            print(f"  EPOCH {epoch} (turns used so far: {turns_used})")
            print(f"  Tools available: {len(_get_all_tools())}")
            print(f"{'─'*60}")

        # Stream through LangGraph
        config = {
            "recursion_limit": min(max_turns - turns_used, 50),
        }

        try:
            result = agent.invoke(
                {"messages": messages},
                config=config,
            )
        except Exception as e:
            if verbose:
                print(f"  Agent error: {e}")
            final_text = f"Agent error: {e}"
            break

        # Extract messages from the result
        output_messages = result.get("messages", [])

        if verbose:
            for msg in output_messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        args_preview = str(tc.get("args", {}))[:200]
                        print(f"  → Tool call: {tc['name']}({args_preview})")
                elif hasattr(msg, 'content') and msg.content and not hasattr(msg, 'tool_call_id'):
                    content_preview = str(msg.content)[:300]
                    if msg.type == "ai":
                        print(f"  Agent: {content_preview}")
                elif hasattr(msg, 'tool_call_id'):
                    content_preview = str(msg.content)[:200]
                    print(f"    Result: {content_preview}")

        # Count tool-call turns
        tool_turns = sum(
            1 for m in output_messages
            if hasattr(m, 'tool_calls') and m.tool_calls
        )
        turns_used += max(tool_turns, 1)

        # Get the final AI message
        ai_messages = [m for m in output_messages if m.type == "ai"]
        if ai_messages:
            last_ai = ai_messages[-1]
            if hasattr(last_ai, 'content') and last_ai.content:
                final_text = last_ai.content
            # Check if the last AI message has tool calls (meaning it wants to continue)
            if hasattr(last_ai, 'tool_calls') and last_ai.tool_calls:
                # Agent was cut off by recursion limit — carry over messages
                messages = output_messages
                continue

        # Check if dynamic tools were created in this epoch
        dynamic_tools = get_dynamic_lc_tools()
        if dynamic_tools and tool_turns > 0:
            # New tools were created — start a new epoch so they're available
            messages = output_messages + [
                HumanMessage(content="New tools are now available. Continue with the workflow.")
            ]
            continue

        # Agent finished naturally
        break

    elapsed = time.time() - start
    if verbose:
        print(f"\n{'═'*60}")
        print(f"  Agent finished in {elapsed:.1f}s ({turns_used} turn(s), {epoch} epoch(s))")
        print(f"  View traces at: https://smith.langchain.com")
        print(f"{'═'*60}")

    return final_text if isinstance(final_text, str) else str(final_text)
