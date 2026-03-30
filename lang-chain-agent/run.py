#!/usr/bin/env python3
"""
CLI entry point for the LangChain PI-Agent.

Usage:
    python run.py                        # Run full pipeline (default)
    python run.py --state                # Print current pipeline state
    python run.py --report               # Print final Q&A report
    python run.py --tool-audit           # Print tool creation audit log
    python run.py --ask "your question"  # Send a custom instruction to the agent
    python run.py --quiet                # Suppress verbose output
"""

from __future__ import annotations

import argparse
import sys

# Ensure the lang-chain-agent package directory is on the path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import run_agent
from tools import check_pipeline_state, get_final_report
from tool_forge import get_tool_audit_log


def main():
    parser = argparse.ArgumentParser(
        description="LangChain PI-Agent — Pipeline Intelligence agent with LangSmith observability.",
    )
    parser.add_argument(
        "--state", action="store_true",
        help="Print current pipeline state and exit.",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print final Q&A report and exit.",
    )
    parser.add_argument(
        "--tool-audit", action="store_true",
        help="Print the tool creation audit log and exit.",
    )
    parser.add_argument(
        "--ask", type=str, default=None,
        help="Custom instruction for the agent (overrides default full-pipeline run).",
    )
    parser.add_argument(
        "--max-turns", type=int, default=30,
        help="Maximum agent turns (default: 30).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose step-by-step output.",
    )
    args = parser.parse_args()

    if args.state:
        print(check_pipeline_state.invoke({}))
        return

    if args.report:
        print(get_final_report.invoke({}))
        return

    if args.tool_audit:
        print(get_tool_audit_log.invoke({}))
        return

    user_msg = args.ask or "Run the full research pipeline end-to-end."

    result = run_agent(
        user_message=user_msg,
        max_turns=args.max_turns,
        verbose=not args.quiet,
    )

    if args.quiet:
        print(result)


if __name__ == "__main__":
    main()
