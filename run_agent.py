#!/usr/bin/env python3
"""
run_agent.py — single entrypoint for all four pipeline implementations.

Pick an engine, point it at a document set and a question list (each
either a local path or the existing GCS convention), and run it.

    python run_agent.py --agent sequential \\
        --docs tests/fixtures --questions tests/fixtures/questions.csv

    python run_agent.py --agent langchain --docs gs://my-bucket/docs \\
        --questions gs://my-bucket/questions.xlsx --ask "..."

Each engine already has its own CLI (plan_questions.py --runAll,
parallelized/plan_questions_parallelized.py --runAll, pi-agent/run.py,
lang-chain-agent/run.py) — this just resolves --docs/--questions into
DATA_PATH/QUESTIONS_PATH env vars (falling back to whatever's already in
.env when omitted) and dispatches to the right one as a subprocess, so
each engine's own module-level config loading works unmodified.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

ENGINES = {
    "sequential": {
        "script": REPO_ROOT / "plan_questions.py",
        "full_run_args": ["--runAll"],
        "supports_agent_flags": False,
    },
    "parallel": {
        "script": REPO_ROOT / "parallelized" / "plan_questions_parallelized.py",
        "full_run_args": ["--runAll"],
        "supports_agent_flags": False,
    },
    "pi-agent": {
        "script": REPO_ROOT / "pi-agent" / "run.py",
        "full_run_args": [],
        "supports_agent_flags": True,
    },
    "langchain": {
        "script": REPO_ROOT / "lang-chain-agent" / "run.py",
        "full_run_args": [],
        "supports_agent_flags": True,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one of the four document-intelligence pipeline implementations."
    )
    parser.add_argument(
        "--agent", required=True, choices=sorted(ENGINES),
        help="Which implementation to run.",
    )
    parser.add_argument(
        "--docs", default=None,
        help="Document source: a local folder, or the existing GCS convention "
             "(bucket/prefix, optionally gs://bucket/prefix). "
             "Overrides DATA_PATH from .env for this run.",
    )
    parser.add_argument(
        "--questions", default=None,
        help="Question source: a local .csv/.xlsx file, or a GCS .xlsx "
             "(bucket/prefix/file.xlsx, optionally gs://...). "
             "Overrides QUESTIONS_PATH from .env for this run.",
    )
    # pi-agent / langchain only — ignored (with a warning) for the others.
    parser.add_argument("--ask", default=None, help="[pi-agent/langchain] natural-language instruction")
    parser.add_argument("--state", action="store_true", help="[pi-agent/langchain] print pipeline state")
    parser.add_argument("--report", action="store_true", help="[pi-agent/langchain] print final report")
    parser.add_argument("--quiet", action="store_true", help="[pi-agent/langchain] suppress verbose output")
    args = parser.parse_args()

    engine = ENGINES[args.agent]
    script: Path = engine["script"]
    if not script.exists():
        print(f"error: expected entrypoint not found: {script}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    if args.docs is not None:
        env["DATA_PATH"] = args.docs
    if args.questions is not None:
        env["QUESTIONS_PATH"] = args.questions

    cmd = [sys.executable, str(script)]
    if engine["supports_agent_flags"]:
        if args.ask is not None:
            cmd += ["--ask", args.ask]
        if args.state:
            cmd.append("--state")
        if args.report:
            cmd.append("--report")
        if args.quiet:
            cmd.append("--quiet")
    else:
        for flag_name in ("ask", "state", "report", "quiet"):
            if getattr(args, flag_name):
                print(
                    f"warning: --{flag_name} is only supported by pi-agent/langchain; "
                    f"ignoring for --agent {args.agent}",
                    file=sys.stderr,
                )
        cmd += engine["full_run_args"]

    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
