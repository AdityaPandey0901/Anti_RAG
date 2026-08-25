"""
Tests for run_agent.py's dispatch logic: given --agent/--docs/--questions/
agent-only flags, does it build the right subprocess command and env?

subprocess.run is monkeypatched so these never actually invoke Gemini/GCS —
each engine's own pipeline is tested separately (tests/test_sources.py plus
the manual end-to-end verifications recorded in the commit history).
"""

import subprocess
import sys

import pytest

import run_agent


class _FakeCompletedProcess:
    def __init__(self, returncode=0):
        self.returncode = returncode


@pytest.fixture
def captured_run(monkeypatch):
    calls = []

    def fake_run(cmd, cwd=None, env=None):
        calls.append({"cmd": cmd, "cwd": cwd, "env": env})
        return _FakeCompletedProcess(0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _invoke(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["run_agent.py"] + argv)
    return run_agent.main()


def test_sequential_full_run_with_local_docs_and_questions(monkeypatch, captured_run):
    rc = _invoke(monkeypatch, [
        "--agent", "sequential",
        "--docs", "tests/fixtures",
        "--questions", "tests/fixtures/questions.csv",
    ])
    assert rc == 0
    assert len(captured_run) == 1
    cmd = captured_run[0]["cmd"]
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("plan_questions.py")
    assert cmd[2:] == ["--runAll"]
    assert captured_run[0]["env"]["DATA_PATH"] == "tests/fixtures"
    assert captured_run[0]["env"]["QUESTIONS_PATH"] == "tests/fixtures/questions.csv"


def test_parallel_dispatches_to_parallelized_script(monkeypatch, captured_run):
    _invoke(monkeypatch, ["--agent", "parallel", "--docs", "tests/fixtures"])
    cmd = captured_run[0]["cmd"]
    assert cmd[1].endswith("parallelized/plan_questions_parallelized.py") or \
        cmd[1].replace("\\", "/").endswith("parallelized/plan_questions_parallelized.py")
    assert cmd[2:] == ["--runAll"]


def test_pi_agent_passes_through_ask_flag(monkeypatch, captured_run):
    _invoke(monkeypatch, ["--agent", "pi-agent", "--ask", "how many docs?"])
    cmd = captured_run[0]["cmd"]
    assert cmd[1].endswith("pi-agent/run.py") or cmd[1].replace("\\", "/").endswith("pi-agent/run.py")
    assert "--ask" in cmd and cmd[cmd.index("--ask") + 1] == "how many docs?"
    # sequential/parallel-only --runAll must never leak into agent-mode calls
    assert "--runAll" not in cmd


def test_langchain_passes_through_state_and_quiet(monkeypatch, captured_run):
    _invoke(monkeypatch, ["--agent", "langchain", "--state", "--quiet"])
    cmd = captured_run[0]["cmd"]
    assert "--state" in cmd
    assert "--quiet" in cmd


def test_omitting_docs_and_questions_leaves_env_untouched(monkeypatch, captured_run):
    monkeypatch.delenv("DATA_PATH", raising=False)
    monkeypatch.delenv("QUESTIONS_PATH", raising=False)
    _invoke(monkeypatch, ["--agent", "sequential"])
    env = captured_run[0]["env"]
    assert "DATA_PATH" not in env
    assert "QUESTIONS_PATH" not in env


def test_agent_only_flags_ignored_with_warning_for_sequential(monkeypatch, captured_run, capsys):
    _invoke(monkeypatch, ["--agent", "sequential", "--ask", "irrelevant"])
    cmd = captured_run[0]["cmd"]
    assert "--ask" not in cmd
    assert "warning" in capsys.readouterr().err.lower()
