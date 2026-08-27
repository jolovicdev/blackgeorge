import json
from pathlib import Path

import pytest

from blackgeorge.cli import main
from blackgeorge.core.job import Job
from blackgeorge.store.sqlite import SQLiteRunStore
from blackgeorge.store.sqlite_session_store import SQLiteSessionStore
from blackgeorge.store.state import RunState


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    path = str(tmp_path / "cli.db")
    store = SQLiteRunStore(path)
    store.create_run("run-complete", {"input": "first"})
    store.update_run("run-complete", "completed", "done", None, None)
    store.create_run("run-paused", {"input": "second"})
    state = RunState(
        run_id="run-paused",
        status="paused",
        runner_type="worker",
        runner_name="Worker",
        job=Job(input="second"),
        messages=[],
        tool_calls=[],
        pending_action=None,
        metrics={"cost_usd": 0.001, "usage": {"total_tokens": 100}},
        iteration=1,
        payload={},
    )
    store.update_run("run-paused", "paused", None, None, state)
    store.close()
    sessions = SQLiteSessionStore(path)
    sessions.create_session("session-1", "ChatBot")
    sessions.close()
    return path


def test_runs_list(db_path: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--db", db_path, "runs", "list"]) == 0
    out = capsys.readouterr().out
    assert "run-complete" in out and "run-paused" in out
    assert out.index("run-paused") < out.index("run-complete")


def test_runs_list_filters_status(db_path: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--db", db_path, "runs", "list", "--status", "failed"]) == 0
    out = capsys.readouterr().out
    assert "run-complete" not in out and "run-paused" not in out
    assert main(["--db", db_path, "runs", "list", "--status", "paused"]) == 0
    out = capsys.readouterr().out
    assert "run-paused" in out and "run-complete" not in out


def test_runs_show(db_path: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--db", db_path, "runs", "show", "run-paused"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "run-paused"
    assert payload["status"] == "paused"
    assert payload["metrics"]["cost_usd"] == 0.001


def test_runs_show_missing(db_path: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--db", db_path, "runs", "show", "nope"]) == 1
    assert "run not found" in capsys.readouterr().err


def test_sessions_list(db_path: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--db", db_path, "sessions", "list"]) == 0
    out = capsys.readouterr().out
    assert "session-1" in out and "ChatBot" in out


def test_sessions_list_filters_worker(db_path: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--db", db_path, "sessions", "list", "--worker", "Other"]) == 0
    assert "session-1" not in capsys.readouterr().out


def test_missing_db_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = str(tmp_path / "missing.db")
    assert main(["--db", missing, "runs", "list"]) == 1
    assert missing in capsys.readouterr().err
