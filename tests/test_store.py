import sqlite3
import threading
from pathlib import Path

import pytest
from pydantic import BaseModel

from blackgeorge.core.event import Event
from blackgeorge.core.job import Job
from blackgeorge.core.report import Report as ReportModel
from blackgeorge.desk import Desk
from blackgeorge.memory.sqlite import SQLiteMemoryStore
from blackgeorge.store.in_memory import InMemoryRunStore
from blackgeorge.store.sqlite import SQLiteRunStore
from blackgeorge.store.state import RunState
from blackgeorge.utils import new_id, utc_now


class ExampleModel(BaseModel):
    value: int


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_run_stores_reject_duplicate_ids(tmp_path: Path, store_kind: str) -> None:
    store = (
        InMemoryRunStore()
        if store_kind == "memory"
        else SQLiteRunStore(str(tmp_path / "duplicate.db"))
    )
    store.create_run("duplicate", {"input": "first"})

    with pytest.raises(ValueError, match="already exists"):
        store.create_run("duplicate", {"input": "second"})

    record = store.get_run("duplicate")
    assert record is not None
    assert record.input == {"input": "first"}
    store.close()


def test_sqlite_memory_store(tmp_path) -> None:
    path = tmp_path / "mem.db"
    store = SQLiteMemoryStore(str(path))
    store.write("key", {"value": 1}, "desk")
    assert store.read("key", "desk") == {"value": 1}
    results = store.search("key", "desk")
    assert results


def test_sqlite_memory_store_serializes_base_model(tmp_path) -> None:
    path = tmp_path / "mem_model.db"
    store = SQLiteMemoryStore(str(path))
    store.write("key", ExampleModel(value=3), "desk")
    assert store.read("key", "desk") == {"value": 3}


def test_sqlite_run_store(tmp_path) -> None:
    path = tmp_path / "run.db"
    store = SQLiteRunStore(str(path))
    run_id = "run_1"
    store.create_run(run_id, {"input": "x"})
    event = Event(
        event_id=new_id(),
        type="run.started",
        timestamp=utc_now(),
        run_id=run_id,
        source="test",
        payload={},
    )
    store.add_event(event)
    record = store.get_run(run_id)
    assert record is not None
    assert record.status == "running"
    store.update_run(run_id, "completed", "out", {"value": 2}, None)
    updated = store.get_run(run_id)
    assert updated is not None
    assert updated.status == "completed"
    events = store.get_events(run_id)
    assert events


def test_sqlite_run_store_in_memory() -> None:
    store = SQLiteRunStore(":memory:")
    run_id = "run_memory"
    store.create_run(run_id, {"input": "x"})
    event = Event(
        event_id=new_id(),
        type="run.started",
        timestamp=utc_now(),
        run_id=run_id,
        source="test",
        payload={},
    )
    store.add_event(event)
    record = store.get_run(run_id)
    assert record is not None
    assert record.run_id == run_id
    events = store.get_events(run_id)
    assert len(events) == 1


def test_sqlite_run_store_serializes_base_models(tmp_path) -> None:
    path = tmp_path / "run_models.db"
    store = SQLiteRunStore(str(path))
    run_id = "run_2"
    store.create_run(run_id, {"input": "x"})
    output = {"items": [ExampleModel(value=1), {"nested": ExampleModel(value=2)}]}
    store.update_run(run_id, "completed", "out", output, None)
    updated = store.get_run(run_id)
    assert updated is not None
    assert updated.output_json == {"items": [{"value": 1}, {"nested": {"value": 2}}]}


def test_sqlite_run_store_persists_events_across_close(tmp_path) -> None:
    path = tmp_path / "run_buffered.db"
    run_id = "run_buffered"
    store = SQLiteRunStore(str(path))
    store.create_run(run_id, {"input": "x"})
    store.add_event(
        Event(
            event_id=new_id(),
            type="run.started",
            timestamp=utc_now(),
            run_id=run_id,
            source="test",
            payload={},
        )
    )
    store.close()
    store.close()

    reopened = SQLiteRunStore(str(path))
    events = reopened.get_events(run_id)
    assert len(events) == 1
    reopened.close()


def test_desk_context_closes_default_store(tmp_path) -> None:
    with Desk(model="fake", storage_dir=str(tmp_path)) as desk:
        store = desk.run_store

    assert isinstance(store, SQLiteRunStore)
    assert store._closed is True
    desk.close()


def test_desk_close_leaves_injected_stores_open(tmp_path) -> None:
    run_store = SQLiteRunStore(str(tmp_path / "external_runs.db"))
    memory_store = SQLiteMemoryStore(str(tmp_path / "external_memory.db"))
    desk = Desk(
        model="fake",
        run_store=run_store,
        memory_store=memory_store,
    )

    desk.close()
    run_store.create_run("still-open", {"input": "value"})
    memory_store.write("key", "value", "scope")

    assert run_store.get_run("still-open") is not None
    assert memory_store.read("key", "scope") == "value"
    run_store.close()
    memory_store.close()


def test_sqlite_run_store_add_event_persists_immediately(tmp_path) -> None:
    path = tmp_path / "run_immediate.db"
    run_id = "run_immediate"
    store = SQLiteRunStore(str(path))
    store.create_run(run_id, {"input": "x"})
    store.add_event(
        Event(
            event_id=new_id(),
            type="run.started",
            timestamp=utc_now(),
            run_id=run_id,
            source="test",
            payload={},
        )
    )

    with sqlite3.connect(str(path)) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM events WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()

    assert row is not None
    assert row[0] == 1
    store.close()


def test_sqlite_memory_store_thread_safe(tmp_path) -> None:
    store = SQLiteMemoryStore(str(tmp_path / "mem_thread.db"))
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            store.write("key", {"value": 1}, "desk")
            store.read("key", "desk")
            store.search("key", "desk")
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert not errors


def test_run_state_response_schema_round_trip(tmp_path) -> None:
    path = tmp_path / "run_state.db"
    store = SQLiteRunStore(str(path))
    run_id = "run_schema"
    job = Job(input="x", response_schema=ReportModel)
    state = RunState(
        run_id=run_id,
        status="paused",
        runner_type="worker",
        runner_name="worker",
        job=job,
        messages=[],
        tool_calls=[],
        pending_action=None,
        metrics={},
        iteration=0,
        payload={},
    )
    store.create_run(run_id, {"input": "x"})
    store.update_run(run_id, "paused", None, None, state)
    record = store.get_run(run_id)
    assert record is not None
    assert record.state is not None
    assert record.state.job.response_schema is ReportModel


def _make_run_store(store_kind: str, tmp_path: Path) -> InMemoryRunStore | SQLiteRunStore:
    if store_kind == "memory":
        return InMemoryRunStore()
    return SQLiteRunStore(str(tmp_path / "list_runs.db"))


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_list_runs_orders_newest_first(tmp_path: Path, store_kind: str) -> None:
    store = _make_run_store(store_kind, tmp_path)
    for run_id in ["run-1", "run-2", "run-3"]:
        store.create_run(run_id, {"input": run_id})
    records = store.list_runs()
    assert [record.run_id for record in records] == ["run-3", "run-2", "run-1"]
    store.close()


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_list_runs_filters_by_status(tmp_path: Path, store_kind: str) -> None:
    store = _make_run_store(store_kind, tmp_path)
    store.create_run("run-1", {"input": "a"})
    store.create_run("run-2", {"input": "b"})
    store.update_run("run-1", "completed", "done", None, None)
    completed = store.list_runs(status="completed")
    assert [record.run_id for record in completed] == ["run-1"]
    running = store.list_runs(status="running")
    assert [record.run_id for record in running] == ["run-2"]
    store.close()


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_list_runs_limit_and_offset(tmp_path: Path, store_kind: str) -> None:
    store = _make_run_store(store_kind, tmp_path)
    for index in range(5):
        store.create_run(f"run-{index}", {"input": index})
    page = store.list_runs(limit=2, offset=1)
    assert [record.run_id for record in page] == ["run-3", "run-2"]
    assert len(store.list_runs(limit=0)) == 0
    store.close()


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_list_runs_rejects_negative_pagination(tmp_path: Path, store_kind: str) -> None:
    store = _make_run_store(store_kind, tmp_path)
    with pytest.raises(ValueError, match="limit must be non-negative"):
        store.list_runs(limit=-1)
    with pytest.raises(ValueError, match="offset must be non-negative"):
        store.list_runs(offset=-1)
    store.close()
