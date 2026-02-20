import sqlite3
import threading

from pydantic import BaseModel

from blackgeorge.core.event import Event
from blackgeorge.core.job import Job
from blackgeorge.core.report import Report as ReportModel
from blackgeorge.memory.sqlite import SQLiteMemoryStore
from blackgeorge.store.sqlite import SQLiteRunStore
from blackgeorge.store.state import RunState
from blackgeorge.utils import new_id, utc_now


class ExampleModel(BaseModel):
    value: int


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


def test_sqlite_run_store_flushes_buffer_on_close(tmp_path) -> None:
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

    reopened = SQLiteRunStore(str(path))
    events = reopened.get_events(run_id)
    assert len(events) == 1
    reopened.close()


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
