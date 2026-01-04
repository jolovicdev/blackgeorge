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
