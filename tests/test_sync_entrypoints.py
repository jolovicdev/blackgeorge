import pytest

from blackgeorge import Desk, Job, Worker
from blackgeorge.store.in_memory import InMemoryRunStore
from blackgeorge.store.state import RunState
from blackgeorge.tools.registry import Toolbelt
from blackgeorge.worker_runner import WorkerRunner
from tests.utils import FakeAdapter


async def test_desk_run_raises_in_running_loop(tmp_path) -> None:
    adapter = FakeAdapter([])
    desk = Desk(
        model="fake",
        adapter=adapter,
        run_store=InMemoryRunStore(),
        storage_dir=str(tmp_path),
    )
    worker = Worker(name="AsyncGuard")
    with pytest.raises(RuntimeError, match="run cannot be called from a running event loop"):
        desk.run(worker, Job(input="hi"))


async def test_worker_resume_raises_in_running_loop() -> None:
    runner = WorkerRunner("AsyncGuard", Toolbelt([]), None)
    state = RunState(
        run_id="run",
        status="paused",
        runner_type="worker",
        runner_name="AsyncGuard",
        job=Job(input="hi"),
    )
    with pytest.raises(RuntimeError, match="resume cannot be called from a running event loop"):
        runner.resume(
            adapter=FakeAdapter([]),
            state=state,
            decision_or_input=None,
            events=[],
            emit=lambda *_: None,
            temperature=None,
            max_tokens=None,
            stream=False,
            stream_options=None,
            structured_output_retries=1,
            max_iterations=1,
            max_tool_calls=1,
            model_name="fake",
        )
