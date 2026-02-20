import pytest

from blackgeorge import Desk, Job, Worker
from blackgeorge.adapters.base import ModelResponse
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.store.in_memory import InMemoryRunStore
from blackgeorge.store.state import RunState
from blackgeorge.tools import tool
from blackgeorge.tools.registry import Toolbelt
from blackgeorge.worker_runner import WorkerRunner
from blackgeorge.workforce import Workforce
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


async def test_desk_run_parallel_workforce_raises_in_running_loop(tmp_path) -> None:
    adapter = FakeAdapter([])
    desk = Desk(
        model="fake",
        adapter=adapter,
        run_store=InMemoryRunStore(),
        storage_dir=str(tmp_path),
    )
    workforce = Workforce(
        [Worker(name="A", model="fake"), Worker(name="B", model="fake")],
        mode="collaborate",
        name="team",
    )
    with pytest.raises(RuntimeError, match="run cannot be called from a running event loop"):
        desk.run(workforce, Job(input="hi"))


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


async def test_desk_aresume_resumes_paused_worker(tmp_path) -> None:
    @tool(requires_confirmation=True)
    def risky(action: str) -> str:
        return f"ok:{action}"

    responses = [
        ModelResponse(
            content=None,
            tool_calls=[ToolCall(id="call-1", name="risky", arguments={"action": "ship"})],
            usage={},
            raw={},
        ),
        ModelResponse(content="done", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(
        model="fake",
        adapter=FakeAdapter(responses),
        run_store=InMemoryRunStore(),
        storage_dir=str(tmp_path),
    )
    worker = Worker(name="AsyncResumer", model="fake", tools=[risky])

    paused = await desk.arun(worker, Job(input="ship it"))
    assert paused.status == "paused"
    assert paused.pending_action is not None

    resumed = await desk.aresume(paused, True)
    assert resumed.status == "completed"
    assert resumed.content == "done"

    record = desk.run_store.get_run(paused.run_id)
    assert record is not None
    assert record.status == "completed"
