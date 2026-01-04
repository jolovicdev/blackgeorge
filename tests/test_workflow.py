import asyncio

from blackgeorge.adapters.base import ModelResponse
from blackgeorge.core.job import Job
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.desk import Desk
from blackgeorge.store.in_memory import InMemoryRunStore
from blackgeorge.tools import tool
from blackgeorge.worker import Worker
from blackgeorge.workflow import Step
from blackgeorge.workforce import Workforce
from tests.utils import AsyncOnlyAdapter, FakeAdapter


def test_flow_steps() -> None:
    responses = [
        ModelResponse(content="first", tool_calls=[], usage={}, raw={}),
        ModelResponse(content="second", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker_a = Worker(name="A", model="fake")
    worker_b = Worker(name="B", model="fake")
    flow = desk.flow([Step(worker_a), Step(worker_b)])
    report = flow.run(Job(input="work"))
    assert report.status == "completed"
    assert "[step 1]" in report.content
    assert "[step 2]" in report.content


def test_flow_pause_and_resume() -> None:
    @tool(requires_confirmation=True)
    def risky(action: str) -> str:
        return f"ok:{action}"

    responses = [
        ModelResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="risky", arguments={"action": "go"})],
            usage={},
            raw={},
        ),
        ModelResponse(content="finished", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker = Worker(name="Worker", tools=[risky], model="fake")
    desk.register_worker(worker)
    flow = desk.flow([Step(worker)])

    report = flow.run(Job(input="run"))
    assert report.status == "paused"
    record = desk.run_store.get_run(report.run_id)
    assert record is not None
    assert record.state is not None

    resumed = desk.resume(report, True)
    assert resumed.status == "completed"
    assert resumed.content == "finished"


def test_flow_pause_resume_multi_step() -> None:
    @tool(requires_confirmation=True)
    def risky(action: str) -> str:
        return f"ok:{action}"

    responses = [
        ModelResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="risky", arguments={"action": "go"})],
            usage={},
            raw={},
        ),
        ModelResponse(content="step-one", tool_calls=[], usage={}, raw={}),
        ModelResponse(content="step-two", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker_a = Worker(name="WorkerA", tools=[risky], model="fake")
    worker_b = Worker(name="WorkerB", model="fake")
    flow = desk.flow([Step(worker_a), Step(worker_b)])

    report = flow.run(Job(input="run"))
    assert report.status == "paused"
    record = desk.run_store.get_run(report.run_id)
    assert record is not None
    assert record.state is not None
    assert record.state.runner_type == "flow"
    assert record.state.payload.get("step_index") == 0

    resumed = desk.resume(report, True)
    assert resumed.status == "completed"
    assert resumed.content is not None
    assert "step-one" in resumed.content
    assert "step-two" in resumed.content


def test_flow_arun_uses_async_adapter() -> None:
    responses = [ModelResponse(content="async", tool_calls=[], usage={}, raw={})]
    desk = Desk(model="fake", adapter=AsyncOnlyAdapter(responses), run_store=InMemoryRunStore())
    worker = Worker(name="AsyncWorker", model="fake")
    flow = desk.flow([Step(worker)])

    report = asyncio.run(flow.arun(Job(input="work")))
    assert report.status == "completed"
    assert report.content == "async"


def test_flow_aresume_uses_async_adapter() -> None:
    @tool(requires_confirmation=True)
    def risky(action: str) -> str:
        return f"ok:{action}"

    responses = [
        ModelResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="risky", arguments={"action": "go"})],
            usage={},
            raw={},
        ),
        ModelResponse(content="done", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(model="fake", adapter=AsyncOnlyAdapter(responses), run_store=InMemoryRunStore())
    worker = Worker(name="AsyncWorker", tools=[risky], model="fake")
    flow = desk.flow([Step(worker)])

    report = asyncio.run(flow.arun(Job(input="run")))
    assert report.status == "paused"
    resumed = asyncio.run(flow.aresume(report, True))
    assert resumed.status == "completed"
    assert resumed.content == "done"


def test_flow_arun_with_workforce_uses_async_adapter() -> None:
    responses = [
        ModelResponse(content="alpha", tool_calls=[], usage={}, raw={}),
        ModelResponse(content="beta", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(model="fake", adapter=AsyncOnlyAdapter(responses), run_store=InMemoryRunStore())
    worker_a = Worker(name="A", model="fake")
    worker_b = Worker(name="B", model="fake")
    workforce = Workforce([worker_a, worker_b], mode="collaborate", name="team")
    flow = desk.flow([Step(workforce)])

    report = asyncio.run(flow.arun(Job(input="work")))
    assert report.status == "completed"
    assert "[A]" in report.content
    assert "[B]" in report.content
