import asyncio
from pathlib import Path
from typing import Any

import pytest

from blackgeorge.adapters.base import ModelResponse
from blackgeorge.core.job import Job
from blackgeorge.core.report import Report
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.desk import Desk
from blackgeorge.store.in_memory import InMemoryRunStore
from blackgeorge.store.sqlite import SQLiteRunStore
from blackgeorge.tools import tool
from blackgeorge.worker import Worker
from blackgeorge.workflow import Condition, Loop, Parallel, Step
from blackgeorge.workflow.context import WorkflowContext
from blackgeorge.workflow.result import StepOutput
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


def test_workflow_nodes_reject_invalid_empty_execution() -> None:
    worker = Worker(name="Worker", model="fake")

    with pytest.raises(ValueError, match="Parallel requires at least one step"):
        Parallel()
    with pytest.raises(ValueError, match="Loop requires at least one step"):
        Loop([], stop=lambda context: True)
    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        Loop([Step(worker)], stop=lambda context: True, max_iterations=0)
    with pytest.raises(ValueError, match="Loop name must not be empty"):
        Loop([Step(worker)], stop=lambda context: True, name=" ")


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


def test_unregistered_flow_resume_updates_failed_state() -> None:
    @tool(requires_confirmation=True)
    def risky(action: str) -> str:
        return f"ok:{action}"

    responses = [
        ModelResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="risky", arguments={"action": "go"})],
            usage={},
            raw={},
        )
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker = Worker(name="Worker", tools=[risky], model="fake")
    flow = desk.flow([Step(worker)])

    report = flow.run(Job(input="run"))
    assert report.status == "paused"
    desk.unregister_flow_run(report.run_id)

    resumed = desk.resume(report, True)
    assert resumed.status == "failed"
    assert "Flow not registered" in resumed.errors
    record = desk.run_store.get_run(report.run_id)
    assert record is not None
    assert record.status == "failed"
    assert any(event.type == "run.failed" for event in desk.run_store.get_events(report.run_id))


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


def test_condition_stops_on_pause() -> None:
    side_effect = {"ran": False}

    class SideStep:
        async def execute(
            self,
            flow: Any,
            context: WorkflowContext,
        ) -> list[StepOutput]:
            side_effect["ran"] = True
            return [Report(run_id="side", status="completed")]

    @tool(requires_confirmation=True)
    def risky(action: str) -> str:
        return f"ok:{action}"

    responses = [
        ModelResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="risky", arguments={"action": "go"})],
            usage={},
            raw={},
        )
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker = Worker(name="Worker", tools=[risky], model="fake")
    flow = desk.flow([Condition(lambda ctx: True, [Step(worker), SideStep()])])
    report = flow.run(Job(input="run"))
    assert report.status == "paused"
    assert side_effect["ran"] is False


def test_condition_resume_runs_remaining_branch_steps() -> None:
    executed: list[str] = []

    class SideStep:
        async def execute(
            self,
            flow: Any,
            context: WorkflowContext,
        ) -> list[StepOutput]:
            executed.append("branch-tail")
            return [Report(run_id="side", status="completed", content="branch-tail")]

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
        ModelResponse(content="confirmed", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker = Worker(name="Worker", tools=[risky], model="fake")
    flow = desk.flow([Condition(lambda context: True, [Step(worker), SideStep()])])

    paused = flow.run(Job(input="run"))
    resumed = flow.resume(paused, True)

    assert resumed.status == "completed"
    assert executed == ["branch-tail"]
    assert resumed.content is not None
    assert "confirmed" in resumed.content
    assert "branch-tail" in resumed.content


def test_condition_resume_restores_continuation_on_recreated_flow() -> None:
    executed: list[str] = []

    class SideStep:
        async def execute(
            self,
            flow: Any,
            context: WorkflowContext,
        ) -> list[StepOutput]:
            executed.append("branch-tail")
            return [Report(run_id="side", status="completed", content="branch-tail")]

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
        ModelResponse(content="confirmed", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker = Worker(name="Worker", tools=[risky], model="fake")
    original = desk.flow([Condition(lambda context: True, [Step(worker), SideStep()])])

    paused = original.run(Job(input="run"))
    recreated = desk.flow([Condition(lambda context: True, [Step(worker), SideStep()])])
    resumed = recreated.resume(paused, True)

    assert resumed.status == "completed"
    assert executed == ["branch-tail"]
    assert resumed.content is not None
    assert "confirmed" in resumed.content
    assert "branch-tail" in resumed.content


def test_parallel_resume_preserves_completed_siblings() -> None:
    class SideStep:
        async def execute(
            self,
            flow: Any,
            context: WorkflowContext,
        ) -> list[StepOutput]:
            return [Report(run_id="side", status="completed", content="sibling")]

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
        ModelResponse(content="confirmed", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker = Worker(name="Worker", tools=[risky], model="fake")
    flow = desk.flow([Parallel(Step(worker), SideStep())])

    paused = flow.run(Job(input="run"))
    recreated = desk.flow([Parallel(Step(worker), SideStep())])
    resumed = recreated.resume(paused, True)

    assert resumed.status == "completed"
    assert resumed.content is not None
    assert "confirmed" in resumed.content
    assert "sibling" in resumed.content


@pytest.mark.parametrize("pause_first", [True, False])
def test_parallel_failure_takes_precedence_over_pause(pause_first: bool) -> None:
    executed: list[str] = []

    class FailedStep:
        async def execute(
            self,
            flow: Any,
            context: WorkflowContext,
        ) -> list[StepOutput]:
            return [Report(run_id="failed", status="failed", errors=["sibling failed"])]

    @tool(requires_confirmation=True)
    def risky(action: str) -> str:
        executed.append(action)
        return action

    responses = [
        ModelResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="risky", arguments={"action": "write"})],
            usage={},
            raw={},
        )
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    paused_step = Step(Worker(name="Worker", tools=[risky], model="fake"))
    steps = (paused_step, FailedStep()) if pause_first else (FailedStep(), paused_step)
    flow = desk.flow([Parallel(*steps)])

    report = flow.run(Job(input="run"))

    assert report.status == "failed"
    assert report.pending_action is None
    assert report.errors == ["sibling failed"]
    assert executed == []


def test_loop_resume_finishes_paused_iteration() -> None:
    executed: list[int] = []

    class SideStep:
        async def execute(
            self,
            flow: Any,
            context: WorkflowContext,
        ) -> list[StepOutput]:
            executed.append(context.loop_iteration("review"))
            return [Report(run_id="side", status="completed", content="loop-tail")]

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
        ModelResponse(content="confirmed", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker = Worker(name="Worker", tools=[risky], model="fake")
    loop = Loop(
        [Step(worker), SideStep()],
        stop=lambda context: context.loop_iteration("review") == 1,
        name="review",
    )
    flow = desk.flow([loop])

    paused = flow.run(Job(input="run"))
    recreated_loop = Loop(
        [Step(worker), SideStep()],
        stop=lambda context: context.loop_iteration("review") == 1,
        name="review",
    )
    recreated = desk.flow([recreated_loop])
    resumed = recreated.resume(paused, True)

    assert resumed.status == "completed"
    assert executed == [1]
    assert resumed.content is not None
    assert "loop-tail" in resumed.content


def test_unnamed_loop_resume_remaps_iteration_on_recreated_flow() -> None:
    active_loop_name = [""]
    executed: list[int] = []

    class SideStep:
        async def execute(
            self,
            flow: Any,
            context: WorkflowContext,
        ) -> list[StepOutput]:
            executed.append(context.loop_iteration(active_loop_name[0]))
            return [Report(run_id="side", status="completed", content="loop-tail")]

    @tool(requires_confirmation=True)
    def risky(action: str) -> str:
        return f"ok:{action}"

    adapter = FakeAdapter(
        [
            ModelResponse(
                content=None,
                tool_calls=[ToolCall(id="1", name="risky", arguments={"action": "go"})],
                usage={},
                raw={},
            ),
            ModelResponse(content="confirmed", tool_calls=[], usage={}, raw={}),
        ]
    )
    desk = Desk(model="fake", adapter=adapter, run_store=InMemoryRunStore())
    worker = Worker(name="Worker", tools=[risky], model="fake")
    original_loop = Loop(
        [Step(worker), SideStep()],
        stop=lambda context: context.loop_iteration(active_loop_name[0]) == 1,
    )
    active_loop_name[0] = original_loop.name
    original = desk.flow([original_loop])

    paused = original.run(Job(input="run"))
    recreated_loop = Loop(
        [Step(worker), SideStep()],
        stop=lambda context: context.loop_iteration(active_loop_name[0]) == 1,
    )
    active_loop_name[0] = recreated_loop.name
    recreated = desk.flow([recreated_loop])
    resumed = recreated.resume(paused, True)

    assert original_loop.name != recreated_loop.name
    assert resumed.status == "completed"
    assert executed == [1]


def test_nested_resume_survives_sqlite_reopen_with_context(tmp_path: Path) -> None:
    class StoreArtifact:
        async def execute(
            self,
            flow: Any,
            context: WorkflowContext,
        ) -> list[StepOutput]:
            context.artifacts["review"] = "persisted"
            return [Report(run_id="store", status="completed", content="stored")]

    class ReadArtifact:
        async def execute(
            self,
            flow: Any,
            context: WorkflowContext,
        ) -> list[StepOutput]:
            value = context.artifacts["review"]
            return [Report(run_id="read", status="completed", content=str(value))]

    @tool(requires_confirmation=True)
    def risky(action: str) -> str:
        return f"ok:{action}"

    database = tmp_path / "runs.db"
    first_adapter = FakeAdapter(
        [
            ModelResponse(
                content=None,
                tool_calls=[ToolCall(id="1", name="risky", arguments={"action": "go"})],
                usage={},
                raw={},
            )
        ]
    )
    first_store = SQLiteRunStore(str(database))
    first_desk = Desk(model="fake", adapter=first_adapter, run_store=first_store)
    first_worker = Worker(name="Worker", tools=[risky], model="fake")
    original = first_desk.flow(
        [
            Condition(
                lambda context: True,
                [StoreArtifact(), Step(first_worker), ReadArtifact()],
            )
        ]
    )

    paused = original.run(Job(input="run"))
    first_store.close()

    second_store = SQLiteRunStore(str(database))
    second_desk = Desk(
        model="fake",
        adapter=FakeAdapter([ModelResponse(content="confirmed", tool_calls=[], usage={}, raw={})]),
        run_store=second_store,
    )
    second_worker = Worker(name="Worker", tools=[risky], model="fake")
    second_desk.register_worker(second_worker)
    recreated = second_desk.flow(
        [
            Condition(
                lambda context: True,
                [StoreArtifact(), Step(second_worker), ReadArtifact()],
            )
        ]
    )

    resumed = recreated.resume(paused, True)

    assert resumed.status == "completed"
    assert resumed.content is not None
    assert "stored" in resumed.content
    assert "confirmed" in resumed.content
    assert "persisted" in resumed.content
    second_store.close()


def test_flow_fails_cleanly_when_paused_context_cannot_be_serialized() -> None:
    class StoreArtifact:
        async def execute(
            self,
            flow: Any,
            context: WorkflowContext,
        ) -> list[StepOutput]:
            context.artifacts["resource"] = object()
            return [Report(run_id="store", status="completed", content="stored")]

    @tool(requires_confirmation=True)
    def risky(action: str) -> str:
        return f"ok:{action}"

    adapter = FakeAdapter(
        [
            ModelResponse(
                content=None,
                tool_calls=[ToolCall(id="1", name="risky", arguments={"action": "go"})],
                usage={},
                raw={},
            )
        ]
    )
    run_store = InMemoryRunStore()
    desk = Desk(model="fake", adapter=adapter, run_store=run_store)
    worker = Worker(name="Worker", tools=[risky], model="fake")
    flow = desk.flow([Condition(lambda context: True, [StoreArtifact(), Step(worker)])])

    report = flow.run(Job(input="run"))

    assert report.status == "failed"
    assert any("JSON-serializable" in error for error in report.errors)
    record = run_store.get_run(report.run_id)
    assert record is not None
    assert record.status == "failed"
    assert not any(event.type == "run.paused" for event in run_store.get_events(report.run_id))
