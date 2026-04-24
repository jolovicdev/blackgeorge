import asyncio
import threading
import time
from typing import Any

from blackgeorge.adapters.base import BaseModelAdapter, ModelResponse
from blackgeorge.config import RunConfig
from blackgeorge.core.job import Job
from blackgeorge.core.pending_action import PendingAction
from blackgeorge.core.report import Report
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.desk import Desk
from blackgeorge.store.in_memory import InMemoryRunStore
from blackgeorge.store.state import RunState
from blackgeorge.tools import tool, transfer_to_agent_tool
from blackgeorge.worker import Worker
from blackgeorge.workforce import WorkerDecision, Workforce
from tests.utils import FakeAdapter


class FailingAdapter(BaseModelAdapter):
    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
        num_retries: int | None = None,
    ) -> ModelResponse:
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(content="alpha", tool_calls=[], usage={}, raw={})
        raise RuntimeError("context length exceeded")

    async def acomplete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
        num_retries: int | None = None,
    ) -> ModelResponse:
        return self.complete(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stream_options=stream_options,
            thinking=thinking,
            drop_params=drop_params,
            extra_body=extra_body,
        )


class StructuredFailAdapter(BaseModelAdapter):
    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
        num_retries: int | None = None,
    ) -> ModelResponse:
        return ModelResponse(content="", tool_calls=[], usage={}, raw={})

    async def acomplete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
        num_retries: int | None = None,
    ) -> ModelResponse:
        return self.complete(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stream_options=stream_options,
            thinking=thinking,
            drop_params=drop_params,
            extra_body=extra_body,
        )

    def structured_complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_schema: object,
        retries: int,
    ) -> object:
        raise RuntimeError("boom")

    async def astructured_complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_schema: object,
        retries: int,
    ) -> object:
        raise RuntimeError("boom")


class SlowAsyncAdapter(BaseModelAdapter):
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.calls = 0

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
        num_retries: int | None = None,
    ) -> ModelResponse:
        raise RuntimeError("sync path not used in this test")

    async def acomplete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
        num_retries: int | None = None,
    ) -> ModelResponse:
        self.calls += 1
        await asyncio.sleep(self.delay)
        return ModelResponse(content="ok", tool_calls=[], usage={}, raw={})


class SyncOnlyAdapter(BaseModelAdapter):
    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
        num_retries: int | None = None,
    ) -> ModelResponse:
        with self._lock:
            self.calls += 1
        return ModelResponse(content="ok", tool_calls=[], usage={}, raw={})


class MessageCaptureAdapter(BaseModelAdapter):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
        num_retries: int | None = None,
    ) -> ModelResponse:
        raise RuntimeError("sync path not used in this test")

    async def acomplete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
        num_retries: int | None = None,
    ) -> ModelResponse:
        self.calls.append(list(messages))
        if not self._responses:
            return ModelResponse(content="", tool_calls=[], usage={}, raw={})
        return self._responses.pop(0)


def test_workforce_collaborate_default_reducer() -> None:
    responses = [
        ModelResponse(content="alpha", tool_calls=[], usage={}, raw={}),
        ModelResponse(content="beta", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker_a = Worker(name="A", model="fake")
    worker_b = Worker(name="B", model="fake")
    workforce = Workforce([worker_a, worker_b], mode="collaborate", name="team")
    report = desk.run(workforce, Job(input="work"))
    assert report.status == "completed"
    assert "[A]" in report.content
    assert "[B]" in report.content


async def test_workforce_collaborate_parallelizes_toolless_workers() -> None:
    adapter = SlowAsyncAdapter(delay=0.05)
    desk = Desk(model="fake", adapter=adapter, run_store=InMemoryRunStore())
    worker_a = Worker(name="A", model="fake")
    worker_b = Worker(name="B", model="fake")
    worker_c = Worker(name="C", model="fake")
    workforce = Workforce([worker_a, worker_b, worker_c], mode="collaborate", name="team")

    started = time.perf_counter()
    report = await desk.arun(workforce, Job(input="work"))
    elapsed = time.perf_counter() - started

    assert report.status == "completed"
    assert adapter.calls == 3
    assert elapsed < 0.14
    assert "[A]" in report.content
    assert "[B]" in report.content
    assert "[C]" in report.content


def test_workforce_sync_parallel_drains_async_event_handlers() -> None:
    responses = [
        ModelResponse(content="alpha", tool_calls=[], usage={}, raw={}),
        ModelResponse(content="beta", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker_a = Worker(name="A", model="fake")
    worker_b = Worker(name="B", model="fake")
    workforce = Workforce([worker_a, worker_b], mode="collaborate", name="team")
    started_sources: list[str] = []
    completed_sources: list[str] = []

    async def on_worker_started(event: Any) -> None:
        await asyncio.sleep(0.02)
        started_sources.append(event.source)

    async def on_worker_completed(event: Any) -> None:
        await asyncio.sleep(0.02)
        completed_sources.append(event.source)

    desk.event_bus.subscribe("worker.started", on_worker_started)
    desk.event_bus.subscribe("worker.completed", on_worker_completed)
    report = desk.run(workforce, Job(input="work"))

    assert report.status == "completed"
    assert sorted(started_sources) == ["A", "B"]
    assert sorted(completed_sources) == ["A", "B"]


def test_workforce_sync_parallel_supports_sync_only_adapter() -> None:
    adapter = SyncOnlyAdapter()
    desk = Desk(model="fake", adapter=adapter, run_store=InMemoryRunStore())
    worker_a = Worker(name="A", model="fake")
    worker_b = Worker(name="B", model="fake")
    workforce = Workforce([worker_a, worker_b], mode="collaborate", name="team")

    report = desk.run(workforce, Job(input="work"))

    assert report.status == "completed"
    assert adapter.calls == 2
    assert "[A]" in report.content
    assert "[B]" in report.content


def test_workforce_collaborate_parallel_guard_requires_no_tools() -> None:
    @tool()
    def noop(value: str) -> str:
        return value

    without_tools = Worker(name="A", model="fake")
    with_tools = Worker(name="B", model="fake", tools=[noop])
    plain = Workforce([without_tools], mode="collaborate", name="plain")
    mixed = Workforce([with_tools], mode="collaborate", name="mixed")
    default_job = Job(input="work")
    nonempty_override = Job(input="work", tools_override=["missing_tool"])
    empty_override = Job(input="work", tools_override=[])

    assert plain._can_parallelize_collaborate(default_job) is True
    assert mixed._can_parallelize_collaborate(default_job) is False
    assert plain._can_parallelize_collaborate(nonempty_override) is False
    assert mixed._can_parallelize_collaborate(empty_override) is True


def test_workforce_collaborate_nonempty_job_tools_override_skips_parallel_sync() -> None:
    responses = [
        ModelResponse(content="alpha", tool_calls=[], usage={}, raw={}),
        ModelResponse(content="beta", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    workforce = Workforce(
        [Worker(name="A", model="fake"), Worker(name="B", model="fake")],
        mode="collaborate",
        name="team",
    )

    async def fail_parallel(**kwargs: Any) -> tuple[Report, RunState | None]:
        raise AssertionError("parallel collaborate path should not run")

    workforce._arun_collaborate_parallel = fail_parallel
    report = desk.run(workforce, Job(input="work", tools_override=["missing_tool"]))

    assert report.status == "completed"
    assert "[A]" in report.content
    assert "[B]" in report.content


def test_workforce_collaborate_failure_propagates() -> None:
    desk = Desk(
        model="fake",
        adapter=FailingAdapter(),
        run_store=InMemoryRunStore(),
        respect_context_window=False,
    )
    worker_a = Worker(name="A", model="fake")
    worker_b = Worker(name="B", model="fake")
    workforce = Workforce([worker_a, worker_b], mode="collaborate", name="team")
    report = desk.run(workforce, Job(input="work"))
    assert report.status == "failed"
    assert "[A]" in report.content


def test_workforce_managed_manager_failure() -> None:
    desk = Desk(model="fake", adapter=StructuredFailAdapter(), run_store=InMemoryRunStore())
    manager = Worker(name="Manager", model="fake")
    worker = Worker(name="Worker", model="fake")
    workforce = Workforce([worker], mode="managed", name="team", manager=manager)
    report = desk.run(workforce, Job(input="work"))
    assert report.status == "failed"


def test_workforce_managed_does_not_fall_through_to_collaborate_sync() -> None:
    @tool()
    def noop(value: str) -> str:
        return value

    manager = Worker(name="Manager", model="fake")
    worker_a = Worker(name="A", model="fake")
    worker_b = Worker(name="B", model="fake", tools=[noop])
    workforce = Workforce([worker_a, worker_b], mode="managed", name="team", manager=manager)
    calls: list[tuple[str, Any]] = []

    async def run_worker(config, worker, job) -> tuple[Report, RunState | None]:

        run_id = config.run_id

        calls.append((worker.name, job.input))
        if worker.name == "Manager":
            manager_report = Report(
                run_id=run_id,
                status="completed",
                content=None,
                data=WorkerDecision(worker="A"),
                messages=[],
                tool_calls=[],
                metrics={},
                events=[],
                pending_action=None,
                errors=[],
            )
            return manager_report, None
        worker_report = Report(
            run_id=run_id,
            status="completed",
            content=f"selected-{worker.name}",
            data=None,
            messages=[],
            tool_calls=[],
            metrics={},
            events=[],
            pending_action=None,
            errors=[],
        )
        return worker_report, None

    workforce._arun_worker = run_worker
    desk = Desk(model="fake", adapter=FakeAdapter([]), run_store=InMemoryRunStore())
    report = desk.run(workforce, Job(input="work"))

    assert report.status == "completed"
    assert report.content == "selected-A"
    assert calls == [
        ("Manager", {"task": "work", "workers": ["A", "B"]}),
        ("A", "work"),
    ]


async def test_workforce_managed_does_not_fall_through_to_collaborate_async() -> None:
    @tool()
    def noop(value: str) -> str:
        return value

    manager = Worker(name="Manager", model="fake")
    worker_a = Worker(name="A", model="fake")
    worker_b = Worker(name="B", model="fake", tools=[noop])
    workforce = Workforce([worker_a, worker_b], mode="managed", name="team", manager=manager)
    calls: list[tuple[str, Any]] = []

    async def arun_worker(config, worker, job) -> tuple[Report, RunState | None]:

        run_id = config.run_id

        calls.append((worker.name, job.input))
        if worker.name == "Manager":
            manager_report = Report(
                run_id=run_id,
                status="completed",
                content=None,
                data=WorkerDecision(worker="A"),
                messages=[],
                tool_calls=[],
                metrics={},
                events=[],
                pending_action=None,
                errors=[],
            )
            return manager_report, None
        worker_report = Report(
            run_id=run_id,
            status="completed",
            content=f"selected-{worker.name}",
            data=None,
            messages=[],
            tool_calls=[],
            metrics={},
            events=[],
            pending_action=None,
            errors=[],
        )
        return worker_report, None

    workforce._arun_worker = arun_worker
    desk = Desk(model="fake", adapter=FakeAdapter([]), run_store=InMemoryRunStore())
    report = await desk.arun(workforce, Job(input="work"))

    assert report.status == "completed"
    assert report.content == "selected-A"
    assert calls == [
        ("Manager", {"task": "work", "workers": ["A", "B"]}),
        ("A", "work"),
    ]


def test_unregister_workforce_blocks_resume() -> None:
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
    workforce = Workforce([worker], mode="collaborate", name="team")
    report = desk.run(workforce, Job(input="work"))
    assert report.status == "paused"
    desk.unregister_workforce(workforce)
    resumed = desk.resume(report, True)
    assert resumed.status == "failed"
    assert "Workforce not registered" in resumed.errors
    record = desk.run_store.get_run(report.run_id)
    assert record is not None
    assert record.status == "failed"
    assert any(event.type == "run.failed" for event in desk.run_store.get_events(report.run_id))


def test_managed_workforce_disables_manager_tools() -> None:
    @tool()
    def manager_tool(info: str) -> str:
        return info

    manager = Worker(name="Manager", model="fake", tools=[manager_tool])
    worker = Worker(name="Worker", model="fake")
    workforce = Workforce([worker], mode="managed", name="team", manager=manager)
    captured: list[list[Any] | None] = []

    async def run_worker(config, worker, job) -> tuple[Report, RunState | None]:

        captured.append(job.tools_override)
        report = Report(
            run_id=config.run_id,
            status="completed",
            content=None,
            data=WorkerDecision(worker=worker.name),
            messages=[],
            tool_calls=[],
            metrics={},
            events=[],
            pending_action=None,
            errors=[],
        )
        return report, None

    workforce._arun_worker = run_worker
    desk = Desk(model="fake", adapter=FakeAdapter([]), run_store=InMemoryRunStore())
    report = desk.run(workforce, Job(input="work"))
    assert report.status == "completed"
    assert captured[0] == []


def test_workforce_pending_index_invalid() -> None:
    workforce = Workforce([Worker(name="A", model="fake")], mode="collaborate", name="team")
    job = Job(input="x")
    worker_state = RunState(
        run_id="r1",
        status="paused",
        runner_type="worker",
        runner_name="A",
        job=job,
        messages=[],
        tool_calls=[],
        pending_action=None,
        metrics={},
        iteration=0,
        payload={},
    )
    state = RunState(
        run_id="r1",
        status="paused",
        runner_type="workforce",
        runner_name="team",
        job=job,
        messages=[],
        tool_calls=[],
        pending_action=None,
        metrics={},
        iteration=0,
        payload={
            "stage": "collaborate",
            "worker_state": worker_state.model_dump(mode="json"),
            "completed_reports": [],
            "pending_worker_index": 5,
        },
    )

    def emit(event_type: str, source: str, payload: dict[str, Any]) -> None:
        return None

    from blackgeorge.config import RunConfig

    config = RunConfig(
        adapter=FakeAdapter([ModelResponse(content="ok", tool_calls=[], usage={}, raw={})]),
        emit=emit,
        run_id="r1",
        events=[],
        structured_output_retries=0,
        max_iterations=1,
        max_tool_calls=1,
        respect_context_window=True,
        default_model="fake",
    )
    report, next_state = workforce.resume(
        config=config,
        state=state,
        decision_or_input="go",
    )
    assert report.status == "failed"
    assert "Invalid pending worker index" in report.errors
    assert next_state is None


def test_workforce_swarm_invalid_handoff_target_fails() -> None:
    responses = [
        ModelResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="handoff-1",
                    name="transfer_to_agent",
                    arguments={"agent_name": "ghost", "context": "route"},
                )
            ],
            usage={},
            raw={},
        )
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    handoff_tool = transfer_to_agent_tool(["coder", "reviewer"])
    coder = Worker(name="coder", model="fake", tools=[handoff_tool])
    reviewer = Worker(name="reviewer", model="fake")
    workforce = Workforce([coder, reviewer], mode="swarm", name="team")

    report = desk.run(workforce, Job(input="start"))

    assert report.status == "failed"
    assert any("handoff target" in error.lower() for error in report.errors)


def test_workforce_swarm_handoff_can_target_manager() -> None:
    responses = [
        ModelResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="handoff-manager-to-worker",
                    name="transfer_to_agent",
                    arguments={"agent_name": "worker", "context": "ctx-worker"},
                )
            ],
            usage={},
            raw={},
        ),
        ModelResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="handoff-worker-to-manager",
                    name="transfer_to_agent",
                    arguments={"agent_name": "manager", "context": "ctx-manager"},
                )
            ],
            usage={},
            raw={},
        ),
        ModelResponse(content="done", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    manager = Worker(name="manager", model="fake", tools=[transfer_to_agent_tool(["worker"])])
    worker = Worker(name="worker", model="fake", tools=[transfer_to_agent_tool(["manager"])])
    workforce = Workforce([worker], mode="swarm", name="team", manager=manager)

    report = desk.run(workforce, Job(input="start"))

    assert report.status == "completed"
    assert report.content == "done"


def test_workforce_swarm_resume_preserves_active_handoff_context() -> None:
    @tool(requires_confirmation=True)
    def risky(action: str) -> str:
        return f"ok:{action}"

    responses = [
        ModelResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="handoff-a-to-b",
                    name="transfer_to_agent",
                    arguments={"agent_name": "B", "context": "ctx1"},
                )
            ],
            usage={},
            raw={},
        ),
        ModelResponse(
            content=None,
            tool_calls=[ToolCall(id="confirm-b", name="risky", arguments={"action": "hold"})],
            usage={},
            raw={},
        ),
        ModelResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="handoff-b-to-c",
                    name="transfer_to_agent",
                    arguments={"agent_name": "C", "context": ""},
                )
            ],
            usage={},
            raw={},
        ),
        ModelResponse(content="done", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    handoff_from_a = transfer_to_agent_tool(["B"])
    handoff_from_b = transfer_to_agent_tool(["C"])
    worker_a = Worker(name="A", model="fake", tools=[handoff_from_a])
    worker_b = Worker(name="B", model="fake", tools=[handoff_from_b, risky])
    worker_c = Worker(name="C", model="fake")
    workforce = Workforce([worker_a, worker_b, worker_c], mode="swarm", name="team")
    observed_inputs: list[tuple[str, Any]] = []
    original_arun_worker = workforce._arun_worker

    async def track_worker_input(config, worker, job) -> tuple[Report, RunState | None]:
        observed_inputs.append((worker.name, job.input))
        return await original_arun_worker(config, worker, job)

    workforce._arun_worker = track_worker_input
    paused = desk.run(workforce, Job(input="root"))

    assert paused.status == "paused"
    assert paused.pending_action is not None
    assert paused.pending_action.type == "confirmation"

    resumed = desk.resume(paused, True)

    assert resumed.status == "completed"
    assert resumed.content == "done"
    assert [worker_input for name, worker_input in observed_inputs if name == "C"] == ["ctx1"]


async def test_workforce_swarm_handoff_drops_prior_worker_system_messages() -> None:
    adapter = MessageCaptureAdapter(
        [
            ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="handoff-a-to-b",
                        name="transfer_to_agent",
                        arguments={"agent_name": "B", "context": "ctx"},
                    )
                ],
                usage={},
                raw={},
            ),
            ModelResponse(content="done", tool_calls=[], usage={}, raw={}),
        ]
    )
    desk = Desk(model="fake", adapter=adapter, run_store=InMemoryRunStore())
    handoff = transfer_to_agent_tool(["B"])
    worker_a = Worker(name="A", model="fake", tools=[handoff], instructions="coder instructions")
    worker_b = Worker(name="B", model="fake", instructions="reviewer instructions")
    workforce = Workforce([worker_a, worker_b], mode="swarm", name="team")

    report = await desk.arun(workforce, Job(input="start"))

    assert report.status == "completed"
    assert len(adapter.calls) >= 2
    second_turn_messages = adapter.calls[1]
    system_contents = [
        content
        for message in second_turn_messages
        if message.get("role") == "system"
        for content in [message.get("content")]
        if isinstance(content, str)
    ]
    combined_system = "\n".join(system_contents)
    assert "reviewer instructions" in combined_system
    assert "coder instructions" not in combined_system


def test_workforce_swarm_handoff_respects_transfer_allowlist() -> None:
    responses = [
        ModelResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="handoff-1",
                    name="transfer_to_agent",
                    arguments={"agent_name": "reviewer", "context": "route"},
                )
            ],
            usage={},
            raw={},
        )
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    handoff_tool = transfer_to_agent_tool(["coder"])
    coder = Worker(name="coder", model="fake", tools=[handoff_tool])
    reviewer = Worker(name="reviewer", model="fake")
    workforce = Workforce([coder, reviewer], mode="swarm", name="team")

    report = desk.run(workforce, Job(input="start"))

    assert report.status == "failed"
    assert any("not allowed" in error.lower() for error in report.errors)


def test_workforce_swarm_handoff_respects_override_transfer_allowlist() -> None:
    responses = [
        ModelResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="handoff-override-1",
                    name="transfer_to_agent",
                    arguments={"agent_name": "C", "context": "route"},
                )
            ],
            usage={},
            raw={},
        ),
        ModelResponse(content="done", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker_a = Worker(name="A", model="fake")
    worker_b = Worker(name="B", model="fake")
    worker_c = Worker(name="C", model="fake")
    workforce = Workforce([worker_a, worker_b, worker_c], mode="swarm", name="team")
    restricted_handoff = transfer_to_agent_tool(["B"])

    report = desk.run(workforce, Job(input="start", tools_override=[restricted_handoff]))

    assert report.status == "failed"
    assert any("not allowed" in error.lower() for error in report.errors)


def test_workforce_swarm_handoff_override_uses_effective_tool_order() -> None:
    responses = [
        ModelResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="handoff-override-order-1",
                    name="transfer_to_agent",
                    arguments={"agent_name": "C", "context": "route"},
                )
            ],
            usage={},
            raw={},
        )
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker_a = Worker(name="A", model="fake")
    worker_b = Worker(name="B", model="fake")
    worker_c = Worker(name="C", model="fake")
    workforce = Workforce([worker_a, worker_b, worker_c], mode="swarm", name="team")
    permissive_handoff = transfer_to_agent_tool(["B", "C"])
    restrictive_handoff = transfer_to_agent_tool(["B"])

    report = desk.run(
        workforce,
        Job(input="start", tools_override=[permissive_handoff, restrictive_handoff]),
    )

    assert report.status == "failed"
    assert any("not allowed" in error.lower() for error in report.errors)


def test_workforce_swarm_resume_uses_paused_worker_identity() -> None:
    @tool(requires_confirmation=True)
    def risky(action: str) -> str:
        return f"ok:{action}"

    responses = [
        ModelResponse(
            content=None,
            tool_calls=[ToolCall(id="confirm-1", name="risky", arguments={"action": "go"})],
            usage={},
            raw={},
        ),
        ModelResponse(content="done", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    manager = Worker(name="manager", model="fake", tools=[risky])
    worker = Worker(name="worker", model="fake")
    workforce = Workforce([worker], mode="swarm", name="team", manager=manager)

    paused = desk.run(workforce, Job(input="start"))

    assert paused.status == "paused"
    assert paused.pending_action is not None
    assert paused.pending_action.type == "confirmation"

    resumed = desk.resume(paused, True)

    assert resumed.status == "completed"
    assert resumed.content == "done"
    assert resumed.tool_calls
    assert resumed.tool_calls[0].name == "risky"
    assert resumed.tool_calls[0].error is None


def test_workforce_resume_preserves_paused_state_run_id() -> None:
    @tool(requires_confirmation=True)
    def risky(action: str) -> str:
        return f"ok:{action}"

    responses = [
        ModelResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="handoff-a-to-b",
                    name="transfer_to_agent",
                    arguments={"agent_name": "B", "context": "ctx"},
                )
            ],
            usage={},
            raw={},
        ),
        ModelResponse(content="done", tool_calls=[], usage={}, raw={}),
    ]
    workforce = Workforce(
        [
            Worker(name="A", model="fake", tools=[risky, transfer_to_agent_tool(["B"])]),
            Worker(name="B", model="fake"),
        ],
        mode="swarm",
        name="team",
    )
    call = ToolCall(id="confirm-1", name="risky", arguments={"action": "go"})
    worker_state = RunState(
        run_id="paused-run",
        status="paused",
        runner_type="worker",
        runner_name="A",
        job=Job(input="root"),
        messages=[],
        tool_calls=[call],
        pending_action=PendingAction(
            action_id="pending-1",
            type="confirmation",
            tool_call=call,
            prompt="confirm",
            options=["yes", "no"],
        ),
        metrics={},
        iteration=0,
        payload={},
    )
    state = RunState(
        run_id="paused-run",
        status="paused",
        runner_type="workforce",
        runner_name="team",
        job=Job(input="root"),
        messages=[],
        tool_calls=[],
        pending_action=None,
        metrics={},
        iteration=0,
        payload={
            "stage": "swarm",
            "worker_state": worker_state.model_dump(mode="json"),
            "root_job": Job(input="root").model_dump(mode="json"),
            "current_worker": "A",
            "handoff_count": 0,
        },
    )
    events = []

    def emit(event_type: str, source: str, payload: dict[str, Any]) -> None:
        return None

    config = RunConfig(
        adapter=FakeAdapter(responses),
        emit=emit,
        run_id="fresh-config-run",
        events=events,
        structured_output_retries=0,
        max_iterations=5,
        max_tool_calls=5,
        respect_context_window=True,
        default_model="fake",
    )

    report, next_state = workforce.resume(config, state, True)

    assert next_state is None
    assert report.status == "completed"
    assert report.content == "done"
    assert report.run_id == "paused-run"
    assert report.run_id != "fresh-config-run"


def test_workforce_swarm_handoff_chain_respects_run_budget() -> None:
    responses = [
        ModelResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="h1",
                    name="transfer_to_agent",
                    arguments={"agent_name": "B", "context": "x"},
                )
            ],
            usage={},
            raw={},
        ),
        ModelResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="h2",
                    name="transfer_to_agent",
                    arguments={"agent_name": "A", "context": "x"},
                )
            ],
            usage={},
            raw={},
        ),
        ModelResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="h3",
                    name="transfer_to_agent",
                    arguments={"agent_name": "B", "context": "x"},
                )
            ],
            usage={},
            raw={},
        ),
        ModelResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="h4",
                    name="transfer_to_agent",
                    arguments={"agent_name": "A", "context": "x"},
                )
            ],
            usage={},
            raw={},
        ),
    ]
    desk = Desk(
        model="fake",
        adapter=FakeAdapter(responses),
        run_store=InMemoryRunStore(),
        max_tool_calls=3,
    )
    worker_a = Worker(name="A", model="fake", tools=[transfer_to_agent_tool(["B"])])
    worker_b = Worker(name="B", model="fake", tools=[transfer_to_agent_tool(["A"])])
    workforce = Workforce([worker_a, worker_b], mode="swarm", name="team")

    report = desk.run(workforce, Job(input="start"))

    assert report.status == "failed"
    assert any("max handoff transitions exceeded" in error.lower() for error in report.errors)
