import asyncio
import threading
import time
from typing import Any

from blackgeorge.adapters.base import BaseModelAdapter, ModelResponse
from blackgeorge.core.job import Job
from blackgeorge.core.report import Report
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.desk import Desk
from blackgeorge.store.in_memory import InMemoryRunStore
from blackgeorge.store.state import RunState
from blackgeorge.tools import tool
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
    ) -> ModelResponse:
        with self._lock:
            self.calls += 1
        return ModelResponse(content="ok", tool_calls=[], usage={}, raw={})


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

    def run_worker(**kwargs: Any) -> tuple[Report, RunState | None]:
        worker = kwargs["worker"]
        run_id = kwargs["run_id"]
        job = kwargs["job"]
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

    workforce._run_worker = run_worker
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

    async def arun_worker(**kwargs: Any) -> tuple[Report, RunState | None]:
        worker = kwargs["worker"]
        run_id = kwargs["run_id"]
        job = kwargs["job"]
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

    def run_worker(**kwargs: Any) -> tuple[Report, RunState | None]:
        job = kwargs["job"]
        captured.append(job.tools_override)
        report = Report(
            run_id=kwargs["run_id"],
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

    workforce._run_worker = run_worker
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

    report, next_state = workforce.resume(
        adapter=FakeAdapter([ModelResponse(content="ok", tool_calls=[], usage={}, raw={})]),
        state=state,
        decision_or_input="go",
        events=[],
        emit=emit,
        temperature=None,
        max_tokens=None,
        stream=False,
        stream_options=None,
        structured_output_retries=0,
        max_iterations=1,
        max_tool_calls=1,
        default_model="fake",
        respect_context_window=True,
    )
    assert report.status == "failed"
    assert "Invalid pending worker index" in report.errors
    assert next_state is None
