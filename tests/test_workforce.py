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
