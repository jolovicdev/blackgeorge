import json
import tempfile
import threading
from typing import Any

import pytest
from pydantic import BaseModel

from blackgeorge.adapters.base import BaseModelAdapter, ModelResponse
from blackgeorge.config import RunConfig
from blackgeorge.core.job import Job
from blackgeorge.core.message import Message
from blackgeorge.core.pending_action import PendingAction
from blackgeorge.core.report import Report
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.desk import Desk
from blackgeorge.memory.base import MemoryScope, MemoryStore
from blackgeorge.store.in_memory import InMemoryRunStore
from blackgeorge.store.state import RunState
from blackgeorge.tools import tool, transfer_to_agent_tool
from blackgeorge.worker import Worker
from blackgeorge.worker_messages import replace_tool_call, structured_content
from tests.utils import FakeAdapter


class ContextLimitAdapter(BaseModelAdapter):
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
            raise RuntimeError("context length exceeded")
        if self.calls == 2:
            return ModelResponse(content="summary", tool_calls=[], usage={}, raw={})
        return ModelResponse(content="done", tool_calls=[], usage={}, raw={})

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
        )


class ContextLimitFailingAdapter(BaseModelAdapter):
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
        )


class StructuredAdapter(BaseModelAdapter):
    def __init__(self) -> None:
        self.called = False

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
        raise RuntimeError("complete should not be used")

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
        raise RuntimeError("acomplete should not be used")

    def structured_complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_schema: Any,
        retries: int,
    ) -> Any:
        self.called = True
        return response_schema(answer="ok")


class StructuredStreamPreviewAdapter(BaseModelAdapter):
    def __init__(self, chunks: list[dict[str, Any]], fallback_answer: str) -> None:
        self._chunks = chunks
        self._fallback_answer = fallback_answer
        self.stream_calls = 0
        self.structured_calls = 0

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
        raise RuntimeError("sync path not used")

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
    ) -> ModelResponse | Any:
        if not stream:
            raise RuntimeError("non-stream completion should not be used")
        self.stream_calls += 1
        return iter(self._chunks)

    async def astructured_complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_schema: Any,
        retries: int,
    ) -> Any:
        self.structured_calls += 1
        return response_schema(answer=self._fallback_answer)


class ProactiveSummaryBudgetAdapter(BaseModelAdapter):
    def __init__(self) -> None:
        self.turn_calls = 0
        self.summary_calls = 0

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
        raise RuntimeError("sync path not used")

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
        if (
            len(messages) >= 1
            and isinstance(messages[0].get("content"), str)
            and str(messages[0]["content"]).startswith("You are a summarization assistant.")
        ):
            self.summary_calls += 1
            return ModelResponse(content="summary", tool_calls=[], usage={}, raw={})
        self.turn_calls += 1
        if self.turn_calls <= 2:
            raise RuntimeError("context length exceeded")
        return ModelResponse(content="done", tool_calls=[], usage={}, raw={})


class ToolThenStructuredContextAdapter(BaseModelAdapter):
    def __init__(self) -> None:
        self.acomplete_calls = 0
        self.structured_calls = 0

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
        raise RuntimeError("sync path not used")

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
        self.acomplete_calls += 1
        if self.acomplete_calls == 1:
            return ModelResponse(
                content=None,
                tool_calls=[ToolCall(id="tool-1", name="echo", arguments={"text": "hello"})],
                usage={},
                raw={},
            )
        if self.acomplete_calls == 2:
            return ModelResponse(content="after-tool", tool_calls=[], usage={}, raw={})
        if self.acomplete_calls == 3:
            return ModelResponse(content="summary", tool_calls=[], usage={}, raw={})
        return ModelResponse(content="after-summary", tool_calls=[], usage={}, raw={})

    async def astructured_complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_schema: Any,
        retries: int,
    ) -> Any:
        self.structured_calls += 1
        if self.structured_calls == 1:
            raise RuntimeError("context length exceeded")
        return response_schema(answer="ok")


class AnswerModel(BaseModel):
    answer: str


class RecordingMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def write(self, key: str, value: Any, scope: MemoryScope) -> None:
        self.calls.append(("write", key))

    def read(self, key: str, scope: MemoryScope) -> Any | None:
        self.calls.append(("read", key))
        return "memory"

    def search(self, query: str, scope: MemoryScope) -> list[tuple[str, Any]]:
        return []

    def reset(self, scope: MemoryScope) -> None:
        return None


def test_structured_content_serializes_model_list() -> None:
    class Item(BaseModel):
        value: int

    payload = [Item(value=1), Item(value=2)]
    content = structured_content(payload)

    assert json.loads(content) == [{"value": 1}, {"value": 2}]


def test_worker_tool_loop() -> None:
    @tool()
    def echo(text: str) -> str:
        return text

    responses = [
        ModelResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "hi"})],
            usage={},
            raw={},
        ),
        ModelResponse(content="done", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker = Worker(name="Worker", tools=[echo], model="fake")
    job = Job(input="run")
    report = desk.run(worker, job)
    assert report.status == "completed"
    assert report.content == "done"
    assert report.tool_calls
    assert report.tool_calls[0].result is not None


def test_parallel_tool_calls_execute_concurrently() -> None:
    barrier = threading.Barrier(2)

    @tool()
    def tool_a() -> str:
        barrier.wait(timeout=1)
        return "a"

    @tool()
    def tool_b() -> str:
        barrier.wait(timeout=1)
        return "b"

    responses = [
        ModelResponse(
            content=None,
            tool_calls=[
                ToolCall(id="1", name="tool_a", arguments={}),
                ToolCall(id="2", name="tool_b", arguments={}),
            ],
            usage={},
            raw={},
        ),
        ModelResponse(content="done", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker = Worker(name="Worker", tools=[tool_a, tool_b], model="fake")
    report = desk.run(worker, Job(input="run"))
    assert report.status == "completed"
    assert len(report.tool_calls) == 2
    assert all(call.error is None for call in report.tool_calls)


def test_pause_and_resume_confirmation() -> None:
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
    job = Job(input="run")
    report = desk.run(worker, job)
    assert report.status == "paused"
    assert report.pending_action is not None

    resumed = desk.resume(report, True)
    assert resumed.status == "completed"
    assert resumed.content == "finished"


def test_worker_resume_preserves_paused_state_run_id() -> None:
    @tool(requires_confirmation=True)
    def risky(action: str) -> str:
        return f"ok:{action}"

    call = ToolCall(id="call-1", name="risky", arguments={"action": "go"})
    pending_action = PendingAction(
        action_id="pending-1",
        type="confirmation",
        tool_call=call,
        prompt="confirm",
        options=["yes", "no"],
    )
    state = RunState(
        run_id="paused-run",
        status="paused",
        runner_type="worker",
        runner_name="Worker",
        job=Job(input="run"),
        messages=[],
        tool_calls=[call],
        pending_action=pending_action,
        metrics={},
        iteration=0,
        payload={},
    )
    events = []

    def emit(event_type: str, source: str, payload: dict[str, Any]) -> None:
        return None

    config = RunConfig(
        adapter=FakeAdapter([ModelResponse(content="done", tool_calls=[], usage={}, raw={})]),
        emit=emit,
        run_id="fresh-config-run",
        events=events,
        structured_output_retries=0,
        max_iterations=5,
        max_tool_calls=5,
        respect_context_window=True,
        default_model="fake",
    )
    worker = Worker(name="Worker", tools=[risky], model="fake")

    report, next_state = worker.resume(config, state, True)

    assert next_state is None
    assert report.status == "completed"
    assert report.run_id == "paused-run"
    assert report.run_id != "fresh-config-run"


def test_unexpected_worker_exception_marks_run_failed() -> None:
    class BrokenAdapter(BaseModelAdapter):
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
            raise RuntimeError("provider down")

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
            raise RuntimeError("provider down")

    run_store = InMemoryRunStore()
    desk = Desk(model="fake", adapter=BrokenAdapter(), run_store=run_store)
    run_id = "broken-run"

    with pytest.raises(RuntimeError, match="provider down"):
        desk.run(Worker(name="Worker", model="fake"), Job(input="run"), run_id=run_id)

    record = run_store.get_run(run_id)
    assert record is not None
    assert record.status == "failed"
    assert record.output_json == {"error": "provider down", "error_type": "RuntimeError"}
    assert any(event.type == "run.failed" for event in run_store.get_events(run_id))


def test_pause_emits_worker_paused_not_completed() -> None:
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
    event_types: list[str] = []

    def capture(event: Any) -> None:
        event_types.append(event.type)

    desk.event_bus.subscribe("worker.paused", capture)
    desk.event_bus.subscribe("worker.completed", capture)
    report = desk.run(worker, Job(input="run"))
    assert report.status == "paused"
    assert "worker.paused" in event_types
    assert "worker.completed" not in event_types


def test_confirmation_decline_emits_tool_failed() -> None:
    executed: list[str] = []

    @tool(requires_confirmation=True)
    def risky(action: str) -> str:
        executed.append(action)
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
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker = Worker(name="Worker", tools=[risky], model="fake")
    tool_failed_payloads: list[dict[str, Any]] = []

    def on_tool_failed(event: Any) -> None:
        tool_failed_payloads.append(event.payload)

    desk.event_bus.subscribe("tool.failed", on_tool_failed)
    report = desk.run(worker, Job(input="run"))
    assert report.status == "paused"
    resumed = desk.resume(report, "no")
    assert resumed.status == "completed"
    assert executed == []
    assert any(
        payload.get("error") == "Tool execution declined" for payload in tool_failed_payloads
    )


def test_confirmation_unknown_truthy_string_keeps_approving() -> None:
    executed: list[str] = []

    @tool(requires_confirmation=True)
    def risky(action: str) -> str:
        executed.append(action)
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
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker = Worker(name="Worker", tools=[risky], model="fake")

    report = desk.run(worker, Job(input="run"))
    assert report.status == "paused"
    resumed = desk.resume(report, "ship it")

    assert resumed.status == "completed"
    assert executed == ["go"]
    assert resumed.tool_calls[0].error is None


def test_handoff_pause_does_not_emit_user_input_requested() -> None:
    responses = [
        ModelResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="1",
                    name="transfer_to_agent",
                    arguments={"agent_name": "reviewer", "context": "route"},
                )
            ],
            usage={},
            raw={},
        )
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    handoff_tool = transfer_to_agent_tool(["reviewer"])
    worker = Worker(name="Worker", tools=[handoff_tool], model="fake")
    user_input_events: list[str] = []
    paused_payloads: list[dict[str, Any]] = []

    def on_user_input_requested(event: Any) -> None:
        user_input_events.append(event.type)

    def on_worker_paused(event: Any) -> None:
        paused_payloads.append(event.payload)

    desk.event_bus.subscribe("tool.user_input_requested", on_user_input_requested)
    desk.event_bus.subscribe("worker.paused", on_worker_paused)
    report = desk.run(worker, Job(input="route"))

    assert report.status == "paused"
    assert report.pending_action is not None
    assert report.pending_action.type == "handoff"
    assert user_input_events == []
    assert paused_payloads
    assert paused_payloads[0].get("pending_action_type") == "handoff"


def test_unregister_worker_blocks_resume() -> None:
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
    report = desk.run(worker, Job(input="run"))
    assert report.status == "paused"
    desk.unregister_worker(worker)
    resumed = desk.resume(report, True)
    assert resumed.status == "failed"
    assert "Worker not registered" in resumed.errors
    record = desk.run_store.get_run(report.run_id)
    assert record is not None
    assert record.status == "failed"
    assert any(event.type == "run.failed" for event in desk.run_store.get_events(report.run_id))


def test_tools_override_resolves_by_name() -> None:
    @tool()
    def echo(text: str) -> str:
        return text

    responses = [
        ModelResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "hi"})],
            usage={},
            raw={},
        ),
        ModelResponse(content="done", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker = Worker(name="Worker", tools=[echo], model="fake")
    job = Job(input="run", tools_override=["echo"])
    report = desk.run(worker, job)
    assert report.status == "completed"
    assert report.tool_calls[0].error is None


def test_resume_uses_tools_override_tool_instance() -> None:
    @tool(requires_confirmation=True)
    def temporary(action: str) -> str:
        return f"ok:{action}"

    responses = [
        ModelResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="temporary", arguments={"action": "go"})],
            usage={},
            raw={},
        ),
        ModelResponse(content="done", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker = Worker(name="Worker", model="fake")
    report = desk.run(worker, Job(input="run", tools_override=[temporary]))
    assert report.status == "paused"

    resumed = desk.resume(report, True)

    assert resumed.status == "completed"
    assert resumed.tool_calls[0].error is None
    assert resumed.tool_calls[0].result is not None
    assert resumed.tool_calls[0].result.content == "ok:go"


def test_resume_uses_tools_override_tool_instance_with_default_store() -> None:
    @tool(requires_confirmation=True)
    def temporary(action: str) -> str:
        return f"ok:{action}"

    responses = [
        ModelResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="temporary", arguments={"action": "go"})],
            usage={},
            raw={},
        ),
        ModelResponse(content="done", tool_calls=[], usage={}, raw={}),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        desk = Desk(model="fake", adapter=FakeAdapter(responses), storage_dir=tmpdir)
        worker = Worker(name="Worker", model="fake")
        report = desk.run(worker, Job(input="run", tools_override=[temporary]))
        assert report.status == "paused"

        resumed = desk.resume(report, True)

    assert resumed.status == "completed"
    assert resumed.tool_calls[0].error is None
    assert resumed.tool_calls[0].result is not None
    assert resumed.tool_calls[0].result.content == "ok:go"


def test_missing_tool_call_marks_error() -> None:
    responses = [
        ModelResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="missing", arguments={})],
            usage={},
            raw={},
        ),
        ModelResponse(content="done", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker = Worker(name="Worker", model="fake")
    report = desk.run(worker, Job(input="run"))
    assert report.status == "completed"
    assert report.tool_calls[0].error == "Tool not found: missing"


def test_context_window_summarization() -> None:
    adapter = ContextLimitAdapter()
    desk = Desk(model="fake", adapter=adapter, run_store=InMemoryRunStore())
    worker = Worker(name="Worker", model="fake")
    report = desk.run(worker, Job(input="run"))
    assert report.status == "completed"
    assert report.content == "done"
    assert adapter.calls == 3
    assert any(message.metadata.get("summary") for message in report.messages)


def test_context_window_retry_after_tool_before_structured() -> None:
    @tool()
    def echo(text: str) -> str:
        return text

    adapter = ToolThenStructuredContextAdapter()
    desk = Desk(model="fake", adapter=adapter, run_store=InMemoryRunStore())
    worker = Worker(name="Worker", model="fake", tools=[echo])
    report = desk.run(worker, Job(input="run", response_schema=AnswerModel))
    assert report.status == "completed"
    assert report.data is not None
    assert report.data.answer == "ok"
    assert adapter.structured_calls == 2
    assert adapter.acomplete_calls >= 4
    assert any(message.metadata.get("summary") for message in report.messages)


def test_context_window_respect_disabled() -> None:
    adapter = ContextLimitFailingAdapter()
    desk = Desk(
        model="fake",
        adapter=adapter,
        run_store=InMemoryRunStore(),
        respect_context_window=False,
    )
    worker = Worker(name="Worker", model="fake")
    report = desk.run(worker, Job(input="run"))
    assert report.status == "failed"
    assert report.errors


def test_proactive_summaries_do_not_consume_context_retry_budget() -> None:
    adapter = ProactiveSummaryBudgetAdapter()
    desk = Desk(
        model="fake",
        adapter=adapter,
        run_store=InMemoryRunStore(),
        max_context_messages=5,
    )
    worker = Worker(name="Worker", model="fake")
    initial_messages = [Message(role="user", content=f"history-{index}") for index in range(6)]
    report = desk.run(worker, Job(input="run", initial_messages=initial_messages))

    assert report.status == "completed"
    assert report.content == "done"
    assert adapter.turn_calls == 3
    assert adapter.summary_calls >= 3


def test_max_tool_calls_limit_enforced() -> None:
    @tool()
    def dummy_tool() -> str:
        return "ok"

    responses = [
        ModelResponse(
            content=None,
            tool_calls=[
                ToolCall(id=f"call_{i}", name="dummy_tool", arguments={}) for i in range(5)
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
    worker = Worker(name="Worker", model="fake", tools=[dummy_tool])
    report = desk.run(worker, Job(input="run"))
    assert report.status == "failed"
    assert len(report.tool_calls) == 3
    assert any("Max tool calls exceeded" in error for error in report.errors)


def test_streaming_with_content_only() -> None:
    from tests.utils import StreamingAdapter

    streams = [
        [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
            {"choices": [{"delta": {}}], "usage": {"total_tokens": 10}},
        ]
    ]
    desk = Desk(
        model="fake", adapter=StreamingAdapter(streams), run_store=InMemoryRunStore(), stream=True
    )
    worker = Worker(name="Worker", model="fake")
    report = desk.run(worker, Job(input="run"))
    assert report.status == "completed"
    assert report.content == "Hello world"


def test_streaming_with_reasoning_content() -> None:
    from tests.utils import StreamingAdapter

    streams = [
        [
            {"choices": [{"delta": {"reasoning_content": "Thinking..."}}]},
            {"choices": [{"delta": {"reasoning_content": " done", "content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
            {"choices": [{"delta": {}}], "usage": {"total_tokens": 10}},
        ]
    ]
    desk = Desk(
        model="fake", adapter=StreamingAdapter(streams), run_store=InMemoryRunStore(), stream=True
    )
    worker = Worker(name="Worker", model="fake")
    report = desk.run(worker, Job(input="run"))
    assert report.status == "completed"
    assert report.content == "Hello world"
    assert report.reasoning_content == "Thinking... done"


def test_streaming_with_thinking_blocks() -> None:
    from tests.utils import StreamingAdapter

    streams = [
        [
            {
                "choices": [
                    {
                        "delta": {
                            "thinking_blocks": [
                                {"type": "thinking", "thinking": "first", "signature": "sig1"}
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "thinking_blocks": [
                                {"type": "thinking", "thinking": "second", "signature": "sig2"}
                            ],
                            "content": "Hello",
                        }
                    }
                ]
            },
            {"choices": [{"delta": {"content": " world"}}]},
            {"choices": [{"delta": {}}], "usage": {"total_tokens": 10}},
        ]
    ]
    desk = Desk(
        model="fake", adapter=StreamingAdapter(streams), run_store=InMemoryRunStore(), stream=True
    )
    worker = Worker(name="Worker", model="fake")
    report = desk.run(worker, Job(input="run"))
    assert report.status == "completed"
    assert report.content == "Hello world"
    assert report.messages[-1].thinking_blocks == [
        {"type": "thinking", "thinking": "first", "signature": "sig1"},
        {"type": "thinking", "thinking": "second", "signature": "sig2"},
    ]


def test_streaming_tool_turn_emits_tokens_and_executes_tool() -> None:
    from tests.utils import StreamingAdapter

    @tool()
    async def echo(text: str) -> str:
        return f"echo:{text}"

    streams = [
        [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "echo", "arguments": ""},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": '{"text": '}}]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"hi"}'}}]}}
                ]
            },
            {
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"total_tokens": 9},
            },
        ],
        [
            {"choices": [{"delta": {"content": "done"}}]},
            {"choices": [{"delta": {}}], "usage": {"total_tokens": 4}},
        ],
    ]
    desk = Desk(
        model="fake",
        adapter=StreamingAdapter(streams),
        run_store=InMemoryRunStore(),
        stream=True,
    )
    worker = Worker(name="Worker", model="fake", tools=[echo])
    streamed_tokens: list[str] = []

    def on_token(event: Any) -> None:
        token = event.payload.get("token")
        if isinstance(token, str):
            streamed_tokens.append(token)

    desk.event_bus.subscribe("stream.token", on_token)
    report = desk.run(worker, Job(input="run"), stream=True)

    assert report.status == "completed"
    assert report.content == "done"
    assert len(report.tool_calls) == 1
    assert report.tool_calls[0].name == "echo"
    assert report.tool_calls[0].arguments == {"text": "hi"}
    assert report.tool_calls[0].result is not None
    assert '{"text": ' in streamed_tokens
    assert '"hi"}' in streamed_tokens
    assert "done" in streamed_tokens


def test_streaming_accepts_adapter_model_response_for_tool_turns() -> None:
    @tool()
    async def echo(text: str) -> str:
        return f"echo:{text}"

    responses = [
        ModelResponse(
            content=None,
            tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "hi"})],
            usage={},
            raw={},
        ),
        ModelResponse(content="done", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(
        model="fake",
        adapter=FakeAdapter(responses),
        run_store=InMemoryRunStore(),
        stream=True,
    )
    worker = Worker(name="Worker", model="fake", tools=[echo])
    report = desk.run(worker, Job(input="run"), stream=True)

    assert report.status == "completed"
    assert report.content == "done"
    assert len(report.tool_calls) == 1
    assert report.tool_calls[0].name == "echo"
    assert report.tool_calls[0].arguments == {"text": "hi"}
    assert report.tool_calls[0].result is not None


def test_streaming_accepts_adapter_model_response_on_resume() -> None:
    @tool(requires_confirmation=True)
    async def risky(action: str) -> str:
        return f"ok:{action}"

    responses = [
        ModelResponse(
            content=None,
            tool_calls=[ToolCall(id="call-1", name="risky", arguments={"action": "go"})],
            usage={},
            raw={},
        ),
        ModelResponse(content="finished", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(
        model="fake",
        adapter=FakeAdapter(responses),
        run_store=InMemoryRunStore(),
        stream=True,
    )
    worker = Worker(name="Worker", model="fake", tools=[risky])
    paused = desk.run(worker, Job(input="run"), stream=True)

    assert paused.status == "paused"
    resumed = desk.resume(paused, True, stream=True)
    assert resumed.status == "completed"
    assert resumed.content == "finished"


def test_streaming_falls_back_when_adapter_returns_non_iterable() -> None:
    class NonIterableStreamAdapter(BaseModelAdapter):
        def __init__(self) -> None:
            self.calls: list[bool] = []

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
            raise RuntimeError("sync path not used")

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
        ) -> ModelResponse | Any:
            self.calls.append(stream)
            if stream:
                return {"unsupported_stream_payload": True}
            return ModelResponse(content="done", tool_calls=[], usage={}, raw={})

    adapter = NonIterableStreamAdapter()
    desk = Desk(model="fake", adapter=adapter, run_store=InMemoryRunStore(), stream=True)
    worker = Worker(name="Worker", model="fake")
    report = desk.run(worker, Job(input="run"), stream=True)

    assert report.status == "completed"
    assert report.content == "done"
    assert adapter.calls == [True, False]


def test_streaming_tool_turn_falls_back_when_streaming_unsupported() -> None:
    class StreamUnsupportedToolAdapter(BaseModelAdapter):
        def __init__(self) -> None:
            self.stream_calls = 0
            self.non_stream_calls = 0

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
            raise RuntimeError("sync path not used")

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
            if stream:
                self.stream_calls += 1
                raise RuntimeError("streaming not supported for tool calls")
            self.non_stream_calls += 1
            if self.non_stream_calls == 1:
                return ModelResponse(
                    content=None,
                    tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "hi"})],
                    usage={},
                    raw={},
                )
            return ModelResponse(content="done", tool_calls=[], usage={}, raw={})

    @tool()
    async def echo(text: str) -> str:
        return f"echo:{text}"

    adapter = StreamUnsupportedToolAdapter()
    desk = Desk(model="fake", adapter=adapter, run_store=InMemoryRunStore(), stream=True)
    worker = Worker(name="Worker", model="fake", tools=[echo])
    report = desk.run(worker, Job(input="run"), stream=True)

    assert report.status == "completed"
    assert report.content == "done"
    assert len(report.tool_calls) == 1
    assert report.tool_calls[0].name == "echo"
    assert report.tool_calls[0].arguments == {"text": "hi"}
    assert report.tool_calls[0].error is None
    assert report.tool_calls[0].result is not None
    assert adapter.stream_calls >= 2
    assert adapter.non_stream_calls == 2


def test_streaming_parallel_tool_calls_do_not_collide_on_position() -> None:
    from tests.utils import StreamingAdapter

    @tool()
    async def tool_a(x: str) -> str:
        return f"a:{x}"

    @tool()
    async def tool_b(y: str) -> str:
        return f"b:{y}"

    streams = [
        [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-a",
                                    "function": {"name": "tool_a", "arguments": ""},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 1,
                                    "id": "call-b",
                                    "function": {"name": "tool_b", "arguments": ""},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": '{"x":"1"}'}}]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 1, "function": {"arguments": '{"y":"2"}'}}]
                        }
                    }
                ]
            },
            {
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"total_tokens": 14},
            },
        ],
        [
            {"choices": [{"delta": {"content": "done"}}]},
            {"choices": [{"delta": {}}], "usage": {"total_tokens": 4}},
        ],
    ]
    desk = Desk(
        model="fake",
        adapter=StreamingAdapter(streams),
        run_store=InMemoryRunStore(),
        stream=True,
    )
    worker = Worker(name="Worker", model="fake", tools=[tool_a, tool_b])
    report = desk.run(worker, Job(input="run"), stream=True)

    assert report.status == "completed"
    assert report.content == "done"
    assert len(report.tool_calls) == 2
    parsed = {call.name: call.arguments for call in report.tool_calls}
    assert parsed["tool_a"] == {"x": "1"}
    assert parsed["tool_b"] == {"y": "2"}
    assert all(call.error is None for call in report.tool_calls)


def test_streaming_tool_calls_fallback_to_message_payload() -> None:
    from tests.utils import StreamingAdapter

    @tool()
    async def echo(text: str) -> str:
        return f"echo:{text}"

    streams = [
        [
            {
                "choices": [
                    {
                        "delta": {"tool_calls": []},
                        "message": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-echo",
                                    "function": {
                                        "name": "echo",
                                        "arguments": '{"text":"hi"}',
                                    },
                                }
                            ]
                        },
                    }
                ]
            },
            {
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"total_tokens": 6},
            },
        ],
        [
            {"choices": [{"delta": {"content": "done"}}]},
            {"choices": [{"delta": {}}], "usage": {"total_tokens": 3}},
        ],
    ]
    desk = Desk(
        model="fake",
        adapter=StreamingAdapter(streams),
        run_store=InMemoryRunStore(),
        stream=True,
    )
    worker = Worker(name="Worker", model="fake", tools=[echo])
    report = desk.run(worker, Job(input="run"), stream=True)

    assert report.status == "completed"
    assert report.content == "done"
    assert len(report.tool_calls) == 1
    assert report.tool_calls[0].name == "echo"
    assert report.tool_calls[0].arguments == {"text": "hi"}
    assert report.tool_calls[0].error is None
    assert report.tool_calls[0].result is not None


def test_streaming_message_fallback_overrides_partial_delta_arguments() -> None:
    from tests.utils import StreamingAdapter

    @tool()
    async def echo(text: str) -> str:
        return f"echo:{text}"

    streams = [
        [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "function": {"name": "echo", "arguments": "{"},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": '"text":"hi"}'}}]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {"tool_calls": []},
                        "message": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "function": {
                                        "name": "echo",
                                        "arguments": '{"text":"hi"}',
                                    },
                                }
                            ]
                        },
                    }
                ]
            },
            {
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"total_tokens": 12},
            },
        ],
        [
            {"choices": [{"delta": {"content": "done"}}]},
            {"choices": [{"delta": {}}], "usage": {"total_tokens": 3}},
        ],
    ]
    desk = Desk(
        model="fake",
        adapter=StreamingAdapter(streams),
        run_store=InMemoryRunStore(),
        stream=True,
    )
    worker = Worker(name="Worker", model="fake", tools=[echo])
    report = desk.run(worker, Job(input="run"), stream=True)

    assert report.status == "completed"
    assert report.content == "done"
    assert len(report.tool_calls) == 1
    assert report.tool_calls[0].name == "echo"
    assert report.tool_calls[0].arguments == {"text": "hi"}
    assert report.tool_calls[0].error is None
    assert report.tool_calls[0].result is not None


def test_streaming_tool_turn_rejects_unsupported_argument_types() -> None:
    from tests.utils import StreamingAdapter

    executions = 0

    @tool()
    async def dangerous() -> str:
        nonlocal executions
        executions += 1
        return "ran"

    streams = [
        [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-danger",
                                    "function": {"name": "dangerous", "arguments": 123},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"total_tokens": 8},
            },
        ],
        [
            {"choices": [{"delta": {"content": "done"}}]},
            {"choices": [{"delta": {}}], "usage": {"total_tokens": 3}},
        ],
    ]
    desk = Desk(
        model="fake",
        adapter=StreamingAdapter(streams),
        run_store=InMemoryRunStore(),
        stream=True,
    )
    worker = Worker(name="Worker", model="fake", tools=[dangerous])
    report = desk.run(worker, Job(input="run"), stream=True)

    assert report.status == "completed"
    assert report.content == "done"
    assert len(report.tool_calls) == 1
    assert report.tool_calls[0].name == "dangerous"
    assert report.tool_calls[0].arguments == {}
    assert report.tool_calls[0].error is not None
    assert "Unsupported tool arguments type: int" in report.tool_calls[0].error
    assert executions == 0


def test_structured_stream_preview_validates_streamed_json() -> None:
    adapter = StructuredStreamPreviewAdapter(
        chunks=[
            {"choices": [{"delta": {"content": '{"answer":"ok"}'}}]},
            {"choices": [{"delta": {}}], "usage": {"total_tokens": 6}},
        ],
        fallback_answer="fallback",
    )
    desk = Desk(
        model="fake",
        adapter=adapter,
        run_store=InMemoryRunStore(),
        stream=True,
        structured_stream_mode="preview",
    )
    worker = Worker(name="Worker", model="fake")
    streamed_tokens: list[str] = []

    def on_token(event: Any) -> None:
        token = event.payload.get("token")
        if isinstance(token, str):
            streamed_tokens.append(token)

    desk.event_bus.subscribe("stream.token", on_token)
    report = desk.run(worker, Job(input="run", response_schema=AnswerModel), stream=True)
    assert report.status == "completed"
    assert report.data is not None
    assert report.data.answer == "ok"
    assert "".join(streamed_tokens) == '{"answer":"ok"}'
    assert adapter.stream_calls == 1
    assert adapter.structured_calls == 0


def test_structured_stream_preview_falls_back_on_invalid_json() -> None:
    adapter = StructuredStreamPreviewAdapter(
        chunks=[
            {"choices": [{"delta": {"content": "not-json"}}]},
            {"choices": [{"delta": {}}], "usage": {"total_tokens": 6}},
        ],
        fallback_answer="fixed",
    )
    desk = Desk(
        model="fake",
        adapter=adapter,
        run_store=InMemoryRunStore(),
        stream=True,
        structured_stream_mode="preview",
    )
    worker = Worker(name="Worker", model="fake")
    streamed_tokens: list[str] = []

    def on_token(event: Any) -> None:
        token = event.payload.get("token")
        if isinstance(token, str):
            streamed_tokens.append(token)

    desk.event_bus.subscribe("stream.token", on_token)
    report = desk.run(worker, Job(input="run", response_schema=AnswerModel), stream=True)
    assert report.status == "completed"
    assert report.data is not None
    assert report.data.answer == "fixed"
    assert "".join(streamed_tokens) == "not-json"
    assert adapter.stream_calls == 1
    assert adapter.structured_calls == 1


def test_structured_output_stream_remains_strict_by_default() -> None:
    class AsyncStructuredOnlyAdapter(BaseModelAdapter):
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
            raise RuntimeError("sync path not used")

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
            raise AssertionError(
                "stream completion should not be used for strict structured output"
            )

        async def astructured_complete(
            self,
            *,
            model: str,
            messages: list[dict[str, Any]],
            response_schema: Any,
            retries: int,
        ) -> Any:
            return response_schema(answer="ok")

    adapter = AsyncStructuredOnlyAdapter()
    desk = Desk(model="fake", adapter=adapter, run_store=InMemoryRunStore(), stream=True)
    worker = Worker(name="Worker", model="fake")
    streamed_tokens: list[str] = []

    def on_token(event: Any) -> None:
        token = event.payload.get("token")
        if isinstance(token, str):
            streamed_tokens.append(token)

    desk.event_bus.subscribe("stream.token", on_token)
    report = desk.run(worker, Job(input="run", response_schema=AnswerModel), stream=True)
    assert report.status == "completed"
    assert report.data is not None
    assert report.data.answer == "ok"
    assert streamed_tokens == []


def test_structured_output_uses_adapter() -> None:
    adapter = StructuredAdapter()
    desk = Desk(model="fake", adapter=adapter, run_store=InMemoryRunStore())
    worker = Worker(name="Worker", model="fake")
    report = desk.run(worker, Job(input="run", response_schema=AnswerModel))
    assert adapter.called is True
    assert report.data is not None
    assert report.data.answer == "ok"


def test_user_input_key_is_respected() -> None:
    @tool(requires_user_input=True, input_key="query")
    def ask(query: str) -> str:
        return f"ok:{query}"

    responses = [
        ModelResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="ask", arguments={"question": "Provide input"})],
            usage={},
            raw={},
        ),
        ModelResponse(content="done", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker = Worker(name="Worker", tools=[ask], model="fake")
    report = desk.run(worker, Job(input="run"))
    assert report.status == "paused"
    resumed = desk.resume(report, "hello")
    assert resumed.status == "completed"
    assert resumed.tool_calls[0].error is None


def test_memory_store_used_in_run() -> None:
    store = RecordingMemoryStore()
    desk = Desk(
        model="fake",
        adapter=FakeAdapter([ModelResponse(content="done", tool_calls=[], usage={}, raw={})]),
        run_store=InMemoryRunStore(),
        memory_store=store,
    )
    worker = Worker(name="Worker", model="fake")
    report = desk.run(worker, Job(input="run"))
    assert report.status == "completed"
    assert ("read", "context") in store.calls
    assert ("write", "last_output") in store.calls


def test_memory_message_keeps_primary_system_message_first() -> None:
    store = RecordingMemoryStore()
    desk = Desk(
        model="fake",
        adapter=FakeAdapter([ModelResponse(content="done", tool_calls=[], usage={}, raw={})]),
        run_store=InMemoryRunStore(),
        memory_store=store,
    )
    worker = Worker(name="Worker", model="fake", instructions="Keep strict format")
    job = Job(
        input="run",
        initial_messages=[
            Message(role="system", content="Primary system"),
            Message(role="user", content="history"),
        ],
    )
    report = desk.run(worker, job)
    assert report.status == "completed"
    assert report.messages[0].role == "system"
    assert "Primary system" in report.messages[0].content
    assert "Keep strict format" in report.messages[0].content
    assert report.messages[1].role == "system"
    content = report.messages[1].content
    assert isinstance(content, str) and content.startswith("Memory:\n")


def test_resume_preserves_stream_flag_when_paused_again() -> None:
    @tool(requires_confirmation=True)
    def risky(action: str) -> str:
        return f"ok:{action}"

    responses = [
        ModelResponse(
            content=None,
            tool_calls=[ToolCall(id="first", name="risky", arguments={"action": "one"})],
            usage={},
            raw={},
        ),
        ModelResponse(
            content=None,
            tool_calls=[ToolCall(id="second", name="risky", arguments={"action": "two"})],
            usage={},
            raw={},
        ),
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker = Worker(name="Worker", tools=[risky], model="fake")
    first = desk.run(worker, Job(input="run"))
    assert first.status == "paused"
    second = desk.resume(first, True, stream=True)
    assert second.status == "paused"
    record = desk.run_store.get_run(first.run_id)
    assert record is not None
    assert record.state is not None
    assert record.state.payload.get("stream") is True


def test_resume_unknown_runner_type_updates_store() -> None:
    desk = Desk(
        model="fake",
        adapter=FakeAdapter([ModelResponse(content="done", tool_calls=[], usage={}, raw={})]),
        run_store=InMemoryRunStore(),
    )
    run_id = "run-unknown"
    job = Job(input="run")
    desk.run_store.create_run(run_id, job.model_dump(mode="json"))
    state = RunState(
        run_id=run_id,
        status="paused",
        runner_type="mystery",
        runner_name="mystery",
        job=job,
    )
    desk.run_store.update_run(run_id, "paused", None, None, state)
    paused_report = Report(
        run_id=run_id,
        status="paused",
        content=None,
        data=None,
        messages=[],
        tool_calls=[],
        metrics={},
        events=[],
        pending_action=None,
        errors=[],
    )
    resumed = desk.resume(paused_report, True)
    assert resumed.status == "failed"
    assert "Unknown runner type" in resumed.errors
    record = desk.run_store.get_run(run_id)
    assert record is not None
    assert record.status == "failed"
    assert any(event.type == "run.failed" for event in desk.run_store.get_events(run_id))


def test_replace_tool_call_missing_id_raises() -> None:
    tool_calls = [ToolCall(id="call-1", name="tool", arguments={})]
    updated = ToolCall(id="missing", name="tool", arguments={})
    with pytest.raises(ValueError):
        replace_tool_call(tool_calls, updated)


def test_multiple_confirmation_tool_calls_adds_error_results_for_remaining() -> None:
    @tool(description="Write a file", requires_confirmation=True)
    def write_file(path: str, content: str) -> str:
        return f"Wrote {path}"

    adapter = FakeAdapter(
        [
            ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1", name="write_file", arguments={"path": "a.py", "content": "1"}
                    ),
                    ToolCall(
                        id="call_2", name="write_file", arguments={"path": "b.py", "content": "2"}
                    ),
                ],
                usage={},
                raw={},
            ),
            ModelResponse(content="done", tool_calls=[], usage={}, raw={}),
        ]
    )
    desk = Desk(model="fake", adapter=adapter, run_store=InMemoryRunStore())
    worker = Worker(name="Worker", model="fake", tools=[write_file])

    report = desk.run(worker, Job(input="create two files"))
    assert report.status == "paused"
    assert report.pending_action is not None
    assert report.pending_action.tool_call.id == "call_1"

    tool_msgs = [m for m in report.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "call_2"
    assert "Skipped" in str(tool_msgs[0].content)

    resumed = desk.resume(report, True)
    assert resumed.status == "completed"


def test_stream_token_events_include_type_field() -> None:
    from tests.utils import StreamingAdapter

    @tool()
    async def echo(text: str) -> str:
        return f"echo:{text}"

    streams = [
        [
            {
                "choices": [
                    {
                        "delta": {
                            "content": "Hello",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "echo", "arguments": '{"text": '},
                                }
                            ],
                        }
                    }
                ]
            },
            {
                "choices": [
                    {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"hi"'}}]}}
                ]
            },
            {
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"total_tokens": 9},
            },
        ],
        [
            {"choices": [{"delta": {"content": "done"}}]},
            {"choices": [{"delta": {}}], "usage": {"total_tokens": 4}},
        ],
    ]
    desk = Desk(
        model="fake",
        adapter=StreamingAdapter(streams),
        run_store=InMemoryRunStore(),
        stream=True,
    )
    worker = Worker(name="Worker", model="fake", tools=[echo])
    token_events: list[tuple[str, str]] = []

    def on_token(event: Any) -> None:
        token = event.payload.get("token")
        token_type = event.payload.get("type")
        if isinstance(token, str) and isinstance(token_type, str):
            token_events.append((token, token_type))

    desk.event_bus.subscribe("stream.token", on_token)
    report = desk.run(worker, Job(input="run"), stream=True)

    assert report.status == "completed"
    assert len(token_events) > 0

    content_tokens = [t for t, ty in token_events if ty == "content"]
    tool_tokens = [t for t, ty in token_events if ty == "tool_argument"]

    assert "Hello" in content_tokens
    assert '{"text": ' in tool_tokens
    assert '"hi"' in tool_tokens
