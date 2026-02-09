import json
import threading
from typing import Any

import pytest
from pydantic import BaseModel

from blackgeorge.adapters.base import BaseModelAdapter, ModelResponse
from blackgeorge.core.job import Job
from blackgeorge.core.message import Message
from blackgeorge.core.report import Report
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.desk import Desk
from blackgeorge.memory.base import MemoryScope, MemoryStore
from blackgeorge.store.in_memory import InMemoryRunStore
from blackgeorge.store.state import RunState
from blackgeorge.tools import tool
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
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker = Worker(name="Worker", tools=[risky], model="fake")
    tool_failed_payloads: list[dict[str, Any]] = []

    def on_tool_failed(event: Any) -> None:
        tool_failed_payloads.append(event.payload)

    desk.event_bus.subscribe("tool.failed", on_tool_failed)
    report = desk.run(worker, Job(input="run"))
    assert report.status == "paused"
    resumed = desk.resume(report, False)
    assert resumed.status == "completed"
    assert any(
        payload.get("error") == "Tool execution declined" for payload in tool_failed_payloads
    )


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
