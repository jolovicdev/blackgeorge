from typing import Any

from blackgeorge.adapters.base import BaseModelAdapter, ModelResponse
from blackgeorge.core.job import Job
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.desk import Desk
from blackgeorge.store.in_memory import InMemoryRunStore
from blackgeorge.tools import tool
from blackgeorge.worker import Worker
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


