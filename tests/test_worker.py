from typing import Any

from pydantic import BaseModel

from blackgeorge.adapters.base import BaseModelAdapter, ModelResponse
from blackgeorge.core.job import Job
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.desk import Desk
from blackgeorge.memory.base import MemoryScope, MemoryStore
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


def test_structured_output_uses_adapter() -> None:
    adapter = StructuredAdapter()
    desk = Desk(model="fake", adapter=adapter, run_store=InMemoryRunStore())
    worker = Worker(name="Worker", model="fake")
    report = desk.run(worker, Job(input="run", response_schema=AnswerModel))
    assert adapter.called is True
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
