import threading
import time
from types import SimpleNamespace

import instructor
import litellm
import pytest
from pydantic import BaseModel, TypeAdapter

from blackgeorge.adapters.instructor_client import InstructorClientPool, instructor_clients
from blackgeorge.adapters.litellm import LiteLLMAdapter, _parse_tool_calls


class AnswerModel(BaseModel):
    answer: str


class ItemModel(BaseModel):
    value: int


def test_parse_tool_calls_with_valid_json() -> None:
    message = {
        "tool_calls": [
            {
                "id": "call_123",
                "function": {
                    "name": "test_tool",
                    "arguments": '{"param1": "value1", "param2": 42}',
                },
            }
        ]
    }

    calls = _parse_tool_calls(message)

    assert len(calls) == 1
    assert calls[0].id == "call_123"
    assert calls[0].name == "test_tool"
    assert calls[0].arguments == {"param1": "value1", "param2": 42}
    assert calls[0].error is None


def test_parse_tool_calls_with_invalid_json() -> None:
    message = {
        "tool_calls": [
            {
                "id": "call_456",
                "function": {
                    "name": "test_tool",
                    "arguments": '{"param1": invalid json here}',
                },
            }
        ]
    }

    calls = _parse_tool_calls(message)

    assert len(calls) == 1
    assert calls[0].id == "call_456"
    assert calls[0].name == "test_tool"
    assert calls[0].arguments == {}
    assert calls[0].error is not None
    assert "Invalid JSON" in calls[0].error
    assert "invalid json" in calls[0].error


def test_parse_tool_calls_with_empty_arguments() -> None:
    message = {
        "tool_calls": [
            {
                "id": "call_789",
                "function": {
                    "name": "test_tool",
                    "arguments": "",
                },
            }
        ]
    }

    calls = _parse_tool_calls(message)

    assert len(calls) == 1
    assert calls[0].arguments == {}
    assert calls[0].error is None


def test_parse_tool_calls_with_dict_arguments() -> None:
    message = {
        "tool_calls": [
            {
                "id": "call_abc",
                "function": {
                    "name": "test_tool",
                    "arguments": {"param1": "value1"},
                },
            }
        ]
    }

    calls = _parse_tool_calls(message)

    assert len(calls) == 1
    assert calls[0].arguments == {"param1": "value1"}
    assert calls[0].error is None


def test_parse_tool_calls_with_attribute_objects() -> None:
    message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                id="call_attr",
                function=SimpleNamespace(
                    name="test_tool",
                    arguments='{"param1":"value1","param2":42}',
                ),
            )
        ]
    )

    calls = _parse_tool_calls(message)

    assert len(calls) == 1
    assert calls[0].id == "call_attr"
    assert calls[0].name == "test_tool"
    assert calls[0].arguments == {"param1": "value1", "param2": 42}
    assert calls[0].error is None


def test_parse_tool_calls_generates_id_if_missing() -> None:
    message = {
        "tool_calls": [
            {
                "function": {
                    "name": "test_tool",
                    "arguments": "{}",
                },
            }
        ]
    }

    calls = _parse_tool_calls(message)

    assert len(calls) == 1
    assert calls[0].id is not None
    assert len(calls[0].id) > 0


def test_parse_tool_calls_missing_name_sets_error() -> None:
    message = {
        "tool_calls": [
            {
                "id": "call_001",
                "function": {
                    "arguments": "{}",
                },
            }
        ]
    }

    calls = _parse_tool_calls(message)

    assert len(calls) == 1
    assert calls[0].name == ""
    assert calls[0].error is not None
    assert "Missing tool name" in calls[0].error


def test_parse_tool_calls_non_list_returns_empty() -> None:
    message = {"tool_calls": {"id": "call_001"}}
    calls = _parse_tool_calls(message)
    assert calls == []


def test_parse_tool_calls_unsupported_arguments_type_sets_error() -> None:
    message = {
        "tool_calls": [
            {
                "id": "call_001",
                "function": {
                    "name": "test_tool",
                    "arguments": 123,
                },
            }
        ]
    }

    calls = _parse_tool_calls(message)

    assert len(calls) == 1
    assert calls[0].name == "test_tool"
    assert calls[0].arguments == {}
    assert calls[0].error is not None
    assert "Unsupported tool arguments type" in calls[0].error


def test_instructor_client_pool_thread_safe(monkeypatch) -> None:
    pool = InstructorClientPool()
    call_count = {"value": 0}
    start_event = threading.Event()
    release_event = threading.Event()
    count_lock = threading.Lock()

    def fake_from_provider(provider: str, async_client: bool = False) -> object:
        with count_lock:
            call_count["value"] += 1
            start_event.set()
        release_event.wait(timeout=1)
        return object()

    monkeypatch.setattr(instructor, "from_provider", fake_from_provider)

    results: list[object] = []
    barrier = threading.Barrier(5)

    def worker() -> None:
        barrier.wait()
        client = pool.get("openai/gpt-5-nano", async_client=False)
        results.append(client)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    start_event.wait(timeout=1)
    time.sleep(0.05)
    release_event.set()
    for thread in threads:
        thread.join()

    assert call_count["value"] == 1
    assert len({id(result) for result in results}) == 1


def test_litellm_parallel_tool_calls_enabled(monkeypatch) -> None:
    adapter = LiteLLMAdapter()
    captured: dict[str, object] = {}

    def fake_completion(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok", "tool_calls": []}}], "usage": {}}

    monkeypatch.setattr(
        litellm,
        "model_cost",
        {"openai/gpt-5-nano": {"supports_parallel_function_calling": True}},
    )
    monkeypatch.setattr(litellm, "completion", fake_completion)

    adapter.complete(
        model="openai/gpt-5-nano",
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "function": {"name": "tool", "parameters": {"type": "object", "properties": {}}},
            }
        ],
        tool_choice="auto",
        temperature=None,
        max_tokens=None,
        stream=False,
        stream_options=None,
    )

    assert captured.get("parallel_tool_calls") is True


def test_litellm_marks_async_cleanup_as_configured(monkeypatch) -> None:
    adapter = LiteLLMAdapter()
    events: list[str] = []

    def emit(event_type: str, source: str, payload: dict[str, object]) -> None:
        events.append(event_type)

    def fake_completion(**kwargs: object) -> dict[str, object]:
        return {"choices": [{"message": {"content": "ok", "tool_calls": []}}], "usage": {}}

    monkeypatch.setattr(litellm, "completion", fake_completion)

    adapter.set_callback_context("run-cleanup-check", emit)
    adapter.complete(
        model="openai/gpt-5-nano",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        tool_choice=None,
        temperature=None,
        max_tokens=None,
        stream=False,
        stream_options=None,
    )
    adapter.clear_callback_context()

    assert litellm.__dict__.get("_async_client_cleanup_registered") is True
    assert "llm.completed" in events


def test_litellm_omits_tool_params_when_tools_absent(monkeypatch) -> None:
    adapter = LiteLLMAdapter()
    captured: dict[str, object] = {}

    def fake_completion(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok", "tool_calls": []}}], "usage": {}}

    monkeypatch.setattr(litellm, "completion", fake_completion)

    adapter.complete(
        model="openai/gpt-5-nano",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        tool_choice="auto",
        temperature=None,
        max_tokens=None,
        stream=False,
        stream_options={"include_usage": True},
    )

    assert "tools" not in captured
    assert "tool_choice" not in captured
    assert "stream_options" not in captured


def test_litellm_stream_emits_completed_after_consumption(monkeypatch) -> None:
    adapter = LiteLLMAdapter()
    events: list[str] = []
    payloads: list[dict[str, object]] = []

    def emit(event_type: str, source: str, payload: dict[str, object]) -> None:
        events.append(event_type)
        payloads.append(payload)

    def fake_completion(**kwargs: object) -> object:
        return iter(
            [
                {"choices": [{"delta": {"content": "hello"}}]},
                {"choices": [{"delta": {}}], "usage": {"total_tokens": 7}},
            ]
        )

    monkeypatch.setattr(litellm, "completion", fake_completion)
    adapter.set_callback_context("run-1", emit)
    stream = adapter.complete(
        model="openai/gpt-5-nano",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        tool_choice=None,
        temperature=None,
        max_tokens=None,
        stream=True,
        stream_options={"include_usage": True},
    )
    list(stream)
    adapter.clear_callback_context()

    assert "llm.started" in events
    assert "llm.completed" in events
    completed_payload = payloads[events.index("llm.completed")]
    assert completed_payload.get("total_tokens") == 7


def test_litellm_stream_emits_failed_on_iteration_error(monkeypatch) -> None:
    adapter = LiteLLMAdapter()
    events: list[str] = []

    def emit(event_type: str, source: str, payload: dict[str, object]) -> None:
        events.append(event_type)

    def failing_stream() -> object:
        def iterator() -> object:
            yield {"choices": [{"delta": {"content": "hello"}}]}
            raise RuntimeError("stream exploded")

        return iterator()

    def fake_completion(**kwargs: object) -> object:
        return failing_stream()

    monkeypatch.setattr(litellm, "completion", fake_completion)
    adapter.set_callback_context("run-2", emit)
    stream = adapter.complete(
        model="openai/gpt-5-nano",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        tool_choice=None,
        temperature=None,
        max_tokens=None,
        stream=True,
        stream_options={"include_usage": True},
    )
    with pytest.raises(RuntimeError, match="stream exploded"):
        list(stream)
    adapter.clear_callback_context()

    assert "llm.started" in events
    assert "llm.failed" in events


def test_structured_complete_uses_response_format_base_model(monkeypatch) -> None:
    adapter = LiteLLMAdapter()
    captured: dict[str, object] = {}

    def fake_completion(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"choices": [{"message": {"content": '{"answer": "ok"}'}}], "usage": {}}

    def fake_get(model: str, async_client: bool) -> object:
        raise AssertionError("Instructor should not be used")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(instructor_clients, "get", fake_get)

    result = adapter.structured_complete(
        model="openai/gpt-5-nano",
        messages=[{"role": "user", "content": "hi"}],
        response_schema=AnswerModel,
        retries=0,
    )

    assert isinstance(result, AnswerModel)
    assert result.answer == "ok"
    response_format = captured.get("response_format")
    assert isinstance(response_format, dict)
    assert response_format.get("type") == "json_schema"
    json_schema = response_format.get("json_schema")
    assert isinstance(json_schema, dict)
    assert json_schema.get("strict") is True


def test_structured_complete_uses_response_format_type_adapter(monkeypatch) -> None:
    adapter = LiteLLMAdapter()
    response_schema = TypeAdapter(list[ItemModel])

    def fake_completion(**kwargs: object) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": '[{"value": 1}, {"value": 2}]',
                    }
                }
            ],
            "usage": {},
        }

    def fake_get(model: str, async_client: bool) -> object:
        raise AssertionError("Instructor should not be used")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(instructor_clients, "get", fake_get)

    result = adapter.structured_complete(
        model="openai/gpt-5-nano",
        messages=[{"role": "user", "content": "hi"}],
        response_schema=response_schema,
        retries=0,
    )

    assert isinstance(result, list)
    assert [item.value for item in result] == [1, 2]


def test_structured_complete_type_adapter_manual_fallback(monkeypatch) -> None:
    adapter = LiteLLMAdapter()
    response_schema = TypeAdapter(list[ItemModel])
    calls: list[dict[str, object]] = []

    def fake_completion(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("response_format unsupported")
        return {
            "choices": [
                {
                    "message": {
                        "content": '[{"value": 3}]',
                    }
                }
            ],
            "usage": {},
        }

    def fake_get(model: str, async_client: bool) -> object:
        raise AssertionError("Instructor should not be used")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(instructor_clients, "get", fake_get)

    result = adapter.structured_complete(
        model="openai/gpt-5-nano",
        messages=[{"role": "user", "content": "hi"}],
        response_schema=response_schema,
        retries=0,
    )

    assert [item.value for item in result] == [3]
    assert len(calls) == 2
    assert "response_format" not in calls[1]


@pytest.mark.asyncio
async def test_astructured_complete_type_adapter_manual_fallback(monkeypatch) -> None:
    adapter = LiteLLMAdapter()
    response_schema = TypeAdapter(list[ItemModel])
    calls: list[dict[str, object]] = []

    async def fake_acompletion(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("response_format unsupported")
        return {
            "choices": [
                {
                    "message": {
                        "content": '[{"value": 4}]',
                    }
                }
            ],
            "usage": {},
        }

    def fake_get(model: str, async_client: bool) -> object:
        raise AssertionError("Instructor should not be used")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(instructor_clients, "get", fake_get)

    result = await adapter.astructured_complete(
        model="openai/gpt-5-nano",
        messages=[{"role": "user", "content": "hi"}],
        response_schema=response_schema,
        retries=0,
    )

    assert [item.value for item in result] == [4]
    assert len(calls) == 2
    assert "response_format" not in calls[1]


def test_structured_complete_retry_floor(monkeypatch) -> None:
    adapter = LiteLLMAdapter()

    class CallCounter:
        def __init__(self) -> None:
            self.calls = 0

    class FakeCompletions:
        def __init__(self, counter: CallCounter) -> None:
            self._counter = counter

        def create(self, **kwargs: object) -> object:
            self._counter.calls += 1
            raise ValueError("bad response")

    class FakeChat:
        def __init__(self, counter: CallCounter) -> None:
            self.completions = FakeCompletions(counter)

    class FakeClient:
        def __init__(self, counter: CallCounter) -> None:
            self.chat = FakeChat(counter)

    counter = CallCounter()

    def fake_completion(**kwargs: object) -> dict[str, object]:
        raise RuntimeError("response_format failed")

    def fake_get(model: str, async_client: bool) -> object:
        return FakeClient(counter)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(instructor_clients, "get", fake_get)

    with pytest.raises(ValueError):
        adapter.structured_complete(
            model="openai/gpt-5-nano",
            messages=[{"role": "user", "content": "hi"}],
            response_schema=AnswerModel,
            retries=0,
        )

    assert counter.calls == 4


@pytest.mark.asyncio
async def test_litellm_async_omits_tool_params_when_tools_absent(monkeypatch) -> None:
    adapter = LiteLLMAdapter()
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok", "tool_calls": []}}], "usage": {}}

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    await adapter.acomplete(
        model="openai/gpt-5-nano",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        tool_choice="required",
        temperature=None,
        max_tokens=None,
        stream=False,
        stream_options={"include_usage": True},
    )

    assert "tools" not in captured
    assert "tool_choice" not in captured
    assert "stream_options" not in captured


@pytest.mark.asyncio
async def test_litellm_async_stream_emits_completed_after_consumption(monkeypatch) -> None:
    adapter = LiteLLMAdapter()
    events: list[str] = []
    payloads: list[dict[str, object]] = []

    def emit(event_type: str, source: str, payload: dict[str, object]) -> None:
        events.append(event_type)
        payloads.append(payload)

    async def stream_generator() -> object:
        yield {"choices": [{"delta": {"content": "hello"}}]}
        yield {"choices": [{"delta": {}}], "usage": {"total_tokens": 9}}

    async def fake_acompletion(**kwargs: object) -> object:
        return stream_generator()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    adapter.set_callback_context("run-3", emit)
    stream = await adapter.acomplete(
        model="openai/gpt-5-nano",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        tool_choice=None,
        temperature=None,
        max_tokens=None,
        stream=True,
        stream_options={"include_usage": True},
    )
    async for _ in stream:
        pass
    adapter.clear_callback_context()

    assert "llm.started" in events
    assert "llm.completed" in events
    completed_payload = payloads[events.index("llm.completed")]
    assert completed_payload.get("total_tokens") == 9
