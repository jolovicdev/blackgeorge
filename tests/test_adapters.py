import threading
import time

import instructor
import litellm

from blackgeorge.adapters.instructor_client import InstructorClientPool
from blackgeorge.adapters.litellm import LiteLLMAdapter, _parse_tool_calls


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
    assert calls[0].arguments == {}
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
        client = pool.get("gpt-4", async_client=False)
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

    def fake_supports(model: str) -> bool:
        return True

    def fake_completion(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok", "tool_calls": []}}], "usage": {}}

    monkeypatch.setattr(litellm, "supports_parallel_function_calling", fake_supports)
    monkeypatch.setattr(litellm, "completion", fake_completion)

    adapter.complete(
        model="openai/gpt-4o",
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
