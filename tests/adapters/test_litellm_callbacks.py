from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from blackgeorge.adapters.litellm_callbacks import LiteLLMCallbackHandler, _callback_context


@contextmanager
def _set_callback_context(run_id: str, emit: Callable[[str, str, dict[str, Any]], None]):
    """Helper to set callback context for testing."""
    original = _callback_context.get()
    _callback_context.set({"run_id": run_id, "emit": emit})
    try:
        yield
    finally:
        _callback_context.set(original)


def test_callback_handler_emits_llm_started_event() -> None:
    events: list[dict[str, Any]] = []

    def emit(event_type: str, source: str, payload: dict[str, Any]) -> None:
        events.append({"type": event_type, "source": source, "payload": payload})

    handler = LiteLLMCallbackHandler()

    with _set_callback_context("test-run-123", emit):
        handler.log_pre_api_call(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "hello"}, {"role": "user", "content": "world"}],
            kwargs={"tools": [{"name": "tool1"}, {"name": "tool2"}]},
        )

    assert len(events) == 1
    assert events[0]["type"] == "llm.started"
    assert events[0]["source"] == "litellm_adapter"
    assert events[0]["payload"]["model"] == "gpt-5-nano"
    assert events[0]["payload"]["messages_count"] == 2
    assert events[0]["payload"]["tools_count"] == 2


def test_callback_handler_emits_llm_completed_event() -> None:
    events: list[dict[str, Any]] = []

    def emit(event_type: str, source: str, payload: dict[str, Any]) -> None:
        events.append({"type": event_type, "source": source, "payload": payload})

    handler = LiteLLMCallbackHandler()

    response_obj = {
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }
    }

    with _set_callback_context("test-run-123", emit):
        handler.log_success_event(
            kwargs={"model": "gpt-5-nano"},
            response_obj=response_obj,
            start_time=1000.0,
            end_time=1001.5,
        )

    assert len(events) == 1
    assert events[0]["type"] == "llm.completed"
    assert events[0]["source"] == "litellm_adapter"
    assert events[0]["payload"]["model"] == "gpt-5-nano"
    assert events[0]["payload"]["latency_ms"] == 1500
    assert events[0]["payload"]["prompt_tokens"] == 10
    assert events[0]["payload"]["completion_tokens"] == 20
    assert events[0]["payload"]["total_tokens"] == 30


def test_callback_handler_emits_llm_failed_event() -> None:
    events: list[dict[str, Any]] = []

    def emit(event_type: str, source: str, payload: dict[str, Any]) -> None:
        events.append({"type": event_type, "source": source, "payload": payload})

    handler = LiteLLMCallbackHandler()

    exception = ValueError("test error")

    with _set_callback_context("test-run-123", emit):
        handler.log_failure_event(
            kwargs={"model": "gpt-5-nano", "exception": exception},
            response_obj=None,
            start_time=1000.0,
            end_time=1000.5,
        )

    assert len(events) == 1
    assert events[0]["type"] == "llm.failed"
    assert events[0]["source"] == "litellm_adapter"
    assert events[0]["payload"]["model"] == "gpt-5-nano"
    assert events[0]["payload"]["error_type"] == "ValueError"
    assert events[0]["payload"]["error_message"] == "test error"
    assert events[0]["payload"]["latency_ms"] == 500


def test_callback_handler_with_no_tools() -> None:
    events: list[dict[str, Any]] = []

    def emit(event_type: str, source: str, payload: dict[str, Any]) -> None:
        events.append({"type": event_type, "source": source, "payload": payload})

    handler = LiteLLMCallbackHandler()

    with _set_callback_context("test-run-123", emit):
        handler.log_pre_api_call(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "hello"}],
            kwargs={},
        )

    assert events[0]["payload"]["tools_count"] == 0


def test_callback_handler_async_methods() -> None:
    import asyncio

    events: list[dict[str, Any]] = []

    def emit(event_type: str, source: str, payload: dict[str, Any]) -> None:
        events.append({"type": event_type, "source": source, "payload": payload})

    handler = LiteLLMCallbackHandler()

    async def run_async_tests() -> None:
        with _set_callback_context("test-run-123", emit):
            await handler.async_log_pre_api_call(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": "hello"}],
                kwargs={},
            )

            response_obj = {
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 10,
                    "total_tokens": 15,
                }
            }

            await handler.async_log_success_event(
                kwargs={"model": "gpt-5-nano"},
                response_obj=response_obj,
                start_time=1000.0,
                end_time=1001.0,
            )

    asyncio.run(run_async_tests())

    assert len(events) == 2
    assert events[0]["type"] == "llm.started"
    assert events[1]["type"] == "llm.completed"


def test_callback_handler_with_pydantic_usage() -> None:
    from pydantic import BaseModel

    class Usage(BaseModel):
        prompt_tokens: int
        completion_tokens: int
        total_tokens: int

    events: list[dict[str, Any]] = []

    def emit(event_type: str, source: str, payload: dict[str, Any]) -> None:
        events.append({"type": event_type, "source": source, "payload": payload})

    handler = LiteLLMCallbackHandler()

    response_obj_with_pydantic = {
        "usage": Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    }

    with _set_callback_context("test-run-123", emit):
        handler.log_success_event(
            kwargs={"model": "gpt-5-nano"},
            response_obj=response_obj_with_pydantic,
            start_time=1000.0,
            end_time=1001.0,
        )

    assert events[0]["payload"]["prompt_tokens"] == 10
    assert events[0]["payload"]["completion_tokens"] == 20
    assert events[0]["payload"]["total_tokens"] == 30


def test_callback_handler_with_dict_response() -> None:
    events: list[dict[str, Any]] = []

    def emit(event_type: str, source: str, payload: dict[str, Any]) -> None:
        events.append({"type": event_type, "source": source, "payload": payload})

    handler = LiteLLMCallbackHandler()

    response_obj_dict = {
        "usage": {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40}
    }

    with _set_callback_context("test-run-123", emit):
        handler.log_success_event(
            kwargs={"model": "gpt-5-nano"},
            response_obj=response_obj_dict,
            start_time=1000.0,
            end_time=1001.0,
        )

    assert events[0]["payload"]["prompt_tokens"] == 15
    assert events[0]["payload"]["completion_tokens"] == 25
    assert events[0]["payload"]["total_tokens"] == 40
