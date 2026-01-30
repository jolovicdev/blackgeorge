from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from blackgeorge.adapters.litellm_callbacks import (
    _callback_context,
    emit_llm_completed,
    emit_llm_failed,
    emit_llm_started,
)


@contextmanager
def _set_callback_context(run_id: str, emit: Callable[[str, str, dict[str, Any]], None]):
    original = _callback_context.get()
    _callback_context.set({"run_id": run_id, "emit": emit})
    try:
        yield
    finally:
        _callback_context.set(original)


def test_emit_llm_started_event() -> None:
    events: list[dict[str, Any]] = []

    def emit(event_type: str, source: str, payload: dict[str, Any]) -> None:
        events.append({"type": event_type, "source": source, "payload": payload})

    with _set_callback_context("test-run-123", emit):
        emit_llm_started("gpt-5-nano", messages_count=2, tools_count=2)

    assert len(events) == 1
    assert events[0]["type"] == "llm.started"
    assert events[0]["source"] == "litellm_adapter"
    assert events[0]["payload"]["model"] == "gpt-5-nano"
    assert events[0]["payload"]["messages_count"] == 2
    assert events[0]["payload"]["tools_count"] == 2


def test_emit_llm_completed_event() -> None:
    events: list[dict[str, Any]] = []

    def emit(event_type: str, source: str, payload: dict[str, Any]) -> None:
        events.append({"type": event_type, "source": source, "payload": payload})

    response_obj = {
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }
    }

    with _set_callback_context("test-run-123", emit):
        emit_llm_started("gpt-5-nano", messages_count=1, tools_count=0)
        emit_llm_completed("gpt-5-nano", response_obj)

    assert len(events) == 2
    assert events[1]["type"] == "llm.completed"
    assert events[1]["source"] == "litellm_adapter"
    assert events[1]["payload"]["model"] == "gpt-5-nano"
    assert events[1]["payload"]["prompt_tokens"] == 10
    assert events[1]["payload"]["completion_tokens"] == 20
    assert events[1]["payload"]["total_tokens"] == 30
    assert "latency_ms" in events[1]["payload"]


def test_emit_llm_failed_event() -> None:
    events: list[dict[str, Any]] = []

    def emit(event_type: str, source: str, payload: dict[str, Any]) -> None:
        events.append({"type": event_type, "source": source, "payload": payload})

    exception = ValueError("test error")

    with _set_callback_context("test-run-123", emit):
        emit_llm_started("gpt-5-nano", messages_count=1, tools_count=0)
        emit_llm_failed("gpt-5-nano", exception)

    assert len(events) == 2
    assert events[1]["type"] == "llm.failed"
    assert events[1]["source"] == "litellm_adapter"
    assert events[1]["payload"]["model"] == "gpt-5-nano"
    assert events[1]["payload"]["error_type"] == "ValueError"
    assert events[1]["payload"]["error_message"] == "test error"
    assert "latency_ms" in events[1]["payload"]


def test_emit_with_no_context() -> None:
    emit_llm_started("gpt-5-nano", messages_count=1, tools_count=0)
    emit_llm_completed("gpt-5-nano", {})
    emit_llm_failed("gpt-5-nano", None)


def test_emit_with_pydantic_usage() -> None:
    from pydantic import BaseModel

    class Usage(BaseModel):
        prompt_tokens: int
        completion_tokens: int
        total_tokens: int

    events: list[dict[str, Any]] = []

    def emit(event_type: str, source: str, payload: dict[str, Any]) -> None:
        events.append({"type": event_type, "source": source, "payload": payload})

    response_obj = {"usage": Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)}

    with _set_callback_context("test-run-123", emit):
        emit_llm_started("gpt-5-nano", messages_count=1, tools_count=0)
        emit_llm_completed("gpt-5-nano", response_obj)

    assert events[1]["payload"]["prompt_tokens"] == 10
    assert events[1]["payload"]["completion_tokens"] == 20
    assert events[1]["payload"]["total_tokens"] == 30


def test_emit_with_dict_response() -> None:
    events: list[dict[str, Any]] = []

    def emit(event_type: str, source: str, payload: dict[str, Any]) -> None:
        events.append({"type": event_type, "source": source, "payload": payload})

    response_obj = {"usage": {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40}}

    with _set_callback_context("test-run-123", emit):
        emit_llm_started("gpt-5-nano", messages_count=1, tools_count=0)
        emit_llm_completed("gpt-5-nano", response_obj)

    assert events[1]["payload"]["prompt_tokens"] == 15
    assert events[1]["payload"]["completion_tokens"] == 25
    assert events[1]["payload"]["total_tokens"] == 40
