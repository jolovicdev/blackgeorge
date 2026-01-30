import contextvars
import time
import warnings
from contextlib import suppress
from datetime import datetime
from typing import Any

import litellm


def _compute_latency_ms(
    start_time: float | datetime | None,
    end_time: float | datetime | None,
    fallback_start: float | None = None,
) -> int:
    if start_time is None:
        start_time = fallback_start
    if end_time is None:
        end_time = time.time()
    if start_time is None:
        start_time = end_time

    if isinstance(start_time, datetime):
        start_time = start_time.timestamp()
    if isinstance(end_time, datetime):
        end_time = end_time.timestamp()

    return int((end_time - start_time) * 1000)


_callback_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "_callback_context", default=None
)


def emit_llm_started(model: str, messages_count: int, tools_count: int) -> None:
    context = _callback_context.get()
    if not context:
        return
    emit = context.get("emit")
    if not emit:
        return
    context["start_time"] = time.time()
    emit(
        "llm.started",
        "litellm_adapter",
        {
            "model": model,
            "messages_count": messages_count,
            "tools_count": tools_count,
        },
    )


def emit_llm_completed(model: str, response: Any) -> None:
    context = _callback_context.get()
    if not context:
        return
    emit = context.get("emit")
    if not emit:
        return

    start_time = context.get("start_time")
    end_time = time.time()
    latency_ms = _compute_latency_ms(start_time, end_time, None)

    usage: dict[str, Any] = {}
    if hasattr(response, "usage"):
        usage_data = response.usage
        if isinstance(usage_data, dict):
            usage = usage_data
        else:
            usage = (
                usage_data.model_dump(mode="json", warnings=False)
                if hasattr(usage_data, "model_dump")
                else {}
            )
    elif isinstance(response, dict):
        usage_data = response.get("usage", {}) or {}
        if isinstance(usage_data, dict):
            usage = usage_data
        else:
            usage = (
                usage_data.model_dump(mode="json", warnings=False)
                if hasattr(usage_data, "model_dump")
                else {}
            )

    cost: float | None = None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
        with suppress(Exception):
            cost = litellm.completion_cost(completion_response=response)

    payload: dict[str, Any] = {
        "model": model,
        "latency_ms": latency_ms,
    }
    if usage:
        payload["prompt_tokens"] = usage.get("prompt_tokens", 0)
        payload["completion_tokens"] = usage.get("completion_tokens", 0)
        payload["total_tokens"] = usage.get("total_tokens", 0)
    if cost is not None:
        payload["cost"] = cost

    emit("llm.completed", "litellm_adapter", payload)


def emit_llm_failed(model: str, exception: Exception | None) -> None:
    context = _callback_context.get()
    if not context:
        return
    emit = context.get("emit")
    if not emit:
        return

    start_time = context.get("start_time")
    end_time = time.time()
    latency_ms = _compute_latency_ms(start_time, end_time, None)

    error_type = type(exception).__name__ if exception else "UnknownError"
    error_message = str(exception) if exception else "Unknown error"

    emit(
        "llm.failed",
        "litellm_adapter",
        {
            "model": model,
            "error_type": error_type,
            "error_message": error_message,
            "latency_ms": latency_ms,
        },
    )
