import contextvars
import time
import warnings
from contextlib import suppress
from typing import Any

import litellm
from litellm.integrations.custom_logger import CustomLogger

_callback_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "_callback_context", default=None
)


class LiteLLMCallbackHandler(CustomLogger):
    def log_pre_api_call(
        self, model: str, messages: list[dict[str, Any]], kwargs: dict[str, Any]
    ) -> None:
        context = _callback_context.get()
        if not context:
            return

        self.start_time = time.time()
        emit = context.get("emit")
        if not emit:
            return
        tools = kwargs.get("tools")
        emit(
            "llm.started",
            "litellm_adapter",
            {
                "model": model,
                "messages_count": len(messages),
                "tools_count": len(tools) if tools else 0,
            },
        )

    async def async_log_pre_api_call(
        self, model: str, messages: list[dict[str, Any]], kwargs: dict[str, Any]
    ) -> None:
        self.log_pre_api_call(model, messages, kwargs)

    def log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float | None,
        end_time: float | None,
    ) -> None:
        context = _callback_context.get()
        if not context:
            return

        emit = context.get("emit")
        if not emit:
            return

        if start_time is None and hasattr(self, "start_time"):
            start_time = self.start_time
        if end_time is None:
            end_time = time.time()
        if start_time is None:
            start_time = end_time

        latency_ms = int((end_time - start_time) * 1000)
        model = kwargs.get("model", "unknown")

        usage: dict[str, Any] = {}
        if hasattr(response_obj, "usage"):
            usage_data = response_obj.usage
            if isinstance(usage_data, dict):
                usage = usage_data
            else:
                usage = (
                    usage_data.model_dump(mode="json", warnings=False)
                    if hasattr(usage_data, "model_dump")
                    else {}
                )
        elif isinstance(response_obj, dict):
            usage_data = response_obj.get("usage", {}) or {}
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
                cost = litellm.completion_cost(completion_response=response_obj)

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

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float | None,
        end_time: float | None,
    ) -> None:
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    def log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float | None,
        end_time: float | None,
    ) -> None:
        context = _callback_context.get()
        if not context:
            return

        emit = context.get("emit")
        if not emit:
            return

        if start_time is None and hasattr(self, "start_time"):
            start_time = self.start_time
        if end_time is None:
            end_time = time.time()
        if start_time is None:
            start_time = end_time

        latency_ms = int((end_time - start_time) * 1000)
        model = kwargs.get("model", "unknown")

        exception = kwargs.get("exception")
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

    async def async_log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float | None,
        end_time: float | None,
    ) -> None:
        self.log_failure_event(kwargs, response_obj, start_time, end_time)


_global_handler: LiteLLMCallbackHandler | None = None


def get_global_callback_handler() -> LiteLLMCallbackHandler:
    global _global_handler
    if _global_handler is None:
        _global_handler = LiteLLMCallbackHandler()
        if not litellm.callbacks:
            litellm.callbacks = []
        if _global_handler not in litellm.callbacks:
            litellm.callbacks.append(_global_handler)
    return _global_handler
