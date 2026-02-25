from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from blackgeorge.adapters.base import BaseModelAdapter
from blackgeorge.core.event import Event
from blackgeorge.core.event_types import EventType
from blackgeorge.core.message import Message
from blackgeorge.core.report import Report
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.worker_context import aapply_context_summary
from blackgeorge.worker_runner_helpers import (
    EventEmitter,
    _acontext_retry,
    _fail_report,
    _finalize_plain_response,
    _finalize_structured_response,
    _record_usage,
)

ContextRetryHandler = Callable[[], Any]


@dataclass
class LoopState:
    run_id: str
    worker_name: str
    messages: list[Message]
    tool_calls: list[ToolCall]
    metrics: dict[str, Any]
    events: list[Event]
    errors: list[str]
    iteration: int = 0
    context_summaries: int = 0
    model_registered: bool = False

    def increment_iteration(self) -> int:
        self.iteration += 1
        return self.iteration

    def increment_context_summaries(self) -> int:
        self.context_summaries += 1
        return self.context_summaries


@dataclass
class CompletionContext:
    adapter: BaseModelAdapter
    model_name: str
    temperature: float | None
    max_tokens: int | None
    stream_options: dict[str, Any] | None
    run_id: str
    emit: EventEmitter
    state: LoopState
    respect_context_window: bool = True

    def make_on_token(self) -> Callable[[str], None]:
        def on_token(token: str) -> None:
            self.emit(EventType.STREAM_TOKEN, self.state.worker_name, {"token": token})

        return on_token

    async def handle_context_limit(
        self,
        exc: Exception,
        apply_summary: Callable[[], Any],
    ) -> Report | None:
        decision = await _acontext_retry(
            run_id=self.state.run_id,
            worker_name=self.state.worker_name,
            messages=self.state.messages,
            tool_calls=self.state.tool_calls,
            metrics=self.state.metrics,
            events=self.state.events,
            errors=self.state.errors,
            emit=self.emit,
            model_registered=self.state.model_registered,
            respect_context_window=self.respect_context_window,
            context_summaries=self.state.context_summaries,
            apply_summary=apply_summary,
        )
        if decision.report is not None:
            return decision.report
        self.state.increment_context_summaries()
        return None

    def fail(self, message: str) -> Report:
        return _fail_report(
            run_id=self.state.run_id,
            worker_name=self.state.worker_name,
            message=message,
            messages=self.state.messages,
            tool_calls=self.state.tool_calls,
            metrics=self.state.metrics,
            events=self.state.events,
            errors=self.state.errors,
            emit=self.emit,
        )

    def finalize_structured(self, data: Any) -> Report:
        return _finalize_structured_response(
            run_id=self.state.run_id,
            data=data,
            messages=self.state.messages,
            tool_calls=self.state.tool_calls,
            metrics=self.state.metrics,
            events=self.state.events,
            errors=self.state.errors,
            emit=self.emit,
            worker_name=self.state.worker_name,
        )

    def finalize_plain(self, response: Any) -> Report:
        return _finalize_plain_response(
            run_id=self.state.run_id,
            response=response,
            messages=self.state.messages,
            tool_calls=self.state.tool_calls,
            metrics=self.state.metrics,
            events=self.state.events,
            errors=self.state.errors,
            emit=self.emit,
            worker_name=self.state.worker_name,
        )

    def record_usage(self, response: Any) -> None:
        _record_usage(self.state.metrics, response)


def make_apply_summary(
    adapter: BaseModelAdapter,
    model_name: str,
    messages: list[Message],
    temperature: float | None,
    metrics: dict[str, Any],
    emit: EventEmitter,
    worker_name: str,
    model_registered: bool,
) -> Callable[[], Any]:
    async def apply_summary() -> Any:
        return await aapply_context_summary(
            adapter=adapter,
            model_name=model_name,
            messages=messages,
            temperature=temperature,
            metrics=metrics,
            emit=emit,
            worker_name=worker_name,
            model_registered=model_registered,
        )

    return apply_summary
