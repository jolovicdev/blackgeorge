from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from blackgeorge.config import RunConfig
from blackgeorge.core.event_types import EventType
from blackgeorge.core.message import Message
from blackgeorge.core.report import Report
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.worker_runner_helpers import (
    _acontext_retry,
    _build_report,
    _build_state,
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
    events: list[Any]
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
    config: RunConfig
    model_name: str
    state: LoopState

    def make_on_token(self) -> Callable[[str, str], None]:
        def on_token(token: str, token_type: str) -> None:
            self.config.emit(
                EventType.STREAM_TOKEN,
                self.state.worker_name,
                {"token": token, "type": token_type},
            )

        return on_token

    def run_config(self) -> RunConfig:
        return self.config

    async def handle_context_limit(
        self,
        apply_summary: Callable[[], Any],
    ) -> Report | None:
        decision = await _acontext_retry(
            config=self.config,
            worker_name=self.state.worker_name,
            messages=self.state.messages,
            tool_calls=self.state.tool_calls,
            metrics=self.state.metrics,
            errors=self.state.errors,
            model_registered=self.state.model_registered,
            context_summaries=self.state.context_summaries,
            apply_summary=apply_summary,
        )
        if decision.report is not None:
            return decision.report
        self.state.increment_context_summaries()
        return None

    def fail(self, message: str) -> Report:
        return _fail_report(
            config=self.config,
            worker_name=self.state.worker_name,
            message=message,
            messages=self.state.messages,
            tool_calls=self.state.tool_calls,
            metrics=self.state.metrics,
            errors=self.state.errors,
        )

    def finalize_structured(self, data: Any) -> Report:
        return _finalize_structured_response(
            config=self.config,
            data=data,
            messages=self.state.messages,
            tool_calls=self.state.tool_calls,
            metrics=self.state.metrics,
            errors=self.state.errors,
            worker_name=self.state.worker_name,
        )

    def finalize_plain(self, response: Any) -> Report:
        return _finalize_plain_response(
            config=self.config,
            response=response,
            messages=self.state.messages,
            tool_calls=self.state.tool_calls,
            metrics=self.state.metrics,
            errors=self.state.errors,
            worker_name=self.state.worker_name,
        )

    def record_usage(self, response: Any) -> None:
        _record_usage(self.state.metrics, response)

    def build_paused_report(self, pending: Any) -> Report:
        return _build_report(
            self.state.run_id,
            "paused",
            None,
            None,
            None,
            self.state.messages,
            self.state.tool_calls,
            self.state.metrics,
            self.config.events,
            pending,
            self.state.errors,
        )

    def build_paused_state(self, job: Any, pending: Any) -> Any:
        return _build_state(
            self.state.run_id,
            "paused",
            self.state.worker_name,
            job,
            self.state.messages,
            self.state.tool_calls,
            pending,
            self.state.metrics,
            self.state.iteration,
        )
