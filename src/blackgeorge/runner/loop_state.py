from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from blackgeorge.adapters.base import ModelResponse
from blackgeorge.config import RunConfig
from blackgeorge.core.event import Event
from blackgeorge.core.event_types import EventType
from blackgeorge.core.job import Job
from blackgeorge.core.message import Message
from blackgeorge.core.pending_action import PendingAction
from blackgeorge.core.report import Report
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.store.state import RunState
from blackgeorge.worker_runner_helpers import (
    aresolve_context_retry,
    build_report,
    build_worker_state,
    fail_report,
    finalize_plain_response,
    finalize_structured_response,
)


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
    config: RunConfig
    model_name: str
    state: LoopState

    def emit_token(self, token: str, token_type: str) -> None:
        self.config.emit(
            EventType.STREAM_TOKEN,
            self.state.worker_name,
            {"token": token, "type": token_type},
        )

    async def handle_context_limit(
        self,
        apply_summary: Callable[[], Awaitable[bool]],
    ) -> Report | None:
        decision = await aresolve_context_retry(
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
        return fail_report(
            config=self.config,
            worker_name=self.state.worker_name,
            message=message,
            messages=self.state.messages,
            tool_calls=self.state.tool_calls,
            metrics=self.state.metrics,
            errors=self.state.errors,
        )

    def finalize_structured(self, data: Any) -> Report:
        return finalize_structured_response(
            config=self.config,
            data=data,
            messages=self.state.messages,
            tool_calls=self.state.tool_calls,
            metrics=self.state.metrics,
            errors=self.state.errors,
            worker_name=self.state.worker_name,
        )

    def finalize_plain(self, response: ModelResponse) -> Report:
        return finalize_plain_response(
            config=self.config,
            response=response,
            messages=self.state.messages,
            tool_calls=self.state.tool_calls,
            metrics=self.state.metrics,
            errors=self.state.errors,
            worker_name=self.state.worker_name,
        )

    def pause(self, job: Job, pending: PendingAction) -> tuple[Report, RunState]:
        report = build_report(
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
        state = build_worker_state(
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
        return report, state
