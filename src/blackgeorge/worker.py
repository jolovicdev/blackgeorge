from collections.abc import Callable
from typing import Any

from blackgeorge.adapters.base import BaseModelAdapter
from blackgeorge.core.event import Event
from blackgeorge.core.job import Job
from blackgeorge.core.report import Report
from blackgeorge.store.state import RunState
from blackgeorge.tools.base import Tool
from blackgeorge.tools.registry import Toolbelt
from blackgeorge.worker_runner import WorkerRunner

EventEmitter = Callable[[str, str, dict[str, Any]], None]


class Worker:
    def __init__(
        self,
        name: str,
        tools: list[Tool] | None = None,
        model: str | None = None,
        instructions: str | None = None,
        memory_scope: str | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.instructions = instructions
        self.toolbelt = Toolbelt(tools)
        self.memory_scope = memory_scope or f"worker:{name}"

    def tools(self) -> list[Tool]:
        return self.toolbelt.list()

    def _runner(self) -> WorkerRunner:
        return WorkerRunner(self.name, self.toolbelt, self.instructions)

    def run(
        self,
        *,
        adapter: BaseModelAdapter,
        job: Job,
        run_id: str,
        events: list[Event],
        emit: EventEmitter,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        structured_output_retries: int,
        max_iterations: int,
        max_tool_calls: int,
        model_name: str,
        respect_context_window: bool = True,
    ) -> tuple[Report, RunState | None]:
        return self._runner().run(
            adapter=adapter,
            job=job,
            run_id=run_id,
            events=events,
            emit=emit,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stream_options=stream_options,
            structured_output_retries=structured_output_retries,
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            model_name=model_name,
            respect_context_window=respect_context_window,
        )

    async def arun(
        self,
        *,
        adapter: BaseModelAdapter,
        job: Job,
        run_id: str,
        events: list[Event],
        emit: EventEmitter,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        structured_output_retries: int,
        max_iterations: int,
        max_tool_calls: int,
        model_name: str,
        respect_context_window: bool = True,
    ) -> tuple[Report, RunState | None]:
        return await self._runner().arun(
            adapter=adapter,
            job=job,
            run_id=run_id,
            events=events,
            emit=emit,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stream_options=stream_options,
            structured_output_retries=structured_output_retries,
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            model_name=model_name,
            respect_context_window=respect_context_window,
        )

    def resume(
        self,
        *,
        adapter: BaseModelAdapter,
        state: RunState,
        decision_or_input: Any,
        events: list[Event],
        emit: EventEmitter,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        structured_output_retries: int,
        max_iterations: int,
        max_tool_calls: int,
        model_name: str,
        respect_context_window: bool = True,
    ) -> tuple[Report, RunState | None]:
        return self._runner().resume(
            adapter=adapter,
            state=state,
            decision_or_input=decision_or_input,
            events=events,
            emit=emit,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stream_options=stream_options,
            structured_output_retries=structured_output_retries,
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            model_name=model_name,
            respect_context_window=respect_context_window,
        )

    async def aresume(
        self,
        *,
        adapter: BaseModelAdapter,
        state: RunState,
        decision_or_input: Any,
        events: list[Event],
        emit: EventEmitter,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        structured_output_retries: int,
        max_iterations: int,
        max_tool_calls: int,
        model_name: str,
        respect_context_window: bool = True,
    ) -> tuple[Report, RunState | None]:
        return await self._runner().aresume(
            adapter=adapter,
            state=state,
            decision_or_input=decision_or_input,
            events=events,
            emit=emit,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stream_options=stream_options,
            structured_output_retries=structured_output_retries,
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            model_name=model_name,
            respect_context_window=respect_context_window,
        )
