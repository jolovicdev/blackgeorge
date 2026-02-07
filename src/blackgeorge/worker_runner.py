import asyncio
import warnings
from collections.abc import Callable, Iterable
from typing import Any, cast

from blackgeorge.adapters.base import BaseModelAdapter, ModelResponse
from blackgeorge.core.event import Event
from blackgeorge.core.job import Job
from blackgeorge.core.message import Message
from blackgeorge.core.report import Report
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.store.state import RunState
from blackgeorge.tools.base import Tool, ToolResult
from blackgeorge.tools.execution import aexecute_tool
from blackgeorge.tools.registry import Toolbelt
from blackgeorge.worker_context import (
    aapply_context_summary,
    is_context_limit_error,
    litellm_model_registered,
)
from blackgeorge.worker_messages import (
    chunk_content,
    chunk_reasoning_content,
    chunk_thinking_blocks,
    chunk_usage,
    emit_assistant_message,
    ensure_content,
    messages_to_payload,
    render_input,
    replace_tool_call,
    system_message,
    tool_call_with_result,
    tool_message,
    tool_schemas,
)
from blackgeorge.worker_runner_helpers import (
    EventEmitter,
    _acontext_retry,
    _build_report,
    _build_state,
    _ensure_not_running_loop,
    _execute_tool_calls_async,
    _fail_report,
    _finalize_plain_response,
    _finalize_structured_response,
    _plan_tool_calls,
    _record_usage,
    _report_error,
    _should_stream,
    _tool_event_payload,
)
from blackgeorge.worker_tools import resume_argument_key, update_arguments


class WorkerRunner:
    def __init__(self, name: str, toolbelt: Toolbelt, instructions: str | None) -> None:
        self.name = name
        self.toolbelt = toolbelt
        self.instructions = instructions

    def _build_messages(self, job: Job) -> list[Message]:
        messages: list[Message] = []

        if job.initial_messages:
            messages.extend(job.initial_messages)

        system_content = system_message(self.instructions, job)
        if system_content:
            if not messages or messages[0].role != "system":
                messages.insert(0, Message(role="system", content=system_content))
            else:
                messages[0] = Message(
                    role="system",
                    content=f"{messages[0].content}\n\n{system_content}",
                )

        messages.append(Message(role="user", content=render_input(job.input)))
        return messages

    def _resolve_tools(self, job: Job) -> list[Tool]:
        if job.tools_override is not None:
            resolved: list[Tool] = []
            for item in job.tools_override:
                if isinstance(item, Tool):
                    resolved.append(item)
                    continue
                if isinstance(item, str):
                    tool = self.toolbelt.resolve(item)
                    if tool is not None:
                        resolved.append(tool)
            return resolved
        return self.toolbelt.list()

    async def _astructured_completion(
        self,
        *,
        adapter: BaseModelAdapter,
        model: str,
        messages: list[Message],
        response_schema: Any,
        retries: int,
    ) -> Any:
        payload = messages_to_payload(messages)
        try:
            return await adapter.astructured_complete(
                model=model,
                messages=payload,
                response_schema=response_schema,
                retries=retries,
            )
        except NotImplementedError:
            return await asyncio.to_thread(
                adapter.structured_complete,
                model=model,
                messages=payload,
                response_schema=response_schema,
                retries=retries,
            )

    async def _acompletion(
        self,
        *,
        adapter: BaseModelAdapter,
        model: str,
        messages: list[Message],
        tools: list[Tool],
        temperature: float | None,
        max_tokens: int | None,
        stream_options: dict[str, Any] | None,
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
        run_id: str | None = None,
        emit: EventEmitter | None = None,
    ) -> ModelResponse:
        if run_id and emit and hasattr(adapter, "set_callback_context"):
            adapter.set_callback_context(run_id, emit)
        try:
            try:
                response = await adapter.acomplete(
                    model=model,
                    messages=messages_to_payload(messages),
                    tools=tool_schemas(tools) if tools else None,
                    tool_choice="auto" if tools else None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                    stream_options=stream_options,
                    thinking=thinking,
                    drop_params=drop_params,
                    extra_body=extra_body,
                )
            except NotImplementedError:
                response = await asyncio.to_thread(
                    adapter.complete,
                    model=model,
                    messages=messages_to_payload(messages),
                    tools=tool_schemas(tools) if tools else None,
                    tool_choice="auto" if tools else None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                    stream_options=stream_options,
                    thinking=thinking,
                    drop_params=drop_params,
                    extra_body=extra_body,
                )
            if isinstance(response, ModelResponse):
                return response
            return ModelResponse(content=None, tool_calls=[], usage={}, raw=response)
        finally:
            if run_id and emit and hasattr(adapter, "clear_callback_context"):
                adapter.clear_callback_context()

    async def _astream_completion(
        self,
        *,
        adapter: BaseModelAdapter,
        model: str,
        messages: list[Message],
        temperature: float | None,
        max_tokens: int | None,
        stream_options: dict[str, Any] | None,
        on_token: Callable[[str], None],
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
        run_id: str | None = None,
        emit: EventEmitter | None = None,
    ) -> ModelResponse:
        if run_id and emit and hasattr(adapter, "set_callback_context"):
            adapter.set_callback_context(run_id, emit)
        try:
            try:
                stream = await adapter.acomplete(
                    model=model,
                    messages=messages_to_payload(messages),
                    tools=None,
                    tool_choice=None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    stream_options=stream_options,
                    thinking=thinking,
                    drop_params=drop_params,
                    extra_body=extra_body,
                )
            except NotImplementedError:
                stream = await asyncio.to_thread(
                    adapter.complete,
                    model=model,
                    messages=messages_to_payload(messages),
                    tools=None,
                    tool_choice=None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    stream_options=stream_options,
                    thinking=thinking,
                    drop_params=drop_params,
                    extra_body=extra_body,
                )
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            thinking_blocks: list[dict[str, Any]] = []
            usage: dict[str, Any] = {}
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
                try:
                    if hasattr(stream, "__aiter__"):
                        async for chunk in stream:
                            token = chunk_content(chunk)
                            if token:
                                content_parts.append(token)
                                on_token(token)
                            reasoning = chunk_reasoning_content(chunk)
                            if reasoning:
                                reasoning_parts.append(reasoning)
                            blocks = chunk_thinking_blocks(chunk)
                            if blocks:
                                thinking_blocks.extend(blocks)
                            usage_chunk = chunk_usage(chunk)
                            if usage_chunk:
                                usage = usage_chunk
                    else:
                        for chunk in cast(Iterable[Any], stream):
                            token = chunk_content(chunk)
                            if token:
                                content_parts.append(token)
                                on_token(token)
                            reasoning = chunk_reasoning_content(chunk)
                            if reasoning:
                                reasoning_parts.append(reasoning)
                            blocks = chunk_thinking_blocks(chunk)
                            if blocks:
                                thinking_blocks.extend(blocks)
                            usage_chunk = chunk_usage(chunk)
                            if usage_chunk:
                                usage = usage_chunk
                finally:
                    if hasattr(stream, "aclose"):
                        await stream.aclose()
                    elif hasattr(stream, "close"):
                        stream.close()
            return ModelResponse(
                content="".join(content_parts),
                reasoning_content="".join(reasoning_parts) or None,
                thinking_blocks=thinking_blocks or None,
                tool_calls=[],
                usage=usage,
                raw=stream,
            )
        finally:
            if run_id and emit and hasattr(adapter, "clear_callback_context"):
                adapter.clear_callback_context()

    async def _arun_loop(
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
        messages: list[Message],
        tool_calls: list[ToolCall],
        metrics: dict[str, Any],
        errors: list[str],
        iteration: int,
        model_name: str,
        respect_context_window: bool,
    ) -> tuple[Report, RunState | None]:
        tools = self._resolve_tools(job)
        allowed_tools = {tool.name: tool for tool in tools}
        response_schema = job.response_schema
        context_summaries = 0
        model_registered = litellm_model_registered(model_name)

        while iteration < max_iterations:
            iteration += 1
            if _should_stream(stream, tools, response_schema):
                try:
                    response = await self._astream_completion(
                        adapter=adapter,
                        model=model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream_options=stream_options,
                        on_token=lambda token: emit("stream.token", self.name, {"token": token}),
                        thinking=job.thinking,
                        drop_params=job.drop_params,
                        extra_body=job.extra_body,
                        run_id=run_id,
                        emit=emit,
                    )
                except Exception as exc:
                    if not is_context_limit_error(exc):
                        raise
                    decision = await _acontext_retry(
                        run_id=run_id,
                        worker_name=self.name,
                        messages=messages,
                        tool_calls=tool_calls,
                        metrics=metrics,
                        events=events,
                        errors=errors,
                        emit=emit,
                        model_registered=model_registered,
                        respect_context_window=respect_context_window,
                        context_summaries=context_summaries,
                        apply_summary=lambda: aapply_context_summary(
                            adapter=adapter,
                            model_name=model_name,
                            messages=messages,
                            temperature=temperature,
                            metrics=metrics,
                            emit=emit,
                            worker_name=self.name,
                            model_registered=model_registered,
                        ),
                    )
                    if decision.report is not None:
                        return decision.report, None
                    context_summaries += 1
                    continue
            elif response_schema is not None and not tools:
                try:
                    data = await self._astructured_completion(
                        adapter=adapter,
                        model=model_name,
                        messages=messages,
                        response_schema=response_schema,
                        retries=structured_output_retries,
                    )
                except Exception as exc:
                    if is_context_limit_error(exc):
                        decision = await _acontext_retry(
                            run_id=run_id,
                            worker_name=self.name,
                            messages=messages,
                            tool_calls=tool_calls,
                            metrics=metrics,
                            events=events,
                            errors=errors,
                            emit=emit,
                            model_registered=model_registered,
                            respect_context_window=respect_context_window,
                            context_summaries=context_summaries,
                            apply_summary=lambda: aapply_context_summary(
                                adapter=adapter,
                                model_name=model_name,
                                messages=messages,
                                temperature=temperature,
                                metrics=metrics,
                                emit=emit,
                                worker_name=self.name,
                                model_registered=model_registered,
                            ),
                        )
                        if decision.report is not None:
                            return decision.report, None
                        context_summaries += 1
                        continue
                    return (
                        _fail_report(
                            run_id=run_id,
                            worker_name=self.name,
                            message=str(exc),
                            messages=messages,
                            tool_calls=tool_calls,
                            metrics=metrics,
                            events=events,
                            errors=errors,
                            emit=emit,
                        ),
                        None,
                    )
                return (
                    _finalize_structured_response(
                        run_id=run_id,
                        data=data,
                        messages=messages,
                        tool_calls=tool_calls,
                        metrics=metrics,
                        events=events,
                        errors=errors,
                        emit=emit,
                        worker_name=self.name,
                    ),
                    None,
                )
            else:
                try:
                    response = await self._acompletion(
                        adapter=adapter,
                        model=model_name,
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream_options=stream_options,
                        thinking=job.thinking,
                        drop_params=job.drop_params,
                        extra_body=job.extra_body,
                        run_id=run_id,
                        emit=emit,
                    )
                except Exception as exc:
                    if not is_context_limit_error(exc):
                        raise
                    decision = await _acontext_retry(
                        run_id=run_id,
                        worker_name=self.name,
                        messages=messages,
                        tool_calls=tool_calls,
                        metrics=metrics,
                        events=events,
                        errors=errors,
                        emit=emit,
                        model_registered=model_registered,
                        respect_context_window=respect_context_window,
                        context_summaries=context_summaries,
                        apply_summary=lambda: aapply_context_summary(
                            adapter=adapter,
                            model_name=model_name,
                            messages=messages,
                            temperature=temperature,
                            metrics=metrics,
                            emit=emit,
                            worker_name=self.name,
                            model_registered=model_registered,
                        ),
                    )
                    if decision.report is not None:
                        return decision.report, None
                    context_summaries += 1
                    continue

            _record_usage(metrics, response)

            if response.tool_calls:
                assistant_message = Message(
                    role="assistant",
                    content=ensure_content(response.content),
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                    tool_calls=response.tool_calls,
                )
                messages.append(assistant_message)
                emit_assistant_message(emit, self.name, assistant_message)
                plan = _plan_tool_calls(
                    response=response,
                    allowed_tools=allowed_tools,
                    tool_calls=tool_calls,
                    max_tool_calls=max_tool_calls,
                )

                await _execute_tool_calls_async(
                    plan.ordered_calls,
                    plan.executable_calls,
                    plan.immediate_results,
                    messages,
                    tool_calls,
                    emit,
                )

                if plan.max_tool_calls_exceeded:
                    return (
                        _fail_report(
                            run_id=run_id,
                            worker_name=self.name,
                            message="Max tool calls exceeded",
                            messages=messages,
                            tool_calls=tool_calls,
                            metrics=metrics,
                            events=events,
                            errors=errors,
                            emit=emit,
                        ),
                        None,
                    )

                if plan.pending is not None:
                    emit(
                        f"tool.{plan.pending.type}_requested",
                        plan.pending.tool_call.name,
                        {"tool_call_id": plan.pending.tool_call.id},
                    )
                    emit(
                        "worker.paused",
                        self.name,
                        {"pending_action_type": plan.pending.type},
                    )
                    report = _build_report(
                        run_id,
                        "paused",
                        None,
                        None,
                        None,
                        messages,
                        tool_calls,
                        metrics,
                        events,
                        plan.pending,
                        errors,
                    )
                    state = _build_state(
                        run_id,
                        "paused",
                        self.name,
                        job,
                        messages,
                        tool_calls,
                        plan.pending,
                        metrics,
                        iteration,
                    )
                    return report, state
                continue

            if response_schema is not None:
                try:
                    data = await self._astructured_completion(
                        adapter=adapter,
                        model=model_name,
                        messages=messages,
                        response_schema=response_schema,
                        retries=structured_output_retries,
                    )
                except Exception as exc:
                    if is_context_limit_error(exc):
                        decision = await _acontext_retry(
                            run_id=run_id,
                            worker_name=self.name,
                            messages=messages,
                            tool_calls=tool_calls,
                            metrics=metrics,
                            events=events,
                            errors=errors,
                            emit=emit,
                            model_registered=model_registered,
                            respect_context_window=respect_context_window,
                            context_summaries=context_summaries,
                            apply_summary=lambda: aapply_context_summary(
                                adapter=adapter,
                                model_name=model_name,
                                messages=messages,
                                temperature=temperature,
                                metrics=metrics,
                                emit=emit,
                                worker_name=self.name,
                                model_registered=model_registered,
                            ),
                        )
                        if decision.report is not None:
                            return decision.report, None
                        context_summaries += 1
                        continue
                    return (
                        _fail_report(
                            run_id=run_id,
                            worker_name=self.name,
                            message=str(exc),
                            messages=messages,
                            tool_calls=tool_calls,
                            metrics=metrics,
                            events=events,
                            errors=errors,
                            emit=emit,
                        ),
                        None,
                    )
                return (
                    _finalize_structured_response(
                        run_id=run_id,
                        data=data,
                        messages=messages,
                        tool_calls=tool_calls,
                        metrics=metrics,
                        events=events,
                        errors=errors,
                        emit=emit,
                        worker_name=self.name,
                    ),
                    None,
                )

            return (
                _finalize_plain_response(
                    run_id=run_id,
                    response=response,
                    messages=messages,
                    tool_calls=tool_calls,
                    metrics=metrics,
                    events=events,
                    errors=errors,
                    emit=emit,
                    worker_name=self.name,
                ),
                None,
            )

        return _fail_report(
            run_id=run_id,
            worker_name=self.name,
            message="Max iterations exceeded",
            messages=messages,
            tool_calls=tool_calls,
            metrics=metrics,
            events=events,
            errors=errors,
            emit=emit,
        ), None

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
        _ensure_not_running_loop("run", "arun")
        return asyncio.run(
            self.arun(
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
        messages = self._build_messages(job)
        tool_calls: list[ToolCall] = []
        metrics: dict[str, Any] = {}
        errors: list[str] = []
        iteration = 0

        if not model_name:
            errors.append("Worker model not set")
            emit("worker.failed", self.name, {"error": errors[-1]})
            report = _report_error(run_id, messages, errors, events)
            return report, None

        emit("worker.started", self.name, {})
        return await self._arun_loop(
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
            messages=messages,
            tool_calls=tool_calls,
            metrics=metrics,
            errors=errors,
            iteration=iteration,
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
        _ensure_not_running_loop("resume", "aresume")
        return asyncio.run(
            self.aresume(
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
        pending = state.pending_action
        if pending is None:
            report = _build_report(
                state.run_id,
                "failed",
                None,
                None,
                None,
                state.messages,
                state.tool_calls,
                state.metrics,
                events,
                None,
                ["No pending action"],
            )
            return report, None

        messages = list(state.messages)
        tool_calls = list(state.tool_calls)
        iteration = state.iteration
        metrics = dict(state.metrics)
        errors: list[str] = []

        tool = self.toolbelt.resolve(pending.tool_call.name)
        if tool is None:
            result = ToolResult(error=f"Tool not found: {pending.tool_call.name}")
            emit(
                "tool.failed",
                pending.tool_call.name,
                {"tool_call_id": pending.tool_call.id, "error": result.error},
            )
            messages.append(tool_message(result, pending.tool_call))
            replace_tool_call(tool_calls, tool_call_with_result(pending.tool_call, result))
        else:
            if pending.type == "confirmation" and not decision_or_input:
                result = ToolResult(error="Tool execution declined")
                emit(
                    "tool.failed",
                    pending.tool_call.name,
                    {"tool_call_id": pending.tool_call.id, "error": result.error},
                )
                tool_result_message = tool_message(result, pending.tool_call)
                messages.append(tool_result_message)
                replace_tool_call(tool_calls, tool_call_with_result(pending.tool_call, result))
            else:
                call = pending.tool_call
                if pending.type == "user_input":
                    key = resume_argument_key(pending)
                    call = update_arguments(call, key, decision_or_input)
                emit("tool.started", tool.name, {"tool_call_id": call.id})
                result = await aexecute_tool(tool, call)
                if result.error:
                    emit("tool.failed", tool.name, {"tool_call_id": call.id, "error": result.error})
                else:
                    emit("tool.completed", tool.name, _tool_event_payload(call, result))
                tool_result_message = tool_message(result, call)
                messages.append(tool_result_message)
                replace_tool_call(tool_calls, tool_call_with_result(call, result))

        if not model_name:
            errors.append("Worker model not set")
            emit("worker.failed", self.name, {"error": errors[-1]})
            report = _report_error(state.run_id, messages, errors, events)
            return report, None

        emit("worker.started", self.name, {})
        return await self._arun_loop(
            adapter=adapter,
            job=state.job,
            run_id=state.run_id,
            events=events,
            emit=emit,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stream_options=stream_options,
            structured_output_retries=structured_output_retries,
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            messages=messages,
            tool_calls=tool_calls,
            metrics=metrics,
            errors=errors,
            iteration=iteration,
            model_name=model_name,
            respect_context_window=respect_context_window,
        )
