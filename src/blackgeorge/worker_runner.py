import asyncio
import json
import warnings
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

from blackgeorge.adapters.base import BaseModelAdapter, ModelResponse
from blackgeorge.adapters.litellm_callbacks import emit_llm_completed, emit_llm_failed
from blackgeorge.core.event import Event
from blackgeorge.core.job import Job
from blackgeorge.core.message import Message
from blackgeorge.core.pending_action import PendingAction
from blackgeorge.core.report import Report
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.core.types import RunStatus
from blackgeorge.store.state import RunState
from blackgeorge.tools.base import Tool, ToolResult
from blackgeorge.tools.execution import aexecute_tool
from blackgeorge.tools.registry import Toolbelt
from blackgeorge.utils import new_id
from blackgeorge.worker_context import (
    SUMMARY_ATTEMPT_LIMIT,
    aapply_context_summary,
    context_error_message,
    is_context_limit_error,
    litellm_model_registered,
)
from blackgeorge.worker_messages import (
    chunk_content,
    chunk_usage,
    emit_assistant_message,
    ensure_content,
    messages_to_payload,
    render_input,
    replace_tool_call,
    structured_content,
    system_message,
    tool_call_with_result,
    tool_message,
    tool_schemas,
)
from blackgeorge.worker_tools import (
    pending_options,
    resume_argument_key,
    tool_action_type,
    tool_prompt,
    update_arguments,
)

EventEmitter = Callable[[str, str, dict[str, Any]], None]


@dataclass(frozen=True)
class ToolPlan:
    ordered_calls: list[ToolCall]
    executable_calls: list[tuple[ToolCall, Tool]]
    immediate_results: dict[str, ToolResult]
    pending: PendingAction | None
    max_tool_calls_exceeded: bool


@dataclass(frozen=True)
class ContextDecision:
    retry: bool
    report: Report | None


def _build_report(
    run_id: str,
    status: RunStatus,
    content: str | None = None,
    reasoning_content: str | None = None,
    data: Any | None = None,
    messages: list[Message] | None = None,
    tool_calls: list[ToolCall] | None = None,
    metrics: dict[str, Any] | None = None,
    events: list[Event] | None = None,
    pending_action: PendingAction | None = None,
    errors: list[str] | None = None,
) -> Report:
    return Report(
        run_id=run_id,
        status=status,
        content=content,
        reasoning_content=reasoning_content,
        data=data,
        messages=list(messages) if messages else [],
        tool_calls=list(tool_calls) if tool_calls else [],
        metrics=metrics or {},
        events=list(events) if events else [],
        pending_action=pending_action,
        errors=list(errors) if errors else [],
    )


def _build_state(
    run_id: str,
    status: RunStatus,
    runner_name: str,
    job: Job,
    messages: list[Message],
    tool_calls: list[ToolCall],
    pending_action: PendingAction | None,
    metrics: dict[str, Any],
    iteration: int,
    payload: dict[str, Any] | None = None,
) -> RunState:
    return RunState(
        run_id=run_id,
        status=status,
        runner_type="worker",
        runner_name=runner_name,
        job=job,
        messages=messages,
        tool_calls=tool_calls,
        pending_action=pending_action,
        metrics=metrics,
        iteration=iteration,
        payload=payload or {},
    )


def _report_error(
    run_id: str,
    messages: list[Message],
    errors: list[str],
    events: list[Event],
) -> Report:
    return Report(
        run_id=run_id,
        status="failed",
        content=None,
        reasoning_content=None,
        data=None,
        messages=messages,
        tool_calls=[],
        metrics={},
        events=events,
        pending_action=None,
        errors=errors,
    )


def _ensure_not_running_loop(action: str, async_action: str) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        f"{action} cannot be called from a running event loop. Use {async_action} instead."
    )


def _fail_report(
    *,
    run_id: str,
    worker_name: str,
    message: str,
    messages: list[Message],
    tool_calls: list[ToolCall],
    metrics: dict[str, Any],
    events: list[Event],
    errors: list[str],
    emit: EventEmitter,
) -> Report:
    errors.append(message)
    emit("worker.failed", worker_name, {"error": message})
    return _build_report(
        run_id,
        "failed",
        None,
        None,
        None,
        messages,
        tool_calls,
        metrics,
        events,
        None,
        errors,
    )


async def _acontext_retry(
    *,
    run_id: str,
    worker_name: str,
    messages: list[Message],
    tool_calls: list[ToolCall],
    metrics: dict[str, Any],
    events: list[Event],
    errors: list[str],
    emit: EventEmitter,
    model_registered: bool,
    respect_context_window: bool,
    context_summaries: int,
    apply_summary: Callable[[], Awaitable[bool]],
) -> ContextDecision:
    if not respect_context_window:
        message = context_error_message(model_registered, False)
        return ContextDecision(
            False,
            _fail_report(
                run_id=run_id,
                worker_name=worker_name,
                message=message,
                messages=messages,
                tool_calls=tool_calls,
                metrics=metrics,
                events=events,
                errors=errors,
                emit=emit,
            ),
        )
    if context_summaries >= SUMMARY_ATTEMPT_LIMIT:
        message = context_error_message(model_registered, True)
        return ContextDecision(
            False,
            _fail_report(
                run_id=run_id,
                worker_name=worker_name,
                message=message,
                messages=messages,
                tool_calls=tool_calls,
                metrics=metrics,
                events=events,
                errors=errors,
                emit=emit,
            ),
        )
    if not await apply_summary():
        message = context_error_message(model_registered, True)
        return ContextDecision(
            False,
            _fail_report(
                run_id=run_id,
                worker_name=worker_name,
                message=message,
                messages=messages,
                tool_calls=tool_calls,
                metrics=metrics,
                events=events,
                errors=errors,
                emit=emit,
            ),
        )
    return ContextDecision(True, None)


def _should_stream(stream: bool, tools: list[Tool], response_schema: Any | None) -> bool:
    return stream and not tools and response_schema is None


def _tool_result_preview(result: ToolResult, limit: int) -> tuple[str | None, bool]:
    if result.content is not None:
        text = result.content
    elif result.data is not None:
        try:
            text = json.dumps(result.data, ensure_ascii=True)
        except (TypeError, ValueError):
            text = str(result.data)
    elif result.error is not None:
        text = result.error
    else:
        return None, False
    if len(text) > limit:
        return f"{text[:limit]}...", True
    return text, False


def _tool_event_payload(call: ToolCall, result: ToolResult, limit: int = 200) -> dict[str, Any]:
    payload: dict[str, Any] = {"tool_call_id": call.id}
    preview, truncated = _tool_result_preview(result, limit)
    if preview is not None:
        payload["result_preview"] = preview
        payload["result_truncated"] = truncated
    if result.timed_out:
        payload["timed_out"] = True
    if result.cancelled:
        payload["cancelled"] = True
    return payload


def _record_usage(metrics: dict[str, Any], response: object) -> None:
    if isinstance(response, ModelResponse):
        metrics["usage"] = response.usage
    else:
        metrics["usage"] = {}


def _finalize_structured_response(
    *,
    run_id: str,
    data: Any,
    messages: list[Message],
    tool_calls: list[ToolCall],
    metrics: dict[str, Any],
    events: list[Event],
    errors: list[str],
    emit: EventEmitter,
    worker_name: str,
) -> Report:
    content = structured_content(data)
    assistant_message = Message(role="assistant", content=content)
    messages.append(assistant_message)
    emit_assistant_message(emit, worker_name, assistant_message)
    emit("worker.completed", worker_name, {})
    return _build_report(
        run_id,
        "completed",
        content,
        None,
        data,
        messages,
        tool_calls,
        metrics,
        events,
        None,
        errors,
    )


def _finalize_plain_response(
    *,
    run_id: str,
    response: ModelResponse,
    messages: list[Message],
    tool_calls: list[ToolCall],
    metrics: dict[str, Any],
    events: list[Event],
    errors: list[str],
    emit: EventEmitter,
    worker_name: str,
) -> Report:
    assistant_message = Message(role="assistant", content=response.content or "")
    messages.append(assistant_message)
    emit_assistant_message(emit, worker_name, assistant_message)
    emit("worker.completed", worker_name, {})
    return _build_report(
        run_id,
        "completed",
        response.content,
        response.reasoning_content,
        None,
        messages,
        tool_calls,
        metrics,
        events,
        None,
        errors,
    )


def _plan_tool_calls(
    *,
    response: ModelResponse,
    allowed_tools: dict[str, Tool],
    tool_calls: list[ToolCall],
    max_tool_calls: int,
) -> ToolPlan:
    ordered_calls: list[ToolCall] = []
    executable_calls: list[tuple[ToolCall, Tool]] = []
    immediate_results: dict[str, ToolResult] = {}
    pending: PendingAction | None = None
    max_tool_calls_exceeded = False

    for call in response.tool_calls:
        if len(tool_calls) >= max_tool_calls:
            max_tool_calls_exceeded = True
            break
        tool_calls.append(call)
        if call.error:
            ordered_calls.append(call)
            immediate_results[call.id] = ToolResult(error=call.error)
            continue

        tool = allowed_tools.get(call.name)
        if tool is None:
            ordered_calls.append(call)
            immediate_results[call.id] = ToolResult(error=f"Tool not found: {call.name}")
            continue
        action_type = tool_action_type(tool)
        if action_type:
            metadata = {"tool": tool.name}
            if tool.input_key:
                metadata["input_key"] = tool.input_key
            pending = PendingAction(
                action_id=new_id(),
                type=action_type,
                tool_call=call,
                prompt=tool_prompt(tool, action_type, call),
                options=pending_options(action_type),
                metadata=metadata,
            )
            break
        ordered_calls.append(call)
        executable_calls.append((call, tool))

    return ToolPlan(
        ordered_calls=ordered_calls,
        executable_calls=executable_calls,
        immediate_results=immediate_results,
        pending=pending,
        max_tool_calls_exceeded=max_tool_calls_exceeded,
    )


async def _execute_tool_calls_async(
    ordered_calls: list[ToolCall],
    executable_calls: list[tuple[ToolCall, Tool]],
    immediate_results: dict[str, ToolResult],
    messages: list[Message],
    tool_calls: list[ToolCall],
    emit: EventEmitter,
) -> None:
    results: dict[str, ToolResult] = dict(immediate_results)
    if executable_calls:
        for call, tool in executable_calls:
            emit("tool.started", tool.name, {"tool_call_id": call.id})
        if len(executable_calls) == 1:
            call, tool = executable_calls[0]
            results[call.id] = await aexecute_tool(tool, call)
        else:
            tasks = [aexecute_tool(tool, call) for call, tool in executable_calls]
            tool_results = await asyncio.gather(*tasks)
            for (call, _), result in zip(executable_calls, tool_results, strict=True):
                results[call.id] = result
    for call in ordered_calls:
        result = results.get(call.id, ToolResult(error="Tool execution failed"))
        if result.error:
            emit("tool.failed", call.name, {"tool_call_id": call.id, "error": result.error})
        else:
            emit("tool.completed", call.name, _tool_event_payload(call, result))
        tool_result_message = tool_message(result, call)
        messages.append(tool_result_message)
        replace_tool_call(tool_calls, tool_call_with_result(call, result))


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
                            usage_chunk = chunk_usage(chunk)
                            if usage_chunk:
                                usage = usage_chunk
                    else:
                        for chunk in cast(Iterable[Any], stream):
                            token = chunk_content(chunk)
                            if token:
                                content_parts.append(token)
                                on_token(token)
                            usage_chunk = chunk_usage(chunk)
                            if usage_chunk:
                                usage = usage_chunk
                except Exception as exc:
                    emit_llm_failed(model, exc)
                    raise
                finally:
                    if hasattr(stream, "aclose"):
                        await stream.aclose()
                    elif hasattr(stream, "close"):
                        stream.close()
            emit_llm_completed(model, {"usage": usage})
            return ModelResponse(
                content="".join(content_parts),
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
                    emit("worker.completed", self.name, {})
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
