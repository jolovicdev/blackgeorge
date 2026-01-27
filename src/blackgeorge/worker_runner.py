import asyncio
import concurrent.futures
import json
from collections.abc import Callable, Iterable
from typing import Any, cast

from blackgeorge.adapters.base import BaseModelAdapter, ModelResponse
from blackgeorge.adapters.instructor_client import instructor_clients
from blackgeorge.core.event import Event
from blackgeorge.core.job import Job
from blackgeorge.core.message import Message
from blackgeorge.core.pending_action import PendingAction
from blackgeorge.core.report import Report
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.core.types import RunStatus
from blackgeorge.store.state import RunState
from blackgeorge.tools.base import Tool, ToolResult
from blackgeorge.tools.execution import aexecute_tool, execute_tool
from blackgeorge.tools.registry import Toolbelt
from blackgeorge.utils import new_id
from blackgeorge.worker_context import (
    CONTEXT_SUMMARY_MAX_ATTEMPTS,
    aapply_context_summary,
    apply_context_summary,
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


def _execute_tool_calls_sync(
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
            results[call.id] = execute_tool(tool, call)
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(executable_calls)
            ) as executor:
                futures = [
                    executor.submit(execute_tool, tool, call) for call, tool in executable_calls
                ]
                for (call, _), future in zip(executable_calls, futures):
                    try:
                        results[call.id] = future.result()
                    except Exception as exc:
                        results[call.id] = ToolResult(error=str(exc))
    for call in ordered_calls:
        result = results.get(call.id, ToolResult(error="Tool execution failed"))
        if result.error:
            emit("tool.failed", call.name, {"tool_call_id": call.id, "error": result.error})
        else:
            emit("tool.completed", call.name, _tool_event_payload(call, result))
        tool_result_message = tool_message(result, call)
        messages.append(tool_result_message)
        replace_tool_call(tool_calls, tool_call_with_result(call, result))


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
            for (call, _), result in zip(executable_calls, tool_results):
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

    def _structured_completion(
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
            return adapter.structured_complete(
                model=model,
                messages=payload,
                response_schema=response_schema,
                retries=retries,
            )
        except NotImplementedError:
            pass
        client = instructor_clients.get(model, async_client=False)
        attempts = 0
        while True:
            try:
                return client.chat.completions.create(
                    model=model,
                    messages=payload,
                    response_model=response_schema,
                )
            except Exception as exc:
                if attempts >= retries:
                    raise
                payload.append(
                    {
                        "role": "user",
                        "content": f"Fix validation errors: {exc}",
                    }
                )
                attempts += 1

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
            pass
        client = instructor_clients.get(model, async_client=True)
        attempts = 0
        while True:
            try:
                return await client.chat.completions.create(
                    model=model,
                    messages=payload,
                    response_model=response_schema,
                )
            except Exception as exc:
                if attempts >= retries:
                    raise
                payload.append(
                    {
                        "role": "user",
                        "content": f"Fix validation errors: {exc}",
                    }
                )
                attempts += 1

    def _completion(
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
    ) -> ModelResponse:
        response = adapter.complete(
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
    ) -> ModelResponse:
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
        if isinstance(response, ModelResponse):
            return response
        return ModelResponse(content=None, tool_calls=[], usage={}, raw=response)

    def _stream_completion(
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
    ) -> ModelResponse:
        stream = cast(
            Iterable[Any],
            adapter.complete(
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
            ),
        )
        content_parts: list[str] = []
        usage: dict[str, Any] = {}
        for chunk in stream:
            token = chunk_content(chunk)
            if token:
                content_parts.append(token)
                on_token(token)
            usage_chunk = chunk_usage(chunk)
            if usage_chunk:
                usage = usage_chunk
        return ModelResponse(
            content="".join(content_parts),
            tool_calls=[],
            usage=usage,
            raw=stream,
        )

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
    ) -> ModelResponse:
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
        content_parts: list[str] = []
        usage: dict[str, Any] = {}
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
        return ModelResponse(
            content="".join(content_parts),
            tool_calls=[],
            usage=usage,
            raw=stream,
        )

    def _run_loop(
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
                    response = self._stream_completion(
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
                    )
                except Exception as exc:
                    if not is_context_limit_error(exc):
                        raise
                    if not respect_context_window:
                        errors.append(context_error_message(model_registered, False))
                        emit("worker.failed", self.name, {"error": errors[-1]})
                        report = _build_report(
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
                        return report, None
                    if context_summaries >= CONTEXT_SUMMARY_MAX_ATTEMPTS:
                        errors.append(context_error_message(model_registered, True))
                        emit("worker.failed", self.name, {"error": errors[-1]})
                        report = _build_report(
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
                        return report, None
                    if not apply_context_summary(
                        adapter=adapter,
                        model_name=model_name,
                        messages=messages,
                        temperature=temperature,
                        metrics=metrics,
                        emit=emit,
                        worker_name=self.name,
                        model_registered=model_registered,
                    ):
                        errors.append(context_error_message(model_registered, True))
                        emit("worker.failed", self.name, {"error": errors[-1]})
                        report = _build_report(
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
                        return report, None
                    context_summaries += 1
                    continue
            elif response_schema is not None and not tools:
                try:
                    data = self._structured_completion(
                        adapter=adapter,
                        model=model_name,
                        messages=messages,
                        response_schema=response_schema,
                        retries=structured_output_retries,
                    )
                except Exception as exc:
                    if is_context_limit_error(exc):
                        if not respect_context_window:
                            errors.append(context_error_message(model_registered, False))
                            emit("worker.failed", self.name, {"error": errors[-1]})
                            report = _build_report(
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
                            return report, None
                        if context_summaries >= CONTEXT_SUMMARY_MAX_ATTEMPTS:
                            errors.append(context_error_message(model_registered, True))
                            emit("worker.failed", self.name, {"error": errors[-1]})
                            report = _build_report(
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
                            return report, None
                        if not apply_context_summary(
                            adapter=adapter,
                            model_name=model_name,
                            messages=messages,
                            temperature=temperature,
                            metrics=metrics,
                            emit=emit,
                            worker_name=self.name,
                            model_registered=model_registered,
                        ):
                            errors.append(context_error_message(model_registered, True))
                            emit("worker.failed", self.name, {"error": errors[-1]})
                            report = _build_report(
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
                            return report, None
                        context_summaries += 1
                        continue
                    errors.append(str(exc))
                    emit("worker.failed", self.name, {"error": errors[-1]})
                    report = _build_report(
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
                    return report, None
                content = structured_content(data)
                assistant_message = Message(role="assistant", content=content)
                messages.append(assistant_message)
                emit_assistant_message(emit, self.name, assistant_message)
                emit("worker.completed", self.name, {})
                report = _build_report(
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
                return report, None
            else:
                try:
                    response = self._completion(
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
                    )
                except Exception as exc:
                    if not is_context_limit_error(exc):
                        raise
                    if not respect_context_window:
                        errors.append(context_error_message(model_registered, False))
                        emit("worker.failed", self.name, {"error": errors[-1]})
                        report = _build_report(
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
                        return report, None
                    if context_summaries >= CONTEXT_SUMMARY_MAX_ATTEMPTS:
                        errors.append(context_error_message(model_registered, True))
                        emit("worker.failed", self.name, {"error": errors[-1]})
                        report = _build_report(
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
                        return report, None
                    if not apply_context_summary(
                        adapter=adapter,
                        model_name=model_name,
                        messages=messages,
                        temperature=temperature,
                        metrics=metrics,
                        emit=emit,
                        worker_name=self.name,
                        model_registered=model_registered,
                    ):
                        errors.append(context_error_message(model_registered, True))
                        emit("worker.failed", self.name, {"error": errors[-1]})
                        report = _build_report(
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
                        return report, None
                    context_summaries += 1
                    continue

            metrics["usage"] = response.usage if isinstance(response, ModelResponse) else {}

            if response.tool_calls:
                assistant_message = Message(
                    role="assistant",
                    content=ensure_content(response.content),
                    tool_calls=response.tool_calls,
                )
                messages.append(assistant_message)
                emit_assistant_message(emit, self.name, assistant_message)
                ordered_calls: list[ToolCall] = []
                executable_calls: list[tuple[ToolCall, Tool]] = []
                immediate_results: dict[str, ToolResult] = {}
                pending: PendingAction | None = None
                max_tool_calls_exceeded = False

                for call in response.tool_calls:
                    tool_calls.append(call)
                    if len(tool_calls) >= max_tool_calls:
                        errors.append("Max tool calls exceeded")
                        max_tool_calls_exceeded = True
                        break
                    if call.error:
                        ordered_calls.append(call)
                        immediate_results[call.id] = ToolResult(error=call.error)
                        continue

                    tool = allowed_tools.get(call.name)
                    if tool is None:
                        ordered_calls.append(call)
                        immediate_results[call.id] = ToolResult(
                            error=f"Tool not found: {call.name}"
                        )
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

                _execute_tool_calls_sync(
                    ordered_calls,
                    executable_calls,
                    immediate_results,
                    messages,
                    tool_calls,
                    emit,
                )

                if max_tool_calls_exceeded:
                    emit("worker.failed", self.name, {"error": errors[-1]})
                    report = _build_report(
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
                    return report, None

                if pending is not None:
                    emit(
                        f"tool.{pending.type}_requested",
                        pending.tool_call.name,
                        {"tool_call_id": pending.tool_call.id},
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
                        pending,
                        errors,
                    )
                    state = _build_state(
                        run_id,
                        "paused",
                        self.name,
                        job,
                        messages,
                        tool_calls,
                        pending,
                        metrics,
                        iteration,
                    )
                    return report, state
                continue

            if response_schema is not None:
                try:
                    data = self._structured_completion(
                        adapter=adapter,
                        model=model_name,
                        messages=messages,
                        response_schema=response_schema,
                        retries=structured_output_retries,
                    )
                except Exception as exc:
                    errors.append(str(exc))
                    emit("worker.failed", self.name, {"error": errors[-1]})
                    report = _build_report(
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
                    return report, None
                content = structured_content(data)
                assistant_message = Message(role="assistant", content=content)
                messages.append(assistant_message)
                emit_assistant_message(emit, self.name, assistant_message)
                emit("worker.completed", self.name, {})
                report = _build_report(
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
                return report, None

            assistant_message = Message(role="assistant", content=response.content or "")
            messages.append(assistant_message)
            emit_assistant_message(emit, self.name, assistant_message)
            emit("worker.completed", self.name, {})
            report = _build_report(
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
            return report, None

        errors.append("Max iterations exceeded")
        emit("worker.failed", self.name, {"error": errors[-1]})
        report = _build_report(
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
        return report, None

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
                    )
                except Exception as exc:
                    if not is_context_limit_error(exc):
                        raise
                    if not respect_context_window:
                        errors.append(context_error_message(model_registered, False))
                        emit("worker.failed", self.name, {"error": errors[-1]})
                        report = _build_report(
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
                        return report, None
                    if context_summaries >= CONTEXT_SUMMARY_MAX_ATTEMPTS:
                        errors.append(context_error_message(model_registered, True))
                        emit("worker.failed", self.name, {"error": errors[-1]})
                        report = _build_report(
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
                        return report, None
                    if not await aapply_context_summary(
                        adapter=adapter,
                        model_name=model_name,
                        messages=messages,
                        temperature=temperature,
                        metrics=metrics,
                        emit=emit,
                        worker_name=self.name,
                        model_registered=model_registered,
                    ):
                        errors.append(context_error_message(model_registered, True))
                        emit("worker.failed", self.name, {"error": errors[-1]})
                        report = _build_report(
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
                        return report, None
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
                        if not respect_context_window:
                            errors.append(context_error_message(model_registered, False))
                            emit("worker.failed", self.name, {"error": errors[-1]})
                            report = _build_report(
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
                            return report, None
                        if context_summaries >= CONTEXT_SUMMARY_MAX_ATTEMPTS:
                            errors.append(context_error_message(model_registered, True))
                            emit("worker.failed", self.name, {"error": errors[-1]})
                            report = _build_report(
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
                            return report, None
                        if not await aapply_context_summary(
                            adapter=adapter,
                            model_name=model_name,
                            messages=messages,
                            temperature=temperature,
                            metrics=metrics,
                            emit=emit,
                            worker_name=self.name,
                            model_registered=model_registered,
                        ):
                            errors.append(context_error_message(model_registered, True))
                            emit("worker.failed", self.name, {"error": errors[-1]})
                            report = _build_report(
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
                            return report, None
                        context_summaries += 1
                        continue
                    errors.append(str(exc))
                    emit("worker.failed", self.name, {"error": errors[-1]})
                    report = _build_report(
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
                    return report, None
                content = structured_content(data)
                assistant_message = Message(role="assistant", content=content)
                messages.append(assistant_message)
                emit_assistant_message(emit, self.name, assistant_message)
                emit("worker.completed", self.name, {})
                report = _build_report(
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
                return report, None
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
                    )
                except Exception as exc:
                    if not is_context_limit_error(exc):
                        raise
                    if not respect_context_window:
                        errors.append(context_error_message(model_registered, False))
                        emit("worker.failed", self.name, {"error": errors[-1]})
                        report = _build_report(
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
                        return report, None
                    if context_summaries >= CONTEXT_SUMMARY_MAX_ATTEMPTS:
                        errors.append(context_error_message(model_registered, True))
                        emit("worker.failed", self.name, {"error": errors[-1]})
                        report = _build_report(
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
                        return report, None
                    if not await aapply_context_summary(
                        adapter=adapter,
                        model_name=model_name,
                        messages=messages,
                        temperature=temperature,
                        metrics=metrics,
                        emit=emit,
                        worker_name=self.name,
                        model_registered=model_registered,
                    ):
                        errors.append(context_error_message(model_registered, True))
                        emit("worker.failed", self.name, {"error": errors[-1]})
                        report = _build_report(
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
                        return report, None
                    context_summaries += 1
                    continue

            metrics["usage"] = response.usage if isinstance(response, ModelResponse) else {}

            if response.tool_calls:
                assistant_message = Message(
                    role="assistant",
                    content=ensure_content(response.content),
                    tool_calls=response.tool_calls,
                )
                messages.append(assistant_message)
                emit_assistant_message(emit, self.name, assistant_message)
                ordered_calls: list[ToolCall] = []
                executable_calls: list[tuple[ToolCall, Tool]] = []
                immediate_results: dict[str, ToolResult] = {}
                pending: PendingAction | None = None
                max_tool_calls_exceeded = False

                for call in response.tool_calls:
                    tool_calls.append(call)
                    if len(tool_calls) >= max_tool_calls:
                        errors.append("Max tool calls exceeded")
                        max_tool_calls_exceeded = True
                        break
                    if call.error:
                        ordered_calls.append(call)
                        immediate_results[call.id] = ToolResult(error=call.error)
                        continue

                    tool = allowed_tools.get(call.name)
                    if tool is None:
                        ordered_calls.append(call)
                        immediate_results[call.id] = ToolResult(
                            error=f"Tool not found: {call.name}"
                        )
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

                await _execute_tool_calls_async(
                    ordered_calls,
                    executable_calls,
                    immediate_results,
                    messages,
                    tool_calls,
                    emit,
                )

                if max_tool_calls_exceeded:
                    emit("worker.failed", self.name, {"error": errors[-1]})
                    report = _build_report(
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
                    return report, None

                if pending is not None:
                    emit(
                        f"tool.{pending.type}_requested",
                        pending.tool_call.name,
                        {"tool_call_id": pending.tool_call.id},
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
                        pending,
                        errors,
                    )
                    state = _build_state(
                        run_id,
                        "paused",
                        self.name,
                        job,
                        messages,
                        tool_calls,
                        pending,
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
                    errors.append(str(exc))
                    emit("worker.failed", self.name, {"error": errors[-1]})
                    report = _build_report(
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
                    return report, None
                content = structured_content(data)
                assistant_message = Message(role="assistant", content=content)
                messages.append(assistant_message)
                emit_assistant_message(emit, self.name, assistant_message)
                emit("worker.completed", self.name, {})
                report = _build_report(
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
                return report, None

            assistant_message = Message(role="assistant", content=response.content or "")
            messages.append(assistant_message)
            emit_assistant_message(emit, self.name, assistant_message)
            emit("worker.completed", self.name, {})
            report = _build_report(
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
            return report, None

        errors.append("Max iterations exceeded")
        emit("worker.failed", self.name, {"error": errors[-1]})
        report = _build_report(
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
        return report, None

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
        return self._run_loop(
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
                result = execute_tool(tool, call)
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
        return self._run_loop(
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
