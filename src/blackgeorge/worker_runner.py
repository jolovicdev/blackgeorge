import asyncio
import json
import warnings
from collections.abc import Callable, Iterable
from typing import Any, cast

from blackgeorge.adapters.base import ModelResponse
from blackgeorge.config import RunConfig
from blackgeorge.core.event_types import EventType
from blackgeorge.core.job import Job
from blackgeorge.core.message import Message
from blackgeorge.core.report import Report
from blackgeorge.runner.loop_state import CompletionContext, LoopState
from blackgeorge.runner.streaming import (
    append_tool_error,
    chunk_tool_call_deltas,
    is_stream_unsupported_error,
    parse_structured_stream_json,
    stream_value,
    streamed_tool_calls,
)
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
    _build_report,
    _ensure_not_running_loop,
    _execute_tool_calls_async,
    _plan_tool_calls,
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
        config: RunConfig,
        model: str,
        messages: list[Message],
        response_schema: Any,
    ) -> Any:
        payload = messages_to_payload(messages)
        try:
            return await config.adapter.astructured_complete(
                model=model,
                messages=payload,
                response_schema=response_schema,
                retries=config.structured_output_retries,
            )
        except NotImplementedError:
            return await asyncio.to_thread(
                config.adapter.structured_complete,
                model=model,
                messages=payload,
                response_schema=response_schema,
                retries=config.structured_output_retries,
            )

    async def _acompletion(
        self,
        *,
        config: RunConfig,
        model: str,
        messages: list[Message],
        tools: list[Tool],
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ModelResponse:
        if hasattr(config.adapter, "set_callback_context"):
            config.adapter.set_callback_context(config.run_id, config.emit)
        try:
            try:
                response = await config.adapter.acomplete(
                    model=model,
                    messages=messages_to_payload(messages),
                    tools=tool_schemas(tools) if tools else None,
                    tool_choice="auto" if tools else None,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    stream=False,
                    stream_options=config.stream_options,
                    thinking=thinking,
                    drop_params=drop_params,
                    extra_body=extra_body,
                )
            except NotImplementedError:
                response = await asyncio.to_thread(
                    config.adapter.complete,
                    model=model,
                    messages=messages_to_payload(messages),
                    tools=tool_schemas(tools) if tools else None,
                    tool_choice="auto" if tools else None,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    stream=False,
                    stream_options=config.stream_options,
                    thinking=thinking,
                    drop_params=drop_params,
                    extra_body=extra_body,
                )
            if isinstance(response, ModelResponse):
                return response
            return ModelResponse(content=None, tool_calls=[], usage={}, raw=response)
        finally:
            if hasattr(config.adapter, "clear_callback_context"):
                config.adapter.clear_callback_context()

    async def _astream_completion(
        self,
        *,
        config: RunConfig,
        model: str,
        messages: list[Message],
        tools: list[Tool],
        on_token: Callable[[str, str], None],
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ModelResponse:
        if hasattr(config.adapter, "set_callback_context"):
            config.adapter.set_callback_context(config.run_id, config.emit)
        try:
            try:
                stream = await config.adapter.acomplete(
                    model=model,
                    messages=messages_to_payload(messages),
                    tools=tool_schemas(tools) if tools else None,
                    tool_choice="auto" if tools else None,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    stream=True,
                    stream_options=config.stream_options,
                    thinking=thinking,
                    drop_params=drop_params,
                    extra_body=extra_body,
                )
            except NotImplementedError:
                try:
                    stream = await asyncio.to_thread(
                        config.adapter.complete,
                        model=model,
                        messages=messages_to_payload(messages),
                        tools=tool_schemas(tools) if tools else None,
                        tool_choice="auto" if tools else None,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                        stream=True,
                        stream_options=config.stream_options,
                        thinking=thinking,
                        drop_params=drop_params,
                        extra_body=extra_body,
                    )
                except Exception as exc:
                    if is_stream_unsupported_error(exc):
                        return await self._acompletion(
                            config=config,
                            model=model,
                            messages=messages,
                            tools=tools,
                            thinking=thinking,
                            drop_params=drop_params,
                            extra_body=extra_body,
                        )
                    raise
            except Exception as exc:
                if is_stream_unsupported_error(exc):
                    return await self._acompletion(
                        config=config,
                        model=model,
                        messages=messages,
                        tools=tools,
                        thinking=thinking,
                        drop_params=drop_params,
                        extra_body=extra_body,
                    )
                raise
            if isinstance(stream, ModelResponse):
                return stream
            is_async_stream = hasattr(stream, "__aiter__")
            is_sync_stream = hasattr(stream, "__iter__") and not isinstance(
                stream, (str, bytes, bytearray, dict)
            )
            if not is_async_stream and not is_sync_stream:
                return await self._acompletion(
                    config=config,
                    model=model,
                    messages=messages,
                    tools=tools,
                    thinking=thinking,
                    drop_params=drop_params,
                    extra_body=extra_body,
                )

            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            thinking_blocks: list[dict[str, Any]] = []
            tool_states: list[dict[str, Any]] = []
            keyed_states: dict[tuple[str, Any], dict[str, Any]] = {}
            usage: dict[str, Any] = {}

            def process_chunk(chunk: Any) -> None:
                nonlocal usage
                token = chunk_content(chunk)
                if token:
                    content_parts.append(token)
                    on_token(token, "content")
                tool_deltas, from_message_payload = chunk_tool_call_deltas(chunk)
                for position, tool_delta in enumerate(tool_deltas):
                    index_value = stream_value(tool_delta, "index")
                    call_id_value = stream_value(tool_delta, "id")
                    stable_keys: list[tuple[str, int | str]] = []
                    if isinstance(index_value, int):
                        stable_keys.append(("index", index_value))
                    if isinstance(call_id_value, str) and call_id_value:
                        stable_keys.append(("id", call_id_value))
                    fallback_key = ("position", position)
                    lookup_keys = stable_keys or [fallback_key]
                    state = None
                    for key in lookup_keys:
                        state = keyed_states.get(key)
                        if state is not None:
                            break
                    if state is None:
                        state = {
                            "id": None,
                            "name": "",
                            "arguments_parts": [],
                            "arguments_obj": None,
                            "error": None,
                        }
                        tool_states.append(state)
                    for key in stable_keys or [fallback_key]:
                        keyed_states[key] = state
                    if isinstance(call_id_value, str) and call_id_value:
                        state["id"] = call_id_value
                    function = stream_value(tool_delta, "function")
                    name_value = stream_value(function, "name")
                    if isinstance(name_value, str):
                        name = name_value.strip()
                        if name:
                            existing_name = cast(str, state["name"])
                            if not existing_name or name.startswith(existing_name):
                                state["name"] = name
                            elif not existing_name.startswith(name):
                                state["name"] = f"{existing_name}{name}"
                    arguments_value = stream_value(function, "arguments")
                    if isinstance(arguments_value, str):
                        argument_parts = cast(list[str], state["arguments_parts"])
                        if from_message_payload:
                            if "".join(argument_parts) != arguments_value:
                                argument_parts.clear()
                                argument_parts.append(arguments_value)
                                if arguments_value:
                                    on_token(arguments_value, "tool_argument")
                        else:
                            argument_parts.append(arguments_value)
                            if arguments_value:
                                on_token(arguments_value, "tool_argument")
                    elif isinstance(arguments_value, dict):
                        merged = dict(state.get("arguments_obj") or {})
                        merged.update(arguments_value)
                        state["arguments_obj"] = merged
                        serialized = json.dumps(arguments_value, ensure_ascii=True)
                        if serialized:
                            on_token(serialized, "tool_argument")
                    elif arguments_value is not None:
                        state["error"] = append_tool_error(
                            cast(str | None, state.get("error")),
                            f"Unsupported tool arguments type: {type(arguments_value).__name__}",
                        )
                reasoning = chunk_reasoning_content(chunk)
                if reasoning:
                    reasoning_parts.append(reasoning)
                blocks = chunk_thinking_blocks(chunk)
                if blocks:
                    thinking_blocks.extend(blocks)
                usage_chunk = chunk_usage(chunk)
                if usage_chunk:
                    usage = usage_chunk

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
                try:
                    if is_async_stream:
                        async for chunk in stream:
                            process_chunk(chunk)
                    else:
                        for chunk in cast(Iterable[Any], stream):
                            process_chunk(chunk)
                finally:
                    if hasattr(stream, "aclose"):
                        await stream.aclose()
                    elif hasattr(stream, "close"):
                        stream.close()
            return ModelResponse(
                content="".join(content_parts),
                reasoning_content="".join(reasoning_parts) or None,
                thinking_blocks=thinking_blocks or None,
                tool_calls=streamed_tool_calls(tool_states),
                usage=usage,
                raw=stream,
            )
        finally:
            if hasattr(config.adapter, "clear_callback_context"):
                config.adapter.clear_callback_context()

    async def _apply_context_summary(self, ctx: CompletionContext) -> bool:
        return await aapply_context_summary(
            adapter=ctx.config.adapter,
            model_name=ctx.model_name,
            messages=ctx.state.messages,
            temperature=ctx.config.temperature,
            metrics=ctx.state.metrics,
            emit=ctx.config.emit,
            worker_name=ctx.state.worker_name,
            model_registered=ctx.state.model_registered,
        )

    async def _retry_context_or_report(self, ctx: CompletionContext) -> Report | None:
        return await ctx.handle_context_limit(lambda: self._apply_context_summary(ctx))

    async def _acquire_turn_response(
        self,
        *,
        ctx: CompletionContext,
        job: Job,
        tools: list[Tool],
        response_schema: Any,
        structured_stream_mode: str,
    ) -> tuple[ModelResponse | None, Report | None]:
        on_token = ctx.make_on_token()
        if _should_stream(ctx.config.stream, tools, response_schema):
            try:
                response = await self._astream_completion(
                    config=ctx.run_config(),
                    model=ctx.model_name,
                    messages=ctx.state.messages,
                    tools=tools,
                    on_token=on_token,
                    thinking=job.thinking,
                    drop_params=job.drop_params,
                    extra_body=job.extra_body,
                )
                return response, None
            except Exception as exc:
                if not is_context_limit_error(exc):
                    raise
                report = await self._retry_context_or_report(ctx)
                return None, report

        if (
            ctx.config.stream
            and response_schema
            and not tools
            and structured_stream_mode == "preview"
        ):
            try:
                streamed = await self._astream_completion(
                    config=ctx.run_config(),
                    model=ctx.model_name,
                    messages=ctx.state.messages,
                    tools=[],
                    on_token=on_token,
                    thinking=job.thinking,
                    drop_params=job.drop_params,
                    extra_body=job.extra_body,
                )
            except Exception as exc:
                if not is_context_limit_error(exc):
                    raise
                report = await self._retry_context_or_report(ctx)
                return None, report
            ctx.state.metrics["usage"] = streamed.usage
            try:
                data = parse_structured_stream_json(response_schema, streamed.content or "")
            except Exception:
                try:
                    data = await self._astructured_completion(
                        config=ctx.run_config(),
                        model=ctx.model_name,
                        messages=ctx.state.messages,
                        response_schema=response_schema,
                    )
                except Exception as exc:
                    if is_context_limit_error(exc):
                        report = await self._retry_context_or_report(ctx)
                        return None, report
                    return None, ctx.fail(str(exc))
            return None, ctx.finalize_structured(data)

        if response_schema and not tools:
            try:
                data = await self._astructured_completion(
                    config=ctx.run_config(),
                    model=ctx.model_name,
                    messages=ctx.state.messages,
                    response_schema=response_schema,
                )
            except Exception as exc:
                if is_context_limit_error(exc):
                    report = await self._retry_context_or_report(ctx)
                    return None, report
                return None, ctx.fail(str(exc))
            return None, ctx.finalize_structured(data)

        try:
            response = await self._acompletion(
                config=ctx.run_config(),
                model=ctx.model_name,
                messages=ctx.state.messages,
                tools=tools,
                thinking=job.thinking,
                drop_params=job.drop_params,
                extra_body=job.extra_body,
            )
            return response, None
        except Exception as exc:
            if not is_context_limit_error(exc):
                raise
            report = await self._retry_context_or_report(ctx)
            return None, report

    async def _process_turn_response(
        self,
        *,
        ctx: CompletionContext,
        job: Job,
        response: ModelResponse,
        response_schema: Any,
        allowed_tools: dict[str, Tool],
    ) -> tuple[Report | None, RunState | None, bool]:
        ctx.record_usage(response)
        if response.tool_calls:
            assistant_message = Message(
                role="assistant",
                content=ensure_content(response.content),
                reasoning_content=response.reasoning_content,
                thinking_blocks=response.thinking_blocks,
                tool_calls=response.tool_calls,
            )
            ctx.state.messages.append(assistant_message)
            emit_assistant_message(ctx.config.emit, self.name, assistant_message)
            plan = _plan_tool_calls(
                response=response,
                allowed_tools=allowed_tools,
                tool_calls=ctx.state.tool_calls,
                max_tool_calls=ctx.config.max_tool_calls,
            )
            await _execute_tool_calls_async(
                ctx.run_config(),
                plan.ordered_calls,
                plan.executable_calls,
                plan.immediate_results,
                ctx.state.messages,
                ctx.state.tool_calls,
            )
            if plan.max_tool_calls_exceeded:
                return ctx.fail("Max tool calls exceeded"), None, False
            if plan.pending:
                if plan.pending.type == "confirmation":
                    ctx.config.emit(
                        EventType.TOOL_CONFIRMATION_REQUESTED,
                        plan.pending.tool_call.name,
                        {"tool_call_id": plan.pending.tool_call.id},
                    )
                elif plan.pending.type == "user_input":
                    ctx.config.emit(
                        EventType.TOOL_USER_INPUT_REQUESTED,
                        plan.pending.tool_call.name,
                        {"tool_call_id": plan.pending.tool_call.id},
                    )
                ctx.config.emit(
                    EventType.WORKER_PAUSED,
                    self.name,
                    {"pending_action_type": plan.pending.type},
                )
                return (
                    ctx.build_paused_report(plan.pending),
                    ctx.build_paused_state(job, plan.pending),
                    False,
                )
            return None, None, True

        if response_schema:
            try:
                data = await self._astructured_completion(
                    config=ctx.run_config(),
                    model=ctx.model_name,
                    messages=ctx.state.messages,
                    response_schema=response_schema,
                )
            except Exception as exc:
                if is_context_limit_error(exc):
                    retry_report = await self._retry_context_or_report(ctx)
                    if retry_report:
                        return retry_report, None, False
                    return None, None, True
                return ctx.fail(str(exc)), None, False
            return ctx.finalize_structured(data), None, False

        return ctx.finalize_plain(response), None, False

    async def _arun_loop(
        self, *, ctx: CompletionContext, job: Job
    ) -> tuple[Report, RunState | None]:
        tools = self._resolve_tools(job)
        allowed_tools = {t.name: t for t in tools}
        response_schema = job.response_schema
        structured_stream_mode = job.structured_stream_mode or "off"

        while ctx.state.iteration < ctx.config.max_iterations:
            ctx.state.increment_iteration()
            if (
                ctx.config.max_context_messages is not None
                and len(ctx.state.messages) > ctx.config.max_context_messages
            ):
                await self._apply_context_summary(ctx)

            response, report = await self._acquire_turn_response(
                ctx=ctx,
                job=job,
                tools=tools,
                response_schema=response_schema,
                structured_stream_mode=structured_stream_mode,
            )
            if report:
                return report, None
            if response is None:
                continue
            report, state, should_continue = await self._process_turn_response(
                ctx=ctx,
                job=job,
                response=response,
                response_schema=response_schema,
                allowed_tools=allowed_tools,
            )
            if report:
                return report, state
            if should_continue:
                continue
        return ctx.fail("Max iterations exceeded"), None

    def run(
        self, config: RunConfig, job: Job, worker_model: str | None = None
    ) -> tuple[Report, RunState | None]:
        _ensure_not_running_loop("run", "arun")
        return asyncio.run(self.arun(config, job, worker_model))

    async def arun(
        self, config: RunConfig, job: Job, worker_model: str | None = None
    ) -> tuple[Report, RunState | None]:
        model_name = config.model_name(worker_model)
        messages = self._build_messages(job)
        if not model_name:
            errors = ["Worker model not set"]
            config.emit(EventType.WORKER_FAILED, self.name, {"error": errors[-1]})
            return _report_error(config.run_id, messages, errors, config.events), None
        state = LoopState(
            run_id=config.run_id,
            worker_name=self.name,
            messages=messages,
            tool_calls=[],
            metrics={},
            events=config.events,
            errors=[],
            iteration=0,
            context_summaries=0,
            model_registered=litellm_model_registered(model_name),
        )
        ctx = CompletionContext(config=config, model_name=model_name, state=state)
        config.emit(EventType.WORKER_STARTED, self.name, {})
        return await self._arun_loop(ctx=ctx, job=job)

    def resume(
        self,
        config: RunConfig,
        state: RunState,
        decision_or_input: Any,
        worker_model: str | None = None,
    ) -> tuple[Report, RunState | None]:
        _ensure_not_running_loop("resume", "aresume")
        return asyncio.run(self.aresume(config, state, decision_or_input, worker_model))

    async def aresume(
        self,
        config: RunConfig,
        state: RunState,
        decision_or_input: Any,
        worker_model: str | None = None,
    ) -> tuple[Report, RunState | None]:
        if config.run_id != state.run_id:
            config = config.with_overrides(run_id=state.run_id)
        pending = state.pending_action
        if pending is None:
            return _build_report(
                config.run_id,
                "failed",
                None,
                None,
                None,
                state.messages,
                state.tool_calls,
                state.metrics,
                config.events,
                None,
                ["No pending action"],
            ), None
        messages = list(state.messages)
        tool_calls = list(state.tool_calls)
        tool = self.toolbelt.resolve(pending.tool_call.name)
        if tool is None:
            result = ToolResult(error=f"Tool not found: {pending.tool_call.name}")
            config.emit(
                EventType.TOOL_FAILED,
                pending.tool_call.name,
                {"tool_call_id": pending.tool_call.id, "error": result.error},
            )
            messages.append(tool_message(result, pending.tool_call))
            replace_tool_call(tool_calls, tool_call_with_result(pending.tool_call, result))
        elif pending.type == "confirmation" and not decision_or_input:
            result = ToolResult(error="Tool execution declined")
            config.emit(
                EventType.TOOL_FAILED,
                pending.tool_call.name,
                {"tool_call_id": pending.tool_call.id, "error": result.error},
            )
            messages.append(tool_message(result, pending.tool_call))
            replace_tool_call(tool_calls, tool_call_with_result(pending.tool_call, result))
        else:
            call = pending.tool_call
            if pending.type == "user_input":
                call = update_arguments(call, resume_argument_key(pending), decision_or_input)
            config.emit(EventType.TOOL_STARTED, tool.name, {"tool_call_id": call.id})
            result = await aexecute_tool(tool, call)
            if result.error:
                config.emit(
                    EventType.TOOL_FAILED,
                    tool.name,
                    {"tool_call_id": call.id, "error": result.error},
                )
            else:
                config.emit(EventType.TOOL_COMPLETED, tool.name, _tool_event_payload(call, result))
            messages.append(tool_message(result, call))
            replace_tool_call(tool_calls, tool_call_with_result(call, result))
        model_name = config.model_name(worker_model)
        if not model_name:
            errors = ["Worker model not set"]
            config.emit(EventType.WORKER_FAILED, self.name, {"error": errors[-1]})
            return _report_error(state.run_id, messages, errors, config.events), None
        loop_state = LoopState(
            run_id=config.run_id,
            worker_name=self.name,
            messages=messages,
            tool_calls=tool_calls,
            metrics=dict(state.metrics),
            events=config.events,
            errors=[],
            iteration=state.iteration,
            context_summaries=0,
            model_registered=litellm_model_registered(model_name),
        )
        ctx = CompletionContext(config=config, model_name=model_name, state=loop_state)
        config.emit(EventType.WORKER_STARTED, self.name, {})
        return await self._arun_loop(ctx=ctx, job=state.job)
