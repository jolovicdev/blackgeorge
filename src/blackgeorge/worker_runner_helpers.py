import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from blackgeorge.adapters.base import ModelResponse
from blackgeorge.core.event import Event
from blackgeorge.core.event_types import EventType
from blackgeorge.core.job import Job
from blackgeorge.core.message import Message
from blackgeorge.core.pending_action import PendingAction
from blackgeorge.core.report import Report
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.core.types import RunStatus
from blackgeorge.store.state import RunState
from blackgeorge.tools.base import Tool, ToolResult
from blackgeorge.tools.execution import aexecute_tool
from blackgeorge.utils import new_id
from blackgeorge.worker_context import (
    SUMMARY_ATTEMPT_LIMIT,
    context_error_message,
)
from blackgeorge.worker_messages import (
    emit_assistant_message,
    replace_tool_call,
    structured_content,
    tool_call_with_result,
    tool_message,
)
from blackgeorge.worker_tools import (
    pending_options,
    tool_action_type,
    tool_prompt,
)

if TYPE_CHECKING:
    from blackgeorge.config import RunConfig

EventEmitter = Callable[[str, str, dict[str, Any]], None]


def build_report(
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


def build_worker_state(
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


def build_error_report(
    run_id: str, messages: list[Message], errors: list[str], events: list[Event]
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


def fail_report(
    *,
    config: "RunConfig",
    worker_name: str,
    message: str,
    messages: list[Message],
    tool_calls: list[ToolCall],
    metrics: dict[str, Any],
    errors: list[str],
) -> Report:
    errors.append(message)
    config.emit(EventType.WORKER_FAILED, worker_name, {"error": message})
    return build_report(
        config.run_id,
        "failed",
        None,
        None,
        None,
        messages,
        tool_calls,
        metrics,
        config.events,
        None,
        errors,
    )


async def aresolve_context_retry(
    *,
    config: "RunConfig",
    worker_name: str,
    messages: list[Message],
    tool_calls: list[ToolCall],
    metrics: dict[str, Any],
    errors: list[str],
    model_registered: bool,
    context_summaries: int,
    apply_summary: Callable[[], Awaitable[bool]],
) -> "ContextDecision":
    if not config.respect_context_window:
        return ContextDecision(
            False,
            fail_report(
                config=config,
                worker_name=worker_name,
                message=context_error_message(model_registered, False),
                messages=messages,
                tool_calls=tool_calls,
                metrics=metrics,
                errors=errors,
            ),
        )
    if context_summaries >= SUMMARY_ATTEMPT_LIMIT:
        return ContextDecision(
            False,
            fail_report(
                config=config,
                worker_name=worker_name,
                message=context_error_message(model_registered, True),
                messages=messages,
                tool_calls=tool_calls,
                metrics=metrics,
                errors=errors,
            ),
        )
    if not await apply_summary():
        return ContextDecision(
            False,
            fail_report(
                config=config,
                worker_name=worker_name,
                message=context_error_message(model_registered, True),
                messages=messages,
                tool_calls=tool_calls,
                metrics=metrics,
                errors=errors,
            ),
        )
    return ContextDecision(True, None)


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


def tool_event_payload(call: ToolCall, result: ToolResult, limit: int = 200) -> dict[str, Any]:
    payload: dict[str, Any] = {"tool_call_id": call.id}
    preview, truncated = _tool_result_preview(result, limit)
    if preview is not None:
        payload["result_preview"], payload["result_truncated"] = preview, truncated
    if result.timed_out:
        payload["timed_out"] = True
    if result.cancelled:
        payload["cancelled"] = True
    return payload


def finalize_structured_response(
    *,
    config: "RunConfig",
    data: Any,
    messages: list[Message],
    tool_calls: list[ToolCall],
    metrics: dict[str, Any],
    errors: list[str],
    worker_name: str,
) -> Report:
    content = structured_content(data)
    assistant_message = Message(role="assistant", content=content)
    messages.append(assistant_message)
    emit_assistant_message(config.emit, worker_name, assistant_message)
    config.emit(EventType.WORKER_COMPLETED, worker_name, {})
    return build_report(
        config.run_id,
        "completed",
        content,
        None,
        data,
        messages,
        tool_calls,
        metrics,
        config.events,
        None,
        errors,
    )


def finalize_plain_response(
    *,
    config: "RunConfig",
    response: ModelResponse,
    messages: list[Message],
    tool_calls: list[ToolCall],
    metrics: dict[str, Any],
    errors: list[str],
    worker_name: str,
) -> Report:
    assistant_message = Message(
        role="assistant",
        content=response.content or "",
        reasoning_content=response.reasoning_content,
        thinking_blocks=response.thinking_blocks,
    )
    messages.append(assistant_message)
    emit_assistant_message(config.emit, worker_name, assistant_message)
    config.emit(EventType.WORKER_COMPLETED, worker_name, {})
    return build_report(
        config.run_id,
        "completed",
        response.content,
        response.reasoning_content,
        None,
        messages,
        tool_calls,
        metrics,
        config.events,
        None,
        errors,
    )


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


def plan_tool_calls(
    *,
    response: ModelResponse,
    allowed_tools: dict[str, Tool],
    tool_calls: list[ToolCall],
    max_tool_calls: int,
) -> ToolPlan:
    ordered_calls, executable_calls, immediate_results, pending, max_tool_calls_exceeded = (
        [],
        [],
        {},
        None,
        False,
    )
    for i, call in enumerate(response.tool_calls):
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
            for remaining_call in response.tool_calls[i + 1 :]:
                if len(tool_calls) >= max_tool_calls:
                    max_tool_calls_exceeded = True
                    break
                tool_calls.append(remaining_call)
                ordered_calls.append(remaining_call)
                immediate_results[remaining_call.id] = ToolResult(
                    error="Skipped: another tool requires confirmation first"
                )
            break
        ordered_calls.append(call)
        executable_calls.append((call, tool))
    return ToolPlan(
        ordered_calls, executable_calls, immediate_results, pending, max_tool_calls_exceeded
    )


async def aexecute_tool_calls(
    config: "RunConfig",
    ordered_calls: list[ToolCall],
    executable_calls: list[tuple[ToolCall, Tool]],
    immediate_results: dict[str, ToolResult],
    messages: list[Message],
    tool_calls: list[ToolCall],
) -> None:
    results = dict(immediate_results)
    if executable_calls:
        for call, tool in executable_calls:
            config.emit(EventType.TOOL_STARTED, tool.name, {"tool_call_id": call.id})
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
            config.emit(
                EventType.TOOL_FAILED, call.name, {"tool_call_id": call.id, "error": result.error}
            )
        else:
            config.emit(EventType.TOOL_COMPLETED, call.name, tool_event_payload(call, result))
        messages.append(tool_message(result, call))
        replace_tool_call(tool_calls, tool_call_with_result(call, result))
