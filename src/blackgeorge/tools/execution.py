import asyncio
import concurrent.futures
import contextlib
import json
import time
from inspect import isawaitable, iscoroutinefunction
from typing import Any

from pydantic import BaseModel

from blackgeorge.async_utils import run_coroutine_in_thread, run_coroutine_sync
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.tools.base import ProgressCallback, Tool, ToolResult


def _to_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    try:
        return json.dumps(value, ensure_ascii=True, default=str)
    except TypeError:
        return str(value)


def _run_coroutine_in_thread(coro: Any) -> Any:
    return run_coroutine_in_thread(coro)


def _run_coroutine_sync(coro: Any) -> Any:
    return run_coroutine_sync(coro)


def _run_sync_call(tool: Tool, args: dict[str, Any]) -> Any:
    if iscoroutinefunction(tool.callable):
        return _run_coroutine_sync(tool.callable(**args))
    return tool.callable(**args)


def _validate_tool_call(call: ToolCall, tool: Tool) -> dict[str, Any]:
    validated = tool.input_model.model_validate(call.arguments)
    return validated.model_dump()


def _sync_invoke_hook(hook: Any, *args: Any) -> None:
    result = hook(*args)
    if isawaitable(result):
        _run_coroutine_sync(result)


def _sync_pre_hooks(tool: Tool, call: ToolCall) -> None:
    for pre_hook in tool.pre:
        _sync_invoke_hook(pre_hook, call)


def _sync_post_hooks(tool: Tool, call: ToolCall, result: ToolResult) -> None:
    for post_hook in tool.post:
        _sync_invoke_hook(post_hook, call, result)


async def _async_invoke_hook(hook: Any, *args: Any) -> None:
    result = hook(*args)
    if isawaitable(result):
        await result


async def _async_pre_hooks(tool: Tool, call: ToolCall) -> None:
    for pre_hook in tool.pre:
        await _async_invoke_hook(pre_hook, call)


async def _async_post_hooks(tool: Tool, call: ToolCall, result: ToolResult) -> None:
    for post_hook in tool.post:
        await _async_invoke_hook(post_hook, call, result)


def _backoff_delay(retry_delay: float, attempt: int) -> float:
    return float(retry_delay * (2**attempt))


def _execute_sync_with_retries(
    tool: Tool,
    args: dict[str, Any],
) -> ToolResult:
    retries = tool.retries
    retry_delay = tool.retry_delay
    last_result: ToolResult | None = None

    for attempt in range(retries + 1):
        last_result = _execute_sync_once(tool, args, tool.timeout)
        if last_result.error is None:
            break
        if last_result.cancelled:
            break
        if attempt < retries:
            time.sleep(_backoff_delay(retry_delay, attempt))

    return last_result or ToolResult(error="No execution result")


def _execute_sync_once(tool: Tool, args: dict[str, Any], timeout: float | None) -> ToolResult:
    try:
        if timeout is None:
            result = _run_sync_call(tool, args)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_sync_call, tool, args)
                try:
                    result = future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    return ToolResult(
                        error=f"Tool execution timed out after {timeout}s",
                        timed_out=True,
                    )
        if isinstance(result, ToolResult):
            return result
        return ToolResult(content=_to_content(result), data=result)
    except Exception as exc:
        return ToolResult(error=str(exc))


def execute_tool(tool: Tool, call: ToolCall) -> ToolResult:
    _sync_pre_hooks(tool, call)

    try:
        args = _validate_tool_call(call, tool)
    except Exception as exc:
        tool_result = ToolResult(error=str(exc))
        _sync_post_hooks(tool, call, tool_result)
        return tool_result

    tool_result = _execute_sync_with_retries(tool, args)
    _sync_post_hooks(tool, call, tool_result)

    return tool_result


async def _execute_once(
    tool: Tool,
    args: dict[str, Any],
    timeout: float | None,
    cancel_event: asyncio.Event | None,
) -> ToolResult:
    async def run_callable() -> Any:
        if iscoroutinefunction(tool.callable):
            return await tool.callable(**args)
        return await asyncio.to_thread(tool.callable, **args)

    if cancel_event is not None and cancel_event.is_set():
        return ToolResult(error="Cancelled before execution", cancelled=True)

    task = asyncio.create_task(run_callable())
    cancel_task: asyncio.Task[bool] | None = None
    tasks: set[asyncio.Task[Any]] = {task}
    if cancel_event is not None:
        cancel_task = asyncio.create_task(cancel_event.wait())
        tasks.add(cancel_task)

    try:
        done, _ = await asyncio.wait(
            tasks,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if task in done:
            result = await task
            if isinstance(result, ToolResult):
                return result
            return ToolResult(content=_to_content(result), data=result)

        if cancel_task is not None and cancel_task in done:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            return ToolResult(error="Tool execution cancelled", cancelled=True)

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return ToolResult(error=f"Tool execution timed out after {timeout}s", timed_out=True)
    except asyncio.CancelledError:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return ToolResult(error="Tool execution cancelled", cancelled=True)
    except Exception as exc:
        return ToolResult(error=str(exc))
    finally:
        if cancel_task is not None and not cancel_task.done():
            cancel_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cancel_task


async def aexecute_tool(
    tool: Tool,
    call: ToolCall,
    *,
    cancel_event: asyncio.Event | None = None,
    on_progress: ProgressCallback | None = None,
) -> ToolResult:
    await _async_pre_hooks(tool, call)

    try:
        args = _validate_tool_call(call, tool)
    except Exception as exc:
        tool_result = ToolResult(error=f"Validation error: {exc}")
        await _async_post_hooks(tool, call, tool_result)
        return tool_result

    retries = tool.retries
    retry_delay = tool.retry_delay
    last_result: ToolResult | None = None

    for attempt in range(retries + 1):
        if cancel_event is not None and cancel_event.is_set():
            return ToolResult(error="Cancelled", cancelled=True)

        if on_progress is not None and attempt > 0:
            on_progress(f"Retry attempt {attempt}/{retries}")

        last_result = await _execute_once(tool, args, tool.timeout, cancel_event)

        if last_result.error is None:
            break

        if last_result.cancelled:
            break

        if attempt < retries:
            delay = _backoff_delay(retry_delay, attempt)
            if on_progress is not None:
                on_progress(f"Waiting {delay:.1f}s before retry")
            await asyncio.sleep(delay)

    tool_result = last_result or ToolResult(error="No execution result")

    await _async_post_hooks(tool, call, tool_result)

    return tool_result
