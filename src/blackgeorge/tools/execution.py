import asyncio
import concurrent.futures
import contextlib
import json
import time
import weakref
from inspect import isawaitable, iscoroutinefunction
from typing import Any

from pydantic import BaseModel

from blackgeorge.async_utils import run_coroutine_in_thread, run_coroutine_sync
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.exceptions import ToolExecutionError, ToolTimeoutError, ToolValidationError
from blackgeorge.tools.base import ProgressCallback, Tool, ToolResult

_shared_executor: concurrent.futures.ThreadPoolExecutor | None = None
_executor_refs: weakref.WeakSet[concurrent.futures.ThreadPoolExecutor] = weakref.WeakSet()


def get_shared_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _shared_executor
    if _shared_executor is None or _shared_executor._shutdown:  # type: ignore[attr-defined]
        _shared_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=10,
            thread_name_prefix="blackgeorge-tool-",
        )
        _executor_refs.add(_shared_executor)
    return _shared_executor


def shutdown_executor(wait: bool = True) -> None:
    global _shared_executor
    if _shared_executor is not None:
        _shared_executor.shutdown(wait=wait)
        _shared_executor = None


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


def _validate_output(result: Any, output_type: type[BaseModel] | None) -> tuple[Any, str | None]:
    if output_type is None:
        return result, None
    if isinstance(result, output_type):
        return result, None
    try:
        validated = output_type.model_validate(result)
        return validated, None
    except Exception as exc:
        return result, f"Output validation failed: {exc}"


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


def _execution_error_result(tool: Tool, exc: Exception) -> ToolResult:
    if isinstance(exc, (ToolExecutionError, ToolTimeoutError, ToolValidationError)):
        return ToolResult(error=str(exc), exception_type=type(exc).__name__)
    tool_exc = ToolExecutionError(tool.name, str(exc), exc)
    return ToolResult(error=str(tool_exc), exception_type=type(tool_exc).__name__)


def _post_hook_error_result(tool: Tool, result: ToolResult, exc: Exception) -> ToolResult:
    hook_result = _execution_error_result(tool, exc)
    error = hook_result.error
    if result.error is not None:
        error = f"{result.error}. Post-hook error: {hook_result.error}"
    return ToolResult(
        content=result.content,
        data=result.data,
        error=error,
        timed_out=result.timed_out,
        cancelled=result.cancelled,
        exception_type=result.exception_type or hook_result.exception_type,
    )


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
            executor = get_shared_executor()
            future = executor.submit(_run_sync_call, tool, args)
            try:
                result = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                future.cancel()
                timeout_exc = ToolTimeoutError(tool.name, timeout)
                return ToolResult(
                    error=str(timeout_exc),
                    timed_out=True,
                    exception_type=type(timeout_exc).__name__,
                )
        if isinstance(result, ToolResult):
            return result
        validated_result, validation_error = _validate_output(result, tool.output_type)
        if validation_error:
            validation_exc = ToolValidationError(tool.name, validation_error)
            return ToolResult(
                error=str(validation_exc),
                exception_type=type(validation_exc).__name__,
            )
        return ToolResult(content=_to_content(validated_result), data=validated_result)
    except Exception as original_exc:
        if isinstance(original_exc, (ToolExecutionError, ToolTimeoutError, ToolValidationError)):
            return ToolResult(
                error=str(original_exc),
                exception_type=type(original_exc).__name__,
            )
        tool_exc = ToolExecutionError(tool.name, str(original_exc), original_exc)
        return ToolResult(error=str(tool_exc), exception_type=type(tool_exc).__name__)


def execute_tool(tool: Tool, call: ToolCall) -> ToolResult:
    try:
        _sync_pre_hooks(tool, call)
    except Exception as exc:
        return _execution_error_result(tool, exc)

    try:
        args = _validate_tool_call(call, tool)
    except Exception as exc:
        exc_wrapped = ToolValidationError(tool.name, str(exc))
        tool_result = ToolResult(
            error=str(exc_wrapped),
            exception_type=type(exc_wrapped).__name__,
        )
        try:
            _sync_post_hooks(tool, call, tool_result)
        except Exception as hook_exc:
            return _post_hook_error_result(tool, tool_result, hook_exc)
        return tool_result

    tool_result = _execute_sync_with_retries(tool, args)
    try:
        _sync_post_hooks(tool, call, tool_result)
    except Exception as exc:
        return _post_hook_error_result(tool, tool_result, exc)

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
            validated_result, validation_error = _validate_output(result, tool.output_type)
            if validation_error:
                validation_exc = ToolValidationError(tool.name, validation_error)
                return ToolResult(
                    error=str(validation_exc),
                    exception_type=type(validation_exc).__name__,
                )
            return ToolResult(content=_to_content(validated_result), data=validated_result)

        if cancel_task is not None and cancel_task in done:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            return ToolResult(error="Tool execution cancelled", cancelled=True)

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        timeout_exc = ToolTimeoutError(tool.name, timeout or 0)
        return ToolResult(
            error=str(timeout_exc),
            timed_out=True,
            exception_type=type(timeout_exc).__name__,
        )
    except asyncio.CancelledError:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return ToolResult(error="Tool execution cancelled", cancelled=True)
    except Exception as original_exc:
        if isinstance(original_exc, (ToolExecutionError, ToolTimeoutError, ToolValidationError)):
            return ToolResult(
                error=str(original_exc),
                exception_type=type(original_exc).__name__,
            )
        tool_exc = ToolExecutionError(tool.name, str(original_exc), original_exc)
        return ToolResult(error=str(tool_exc), exception_type=type(tool_exc).__name__)
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
    try:
        await _async_pre_hooks(tool, call)
    except Exception as exc:
        return _execution_error_result(tool, exc)

    try:
        args = _validate_tool_call(call, tool)
    except Exception as exc:
        exc_wrapped = ToolValidationError(tool.name, str(exc))
        tool_result = ToolResult(
            error=str(exc_wrapped),
            exception_type=type(exc_wrapped).__name__,
        )
        try:
            await _async_post_hooks(tool, call, tool_result)
        except Exception as hook_exc:
            return _post_hook_error_result(tool, tool_result, hook_exc)
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

    try:
        await _async_post_hooks(tool, call, tool_result)
    except Exception as exc:
        return _post_hook_error_result(tool, tool_result, exc)

    return tool_result
