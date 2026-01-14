import asyncio
import time

import pytest

from blackgeorge.core.tool_call import ToolCall
from blackgeorge.tools import tool
from blackgeorge.tools.base import Tool
from blackgeorge.tools.execution import aexecute_tool, execute_tool
from blackgeorge.tools.schema import build_input_model, build_schema


@pytest.mark.asyncio
async def test_timeout_triggers() -> None:
    @tool(timeout=0.1)
    async def slow_tool() -> str:
        await asyncio.sleep(5)
        return "done"

    call = ToolCall(id="1", name="slow_tool", arguments={})
    result = await aexecute_tool(slow_tool, call)
    assert result.timed_out is True
    assert result.error is not None
    assert "timed out" in result.error.lower()


@pytest.mark.asyncio
async def test_no_timeout_succeeds() -> None:
    @tool(timeout=5.0)
    async def fast_tool() -> str:
        await asyncio.sleep(0.01)
        return "done"

    call = ToolCall(id="1", name="fast_tool", arguments={})
    result = await aexecute_tool(fast_tool, call)
    assert result.timed_out is False
    assert result.error is None
    assert result.content == "done"


def test_sync_timeout_triggers() -> None:
    @tool(timeout=0.05)
    def slow_tool() -> str:
        time.sleep(0.2)
        return "done"

    call = ToolCall(id="1", name="slow_tool", arguments={})
    result = execute_tool(slow_tool, call)
    assert result.timed_out is True
    assert result.error is not None
    assert "timed out" in result.error.lower()


def test_sync_retry_succeeds() -> None:
    call_count = 0

    @tool(retries=2, retry_delay=0.01)
    def flaky_tool() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("temporary failure")
        return "success"

    call = ToolCall(id="1", name="flaky_tool", arguments={})
    result = execute_tool(flaky_tool, call)
    assert result.error is None
    assert result.content == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_succeeds_after_failure() -> None:
    call_count = 0

    def flaky_tool() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("temporary failure")
        return "success"

    input_model = build_input_model(flaky_tool)
    schema = build_schema(input_model)
    flaky = Tool(
        name="flaky",
        description="flaky tool",
        schema=schema,
        callable=flaky_tool,
        input_model=input_model,
        retries=3,
        retry_delay=0.01,
    )
    call = ToolCall(id="1", name="flaky", arguments={})
    result = await aexecute_tool(flaky, call)
    assert result.error is None
    assert result.content == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_exhausted() -> None:
    call_count = 0

    def always_fails() -> str:
        nonlocal call_count
        call_count += 1
        raise ValueError("always fails")

    input_model = build_input_model(always_fails)
    schema = build_schema(input_model)
    failing = Tool(
        name="failing",
        description="always fails",
        schema=schema,
        callable=always_fails,
        input_model=input_model,
        retries=2,
        retry_delay=0.01,
    )
    call = ToolCall(id="1", name="failing", arguments={})
    result = await aexecute_tool(failing, call)
    assert result.error is not None
    assert "always fails" in result.error
    assert call_count == 3


@pytest.mark.asyncio
async def test_cancellation() -> None:
    cancel_event = asyncio.Event()
    cancel_event.set()

    @tool()
    async def cancellable() -> str:
        await asyncio.sleep(10)
        return "done"

    call = ToolCall(id="1", name="cancellable", arguments={})
    result = await aexecute_tool(cancellable, call, cancel_event=cancel_event)
    assert result.cancelled is True


@pytest.mark.asyncio
async def test_cancellation_during_execution() -> None:
    cancel_event = asyncio.Event()

    @tool()
    async def cancellable() -> str:
        await asyncio.sleep(1)
        return "done"

    call = ToolCall(id="1", name="cancellable", arguments={})
    task = asyncio.create_task(aexecute_tool(cancellable, call, cancel_event=cancel_event))
    await asyncio.sleep(0.05)
    cancel_event.set()
    result = await task
    assert result.cancelled is True


@pytest.mark.asyncio
async def test_execute_tool_runs_async_callable() -> None:
    @tool()
    async def async_tool() -> str:
        await asyncio.sleep(0.01)
        return "ok"

    call = ToolCall(id="1", name="async_tool", arguments={})
    result = execute_tool(async_tool, call)
    assert result.error is None
    assert result.content == "ok"


@pytest.mark.asyncio
async def test_progress_callback() -> None:
    call_count = 0
    progress_messages: list[str] = []

    def flaky() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("fail")
        return "ok"

    input_model = build_input_model(flaky)
    schema = build_schema(input_model)
    tool_obj = Tool(
        name="flaky",
        description="flaky",
        schema=schema,
        callable=flaky,
        input_model=input_model,
        retries=2,
        retry_delay=0.01,
    )
    call = ToolCall(id="1", name="flaky", arguments={})
    result = await aexecute_tool(
        tool_obj, call, on_progress=lambda msg: progress_messages.append(msg)
    )
    assert result.error is None
    assert len(progress_messages) >= 1
