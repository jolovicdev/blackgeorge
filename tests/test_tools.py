import asyncio

import pytest

from blackgeorge.core.tool_call import ToolCall
from blackgeorge.tools import execute_tool, tool
from blackgeorge.tools.execution import _run_coroutine_sync, aexecute_tool
from blackgeorge.worker_messages import tool_message


def test_tool_schema_inference() -> None:
    @tool()
    def add(a: int, b: int) -> int:
        return a + b

    assert "properties" in add.schema
    assert "a" in add.schema["properties"]
    assert "b" in add.schema["properties"]


def test_tool_execution() -> None:
    @tool()
    def add(a: int, b: int) -> int:
        return a + b

    call = ToolCall(id="1", name="add", arguments={"a": 1, "b": 2})
    result = execute_tool(add, call)
    assert result.error is None
    assert result.content == "3"


def test_tool_serializes_non_json_output() -> None:
    @tool()
    def returns_set() -> set[int]:
        return {1, 2}

    call = ToolCall(id="1", name="returns_set", arguments={})
    result = execute_tool(returns_set, call)
    assert result.error is None
    assert result.content is not None
    message = tool_message(result, call)
    assert message.content


def test_tool_decorator_accepts_hooks() -> None:
    executed: list[str] = []

    def pre_hook(call: ToolCall) -> None:
        executed.append(f"pre:{call.id}")

    def post_hook(call: ToolCall, _result) -> None:
        executed.append(f"post:{call.id}")

    @tool(pre=(pre_hook,), post=(post_hook,))
    def add(a: int, b: int) -> int:
        return a + b

    call = ToolCall(id="hook-call", name="add", arguments={"a": 1, "b": 2})
    result = execute_tool(add, call)
    assert result.error is None
    assert result.content == "3"
    assert executed == ["pre:hook-call", "post:hook-call"]


@pytest.mark.asyncio
async def test_async_tool_execution() -> None:
    @tool()
    async def async_add(a: int, b: int) -> int:
        await asyncio.sleep(0.01)
        return a + b

    call = ToolCall(id="1", name="async_add", arguments={"a": 1, "b": 2})
    result = await aexecute_tool(async_add, call)
    assert result.error is None
    assert result.content == "3"


@pytest.mark.asyncio
async def test_async_tool_execution_with_sync_tool() -> None:
    @tool()
    def sync_add(a: int, b: int) -> int:
        return a + b

    call = ToolCall(id="1", name="sync_add", arguments={"a": 1, "b": 2})
    result = await aexecute_tool(sync_add, call)
    assert result.error is None
    assert result.content == "3"


@pytest.mark.asyncio
async def test_async_hooks_with_decorator() -> None:
    executed: list[str] = []

    async def pre_hook(call: ToolCall) -> None:
        executed.append(f"pre:{call.id}")

    async def post_hook(call: ToolCall, _result) -> None:
        executed.append(f"post:{call.id}")

    @tool(pre=(pre_hook,), post=(post_hook,))
    async def async_add(a: int, b: int) -> int:
        await asyncio.sleep(0)
        return a + b

    call = ToolCall(id="ahook-call", name="async_add", arguments={"a": 1, "b": 2})
    result = await aexecute_tool(async_add, call)
    assert result.error is None
    assert result.content == "3"
    assert executed == ["pre:ahook-call", "post:ahook-call"]


@pytest.mark.asyncio
async def test_run_coroutine_sync_from_running_loop() -> None:
    async def coro() -> str:
        await asyncio.sleep(0)
        return "ok"

    result = _run_coroutine_sync(coro())
    assert result == "ok"
