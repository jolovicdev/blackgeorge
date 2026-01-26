import asyncio

import pytest

from blackgeorge.core.tool_call import ToolCall
from blackgeorge.tools import execute_tool, tool
from blackgeorge.tools.execution import aexecute_tool
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
