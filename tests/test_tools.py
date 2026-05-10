import asyncio

import pytest

from blackgeorge.core.tool_call import ToolCall
from blackgeorge.tools import execute_tool, tool, transfer_to_agent_tool
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


def test_sync_pre_hook_failure_returns_tool_result() -> None:
    def pre_hook(call: ToolCall) -> None:
        raise PermissionError("not allowed")

    @tool(pre=(pre_hook,))
    def add(a: int, b: int) -> int:
        return a + b

    call = ToolCall(id="hook-fail", name="add", arguments={"a": 1, "b": 2})
    result = execute_tool(add, call)

    assert result.error is not None
    assert "not allowed" in result.error
    assert result.exception_type == "ToolExecutionError"


def test_sync_validation_error_survives_post_hook_failure() -> None:
    def post_hook(_call: ToolCall, _result) -> None:
        raise RuntimeError("post hook failed")

    @tool(post=(post_hook,))
    def add(a: int) -> int:
        return a

    call = ToolCall(id="validation-post-hook", name="add", arguments={"a": "bad"})
    result = execute_tool(add, call)

    assert result.error is not None
    assert "validation failed" in result.error
    assert "Post-hook error" in result.error
    assert "post hook failed" in result.error
    assert result.exception_type == "ToolValidationError"


def test_sync_success_result_survives_post_hook_failure() -> None:
    def post_hook(_call: ToolCall, _result) -> None:
        raise RuntimeError("post hook failed")

    @tool(post=(post_hook,))
    def add(a: int, b: int) -> int:
        return a + b

    call = ToolCall(id="success-post-hook", name="add", arguments={"a": 1, "b": 2})
    result = execute_tool(add, call)

    assert result.error is not None
    assert "post hook failed" in result.error
    assert result.content == "3"
    assert result.data == 3
    assert result.exception_type == "ToolExecutionError"


def test_transfer_to_agent_tool_snapshots_allowlist_for_runtime_validation() -> None:
    available_agents = ["alpha"]
    handoff_tool = transfer_to_agent_tool(available_agents)
    available_agents.append("beta")

    call = ToolCall(
        id="handoff-1",
        name="transfer_to_agent",
        arguments={"agent_name": "beta", "context": "route"},
    )
    result = execute_tool(handoff_tool, call)

    assert result.error is not None
    assert "not available" in result.error.lower()
    properties = handoff_tool.schema.get("properties")
    assert isinstance(properties, dict)
    agent_name_schema = properties.get("agent_name")
    assert isinstance(agent_name_schema, dict)
    assert agent_name_schema.get("enum") == ["alpha"]


def test_sync_tool_execution_supports_async_hooks() -> None:
    executed: list[str] = []

    async def pre_hook(call: ToolCall) -> None:
        executed.append(f"pre:{call.id}")

    async def post_hook(call: ToolCall, _result) -> None:
        executed.append(f"post:{call.id}")

    @tool(pre=(pre_hook,), post=(post_hook,))
    def add(a: int, b: int) -> int:
        return a + b

    call = ToolCall(id="sync-async-hook", name="add", arguments={"a": 2, "b": 3})
    result = execute_tool(add, call)
    assert result.error is None
    assert result.content == "5"
    assert executed == ["pre:sync-async-hook", "post:sync-async-hook"]


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
async def test_async_pre_hook_failure_returns_tool_result() -> None:
    async def pre_hook(call: ToolCall) -> None:
        raise PermissionError("not allowed")

    @tool(pre=(pre_hook,))
    async def async_add(a: int, b: int) -> int:
        return a + b

    call = ToolCall(id="async-hook-fail", name="async_add", arguments={"a": 1, "b": 2})
    result = await aexecute_tool(async_add, call)

    assert result.error is not None
    assert "not allowed" in result.error
    assert result.exception_type == "ToolExecutionError"


@pytest.mark.asyncio
async def test_async_validation_error_survives_post_hook_failure() -> None:
    async def post_hook(_call: ToolCall, _result) -> None:
        raise RuntimeError("post hook failed")

    @tool(post=(post_hook,))
    async def async_add(a: int) -> int:
        return a

    call = ToolCall(id="async-validation-post-hook", name="async_add", arguments={"a": "bad"})
    result = await aexecute_tool(async_add, call)

    assert result.error is not None
    assert "validation failed" in result.error
    assert "Post-hook error" in result.error
    assert "post hook failed" in result.error
    assert result.exception_type == "ToolValidationError"


@pytest.mark.asyncio
async def test_async_success_result_survives_post_hook_failure() -> None:
    async def post_hook(_call: ToolCall, _result) -> None:
        raise RuntimeError("post hook failed")

    @tool(post=(post_hook,))
    async def async_add(a: int, b: int) -> int:
        return a + b

    call = ToolCall(id="async-success-post-hook", name="async_add", arguments={"a": 1, "b": 2})
    result = await aexecute_tool(async_add, call)

    assert result.error is not None
    assert "post hook failed" in result.error
    assert result.content == "3"
    assert result.data == 3
    assert result.exception_type == "ToolExecutionError"


@pytest.mark.asyncio
async def test_run_coroutine_sync_from_running_loop() -> None:
    async def coro() -> str:
        await asyncio.sleep(0)
        return "ok"

    result = _run_coroutine_sync(coro())
    assert result == "ok"
