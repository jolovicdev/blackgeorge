# Tools

Tools are regular Python functions with type hints. Blackgeorge turns them into validated, schema-backed actions the model can call.

## Tool metadata

A tool includes:

- name
- description
- schema
- callable
- input_model
- requires_confirmation
- requires_user_input
- external_execution
- pre and post hooks
- confirmation_prompt
- user_input_prompt
- input_key
- timeout
- retries
- retry_delay

`external_execution` is available for your own conventions. The core worker does not change behavior based on this flag.

## Define a tool

```python
from blackgeorge.tools import tool

@tool(description="Add two numbers")
def add(a: int, b: int) -> int:
    return a + b
```

The decorator builds:

- a Pydantic input model for validation
- a JSON schema sent to the model

## Tool safety

Tools can require confirmation or user input.

```python
from blackgeorge.tools import tool

@tool(requires_confirmation=True)
def delete_record(record_id: str) -> str:
    return f"deleted:{record_id}"
```

When the model requests this tool, the worker pauses and returns a pending action. Resume the run with a boolean decision.

## User input tools

```python
from blackgeorge.tools import tool

@tool(requires_user_input=True)
def ask(question: str, user_input: str) -> str:
    return user_input
```

When resuming, the provided input is inserted into the tool arguments under `user_input` by default. If you want a different argument name, set `input_key` on the tool.

```python
from blackgeorge.tools import tool

@tool(requires_user_input=True, input_key="answer")
def ask(question: str, answer: str) -> str:
    return answer
```

`input_key` is optional. If you omit it, `user_input` is used.

## Timeouts and retries

Tools can specify timeout and retry behavior for resilient execution.
Retries use exponential backoff based on `retry_delay`.

```python
from blackgeorge.tools import tool

@tool(timeout=5.0, retries=3, retry_delay=1.0)
async def fetch_data(url: str) -> str:
    ...
```

The `ToolResult` includes `timed_out` and `cancelled` flags to detect failure modes.

## Cancellation

Async tool execution supports cancellation via an event:

```python
import asyncio
from blackgeorge.tools.execution import aexecute_tool

cancel_event = asyncio.Event()
result = await aexecute_tool(tool, call, cancel_event=cancel_event)
cancel_event.set()
```

## ToolResult

Tools can return a `ToolResult` to control content, data, and error fields directly.

```python
from blackgeorge.tools import ToolResult, tool

@tool()
def fetch_status(code: int) -> ToolResult:
    if code == 200:
        return ToolResult(content="ok", data={"code": code})
    return ToolResult(error="not ok")
```

## Hooks

Each tool can define pre and post hooks. Pre hooks receive the `ToolCall`. Post hooks receive the `ToolCall` and the `ToolResult`.

## Toolbelt

`Toolbelt` manages tool registration and lookup. `Toolkit` is an alias for `Toolbelt`.

```python
from blackgeorge.tools import Toolbelt

belt = Toolbelt()
```

## MCP Tool Integration

Connect to MCP (Model Context Protocol) servers and use their tools.
Use `connect_sse` when pointing at an SSE endpoint.

```python
from blackgeorge.tools import MCPToolProvider

async with MCPToolProvider() as provider:
    await provider.connect_stdio("uv", ["run", "my-mcp-server"])
    tools = provider.list_tools()
    result = await provider.acall_tool("fetch", {"url": "https://example.com"})
```

MCP tools are automatically converted to the blackgeorge `Tool` format and can be passed to workers.

## Execution path

The worker executes tools using `execute_tool`:

- validate input with the tool input model
- call the function
- convert output to `ToolResult`
- run post hooks

If validation or execution fails, the error is captured in the tool result and the run continues.
