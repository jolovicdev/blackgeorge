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
- output_type (optional)

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

`ToolResult` fields:

- `content`: String content returned to the model
- `data`: Structured data (preserved when using `output_type`)
- `error`: Error message if execution failed
- `timed_out`: Whether the tool timed out
- `cancelled`: Whether the tool was cancelled
- `original_exception`: The original exception if an error occurred

## Output Type Validation

Tools can declare an `output_type` for automatic validation of return values:

```python
from pydantic import BaseModel
from blackgeorge.tools import tool

class AnalysisResult(BaseModel):
    findings: list[str]
    confidence: float

@tool(output_type=AnalysisResult)
def analyze(text: str) -> dict:
    return {
        "findings": ["insight 1", "insight 2"],
        "confidence": 0.85,
    }
```

When `output_type` is set:
- Return values are validated against the Pydantic model
- Invalid outputs produce a `ToolResult` with `error` set
- Valid outputs are available as structured `data`

This enables type-safe tool outputs and automatic schema validation.

## Hooks

Each tool can define pre and post hooks. Pre hooks receive the `ToolCall`. Post hooks receive the `ToolCall` and the `ToolResult`.

## Toolbelt

`Toolbelt` manages tool registration and lookup. `Toolkit` is an alias for `Toolbelt`.

```python
from blackgeorge.tools import Toolbelt

belt = Toolbelt()
```

## Swarm Handoff Tool

For `Workforce` running in `swarm` mode, workers can dynamically transfer control and pass context using the `transfer_to_agent_tool`.

```python
from blackgeorge.tools import transfer_to_agent_tool

handoff_tool = transfer_to_agent_tool(available_agents=["researcher", "coder"])
worker = Worker(name="router", tools=[handoff_tool])
```

When this tool is executed, it signals a `handoff` pending action that the orchestrator uses to transparently switch the active worker mid-run.

`available_agents` is encoded into the tool schema and enforced by swarm routing. A handoff target
must be both:

- in the handoff tool allowlist
- present in the workforce worker list

The same enforcement applies when the handoff tool is provided through `Job.tools_override`.

## Subworker tools

Use `create_subworker_tool` to let a worker delegate to bounded child workers with explicit
guardrails and run-level persistence.

```python
from blackgeorge.tools import create_subworker_tool

spawn_subworker = create_subworker_tool(
    desk=desk,
    max_subworkers=5,
    max_tools_per_subworker=8,
)
```

This tool returns `ToolResult` with a structured payload that includes child `run_id` and `status`.
Child runs are executed through `Desk`, so they use the same adapter, run store, event bus, and
memory integration as top-level runs.

For full API and behavior details, see `subworkers.md`.

## MCP Tool Integration

Connect to MCP (Model Context Protocol) servers and use their tools. MCP tools are automatically converted to the blackgeorge `Tool` format and can be passed to workers.

### stdio transport

For local MCP servers that run as subprocesses.

```python
from blackgeorge.tools import MCPToolProvider

async with MCPToolProvider() as provider:
    await provider.connect_stdio("uv", ["run", "my-mcp-server"])
    tools = provider.list_tools()
    result = await provider.acall_tool("fetch", {"url": "https://example.com"})
```

### Streamable HTTP transport

For remote MCP servers over HTTP.

```python
from blackgeorge.tools import MCPToolProvider

async with MCPToolProvider() as provider:
    await provider.connect_streamable_http("https://api.example.com/mcp")
    tools = provider.list_tools()
    result = await provider.acall_tool("search", {"query": "python"})
```

For servers requiring authentication, pass a custom `httpx.AsyncClient`:

```python
import httpx
from blackgeorge.tools import MCPToolProvider

async with MCPToolProvider() as provider:
    client = httpx.AsyncClient(headers={"Authorization": "Bearer token"})
    await provider.connect_streamable_http("https://api.example.com/mcp", http_client=client)
    tools = provider.list_tools()
```

### SSE transport

For MCP servers that use Server-Sent Events.

```python
from blackgeorge.tools import MCPToolProvider

async with MCPToolProvider() as provider:
    await provider.connect_sse("https://api.example.com/mcp/sse")
    tools = provider.list_tools()
```

## Execution path

The worker executes tools using `execute_tool`:

- validate input with the tool input model
- call the function
- convert output to `ToolResult`
- run post hooks

If validation or execution fails, the error is captured in the tool result and the run continues.
When multiple tool calls are returned in the same model response, the worker executes them in
parallel and then appends tool results in call order.
