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

When resuming, the provided input is inserted into the tool arguments under `user_input` by default.

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

## Execution path

The worker executes tools using `execute_tool`:

- validate input with the tool input model
- call the function
- convert output to `ToolResult`
- run post hooks

If validation or execution fails, the error is captured in the tool result and the run continues.
