# Exceptions

Blackgeorge provides a custom exception hierarchy for structured error handling.

## Exception Hierarchy

```text
BlackgeorgeError (base)
├── ContextLimitError
├── ToolExecutionError
├── ToolValidationError
├── ToolTimeoutError
├── EventHandlerError
├── RunnerNotRegisteredError
└── StreamingUnsupportedError
```

## Base Exception

::: blackgeorge.exceptions.BlackgeorgeError

## Context Errors

::: blackgeorge.exceptions.ContextLimitError

Raised when the LLM context window is exceeded. Provides context about whether the model was registered and whether summarization was attempted.

## Tool Errors

::: blackgeorge.exceptions.ToolExecutionError

Raised when a tool fails during execution. Includes the original exception for debugging.

::: blackgeorge.exceptions.ToolValidationError

Raised when tool arguments fail validation.

::: blackgeorge.exceptions.ToolTimeoutError

Raised when a tool execution exceeds its timeout.

## Event Errors

::: blackgeorge.exceptions.EventHandlerError

Raised when an event handler fails. Wraps the original exception.

## Usage Example

```python
from blackgeorge import BlackgeorgeError, ContextLimitError, ToolExecutionError

try:
    report = await desk.arun(worker, job)
except ContextLimitError as e:
    print(f"Context limit: {e}")
    print(f"Model registered: {e.model_registered}")
    print(f"Summary attempted: {e.summary_attempted}")
except ToolExecutionError as e:
    print(f"Tool {e.tool_name} failed: {e}")
    if e.original_exception:
        raise e.original_exception
except BlackgeorgeError as e:
    print(f"Framework error: {e}")
```

## ToolResult Exception Access

Tool execution errors are captured in `ToolResult` with the exception type preserved:

```python
from blackgeorge import Worker, Desk, Job
from blackgeorge.tools import tool

@tool()
def risky_operation(data: str) -> str:
    raise ValueError("Something went wrong")

worker = Worker(name="worker", tools=[risky_operation])
desk = Desk(model="openai/gpt-5-nano")
report = desk.run(worker, Job(input="Process this"))

# Check tool results in report
for tool_call in report.tool_calls:
    if tool_call.result and tool_call.result.error:
        print(f"Error: {tool_call.result.error}")
        if tool_call.result.exception_type:
            print(f"Exception type: {tool_call.result.exception_type}")
```

## Checking Context Limit Errors

The `is_context_limit_error` helper checks both string patterns and typed exceptions:

```python
from blackgeorge.worker_context import is_context_limit_error

try:
    response = await adapter.acomplete(...)
except Exception as e:
    if is_context_limit_error(e):
        # Handle context limit - retry with smaller context
        ...
    raise
```
