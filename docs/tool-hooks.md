# Tool Hooks

Tool hooks allow you to execute custom logic before and after tool execution. Use hooks for logging, validation, policy checks, and monitoring.

## Hook types

- **Pre-hooks**: Execute before the tool callable runs. Receive the `ToolCall`.
- **Post-hooks**: Execute after the tool callable completes. Receive both the `ToolCall` and `ToolResult`.

## Pre-hooks

Pre-hooks receive the tool call before execution:

```python
from blackgeorge.tools import ToolPreHook, tool
from blackgeorge.core.tool_call import ToolCall

def log_call(call: ToolCall) -> None:
    print(f"Calling {call.name} with {call.arguments}")

@tool(pre=(log_call,))
def calculate(x: int, y: int) -> int:
    return x + y
```

### Pre-hook signature

```python
from typing import Any
from blackgeorge.core.tool_call import ToolCall

ToolPreHook = Callable[[ToolCall], Any]
```

A pre-hook can:
- Inspect the tool call
- Log metadata
- Validate arguments

### Pre-hook examples

#### Argument validation

```python
from blackgeorge.tools import tool

def validate_positive(call):
    for key, value in call.arguments.items():
        if isinstance(value, (int, float)) and value < 0:
            raise ValueError(f"{key} must be positive, got {value}")

@tool(pre=(validate_positive,))
def square_root(x: float) -> float:
    return x ** 0.5
```

#### Timing

```python
import time
from blackgeorge.tools import tool

timing_data = {}

def start_timer(call):
    timing_data[call.id] = time.time()

@tool(pre=(start_timer,))
def expensive_operation(n: int) -> int:
    return sum(range(n))
```

#### Access control

```python
from blackgeorge.tools import tool

ALLOWED_USERS = {"admin", "supervisor"}

def check_permission(call):
    user = call.arguments.get("requested_by", "anonymous")
    if user not in ALLOWED_USERS:
        raise PermissionError(f"User {user} not allowed to call {call.name}")

@tool(pre=(check_permission,), requires_confirmation=True)
def delete_file(file_path: str, requested_by: str) -> str:
    import os
    os.remove(file_path)
    return f"Deleted {file_path}"
```

## Post-hooks

Post-hooks receive the tool call and result after execution:

```python
from blackgeorge.tools import ToolPostHook, tool
from blackgeorge.tools.base import ToolResult

def log_result(call: ToolCall, result: ToolResult) -> None:
    status = "success" if not result.error else "failed"
    print(f"{call.name} {status}: {result.content}")

@tool(post=(log_result,))
def calculate(x: int, y: int) -> int:
    return x + y
```

### Post-hook signature

```python
from typing import Any
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.tools.base import ToolResult

ToolPostHook = Callable[[ToolCall, ToolResult], Any]
```

A post-hook can:
- Log results
- Add metrics
- Trigger side effects
- Inspect error conditions

### Post-hook examples

#### Result logging

```python
from blackgeorge.tools import tool

def log_completion(call, result):
    if result.timed_out:
        print(f"WARNING: {call.name} timed out")
    elif result.cancelled:
        print(f"WARNING: {call.name} was cancelled")
    elif result.error:
        print(f"ERROR in {call.name}: {result.error}")
    else:
        print(f"SUCCESS: {call.name} returned {result.content}")

@tool(
    timeout=5.0,
    post=(log_completion,),
)
def fetch_data(url: str) -> str:
    import requests
    return requests.get(url).text
```

#### Result enrichment (side effects)

```python
from datetime import datetime

from blackgeorge.tools import tool

result_audit = {}

def record_timestamp(call, result):
    if result.content:
        result_audit[call.id] = {
            "recorded_at": datetime.now().isoformat(),
            "content_preview": result.content[:80],
        }

@tool(post=(record_timestamp,))
def get_status() -> str:
    return "OK"
```

#### Metrics collection

```python
from collections import defaultdict
from blackgeorge.tools import tool

tool_metrics = defaultdict(lambda: {"calls": 0, "errors": 0, "total_time": 0})

def track_metrics(call, result):
    metrics = tool_metrics[call.name]
    metrics["calls"] += 1
    if result.error:
        metrics["errors"] += 1

@tool(post=(track_metrics,))
def process_data(data: str) -> str:
    return data.upper()
```

## Combining pre and post hooks

You can use both pre and post hooks on the same tool:

```python
from blackgeorge.tools import tool

def before(call):
    print(f"Starting {call.name}")

def after(call, result):
    print(f"Finished {call.name}: {result.content}")

@tool(pre=(before,), post=(after,))
def multiply(a: int, b: int) -> int:
    return a * b
```

## Async hooks

Async hooks are supported in async tool execution paths:

```python
import asyncio
from blackgeorge.tools import tool

async def async_pre_hook(call):
    await asyncio.sleep(0.1)
    print(f"Async pre-hook for {call.name}")

async def async_post_hook(call, result):
    await asyncio.sleep(0.1)
    print(f"Async post-hook for {call.name}")

@tool(pre=(async_pre_hook,), post=(async_post_hook,))
async def async_tool(value: str) -> str:
    await asyncio.sleep(0.1)
    return value.upper()
```

## Multiple hooks

You can chain multiple hooks:

```python
from blackgeorge.tools import tool

def log_start(call):
    print(f"Start: {call.name}")

def validate_args(call):
    if "x" not in call.arguments:
        raise ValueError("Missing 'x' argument")

def log_end(call, result):
    print(f"End: {call.name} -> {result.content}")

def check_error(call, result):
    if result.error:
        print(f"Error detected: {result.error}")

@tool(
    pre=(log_start, validate_args),
    post=(log_end, check_error),
)
def compute(x: int, y: int) -> int:
    return x + y
```

Hooks are executed in the order they are defined.

## Hook context and state

Hooks can share state through closures or class attributes:

```python
class ToolMonitor:
    def __init__(self):
        self.calls = []
        self.errors = []

    def pre_hook(self, call):
        self.calls.append(call)

    def post_hook(self, call, result):
        if result.error:
            self.errors.append((call, result.error))

monitor = ToolMonitor()

@tool(pre=(monitor.pre_hook,), post=(monitor.post_hook,))
def risky_operation(value: int) -> int:
    if value < 0:
        raise ValueError("Negative values not allowed")
    return value * 2
```

## Advanced patterns

#### Conditional hooks

```python
from blackgeorge.tools import tool

def conditional_pre(call):
    actor = call.arguments.get("requested_by", "unknown")
    if call.name.startswith("admin_"):
        print(f"Admin tool {call.name} called by {actor}")

@tool(pre=(conditional_pre,))
def admin_delete_user(user_id: str) -> str:
    return f"Deleted {user_id}"
```

#### Retry-aware hooks

```python
from blackgeorge.tools import tool

attempt_count = {}

def count_attempts(call):
    attempt_count[call.id] = attempt_count.get(call.id, 0) + 1
    print(f"Attempt {attempt_count[call.id]} for {call.name}")

@tool(retries=3, pre=(count_attempts,))
def flaky_operation(value: str) -> str:
    import random
    if random.random() < 0.5:
        raise Exception("Random failure")
    return value
```

#### Cache telemetry with hooks

```python
from blackgeorge.tools import tool

cache_stats = {"hits": 0, "misses": 0}
cache = set()

def cache_pre(call):
    key = f"{call.name}:{frozenset(call.arguments.items())}"
    if key in cache:
        cache_stats["hits"] += 1
    else:
        cache_stats["misses"] += 1

def cache_post(call, result):
    if result.error is None:
        key = f"{call.name}:{frozenset(call.arguments.items())}"
        cache.add(key)

@tool(pre=(cache_pre,), post=(cache_post,))
def expensive_computation(n: int) -> int:
    return sum(range(n))
```

## Tool result inspection

Post-hooks can inspect detailed result information:

```python
from blackgeorge.tools import tool, ToolResult

def inspect_result(call, result):
    print(f"Tool: {call.name}")
    print(f"Content: {result.content}")
    print(f"Data: {result.data}")
    print(f"Error: {result.error}")
    print(f"Timed out: {result.timed_out}")
    print(f"Cancelled: {result.cancelled}")

@tool(timeout=10.0, post=(inspect_result,))
async def long_running_task(seconds: int) -> str:
    import asyncio
    await asyncio.sleep(seconds)
    return f"Slept for {seconds} seconds"
```

## Error handling in hooks

Exceptions in pre-hooks prevent tool execution:

```python
from blackgeorge.tools import tool

def strict_validation(call):
    if call.arguments.get("value", 0) < 0:
        raise ValueError("Value must be non-negative")

@tool(pre=(strict_validation,))
def process(value: int) -> int:
    return value * 2

# This will fail with ValueError if value < 0
```

Exceptions in post-hooks propagate and fail tool execution:

```python
def failing_post_hook(call, result):
    raise Exception("Post-hook failure")

@tool(post=(failing_post_hook,))
def my_tool(value: str) -> str:
    return value
```

## Best practices

1. **Keep hooks fast**: Hooks run synchronously, so avoid blocking operations
2. **Use hooks for cross-cutting concerns**: Logging, metrics, validation
3. **Don't rely on hook return values**: Pre-hook returns are ignored
4. **Handle exceptions in post-hooks**: Unhandled post-hook errors fail tool execution
5. **Use closures for state**: Maintain hook-specific state with function closures or classes
6. **Chain hooks carefully**: Order matters when using multiple hooks
7. **Consider async for I/O**: Use async hooks for network or database operations
