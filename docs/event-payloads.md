# Event Payloads

Blackgeorge provides typed dataclasses for event payloads, replacing untyped `dict[str, Any]` with structured data.

## Available Payload Types

### Run Events

```python
from blackgeorge import RunStartedPayload, RunFailedPayload

# run.started
payload = RunStartedPayload(job_id="job-123")

# run.failed
payload = RunFailedPayload(errors=["Error 1", "Error 2"])
```

### Worker Events

```python
from blackgeorge import (
    WorkerPausedPayload,
    WorkerFailedPayload,
    WorkerContextSummarizedPayload,
)

# worker.paused
payload = WorkerPausedPayload(pending_action_type="confirmation")

# worker.failed
payload = WorkerFailedPayload(error="Something went wrong")

# worker.context_summarized
payload = WorkerContextSummarizedPayload(
    model="gpt-4",
    summarized_messages=10,
    kept_messages=4,
    unregistered_model=False,
    registration_hint=None,
)
```

### Tool Events

```python
from blackgeorge import ToolStartedPayload, ToolCompletedPayload, ToolFailedPayload

# tool.started
payload = ToolStartedPayload(tool_call_id="call-123")

# tool.completed
payload = ToolCompletedPayload(
    tool_call_id="call-123",
    result_preview="Success...",
    result_truncated=False,
    timed_out=False,
    cancelled=False,
)

# tool.failed
payload = ToolFailedPayload(
    tool_call_id="call-123",
    error="Execution failed",
)
```

### Streaming Events

```python
from blackgeorge import StreamTokenPayload, AssistantMessagePayload

# stream.token
payload = StreamTokenPayload(token="Hello", type="content")

# tool argument delta
payload = StreamTokenPayload(token='{"path":', type="tool_argument")

# assistant.message
payload = AssistantMessagePayload(
    content="Response text",
    tool_calls=[{"id": "call-1", "name": "tool", "arguments": "{}"}],
)
```

### LLM Events

```python
from blackgeorge import LLMCompletedPayload, LLMFailedPayload

# llm.completed
payload = LLMCompletedPayload(
    model="gpt-4",
    latency_ms=150,
    prompt_tokens=100,
    completion_tokens=50,
    total_tokens=150,
    cost=0.001,
)

# llm.failed
payload = LLMFailedPayload(
    model="gpt-4",
    latency_ms=100,
    error_type="RateLimitError",
    error_message="Rate limit exceeded",
)
```

### Step Events

```python
from blackgeorge import StepCompletedPayload, StepPausedPayload

# step.completed
payload = StepCompletedPayload(status="completed")

# step.paused
payload = StepPausedPayload(status="paused")
```

### Workforce Events

```python
from blackgeorge import WorkforcePausedPayload

# workforce paused state
payload = WorkforcePausedPayload(
    root_job={"input": "task"},
    completed_reports=[],
    pending_worker_index=0,
)
```

## Usage

These payload types are for documentation and type hints. Event handlers receive `Event` objects with `payload: dict[str, Any]`:

```python
from blackgeorge import Desk, EventType

def handle_tool_completed(event) -> None:
    # Access payload fields (matches ToolCompletedPayload structure)
    tool_call_id = event.payload.get("tool_call_id")
    result_preview = event.payload.get("result_preview")
    print(f"Tool {tool_call_id}: {result_preview}")

desk = Desk(model="openai/gpt-5-nano")
desk.event_bus.subscribe(EventType.TOOL_COMPLETED, handle_tool_completed)
```

## All Payload Types

| Payload Class | Event Type | Key Fields |
|--------------|------------|------------|
| `RunStartedPayload` | `run.started` | `job_id` |
| `RunFailedPayload` | `run.failed` | `errors` |
| `WorkerPausedPayload` | `worker.paused` | `pending_action_type` |
| `WorkerFailedPayload` | `worker.failed` | `error` |
| `WorkerContextSummarizedPayload` | `worker.context_summarized` | `model`, `summarized_messages`, `kept_messages` |
| `ToolStartedPayload` | `tool.started` | `tool_call_id` |
| `ToolCompletedPayload` | `tool.completed` | `tool_call_id`, `result_preview`, `result_truncated`, `timed_out`, `cancelled` |
| `ToolFailedPayload` | `tool.failed` | `tool_call_id`, `error` |
| `StreamTokenPayload` | `stream.token` | `token`, `type` (`"content"` or `"tool_argument"`)
| `AssistantMessagePayload` | `assistant.message` | `content`, `tool_calls` |
| `LLMCompletedPayload` | `llm.completed` | `model`, `latency_ms`, `total_tokens`, `cost` |
| `LLMFailedPayload` | `llm.failed` | `model`, `latency_ms`, `error_type`, `error_message` |
| `StepCompletedPayload` | `step.completed` | `status` |
| `StepPausedPayload` | `step.paused` | `status` |
| `WorkforcePausedPayload` | workforce state | `root_job`, `completed_reports`, `pending_worker_index` |
