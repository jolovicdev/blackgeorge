# Events and streaming

Blackgeorge emits events for all major lifecycle steps. Subscribe to the event bus to react to changes in real time.

## EventBus API

The `EventBus` manages event subscriptions and emission.

### Creating an event bus

```python
from blackgeorge.event_bus import EventBus

bus = EventBus()
```

### Subscribing to events

```python
def handle_event(event) -> None:
    print(f"{event.type}: {event.payload}")

# Subscribe to a specific event type
bus.subscribe("run.started", handle_event)

# Subscribe to multiple event types
bus.subscribe("run.completed", handle_event)
bus.subscribe("run.failed", handle_event)
```

The handler function receives an `Event` object with the following fields:

- `event_id` (str): Unique event identifier
- `type` (str): Event type (e.g., "run.started")
- `timestamp` (datetime): When the event occurred
- `run_id` (str): Associated run identifier
- `source` (str): Event source (e.g., worker name)
- `payload` (dict): Event-specific data

### Synchronous event emission

```python
from blackgeorge.core.event import Event

event = Event(
    type="custom.event",
    source="my_component",
    run_id="run-123",
    payload={"data": "value"}
)
bus.emit(event)
```

`emit` supports async handlers as well. If a handler is a coroutine function or returns an awaitable (including a Task or Future), it is scheduled on the current event loop when available, or run to completion with a temporary loop when called from a sync context.

### Asynchronous event emission

```python
# For async handlers, use aemit
await bus.aemit(event)
```

`aemit` automatically handles both sync and async handlers:
- Sync handlers are called directly
- Async handlers are awaited
- Sync handlers that return awaitables are also awaited

### Unsubscribing

The EventBus does not provide a built-in unsubscribe method. To stop receiving events, maintain a reference to your handler and wrap it with conditional logic:

```python
class EventHandler:
    def __init__(self):
        self.enabled = True

    def handle(self, event):
        if not self.enabled:
            return
        print(event)

handler = EventHandler()
bus.subscribe("run.started", handler.handle)

# Later, to stop handling events
handler.enabled = False
```

## Event types

### Run events

Emitted during the lifecycle of a run.

| Event Type | Description | Payload Fields |
|------------|-------------|----------------|
| `run.started` | Run has started | `runner_type`, `runner_name` |
| `run.paused` | Run is paused pending action | `pending_action_type` |
| `run.resumed` | Run has been resumed | `pending_action_type` |
| `run.completed` | Run completed successfully | `status`, `metrics` |
| `run.failed` | Run failed with error | `error`, `errors` |

### Worker events

Emitted during worker execution.

| Event Type | Description | Payload Fields |
|------------|-------------|----------------|
| `worker.started` | Worker started processing | `worker_name` |
| `worker.completed` | Worker completed successfully | `worker_name`, `iterations` |
| `worker.failed` | Worker failed | `worker_name`, `error` |
| `worker.context_summarized` | Conversation context was summarized | `model`, `summarized_messages`, `kept_messages` |

### Workforce events

Emitted during workforce execution.

| Event Type | Description | Payload Fields |
|------------|-------------|----------------|
| `workforce.started` | Workforce started | `workforce_name`, `mode` |
| `workforce.completed` | Workforce completed | `workforce_name`, `worker_count` |

### Step events

Emitted during workflow step execution.

| Event Type | Description | Payload Fields |
|------------|-------------|----------------|
| `step.started` | Step started | `step_index`, `step_type` |
| `step.completed` | Step completed | `step_index`, `step_type` |
| `step.paused` | Step paused | `step_index`, `pending_action` |
| `step.failed` | Step failed | `step_index`, `error` |

### Tool events

Emitted during tool execution.

| Event Type | Description | Payload Fields |
|------------|-------------|----------------|
| `tool.started` | Tool execution started | `tool_name`, `tool_call_id` |
| `tool.completed` | Tool completed successfully | `tool_name`, `tool_call_id`, `result_preview`, `timed_out`, `cancelled` |
| `tool.failed` | Tool execution failed | `tool_name`, `tool_call_id`, `error` |
| `tool.confirmation_requested` | Tool requires confirmation | `tool_name`, `prompt` |
| `tool.user_input_requested` | Tool requires user input | `tool_name`, `prompt` |

### Streaming and message events

| Event Type | Description | Payload Fields |
|------------|-------------|----------------|
| `stream.token` | Token emitted during streaming | `token` |
| `assistant.message` | Full assistant message | `content` |

## Subscribing from the desk

The most common way to subscribe to events is through the desk:

```python
from blackgeorge import Desk

def on_run_started(event):
    print(f"Run {event.run_id} started")

def on_tool_completed(event):
    tool_name = event.payload.get("tool_name")
    print(f"Tool {tool_name} completed")

desk = Desk(model="openai/gpt-5-nano")
desk.event_bus.subscribe("run.started", on_run_started)
desk.event_bus.subscribe("tool.completed", on_tool_completed)
```

## Event filtering patterns

### Filter by event type

```python
def handle_all_events(event):
    if event.type.startswith("tool."):
        print(f"Tool event: {event.type}")
        # Process tool events

bus.subscribe("*", handle_all_events)
```

### Filter by payload content

```python
def handle_worker_events(event):
    if "worker_name" in event.payload:
        worker = event.payload["worker_name"]
        if worker == "analyst":
            print(f"Analyst event: {event.type}")

bus.subscribe("worker.started", handle_worker_events)
```

### Filter with a wrapper

```python
def create_filter(handler, event_types):
    def filtered(event):
        if event.type in event_types:
            handler(event)
    return filtered

def my_handler(event):
    print(event)

filtered_handler = create_filter(my_handler, {"run.started", "run.completed"})
bus.subscribe("run.started", filtered_handler)
bus.subscribe("run.completed", filtered_handler)
```

## Async event handlers

You can use async handlers with the event bus:

```python
async def async_handler(event):
    await some_async_operation(event)
    print(f"Processed {event.type}")

# Use aemit for async handlers
await bus.aemit(event)
```

## Streaming events

Streaming emits `stream.token` for each token during response generation. Streaming only occurs when:

- The worker has no tools for the job
- No response schema is set
- `Desk.stream` is enabled or `desk.run(..., stream=True)` is used

```python
def on_token(event):
    token = event.payload.get("token")
    print(token, end="", flush=True)

desk = Desk(model="openai/gpt-5-nano", stream=True)
desk.event_bus.subscribe("stream.token", on_token)

report = desk.run(worker, job)
```

## Tool result previews

Tool completion events include a `result_preview` field that contains a truncated version of the tool result. This is useful for logging without dumping full payloads.

```python
def on_tool_completed(event):
    tool_name = event.payload.get("tool_name")
    preview = event.payload.get("result_preview", "")
    truncated = event.payload.get("result_truncated", False)

    print(f"{tool_name} completed")
    print(f"Preview: {preview}")
    if truncated:
        print("(Result truncated)")
```

## Context summary events

When the worker summarizes conversation history due to context limits, it emits a `worker.context_summarized` event:

```python
def on_context_summarized(event):
    payload = event.payload
    print(f"Summarized {payload['summarized_messages']} messages")
    print(f"Kept {payload['kept_messages']} recent messages")

    if payload.get("unregistered_model"):
        print("Warning: Model not registered in LiteLLM")
        print(payload.get("registration_hint"))
```

## Event storage

Events emitted through the desk are automatically stored in the run store:

```python
# Retrieve events for a run
events = desk.run_store.get_events(run_id)

for event in events:
    print(f"{event.type}: {event.payload}")
```

## Custom events

You can emit custom events from your tools or handlers:

```python
from blackgeorge.core.event import Event
from blackgeorge import Desk

desk = Desk(model="openai/gpt-5-nano")

# Emit a custom event
custom_event = Event(
    type="custom.progress",
    source="my_tool",
    run_id=desk.run_store.get_runs()[-1].run_id,
    payload={"percent": 50, "status": "processing"}
)
desk.event_bus.emit(custom_event)
```

## Performance considerations

- Event handlers are called synchronously on the emitting thread. Keep handlers fast to avoid blocking the run.
- For expensive operations, offload work to a separate thread or queue:
  ```python
  import queue

  event_queue = queue.Queue()

  def queue_handler(event):
      event_queue.put(event)

  bus.subscribe("run.completed", queue_handler)

  # Process events in a separate thread
  def process_events():
      while True:
          event = event_queue.get()
          # Expensive processing here
  ```
- The EventBus is thread-safe for concurrent subscription and emission.
- Avoid blocking operations in async handlers used with `aemit`.

## Event-driven patterns

### Progress tracking

```python
class ProgressTracker:
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.completed_steps = 0

    def on_step_complete(self, event):
        self.completed_steps += 1
        percent = (self.completed_steps / self.total_steps) * 100
        print(f"Progress: {percent:.1f}%")

tracker = ProgressTracker(total_steps=5)
desk.event_bus.subscribe("step.completed", tracker.on_step_complete)
```

### Metrics collection

```python
class MetricsCollector:
    def __init__(self):
        self.metrics = {}

    def on_run_complete(self, event):
        run_id = event.run_id
        self.metrics[run_id] = event.payload.get("metrics", {})

collector = MetricsCollector()
desk.event_bus.subscribe("run.completed", collector.on_run_complete)
```

### Error aggregation

```python
class ErrorLogger:
    def __init__(self):
        self.errors = []

    def on_error(self, event):
        error = event.payload.get("error")
        source = event.source
        self.errors.append({
            "event": event.type,
            "source": source,
            "error": error,
            "timestamp": event.timestamp
        })

logger = ErrorLogger()
desk.event_bus.subscribe("run.failed", logger.on_error)
desk.event_bus.subscribe("worker.failed", logger.on_error)
desk.event_bus.subscribe("tool.failed", logger.on_error)
```
