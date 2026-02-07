# Events and streaming

Blackgeorge emits events for run lifecycle, workers, tools, workflows, and adapter calls.

## EventBus API

### Creating an event bus

```python
from blackgeorge.event_bus import EventBus

bus = EventBus()
```

### Subscribing to events

```python
def handle_event(event) -> None:
    print(f"{event.type}: {event.payload}")

bus.subscribe("run.started", handle_event)
bus.subscribe("run.completed", handle_event)
bus.subscribe("run.failed", handle_event)
```

Event handlers receive an `Event` with these fields:

- `event_id`: unique event id
- `type`: event name, for example `run.started`
- `timestamp`: UTC timestamp
- `run_id`: run identifier
- `source`: emitter name (for example worker/tool/workforce name)
- `payload`: event-specific data

`EventBus.subscribe` matches exact event types only. Wildcard subscriptions like `*` are not supported.

### Emitting events

```python
from blackgeorge.core.event import Event
from blackgeorge.utils import new_id, utc_now

event = Event(
    event_id=new_id(),
    type="custom.event",
    timestamp=utc_now(),
    run_id="run-123",
    source="my_component",
    payload={"data": "value"},
)
bus.emit(event)
```

`emit` runs handlers in-process. If a handler is async (or returns an awaitable), it is scheduled on the current loop when available, or run with a temporary loop from sync contexts.

```python
await bus.aemit(event)
```

`aemit` awaits async handlers and awaitable returns from sync handlers.

### Unsubscribing

EventBus has no built-in unsubscribe API. Use a wrapper with an internal enabled flag when you need dynamic opt-out.

## Event types

### Run events

| Event Type | Description | Payload Fields |
|------------|-------------|----------------|
| `run.started` | Run started | `job_id` |
| `run.paused` | Run paused | none |
| `run.resumed` | Run resumed | none |
| `run.completed` | Run completed | none |
| `run.failed` | Run failed | `errors` when emitted by `Desk`; may be empty for flow-level failures |

### Worker events

| Event Type | Description | Payload Fields |
|------------|-------------|----------------|
| `worker.started` | Worker iteration started/resumed | none |
| `worker.paused` | Worker paused for pending action | `pending_action_type` |
| `worker.completed` | Worker completed successfully | none |
| `worker.failed` | Worker failed | `error` |
| `worker.context_summarized` | Context summary applied | `model`, `summarized_messages`, `kept_messages`, optional `unregistered_model`, optional `registration_hint` |

### Workforce events

| Event Type | Description | Payload Fields |
|------------|-------------|----------------|
| `workforce.started` | Workforce run started | none |
| `workforce.completed` | Workforce run finished | none |

### Workflow step events

| Event Type | Description | Payload Fields |
|------------|-------------|----------------|
| `step.started` | Step execution started | none |
| `step.completed` | Step execution finished | `status` |
| `step.paused` | Step paused | `status` |

### Tool events

| Event Type | Description | Payload Fields |
|------------|-------------|----------------|
| `tool.started` | Tool execution started | `tool_call_id` |
| `tool.completed` | Tool execution completed | `tool_call_id`, optional `result_preview`, optional `result_truncated`, optional `timed_out`, optional `cancelled` |
| `tool.failed` | Tool execution failed | `tool_call_id`, `error` |
| `tool.confirmation_requested` | Tool needs confirmation | `tool_call_id` |
| `tool.user_input_requested` | Tool needs user input | `tool_call_id` |

Tool/workforce/worker names are exposed via `event.source`.

### LLM adapter events

| Event Type | Description | Payload Fields |
|------------|-------------|----------------|
| `llm.started` | LLM call started | `model`, `messages_count`, `tools_count` |
| `llm.completed` | LLM call completed or stream closed | `model`, `latency_ms`, optional `prompt_tokens`, optional `completion_tokens`, optional `total_tokens`, optional `cost` |
| `llm.failed` | LLM call failed | `model`, `latency_ms`, `error_type`, `error_message` |

### Streaming/message events

| Event Type | Description | Payload Fields |
|------------|-------------|----------------|
| `stream.token` | Stream token delta | `token` |
| `assistant.message` | Assistant message appended | `content`, optional `tool_calls` |

## Subscribing from a desk

```python
from blackgeorge import Desk

def on_tool_completed(event) -> None:
    preview = event.payload.get("result_preview")
    print(f"{event.source} completed: {preview}")

desk = Desk(model="openai/gpt-5-nano")
desk.event_bus.subscribe("tool.completed", on_tool_completed)
```

## Filtering patterns

### Filter by prefixes

```python
def handle_tool_events(event) -> None:
    print(f"{event.type} from {event.source}")

for event_type in ("tool.started", "tool.completed", "tool.failed"):
    bus.subscribe(event_type, handle_tool_events)
```

### Filter by source

```python
def handle_analyst_events(event) -> None:
    if event.source == "analyst":
        print(event.type, event.payload)

bus.subscribe("worker.started", handle_analyst_events)
bus.subscribe("worker.failed", handle_analyst_events)
```

### Filter with wrappers

```python
def create_filter(handler, event_types):
    def filtered(event):
        if event.type in event_types:
            handler(event)
    return filtered

filtered = create_filter(print, {"run.started", "run.completed"})
bus.subscribe("run.started", filtered)
bus.subscribe("run.completed", filtered)
```

## Async handlers

```python
async def async_handler(event):
    await some_async_operation(event)

await bus.aemit(event)
```

## Streaming events

`stream.token` emits only when all are true:

- streaming enabled on desk/run
- no tools for that turn
- no response schema for that turn

```python
def on_token(event):
    print(event.payload.get("token", ""), end="", flush=True)

desk.event_bus.subscribe("stream.token", on_token)
```

## Tool result previews

`tool.completed` may include `result_preview` and `result_truncated` for lightweight logging.

```python
def on_tool_completed(event):
    print(event.source, event.payload.get("result_preview"))
```

## Context summary events

```python
def on_context_summarized(event):
    payload = event.payload
    print(payload["summarized_messages"], payload["kept_messages"])
    if payload.get("unregistered_model"):
        print(payload.get("registration_hint"))
```

## Event storage

Events emitted through a desk are persisted in the run store.

```python
events = desk.run_store.get_events(run_id)
for event in events:
    print(event.type, event.payload)
```

## Custom events

```python
from blackgeorge import Desk, Job, Worker
from blackgeorge.core.event import Event
from blackgeorge.utils import new_id, utc_now

desk = Desk(model="openai/gpt-5-nano")
worker = Worker(name="assistant")
report = desk.run(worker, Job(input="hello"))

custom_event = Event(
    event_id=new_id(),
    type="custom.progress",
    timestamp=utc_now(),
    run_id=report.run_id,
    source="my_tool",
    payload={"percent": 50},
)
desk.event_bus.emit(custom_event)
```

## Performance notes

- Handlers run on the emitting path, so keep them fast.
- Offload heavy work to queues/threads/processes.
- Register subscriptions before starting concurrent run execution.

```python
import queue

event_queue = queue.Queue()

def queue_handler(event):
    event_queue.put(event)

bus.subscribe("run.completed", queue_handler)
```

## Event-driven patterns

### Progress tracking

```python
class ProgressTracker:
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.completed_steps = 0

    def on_step_complete(self, _event):
        self.completed_steps += 1
        percent = (self.completed_steps / self.total_steps) * 100
        print(f"{percent:.1f}%")

tracker = ProgressTracker(total_steps=5)
desk.event_bus.subscribe("step.completed", tracker.on_step_complete)
```

### Error aggregation

```python
class ErrorLogger:
    def __init__(self):
        self.errors = []

    def on_error(self, event):
        error = event.payload.get("error")
        if error is None:
            errors = event.payload.get("errors")
            if isinstance(errors, list) and errors:
                error = "; ".join(errors)
        self.errors.append((event.type, event.source, error))

logger = ErrorLogger()
desk.event_bus.subscribe("run.failed", logger.on_error)
desk.event_bus.subscribe("worker.failed", logger.on_error)
desk.event_bus.subscribe("tool.failed", logger.on_error)
```
