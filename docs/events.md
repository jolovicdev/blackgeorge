# Events and streaming

Blackgeorge emits events for all major lifecycle steps. Subscribe to the event bus to react to changes in real time.

## Event bus

```python
from blackgeorge import Desk

def handle_event(event) -> None:
    print(event.type, event.source)

desk = Desk(model="openai/gpt-5-nano")
desk.event_bus.subscribe("run.started", handle_event)
```

`EventBus` also provides `aemit` for async handler execution, but the desk uses the synchronous `emit` path.

## Event payloads

Each event includes:

- event_id
- type
- timestamp
- run_id
- source
- payload

## Common event types

Run events:

- run.started
- run.paused
- run.resumed
- run.completed
- run.failed

Worker events:

- worker.started
- worker.completed
- worker.failed
- worker.context_summarized

Workforce events:

- workforce.started
- workforce.completed

Step events:

- step.started
- step.completed
- step.paused
- step.failed

Tool events:

- tool.started
- tool.completed
- tool.failed
- tool.confirmation_requested
- tool.user_input_requested

`tool.completed` payloads include `tool_call_id` and may include `result_preview`,
`result_truncated`, `timed_out`, and `cancelled` when available.

Streaming and message events:

- stream.token
- assistant.message

## Streaming

Streaming emits `stream.token` for each token and `assistant.message` for full assistant messages. Streaming only occurs when the worker has no tools for the job and no response schema is set.

## Storage

Events emitted through the desk are also stored in the run store so they can be retrieved later.
