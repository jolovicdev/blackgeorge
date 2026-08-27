# Desk

`Desk` is the orchestration layer. It owns the model adapter, event bus, run store, and memory store. You run workers, workforces, or flows through the desk.

## Creating a desk

```python
from blackgeorge import Desk

desk = Desk(
    model="openai/gpt-5-nano",
    temperature=0.2,
    max_tokens=800,
    stream=False,
    structured_stream_mode="off",
    structured_output_retries=3,
    max_iterations=10,
    max_tool_calls=20,
    num_retries=0,
    respect_context_window=True,
    max_context_messages=10,
    max_cost_usd=0.50,
)
```

### Key parameters

- model: default model name for workers without their own model
- temperature: model temperature
- max_tokens: max tokens for completion requests
- stream: enables streaming when the worker is eligible
- structured_stream_mode: "off" (strict structured output) or "preview" (stream preview tokens for schema jobs)
- structured_output_retries: retries for structured output validation
- max_iterations: max model turns per worker run
- max_tool_calls: max tool calls per worker run
- num_retries: LiteLLM-level retry count for failed calls (0 disables retries)
- respect_context_window: when True, auto-summarize and retry on context length errors (Reactive)
- max_context_messages: auto-summarize when message count exceeds this limit (Proactive)
- max_cost_usd: cost budget for the whole run in USD (summed across workers and flow steps); the run fails before another model turn once the accumulated cost in the run totals exceeds it
- event_bus: custom event bus implementation
- run_store: custom run store implementation
- memory_store: custom memory store implementation
- adapter: custom model adapter
- storage_dir: directory for the default SQLite run store

## Memory integration

The desk applies simple read/write memory behavior for workers:

- Before a worker run, it reads `context` from the store using `worker.memory_scope` and inserts it as a system message after any existing leading system messages.
- After a completed run, it writes `last_output` (structured data or content) using the same scope.

This is intentionally minimal so you can build your own memory workflows on top.

## Context window handling

When `respect_context_window` is enabled, workers summarize conversation history on context length errors and retry the call with the summary plus the most recent messages (Reactive).
When `max_context_messages` is configured, workers summarize conversation proactively when the number of messages exceeds the limit to maintain a healthy context window (Proactive).

If you disable `respect_context_window`, reactive retries on context-limit errors are disabled.
If you also do not provide `max_context_messages`, runs fail directly on context-limit errors.
For custom or unmapped models, register model context limits in LiteLLM to avoid repeated overflows.

## Running a worker

```python
from blackgeorge import Desk, Job, Worker

desk = Desk(model="openai/gpt-5-nano")
worker = Worker(name="Researcher")
job = Job(input="Summarize this topic")

report = desk.run(worker, job)
print(report.status)
```

### Async running

Use `arun()` for async execution:

```python
report = await desk.arun(worker, job)
print(report.status)
```

If you already have a running event loop, call `arun()`/`aresume()`. The sync `run()`/`resume()`
raise in that case.

Both `run()` and `arun()` accept a `stream` parameter:

```python
report = desk.run(worker, job, stream=True)
report = await desk.arun(worker, job, stream=True)
```

When `stream=True`, `report.events` contains `stream.token` events with incremental deltas for
eligible turns. On tool turns, `stream.token` contains streamed tool argument deltas. Structured
schema turns remain strict by default, unless
`structured_stream_mode="preview"` is enabled.

## Running a workforce

```python
from blackgeorge import Desk, Job, Worker, Workforce

w1 = Worker(name="Researcher")
w2 = Worker(name="Writer")
workforce = Workforce([w1, w2], mode="managed")

desk = Desk(model="openai/gpt-5-nano")
job = Job(input="Create a market report")
report = desk.run(workforce, job)
```

## Running a flow

```python
from blackgeorge import Desk, Job, Worker
from blackgeorge.workflow import Step

desk = Desk(model="openai/gpt-5-nano")
worker = Worker(name="Analyst")
flow = desk.flow([Step(worker)])

report = flow.run(Job(input="Analyze feedback"))
```

## Resume a paused run

```python
report = desk.run(worker, job)
if report.status == "paused" and report.pending_action is not None:
    report = desk.resume(report, True)
```

For async applications, use `arun()` and `aresume()`.

## Creating sessions

Sessions manage multi-turn conversations with automatic persistence:

```python
session = desk.session(worker)
report = session.run("Hello")
report = session.run("What's my name?")
```

Resume an existing session:

```python
session = desk.session(worker, session_id="user-123")
if session:
    report = session.run("Continue")
```

See [session.md](session.md) for full documentation.

## Cleanup registries

If you create many temporary workers or workforces, unregister them when you are done.

```python
desk.unregister_worker(worker)
desk.unregister_workforce(workforce)
```

## Events

The desk emits run and component events. Subscribe to `desk.event_bus` to observe them.

```python
from blackgeorge import Desk

def handle_event(event) -> None:
    print(event.type, event.source)

desk = Desk(model="openai/gpt-5-nano")
desk.event_bus.subscribe("run.started", handle_event)
```
