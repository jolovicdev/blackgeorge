# Worker

A `Worker` is a single agent loop that talks to the model, optionally calls tools, and returns a `Report`.

## Create a worker

```python
from blackgeorge import Worker

worker = Worker(
    name="Analyst",
    model="openai/gpt-5-nano",
    instructions="You are a concise analyst.",
)
```

### Parameters

- name: worker name used in events and selection
- tools: list of tools this worker can use
- model: model name override for this worker
- instructions: optional system instructions
- memory_scope: a string namespace for external memory usage

## Job input and system message

The worker builds messages from the job.

- Job.input becomes the user message. Non-string values are JSON serialized.
- Job.expected_output and Job.constraints are appended to the system message.

## Tools and tool safety

If the model returns tool calls, the worker executes tools or pauses for a pending action.

- requires_confirmation: the run pauses and waits for an explicit approval decision
- requires_user_input: the run pauses and waits for a string input
- requires_handoff: the run pauses with `pending_action.type="handoff"` (typically intercepted by `Workforce` in swarm mode)

When a tool is paused, the worker returns a `Report` with `status="paused"` and a `PendingAction`.
Paused turns emit `worker.paused` events instead of `worker.completed`.

When a model response includes multiple tool calls in the same turn, the worker executes them in
parallel and records tool results in the original call order. If a tool requires confirmation or
user input, the worker executes prior tool calls and then pauses before that tool. Any remaining
tool calls after the pending one receive an error result so the conversation history stays valid
for strict providers.

## Structured output

Set `Job.response_schema` to a Pydantic model or TypeAdapter to enforce a structured response. Blackgeorge first attempts LiteLLM structured output using a JSON schema response format, then falls back to Instructor and Pydantic validation. The validated object is returned in `Report.data`. TypeAdapter outputs, including lists of models, are serialized into JSON arrays in `Report.content`.

```python
from pydantic import BaseModel

from blackgeorge import Job, Worker

class Result(BaseModel):
    title: str
    score: float

worker = Worker(name="Judge")
job = Job(input="Score this", response_schema=Result)
```

Structured output is used when:

- a response schema is set
- tools are not required for the current step

If tools are present, the worker may call tools first and then request structured output once the model stops emitting tool calls. Structured output uses the adapter's `structured_complete`/`astructured_complete` hooks when implemented, and falls back to the default LiteLLM JSON schema and Instructor pipeline otherwise.

## Streaming

Streaming only happens when all of the following are true:

- `Desk.stream` (or `desk.run(..., stream=True)`) is enabled
- no response schema is set
- or a response schema is set and `structured_stream_mode="preview"`

When streaming is enabled, the worker emits `stream.token` events. On tool turns, these tokens
are streamed tool argument deltas.

With `structured_stream_mode="preview"`, streamed tokens are preview output. The final `Report.data`
is still validated against `response_schema`. If preview JSON is invalid, Blackgeorge falls back to
strict structured completion before returning.

## Async usage

When you already have an event loop, run a single worker through a flow and await it.

```python
from blackgeorge import Desk, Job, Worker
from blackgeorge.workflow import Step

desk = Desk(model="openai/gpt-5-nano")
worker = Worker(name="Analyst")
flow = desk.flow([Step(worker)])

report = await flow.arun(Job(input="Analyze feedback"))
```

Use `flow.aresume` to continue paused runs in async applications.

## Pause and resume

When a worker pauses, resume the run using the report it returned.

```python
report = desk.run(worker, job)
if report.status == "paused":
    report = desk.resume(report, "your input")
```

Confirmation actions accept `True` or strings such as `yes`, `approve`, or `confirm`.
They decline `False`, `None`, blank strings, or strings such as `no`, `decline`, `deny`, or `cancel`.
Other non-empty strings still approve the tool for compatibility with earlier truthy resume values.
Declined confirmations produce a tool result error with message "Tool execution declined".
Declined confirmations emit a `tool.failed` event with that error.

In async applications, use `desk.arun()` and `desk.aresume()`.

User input actions insert the provided value into the tool call arguments under `user_input` unless the tool sets a different `input_key`.

## Limits and failure behavior

The worker stops and fails when:

- max_iterations is exceeded
- max_tool_calls is exceeded
- the model fails to satisfy a structured response after retries
- a context limit is hit and reactive context handling is disabled

## Context window handling

`Desk` supports two context compaction paths:

- Reactive: when `respect_context_window=True`, context-limit errors trigger summarization and retry.
- Proactive: when `max_context_messages` is set, the worker summarizes before model calls when message count exceeds the limit.

Proactive compaction does not consume the reactive retry budget used for context-limit recovery.
Compaction keeps recent tool-call/result groups together so providers that require strict tool
message ordering still receive valid history.
If you are using a custom model, register its context window in LiteLLM for more reliable behavior.

Failures are returned as `Report` objects with status `failed` and error messages in `Report.errors`.

## Tool override

`Job.tools_override` replaces the worker tool list for a single run.

- `Tool` instances are used directly.
- String entries are resolved by name from the worker toolbelt.
- Unknown or unsupported entries are ignored.
- If multiple override entries resolve to the same tool name, the last one is effective.
