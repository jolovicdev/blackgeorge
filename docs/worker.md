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

If the model returns tool calls, the worker will execute tools or pause for confirmation/user input.

- requires_confirmation: the run pauses and waits for a boolean decision
- requires_user_input: the run pauses and waits for a string input

When a tool is paused, the worker returns a `Report` with `status="paused"` and a `PendingAction`.

When a model response includes multiple tool calls in the same turn, the worker executes them in
parallel and records tool results in the original call order. If a tool requires confirmation or
user input, the worker executes prior tool calls and then pauses before that tool.

## Structured output

Set `Job.response_schema` to a Pydantic model or TypeAdapter to enforce a structured response. Blackgeorge uses Instructor and returns the validated model in `Report.data`.

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

If tools are present, the worker may call tools first and then request structured output once the model stops emitting tool calls. Structured output uses the adapter's `structured_complete`/`astructured_complete` hooks when implemented, and falls back to the default LiteLLM + Instructor path otherwise.

## Streaming

Streaming only happens when all of the following are true:

- `Desk.stream` (or `desk.run(..., stream=True)`) is enabled
- the worker has no tools for the job
- no response schema is set

When streaming is enabled, the worker emits `stream.token` events.

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

Confirmation actions treat truthy values as acceptance. If you pass a falsy value for confirmation, the tool result will be an error with message "Tool execution declined".

User input actions insert the provided value into the tool call arguments under `user_input` unless the tool sets a different `input_key`.

## Limits and failure behavior

The worker stops and fails when:

- max_iterations is exceeded
- max_tool_calls is exceeded
- the model fails to satisfy a structured response after retries
- `Desk(respect_context_window=False)` and a context limit is hit

## Context window handling

When `Desk.respect_context_window` is True, the worker summarizes the conversation history and retries the model call if the provider reports a context length error. The summary preserves system instructions, key facts, tool results, and the most recent messages. If you are using a custom model, register its context window in LiteLLM for more reliable behavior.

Failures are returned as `Report` objects with status `failed` and error messages in `Report.errors`.

## Tool override

`Job.tools_override` replaces the worker tool list for a single run. Only `Tool` instances in the list are used; other items are ignored.
