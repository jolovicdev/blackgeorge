# Testing

Blackgeorge ships `ScriptedAdapter` so you can test agents deterministically without calling a
model provider or spending tokens. It implements the full `BaseModelAdapter` interface: it replays
a queue of scripted responses and records every call it receives.

```python
from blackgeorge import Desk, Job, ScriptedAdapter, Worker
from blackgeorge.adapters.base import ModelResponse
from blackgeorge.store.in_memory import InMemoryRunStore

adapter = ScriptedAdapter([
    ModelResponse(content="hello", tool_calls=[], usage={}, raw={}),
])
desk = Desk(model="fake", adapter=adapter, run_store=InMemoryRunStore())

report = desk.run(Worker(name="Assistant"), Job(input="hi"))

assert report.status == "completed"
assert report.content == "hello"
```

Pass `InMemoryRunStore` (and no `storage_dir`) so tests stay hermetic and write nothing to disk.

## Scripting tool calls

Script a `ModelResponse` with `ToolCall`s to exercise tool execution, then script the follow-up
answer for the next model turn:

```python
from blackgeorge.adapters.base import ModelResponse
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.tools import tool

@tool()
def echo(text: str) -> str:
    return text.upper()

adapter = ScriptedAdapter([
    ModelResponse(
        content=None,
        tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "hi"})],
        usage={},
        raw={},
    ),
    ModelResponse(content="done", tool_calls=[], usage={}, raw={}),
])

report = desk.run(Worker(name="Assistant", tools=[echo]), Job(input="go"))
assert report.tool_calls[0].result.content == "HI"
```

Each model turn consumes one scripted response, in order. If the agent makes more calls than you
scripted, `ScriptedAdapter` raises `RuntimeError` so the test fails loudly instead of hanging or
silently passing.

## Structured output

`structured_complete` validates the scripted response content against the job's `response_schema`,
so structured output paths are covered too:

```python
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str

adapter = ScriptedAdapter([
    ModelResponse(content='{"answer": "ok"}', tool_calls=[], usage={}, raw={}),
])

report = desk.run(Worker(name="Assistant"), Job(input="hi", response_schema=Answer))
assert report.data.answer == "ok"
```

Invalid JSON or schema mismatches fail the run, mirroring provider behavior.

## Asserting on calls

Every call is recorded in `adapter.calls` with its kind (`"complete"` or `"structured"`), the model,
and the exact message payload sent:

```python
assert adapter.calls[0]["kind"] == "complete"
assert adapter.calls[0]["model"] == "fake"
assert adapter.calls[0]["messages"][-1]["content"] == "hi"
```

Use this to assert on prompts, tool schemas, iteration counts, and retry behavior.

## Usage and cost metrics

Script `usage` to test token and cost accounting, including `max_cost_usd` budgets:

```python
adapter = ScriptedAdapter([
    ModelResponse(
        content="done",
        tool_calls=[],
        usage={"prompt_tokens": 1000, "completion_tokens": 500},
        raw={},
    ),
])
```

Cost is priced from LiteLLM model metadata, so use a registered model name (for example
`deepseek/deepseek-v4-flash`) on the desk when testing budgets.

## Limitations

- Streaming is not simulated: with `stream=True` the scripted `ModelResponse` is returned directly
  and no `stream.token` events are emitted. Token-level streaming tests should use a custom adapter
  that yields chunks.
- Pause/resume works, but script enough responses for the turns that happen after resuming.

## Eval harness

`evaluate` runs a list of cases against a worker or workforce and reports pass/fail per case. A
case asserts substrings in the report content and/or a custom check; runs that do not complete fail
the case.

```python
from blackgeorge import Desk, EvalCase, Job, Worker, evaluate

desk = Desk(model="openai/gpt-5-nano")
worker = Worker(name="Assistant")

results = evaluate(
    desk,
    worker,
    [
        EvalCase(name="math", job=Job(input="What is 2+2?"), contains=("4",)),
        EvalCase(
            name="length",
            job=Job(input="One sentence about rain."),
            check=lambda report: len(report.content or "") < 200,
        ),
    ],
)

for result in results:
    print(result.name, result.passed, result.failures)
```

Each `EvalResult` carries the full `Report`, so failures can be inspected with the CLI or the run
store. Use `aevaluate` inside an existing event loop.

### Judge scoring

Pass a `judge` worker and a `rubric` to score answers with an LLM. The judge sees the rubric, the
task input, and the answer, and returns a structured score from 0.0 to 1.0. A case fails when its
score is below `min_score` (default 0.7):

```python
results = evaluate(
    desk,
    worker,
    cases,
    judge=Worker(name="Judge", instructions="Score answers strictly by the rubric."),
    rubric="1.0 if the answer is correct and concise, 0.0 otherwise",
    min_score=0.8,
)
```

Judging only runs for completed reports. Judge calls are ordinary runs, so each gets its own
`max_cost_usd` budget and its own entry in the run store.
