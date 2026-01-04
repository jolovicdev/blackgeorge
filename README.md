# Blackgeorge

Blackgeorge is a code-first agentic framework built around the Desk, Worker, and Workforce primitives. It focuses on clear APIs, structured outputs, tool safety, and pause/resume flows.

## Docs

See `docs/README.md` for the full documentation set.
Preview locally with `uv run mkdocs serve`.

## Install

```
uv add blackgeorge
```

```
pip install blackgeorge
```

For development setup, see `docs/development.md`.

## Basic usage

```python
from blackgeorge import Desk, Worker, Job

desk = Desk(model="openai/gpt-5-nano")
worker = Worker(name="Researcher")
job = Job(input="Summarize this topic", expected_output="A short summary")

report = desk.run(worker, job)
print(report.content)
```

## Job input

`Job.input` is the payload sent to the worker as the user message. If it is not a string, it is serialized to JSON. Use a string for simple requests, or a structured dict when you want explicit fields.

```python
job = Job(
    input={
        "task": "Fix calculator behavior and update tests.",
        "context": "Use tools to inspect the project files.",
        "requirements": [
            "Confirm divide-by-zero behavior with the user.",
            "Confirm empty-average behavior with the user.",
            "Apply changes using tools.",
        ],
    },
    expected_output="Updated project files with consistent behavior.",
)
```

## Workforce

```python
from blackgeorge import Desk, Worker, Workforce, Job

desk = Desk(model="openai/gpt-5-nano")
w1 = Worker(name="Researcher")
w2 = Worker(name="Writer")
workforce = Workforce([w1, w2], mode="managed")

job = Job(input="Create a market report")
report = desk.run(workforce, job)
```

## Workflow

```python
from blackgeorge import Desk, Worker, Job
from blackgeorge.workflow import Step, Parallel

desk = Desk(model="openai/gpt-5-nano")
analyst = Worker(name="Analyst")
writer = Worker(name="Writer")

flow = desk.flow([
    Step(analyst),
    Parallel(Step(writer), Step(analyst)),
])

job = Job(input="Analyze product feedback")
report = flow.run(job)
```

## Streaming

```python
report = desk.run(worker, job, stream=True)
```

## Pause and resume

```python
from blackgeorge import Desk, Worker, Job
from blackgeorge.tools import tool

@tool(requires_confirmation=True)
def risky_action(action: str) -> str:
    return f"ran:{action}"

desk = Desk(model="openai/gpt-5-nano")
worker = Worker(name="Ops", tools=[risky_action])
job = Job(input="run risky")

report = desk.run(worker, job)
if report.status == "paused":
    report = desk.resume(report, True)
```
