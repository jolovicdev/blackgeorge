# Quickstart

This guide gets you from zero to a running worker quickly.

## Install

```
uv add blackgeorge
```

```
pip install blackgeorge
```

For development setup, see `development.md`.

## Provider setup

Blackgeorge uses LiteLLM under the hood. Set the API key for the provider you plan to use before running.

```
export OPENAI_API_KEY="your-key"
```

If you use another provider, set its env vars instead:

```
export ANTHROPIC_API_KEY="your-key"
export DEEPSEEK_API_KEY="your-key"
export AZURE_API_KEY="your-key"
export AZURE_API_BASE="https://your-resource.openai.azure.com"
export AZURE_API_VERSION="2023-07-01-preview"
```

Model names follow LiteLLM provider prefixes, for example `openai/gpt-5-nano`.

## Your first run

```python
from blackgeorge import Desk, Job, Worker

desk = Desk(model="openai/gpt-5-nano")
worker = Worker(name="Researcher")
job = Job(input="Summarize this topic", expected_output="A short summary")

report = desk.run(worker, job)
print(report.content)
```

## Add a tool

```python
from blackgeorge import Desk, Job, Worker
from blackgeorge.tools import tool

@tool(description="Echo text back")
def echo(text: str) -> str:
    return text

desk = Desk(model="openai/gpt-5-nano")
worker = Worker(name="Agent", tools=[echo])
job = Job(input="Call echo with text=hello")

report = desk.run(worker, job)
print(report.content)
```

## Pause and resume

When a tool requires confirmation or user input, the run pauses and returns a pending action.

```python
from blackgeorge import Desk, Job, Worker
from blackgeorge.tools import tool

@tool(requires_confirmation=True)
def risky(action: str) -> str:
    return f"ok:{action}"

desk = Desk(model="openai/gpt-5-nano")
worker = Worker(name="Ops", tools=[risky])
job = Job(input="Run the risky action")

report = desk.run(worker, job)
if report.status == "paused" and report.pending_action is not None:
    report = desk.resume(report, True)

print(report.status)
print(report.content)
```

## Structured output

Set a response schema to get validated output. Blackgeorge first attempts LiteLLM structured output with JSON schema, then falls back to Instructor with Pydantic models.

```python
from pydantic import BaseModel

from blackgeorge import Desk, Job, Worker

class Summary(BaseModel):
    title: str
    bullets: list[str]

worker = Worker(name="Writer", model="openai/gpt-5-nano")
desk = Desk(model="openai/gpt-5-nano")
job = Job(input="Summarize the report", response_schema=Summary)

report = desk.run(worker, job)
print(report.data)
```

## Next steps

- Read Concepts to understand the mental model.
- Review Worker and Tools for tool safety and pause/resume flows.
- Use Workforce for multi-agent coordination.
- Use Workflow for multi-step pipelines.
