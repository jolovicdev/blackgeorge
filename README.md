# Blackgeorge: Python Agent Framework for LLM Tool-Calling and Multi-Agent Orchestration

[![PyPI version](https://badge.fury.io/py/blackgeorge.svg)](https://pypi.org/project/blackgeorge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A code-first Python framework for building AI agents, tool-calling workflows, and multi-agent systems with explicit APIs, structured outputs, safe tool execution, and pause/resume flows.

## What you can build with this Python AI agent framework

- tool-calling AI agents with validated inputs
- multi-agent teams that coordinate work
- agentic workflows with parallel and sequential steps
- LLM services with durable run state, events, and resume

## Core primitives for agent orchestration

- **Desk**: orchestrates runs, events, and persistence
- **Worker**: single-agent execution with tools and memory
- **Workforce**: multi-worker coordination and management modes
- **Workflow**: step-based flows with parallel execution

## Feature highlights for tool-calling and multi-agent workflows

- tool execution with confirmation, user input, timeouts, retries, and cancellation
- structured output support with Pydantic models
- event streaming and run store persistence
- collaboration primitives: channel messaging and blackboard state
- memory stores including vector memory with configurable chunking
- LiteLLM adapter for OpenAI-compatible model providers
- MCP tool integration for external tool providers

## Why Blackgeorge

If you want a LangChain alternative that stays close to the metal, Blackgeorge emphasizes small, explicit primitives and clear execution flow. Compared to CrewAI or AutoGen, it keeps orchestration and tool calling predictable while still supporting multi-agent systems, workflows, and OpenAI-compatible function calling through LiteLLM.

## Use cases and examples

- coding agents that edit files with confirmation and audit trails
- research and summarization agents with structured outputs
- support triage and routing across multiple workers
- operational workflows that pause for approvals and resume safely

See `examples/coding_agent` for a full end-to-end example.

## Install

```
uv add blackgeorge
```

For development setup, see `docs/development.md`.

## Quick Start: build your first AI agent

```python
from blackgeorge import Desk, Worker, Job

desk = Desk(model="openai/gpt-5-nano")
worker = Worker(name="Researcher")
job = Job(input="Summarize this topic", expected_output="A short summary")

report = desk.run(worker, job)
print(report.content)
```

## Documentation

See `docs/README.md` for the full documentation set.
Preview locally with `uv run mkdocs serve`.

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
