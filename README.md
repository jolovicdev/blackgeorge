# Blackgeorge: Python Agent Framework for LLM Tool-Calling and Multi-Agent Orchestration

[![PyPI version](https://badge.fury.io/py/blackgeorge.svg)](https://pypi.org/project/blackgeorge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![DeepWiki docs](https://img.shields.io/badge/DeepWiki-docs-2F80ED)](https://deepwiki.com/jolovicdev/blackgeorge)
[![Docs](https://img.shields.io/badge/docs-site-2F80ED)](https://jolovicdev.github.io/blackgeorge/)

A code-first Python framework for building AI agents, tool-calling workflows, and multi-agent systems with explicit APIs, structured outputs, safe tool execution, and pause/resume flows.

Works with OpenAI, Anthropic, DeepSeek, Gemini, Mistral, Ollama, and 100+ other providers through [LiteLLM](https://github.com/BerriAI/litellm). Tools come from plain Python functions or [MCP](https://modelcontextprotocol.io) servers. Structured outputs are Pydantic models.

## What you can build

- tool-calling AI agents with validated inputs
- multi-agent teams that coordinate work
- agentic workflows with parallel and sequential steps
- LLM services with durable run state, events, and resume

## Core primitives

- **Desk**: orchestrates runs, events, and persistence
- **Worker**: single-agent execution with tools and memory
- **Workforce**: multi-worker coordination and management modes
- **Workflow**: step-based flows with parallel execution

## Features

- tool execution with confirmation, user input, timeouts, retries, and cancellation
- structured output support with Pydantic models
- event streaming and run store persistence
- collaboration primitives: channel messaging and blackboard state
- memory stores including vector memory with configurable chunking
- LiteLLM adapter for OpenAI-compatible model providers
- MCP tool integration for external tool providers

## Why Blackgeorge

Most teams start with a hand-written tool loop over the OpenAI or Anthropic SDK. It works until a tool needs human approval, a run has to survive a restart, or someone asks what a run cost. Blackgeorge is that loop with those parts built in: tool calls can pause for confirmation or user input, run state is stored and resumable from another process, every run reports token usage and cost with an optional budget, and structured outputs are validated with Pydantic and retried. The primitives stay small and explicit, so the execution flow reads like the code you would have written yourself.

### How it compares

| | Blackgeorge | LangGraph | CrewAI | AutoGen (AgentChat) |
|---|---|---|---|---|
| Orchestration model | `Desk`, `Worker`, `Workforce`, `Flow` | State graph of nodes and edges | Crews of agents and tasks, plus Flows | Agent teams such as round-robin group chat |
| Pause and resume | Tool-level confirmation and user-input pauses; run state persisted in a run store and resumable from another process | `interrupt()` inside a node with a checkpointer; the node re-runs from its start on resume | `@human_feedback` and `@persist` on Flows | `save_state()` / `load_state()` on agents and teams |
| Structured output | `Job(response_schema=Model)` with validation retries and provider fallbacks | LangChain `with_structured_output` | Pydantic `response_format` on agents | Pydantic `response_format` via model client arguments |
| Provider layer | LiteLLM | LangChain chat models | LiteLLM | Its own model clients (OpenAI, Azure, and others) |

Cells describe each project's documented defaults; all four can be extended beyond them.

## Use cases

- coding agents that edit files with confirmation and audit trails
- research and summarization agents with structured outputs
- support triage and routing across multiple workers
- operational workflows that pause for approvals and resume safely

See `examples/coding_agent` for a full end-to-end example.

## Install

```
uv add blackgeorge
```

Vector memory is optional because ChromaDB adds a substantial dependency tree:

```
uv add "blackgeorge[vector]"
```

For development setup, see `docs/development.md`.

## Quick start

A worker with one tool and a Pydantic output schema:

```python
from pydantic import BaseModel

from blackgeorge import Desk, Job, Worker
from blackgeorge.tools import tool


class Summary(BaseModel):
    title: str
    bullets: list[str]


@tool()
def fetch_notes(topic: str) -> str:
    return f"Key points about {topic}: tool calling, structured output, pause and resume."


desk = Desk(model="openai/gpt-5-nano")
worker = Worker(name="Researcher", tools=[fetch_notes])
job = Job(input="Summarize agent frameworks using fetch_notes", response_schema=Summary)

report = desk.run(worker, job)
print(report.data.title, report.data.bullets)
print(report.metrics["cost_usd"])
```

## Documentation

- Full documentation: [jolovicdev.github.io/blackgeorge](https://jolovicdev.github.io/blackgeorge/)
- Searchable code walkthrough generated from the repository: [DeepWiki](https://deepwiki.com/jolovicdev/blackgeorge)
- Source for the docs lives in `docs/`; preview locally with `uv run mkdocs serve`

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

## Session: multi-turn conversations

```python
from blackgeorge import Desk, Worker

desk = Desk(model="openai/gpt-5-nano")
worker = Worker(name="ChatBot")

session = desk.session(worker)

session.run("My name is Alice")
session.run("What's my name?")

session_id = session.session_id

later_session = desk.session(worker, session_id=session_id)
later_session.run("Where do I live?")
```
