# Blackgeorge Docs

Blackgeorge is a Python LLM agent framework built around three primitives: Desk, Worker, and Workforce. It provides clear APIs for tool calling, multi-agent coordination, workflow orchestration, structured outputs, and run persistence.

If you want a fast start, read the quickstart first. If you want to understand how the system works, read Concepts and then the component guides.

## What this documentation covers

- building tool-using agents and multi-agent teams
- orchestrating workflows with steps and parallel execution
- handling pause/resume flows and persisted run state
- integrating memory, events, LiteLLM adapters, and external tool providers

## Start here

- [Quickstart](quickstart.md): install and run your first worker
- [Concepts](concepts.md): the mental model and core primitives

## Component guides

- [Desk](desk.md): orchestration, configuration, run/resume
- [Worker](worker.md): single-agent execution loop
- [Workforce](workforce.md): multi-worker coordination
- [Workflow](workflow.md): multi-step flows
- [Session](session.md): multi-turn conversations with persistence
- [Tools](tools.md): tool definition, validation, and execution
- [Events and streaming](events.md): event bus and runtime signals
- [Storage](storage.md): run store and run state
- [Memory](memory.md): memory stores and scopes
- [Adapters](adapters.md): model adapters and structured output integration
- [Testing](testing.md): deterministic agent tests with ScriptedAdapter
- [Core types](core-types.md): Job, Report, Message, ToolCall, PendingAction

## Examples and development

- [Examples](examples.md): the coding agent walkthrough
- [Development](development.md): tests, lint, and type checking
