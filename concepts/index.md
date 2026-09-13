# Concepts

This page describes the mental model behind Blackgeorge.

## Primitives

Blackgeorge has three core primitives.

- Desk: the runtime that owns configuration, storage, and events.
- Worker: a single agent that talks to a model and can call tools.
- Workforce: a group of workers coordinated by a simple strategy.

There is also a workflow layer for multi-step pipelines.

## The run lifecycle

A run starts when you call `desk.run(...)`. The desk creates a run record, emits a `run.started` event, and hands control to a worker or workforce.

A run ends in one of three states.

- completed: the model finished the task
- failed: the model or tool flow failed
- paused: the run waits for confirmation or user input

## Pause and resume

When a tool requires confirmation or user input, the worker pauses the run and returns a `PendingAction`. You call `desk.resume(report, decision_or_input)` to continue the run using the stored state.

## Structured output

If you set `Job.response_schema`, Blackgeorge first attempts structured output via LiteLLM `response_format` JSON schema. If that fails, it falls back to Instructor and Pydantic validation. The validated object is returned as `Report.data`.

## Tools

Tools are normal Python callables with type hints. The `@tool` decorator builds a schema and validation model from those hints. Tools can be safe by default and still require explicit confirmation.

## Events

All activity is emitted as events through the `EventBus`. You can subscribe to events to stream tokens, show progress, or persist audit logs.

## Storage

The run store persists run status, state, and events. By default, Blackgeorge uses a SQLite-backed run store in `.blackgeorge/blackgeorge.db`.
