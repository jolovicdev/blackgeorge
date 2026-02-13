# Core types

This page documents the primary data types exposed by Blackgeorge.

## Job

A `Job` is the input to a worker or workforce.

Fields:

- id: unique job id
- input: payload passed to the worker as the user message
- expected_output: appended to the system message
- tools_override: optional list of tools to use for this job only
- response_schema: Pydantic model or TypeAdapter for structured output
- constraints: extra constraints appended to the system message
- metadata: arbitrary metadata for your application
- initial_messages: optional list of messages to pre-populate conversation history
- thinking: enables thinking mode for reasoning models (format varies by provider)
  - DeepSeek: `{"type": "enabled"}`
  - Anthropic: `{"type": "enabled", "budget_tokens": 1024}` - example.
- structured_stream_mode: optional `"off"` or `"preview"` override for schema-job streaming behavior
- drop_params: drops unsupported parameters instead of erroring
- extra_body: provider-specific parameters passed to the model API

`Job` is immutable.

## Report

A `Report` is the result of a run.

Fields:

- run_id
- status: completed, paused, failed, or running
- pending_action: present when the run is paused
- content: assistant output
- reasoning_content: separate reasoning output from thinking models (DeepSeek Reasoner, Claude 3.7, o1, etc.)
- data: structured output if a response schema is set
- messages: full conversation history
- tool_calls: tool calls made during the run
- metrics: usage and other metrics
- events: list of events for the run
- errors: list of error messages

## Message

A `Message` represents a single chat message.

Fields:

- role: system, user, assistant, tool
- content
- reasoning_content: separate reasoning content from thinking models
- thinking_blocks: optional structured reasoning blocks from providers like Anthropic
- tool_calls: tool calls emitted by the assistant
- tool_call_id: set for tool results
- metadata

## ToolCall

A `ToolCall` represents a single tool invocation request.

Fields:

- id
- name
- arguments
- result
- error

`result` is serialized for BaseModel and dataclass values.

## PendingAction

A `PendingAction` is returned when a run pauses.

Fields:

- action_id
- type: confirmation or user_input
- tool_call
- prompt
- options
- metadata

## Event

An `Event` describes a runtime signal.

Fields:

- event_id
- type
- timestamp
- run_id
- source
- payload

## RunState and RunRecord

- `RunState` contains the full resume state for a paused run.
- `RunRecord` is the stored run metadata in the run store.

## Enums and aliases

- MessageRole: system, user, assistant, tool
- PendingActionType: confirmation, user_input
- RunStatus: completed, paused, failed, running
- WorkforceMode: managed, collaborate
- Brief: alias for Job
- RunOutput: alias for Report

## Utilities

- new_id(): creates a random hex identifier
- utc_now(): returns the current UTC time
