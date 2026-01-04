# Adapters

Adapters define how Blackgeorge talks to a model provider.

## BaseModelAdapter

`BaseModelAdapter` defines the interface for model calls.

- complete(...): synchronous completion
- acomplete(...): async completion

Both methods accept OpenAI-style message payloads and optional tool schemas.

## LiteLLMAdapter

`LiteLLMAdapter` is the default adapter. It uses LiteLLM to call models with OpenAI-compatible inputs.

Key behaviors:

- calls `litellm.completion(...)` for synchronous requests
- calls `litellm.acompletion(...)` for async requests
- passes messages, tools, tool_choice, temperature, max_tokens
- supports streaming when requested

Tool calls are parsed from the response and mapped into `ToolCall` objects.

## Instructor integration

Structured output uses Instructor with LiteLLM. Blackgeorge initializes Instructor clients with:

- `instructor.from_provider("litellm/<model>")`
- `instructor.from_provider("litellm/<model>", async_client=True)`

The worker calls `chat.completions.create(..., response_model=YourModel)` and returns the validated Pydantic object as `Report.data`.

## Custom adapters

To use another provider, implement `BaseModelAdapter` and pass it to `Desk(adapter=...)`.
