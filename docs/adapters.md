# Adapters

Adapters define how Blackgeorge talks to a model provider.

## BaseModelAdapter

`BaseModelAdapter` defines the interface for model calls.

- complete(...): synchronous completion
- acomplete(...): async completion
- structured_complete(...): structured output completion
- astructured_complete(...): async structured output completion

Both methods accept OpenAI-style message payloads and optional tool schemas.

## LiteLLMAdapter

`LiteLLMAdapter` is the default adapter. It uses LiteLLM to call models with OpenAI-compatible inputs.

Key behaviors:

- calls `litellm.completion(...)` for synchronous requests
- calls `litellm.acompletion(...)` for async requests
- passes messages, tools, tool_choice, temperature, max_tokens
- supports streaming when requested
- enables `parallel_tool_calls` when LiteLLM reports the model supports parallel function calling

Tool calls are parsed from the response and mapped into `ToolCall` objects.

## Instructor integration

Structured output uses Instructor with LiteLLM. Blackgeorge initializes Instructor clients with:

- `instructor.from_provider("litellm/<model>")`
- `instructor.from_provider("litellm/<model>", async_client=True)`

The worker calls `chat.completions.create(..., response_model=YourModel)` and returns the validated Pydantic object as `Report.data`.

## Adapter hooks for structured output

If your adapter implements `structured_complete`/`astructured_complete`, the worker will call those hooks for response-schema jobs. This lets you route structured output through non-LiteLLM providers or custom pipelines. If the hooks are not implemented, the worker falls back to the LiteLLM + Instructor path.

## Custom adapters

To use another provider, implement `BaseModelAdapter` and pass it to `Desk(adapter=...)`.
