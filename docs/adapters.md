# Adapters

Adapters define how Blackgeorge talks to a model provider.

## BaseModelAdapter

`BaseModelAdapter` defines the interface for model calls.

- complete(...): synchronous completion
- acomplete(...): async completion
- structured_complete(...): structured output completion
- astructured_complete(...): async structured output completion

Both methods accept OpenAI-style message payloads and optional tool schemas.
Blackgeorge uses the async adapter methods internally for both sync and async runs, so custom
adapters should implement `acomplete` and `astructured_complete`.

## LiteLLMAdapter

`LiteLLMAdapter` is the default adapter. It uses LiteLLM to call models with OpenAI-compatible inputs.

Key behaviors:

- calls `litellm.completion(...)` for synchronous requests
- calls `litellm.acompletion(...)` for async requests
- passes messages and optional model parameters (`temperature`, `max_tokens`, `thinking`, `extra_body`)
- only sends `tools` and `tool_choice` when tools are present
- supports streaming when requested
- when `complete`/`acomplete` receive a `response_schema`, requests a `json_schema` response format
  for it and retries with `json_object` plus a schema prompt if the provider rejects `json_schema`
- enables `parallel_tool_calls` when model metadata indicates `supports_parallel_function_calling`
- for streaming calls, emits `llm.completed` on stream exhaustion/close and `llm.failed` if stream iteration raises

### Runtime lifecycle hardening

Blackgeorge configures LiteLLM runtime lifecycle once when `LiteLLMAdapter` is initialized:

- it applies deterministic shutdown cleanup for async LiteLLM clients
- it patches LiteLLM logging-worker enqueue behavior to close dropped coroutines safely
- it avoids registering LiteLLM lazy async cleanup in a way that can emit shutdown warnings

This hardening was added to prevent process-exit warnings observed in real integrations, including DeepSeek (`deepseek/deepseek-chat`):

- `RuntimeWarning: coroutine 'close_litellm_async_clients' was never awaited`
- `RuntimeWarning: coroutine 'Logging.async_success_handler' was never awaited`

Tool calls are parsed from the response and mapped into `ToolCall` objects. Malformed tool payloads are preserved with `ToolCall.error` metadata instead of crashing adapter parsing.

## Structured output pipeline

Structured output uses LiteLLM JSON schema response formats when possible and falls back to Instructor with LiteLLM. Blackgeorge initializes Instructor clients with:

- `instructor.from_provider("litellm/<model>")`
- `instructor.from_provider("litellm/<model>", async_client=True)`

If the LiteLLM structured response fails or is unavailable, the worker calls `chat.completions.create(..., response_model=YourModel)` and returns the validated Pydantic object as `Report.data`.
`structured_output_retries` sets how many extra attempts follow a failed validation. With
`retries=0` a single attempt is made. Provider errors other than an unsupported response format
are raised immediately and never retried.

## Adapter hooks for structured output

If your adapter implements `structured_complete`/`astructured_complete`, the worker will call those hooks for response-schema jobs. This lets you route structured output through non-LiteLLM providers or custom pipelines. If the hooks are not implemented, the worker falls back to the LiteLLM + Instructor path.

The worker also passes the run's `temperature`, `max_tokens`, and `num_retries` plus the job's
`thinking`, `drop_params`, and `extra_body` as keyword arguments. Declare the ones you need in your
hook signature (or accept `**kwargs`); the worker only sends keywords your hook declares, so hooks
with the older four-argument signature keep working.

Return a `StructuredResponse(data, usage)` from `blackgeorge.adapters` to have the worker add the
call's token usage and cost to `Report.metrics` and the run's `max_cost_usd` budget. Returning the
parsed data directly is still accepted, but such calls are not metered. `LiteLLMAdapter` returns
`StructuredResponse` with usage summed over every attempt (schema, JSON fallback, retries) and emits
`llm.started`, `llm.completed`, and `llm.failed` for each attempt.

## Cost tracking

The LiteLLM adapter provides cost tracking through callback events. When using `LiteLLMAdapter`, the following events are automatically emitted:

### LLM events

Subscribe to these events to track costs and usage:

```python
from blackgeorge import Desk, Worker, Job

desk = Desk(model="openai/gpt-5-nano")

# Track LLM costs
def on_llm_completed(event):
    payload = event.payload
    print(f"Model: {payload['model']}")
    print(f"Cost: ${payload.get('cost', 0):.6f}")
    print(f"Tokens: {payload['total_tokens']}")
    print(f"Latency: {payload['latency_ms']}ms")

desk.event_bus.subscribe("llm.completed", on_llm_completed)

worker = Worker(name="assistant")
report = desk.run(worker, Job(input="Hello"))
```

### Event payloads

**llm.started:**
- `model`: Model name being called
- `messages_count`: Number of messages in the request
- `tools_count`: Number of tools available

**llm.completed:**
- `model`: Model name
- `latency_ms`: Request latency in milliseconds
- `prompt_tokens`: Number of prompt tokens
- `completion_tokens`: Number of completion tokens
- `total_tokens`: Total token count
- `cost`: Estimated cost in USD (if available from LiteLLM)

**llm.failed:**
- `model`: Model name
- `latency_ms`: Request latency before failure
- `error_type`: Exception class name
- `error_message`: Exception message

### Cost utilities

The `blackgeorge.adapters.cost` module provides utilities for cost calculation:

```python
from blackgeorge.adapters.cost import calculate_cost, get_model_pricing

# Calculate cost from a response
cost = calculate_cost(response)

# Get pricing info for a model
pricing = get_model_pricing("openai/gpt-5-nano")
```

## Custom adapters

To use another provider, implement `BaseModelAdapter` and pass it to `Desk(adapter=...)`.
