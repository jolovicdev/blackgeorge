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
- passes messages, tools, tool_choice, temperature, max_tokens
- supports streaming when requested
- enables `parallel_tool_calls` when LiteLLM reports the model supports parallel function calling

Tool calls are parsed from the response and mapped into `ToolCall` objects.

## Structured output pipeline

Structured output uses LiteLLM JSON schema response formats when possible and falls back to Instructor with LiteLLM. Blackgeorge initializes Instructor clients with:

- `instructor.from_provider("litellm/<model>")`
- `instructor.from_provider("litellm/<model>", async_client=True)`

If the LiteLLM structured response fails or is unavailable, the worker calls `chat.completions.create(..., response_model=YourModel)` and returns the validated Pydantic object as `Report.data`.

## Adapter hooks for structured output

If your adapter implements `structured_complete`/`astructured_complete`, the worker will call those hooks for response-schema jobs. This lets you route structured output through non-LiteLLM providers or custom pipelines. If the hooks are not implemented, the worker falls back to the LiteLLM + Instructor path.

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
