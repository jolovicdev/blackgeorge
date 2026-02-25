# Configuration

Blackgeorge provides extensive configuration options through environment variables, Desk parameters, and module-level constants.

## Environment variables

### Provider API keys

Set the API key for your model provider:

```bash
# OpenAI
export OPENAI_API_KEY="your-key"

# Anthropic
export ANTHROPIC_API_KEY="your-key"

# DeepSeek
export DEEPSEEK_API_KEY="your-key"

# Azure
export AZURE_API_KEY="your-key"
export AZURE_API_BASE="https://your-resource.openai.azure.com"
export AZURE_API_VERSION="2023-07-01-preview"
```

### Example changes preservation

For the coding agent example, control whether file changes are preserved:

```bash
# Preserve example changes (default: changes are restored)
export PRESERVE_EXAMPLE_CHANGES="1"
```

## Desk configuration

The `Desk` class accepts comprehensive configuration options:

```python
from blackgeorge import Desk

desk = Desk(
    # Model configuration
    model="openai/gpt-5-nano",
    temperature=0.2,
    max_tokens=800,

    # Streaming
    stream=False,

    # Structured output
    structured_stream_mode="off",
    structured_output_retries=3,

    # Limits
    max_iterations=10,
    max_tool_calls=20,

    # Context window handling
    respect_context_window=True,

    # Custom components
    event_bus=custom_event_bus,
    run_store=custom_run_store,
    memory_store=custom_memory_store,
    adapter=custom_adapter,

    # Storage
    storage_dir=".blackgeorge",
)
```

## RunConfig

For programmatic control, use `RunConfig` to bundle run parameters:

```python
from blackgeorge import Desk, Worker, RunConfig
from blackgeorge.adapters import LiteLLMAdapter

adapter = LiteLLMAdapter()
worker = Worker(name="analyst")

config = RunConfig(
    adapter=adapter,
    emit=lambda type, source, payload: print(f"{type}: {source}"),
    run_id="run-123",
    temperature=0.7,
    max_tokens=1000,
    max_iterations=5,
    max_tool_calls=10,
    default_model="openai/gpt-5-nano",
)

# Use config-based methods
report, state = worker.run_with_config(config, job)
report, state = await worker.arun_with_config(config, job)
```

### RunConfig fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `adapter` | BaseModelAdapter | Required | LLM adapter for model calls |
| `emit` | EventEmitter | Required | Event emission callback |
| `run_id` | str | Required | Unique run identifier |
| `events` | list[Event] | [] | Event list for the run |
| `temperature` | float | None | Model temperature |
| `max_tokens` | int | None | Maximum completion tokens |
| `stream` | bool | False | Enable streaming |
| `stream_options` | dict | None | Streaming options |
| `structured_output_retries` | int | 3 | Structured output retry count |
| `max_iterations` | int | 10 | Maximum model turns |
| `max_tool_calls` | int | 20 | Maximum tool calls |
| `respect_context_window` | bool | True | Auto-summarize on context errors |
| `default_model` | str | None | Default model name |

### RunConfig methods

```python
# Create a copy with overrides
new_config = config.with_overrides(temperature=0.5, max_tokens=500)

# Get effective model name (worker model or default)
model = config.model_name(worker.model)
```

### Model configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | Required | Default model name for workers without their own model |
| `temperature` | float | None | Model temperature (0.0-1.0) |
| `max_tokens` | int | None | Maximum tokens for completion requests |

Model names follow LiteLLM provider prefixes:
- `openai/gpt-5-nano`
- `anthropic/claude-3-opus`
- `deepseek/deepseek-chat`

### Streaming

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stream` | bool | False | Enable streaming for eligible workers |

Streaming occurs when:
- Desk or desk.run() has `stream=True`
- No response schema is set
- Or `structured_stream_mode="preview"` with a response schema

On tool turns, streamed `stream.token` events contain tool argument deltas.

### Structured output

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `structured_stream_mode` | str | "off" | Structured output mode: "off" (strict non-stream) or "preview" (stream tokens, then validate/fallback) |
| `structured_output_retries` | int | 3 | Retries for structured output validation failures |

### Execution limits

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_iterations` | int | 10 | Maximum model turns per worker run |
| `max_tool_calls` | int | 20 | Maximum tool calls per worker run |

When these limits are exceeded, the run fails with an error in `Report.errors`.

### Context window handling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `respect_context_window` | bool | True | Auto-summarize on context length errors |

When enabled, workers summarize conversation history and retry on context limit errors.

### Custom components

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `event_bus` | EventBus | None | Custom event bus implementation |
| `run_store` | RunStore | None | Custom run store implementation |
| `memory_store` | MemoryStore | None | Custom memory store implementation |
| `adapter` | BaseModelAdapter | None | Custom model adapter |

### Storage

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `storage_dir` | str | ".blackgeorge" | Directory for SQLite run store |

## Worker configuration

Workers can be configured with several options:

```python
from blackgeorge import Worker

worker = Worker(
    # Identity
    name="Analyst",

    # Model
    model="openai/gpt-5-nano",
    instructions="You are a data analyst.",

    # Tools
    tools=[tool1, tool2],

    # Memory
    memory_scope="analyst:session-1",
)
```

### Worker parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | str | Yes | Worker name used in events and selection |
| `model` | str | No | Model name override (uses desk default if None) |
| `instructions` | str | No | System instructions for the worker |
| `tools` | list[Tool] | No | Tools this worker can use |
| `memory_scope` | str | No | Namespace for external memory usage |

## Workforce configuration

Workforces support multiple coordination modes:

```python
from blackgeorge import Worker, Workforce

workers = [Worker(name="a"), Worker(name="b")]

workforce = Workforce(
    workers=workers,
    mode="managed",           # "managed" or "collaborate"
    name="team",
    manager=manager_worker,   # Only for managed mode
    reducer=custom_reducer,   # Only for collaborate mode
    channel=channel,          # Optional
    blackboard=blackboard,    # Optional
)
```

### Workforce parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `workers` | list[Worker] | Yes | Workers in the workforce |
| `mode` | str | No | "managed" or "collaborate" (default: "managed") |
| `name` | str | No | Workforce name for events |
| `manager` | Worker | No | Manager for worker selection (managed mode) |
| `reducer` | Callable | No | Function to combine worker reports (collaborate mode) |
| `channel` | Channel | No | Communication channel between workers |
| `blackboard` | Blackboard | No | Shared state for workers |

### Managed mode

In managed mode, a manager worker selects one worker, and the selected worker's report is returned.
The `reducer` is not used in managed mode.

### Collaborate mode

In collaborate mode, workers without tools run concurrently, while workers with tools run
sequentially to preserve deterministic pause/resume semantics. Provide a custom reducer to control
how reports are combined:

```python
from blackgeorge import Report

def custom_reducer(reports: list[Report]) -> Report:
    combined_content = "\n\n".join(r.content or "" for r in reports)
    merged_errors = [error for report in reports for error in report.errors]
    return reports[0].model_copy(
        update={
            "status": "failed" if merged_errors else "completed",
            "content": combined_content,
            "errors": merged_errors,
            "pending_action": None,
        }
    )

workforce = Workforce(
    workers=[w1, w2],
    mode="collaborate",
    reducer=custom_reducer,
)
```

## Context window configuration

Module-level constants control context summarization behavior in `src/blackgeorge/worker_context.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `SUMMARY_CHUNK_TOKENS` | 2000 | Target token count for summarization chunks |
| `SUMMARY_TAIL_MESSAGES` | 4 | Number of recent messages to keep unsummarized |
| `SUMMARY_ATTEMPT_LIMIT` | 2 | Maximum summarization attempts before failing |
| `SUMMARY_MAX_OUTPUT_TOKENS` | 800 | Maximum tokens in summary output |

These constants affect how workers handle context length errors when `respect_context_window=True`.

### Model registration

For accurate context window handling, register your models in LiteLLM:

```python
import litellm

litellm.model_cost = {
    **litellm.model_cost,
    "my-model": {
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.00002,
        "max_tokens": 128000,
    },
}
```

You can also register models programmatically:

```python
import litellm

litellm.register_model(
    {
        "my-provider/my-model": {
            "litellm_provider": "my-provider",
            "input_cost_per_token": 0.00001,
            "output_cost_per_token": 0.00002,
            "max_tokens": 128000,
            "mode": "chat",
        },
    }
)
```

Model registration only adds pricing/context metadata. The provider prefix still has to be a
LiteLLM-supported provider (or routed through LiteLLM proxy or a custom adapter) to make calls.

## Tool configuration

Tools accept several configuration options:

```python
from blackgeorge.tools import tool

def pre_hook(call):
    print(f"calling {call.name}")

def post_hook(call, result):
    print(f"{call.name} done: {result.error is None}")

@tool(
    name="my_tool",
    description="Does something useful",
    requires_confirmation=False,
    requires_user_input=False,
    external_execution=False,
    pre=(pre_hook,),
    post=(post_hook,),
    confirmation_prompt="Proceed?",
    user_input_prompt="Enter value:",
    timeout=30.0,
    retries=3,
    retry_delay=1.0,
)
def my_function(param: str) -> str:
    return param
```

### Tool parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | Function name | Tool name |
| `description` | str | Auto-generated | Tool description for the model |
| `requires_confirmation` | bool | False | Requires user confirmation before execution |
| `requires_user_input` | bool | False | Requires user input before execution |
| `external_execution` | bool | False | Mark as external tool |
| `pre` | tuple[ToolPreHook, ...] | () | Pre-execution hooks |
| `post` | tuple[ToolPostHook, ...] | () | Post-execution hooks |
| `confirmation_prompt` | str | Auto-generated | Custom confirmation prompt |
| `user_input_prompt` | str | Auto-generated | Custom input prompt |
| `input_key` | str | None | Field name to receive resume user input (default key: `user_input`) |
| `timeout` | float | None | Execution timeout in seconds |
| `retries` | int | 0 | Number of retry attempts |
| `retry_delay` | float | 1.0 | Delay between retries (exponential backoff) |

## Memory store configuration

### VectorMemoryStore

```python
from blackgeorge.memory import VectorMemoryStore

store = VectorMemoryStore(
    path="/path/to/db",
    chunk_size=8000,
    chunk_overlap=200,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str | Required | Storage path for ChromaDB |
| `chunk_size` | int | 8000 | Chunk size for long documents |
| `chunk_overlap` | int | 200 | Overlap between chunks |

## Run store configuration

### Custom run store

Implement the `RunStore` interface:

```python
from typing import Any

from blackgeorge.core.event import Event
from blackgeorge.core.types import RunStatus
from blackgeorge.store import RunRecord, RunState, RunStore

class CustomRunStore(RunStore):
    def create_run(self, run_id: str, input_payload: Any) -> None:
        # Implementation
        pass

    def update_run(
        self,
        run_id: str,
        status: RunStatus,
        output: str | None,
        output_json: Any | None,
        state: RunState | None,
    ) -> None:
        # Implementation
        pass

    def get_run(self, run_id: str) -> RunRecord | None:
        # Implementation
        pass

    def add_event(self, event: Event) -> None:
        # Implementation
        pass

    def get_events(self, run_id: str) -> list[Event]:
        # Implementation
        pass

desk = Desk(run_store=CustomRunStore())
```

## Event bus configuration

### Custom event bus

```python
from blackgeorge.event_bus import EventBus
from blackgeorge import Desk

bus = EventBus()

# Subscribe to events before passing to desk
def log_events(event):
    print(f"{event.type}: {event.payload}")

for event_type in ("run.started", "run.completed", "run.failed"):
    bus.subscribe(event_type, log_events)

desk = Desk(event_bus=bus)
```

## Adapter configuration

### Custom adapter

Implement the `BaseModelAdapter` interface:

```python
from typing import Any

from blackgeorge.adapters.base import BaseModelAdapter, ModelResponse

class CustomAdapter(BaseModelAdapter):
    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ModelResponse:
        # Implementation
        pass

    async def acomplete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ModelResponse:
        # Implementation
        pass

desk = Desk(adapter=CustomAdapter())
```

## Logging configuration

Blackgeorge uses the `StructuredLogger` for JSON-formatted logs:

```python
import logging

from blackgeorge.logging import get_logger

logger = get_logger("my_app", level=logging.INFO)

logger.info("Application started", user_id="123")
# {"timestamp": "2024-01-19T...", "level": "INFO", "message": "Application started", "user_id": "123"}
```

See [Logging](logging.md) for more details.

## Environment variable reference

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `AZURE_API_KEY` | Azure OpenAI API key |
| `AZURE_API_BASE` | Azure OpenAI endpoint |
| `AZURE_API_VERSION` | Azure API version |
| `PRESERVE_EXAMPLE_CHANGES` | Preserve coding-agent example file changes ("1") |

## Configuration best practices

1. **Use environment variables for secrets**: Never hardcode API keys
2. **Set appropriate limits**: Adjust `max_iterations` and `max_tool_calls` based on your use case
3. **Enable context window handling**: Keep `respect_context_window=True` for production
4. **Configure storage_dir**: Use a dedicated directory for run storage
5. **Use memory_scope**: Isolate worker memory when using multiple workers
6. **Register models**: Register custom models in LiteLLM for accurate context handling
