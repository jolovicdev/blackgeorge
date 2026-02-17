# Subworkers

Subworkers are delegated workers spawned by another worker through a tool call.

## Naming

Use `create_subworker_tool` as the primary API.

`create_swarm_tool` is a compatibility alias that returns the same behavior but names the tool
`spawn_agent` instead of `spawn_subworker`.

## API

```python
from blackgeorge.tools import create_subworker_tool

spawn_subworker = create_subworker_tool(
    desk=desk,
    available_tools=[...],
    default_model="deepseek/deepseek-chat",
    allowed_models={"deepseek/deepseek-chat"},
    max_subworkers=5,
    max_tools_per_subworker=8,
    max_iterations=10,
    max_tool_calls=10,
    structured_output_retries=1,
)
```

Arguments:

- `desk`: Runtime context for child runs. Child runs share adapter, event bus, run store, and memory.
- `available_tools`: Tool allowlist for spawned subworkers.
- `default_model`: Fallback model when a spawn call does not pass `model`.
- `allowed_models`: Optional model policy. If set, spawn is rejected for models outside this set.
- `max_subworkers`: Maximum successful spawn attempts for this tool instance.
- `max_tools_per_subworker`: Maximum number of tools that can be assigned in one spawn call.
- `max_iterations`: Worker iteration budget per subworker run.
- `max_tool_calls`: Tool-call budget per subworker run.
- `structured_output_retries`: Structured response retry budget per subworker run.

## Tool Contract

Input schema for `spawn_subworker`:

- `name: str`
- `instructions: str`
- `task: str`
- `tools: list[str]` (optional)
- `model: str | None` (optional)

Output uses `ToolResult`:

- On success:
  - `content`: subworker textual output
  - `data`: includes `run_id`, `status`, `worker`
- On failure:
  - `error`: policy/runtime error message
  - `data`: includes `run_id` and `status` when a child run exists

## Runtime Behavior

Each spawn creates a normal child run through `Desk.arun`. This means:

- Child runs are persisted in run storage.
- Child events are emitted to the same event bus.
- Memory policies from `Desk` are preserved.

Status handling:

- `completed`: returns success result.
- `failed`: returned as `ToolResult(error=...)`.
- `paused`: returned as `ToolResult(error=...)` with `pending_action_type` in `data`.

Paused children are treated as tool failures in the parent run so the parent does not silently
continue with incomplete delegation.

## Guardrails

Built-in guardrails:

- Spawn budget (`max_subworkers`)
- Model policy (`allowed_models`)
- Per-spawn tool budget (`max_tools_per_subworker`)
- Child run budgets (`max_iterations`, `max_tool_calls`)
- Tool allowlist (`available_tools`)

## Minimal Example

```python
from blackgeorge.core.job import Job
from blackgeorge.desk import Desk
from blackgeorge.tools import create_subworker_tool
from blackgeorge.worker import Worker

desk = Desk(model="deepseek/deepseek-chat")
spawn_subworker = create_subworker_tool(desk=desk, max_subworkers=3)

lead = Worker(
    name="Lead",
    model="deepseek/deepseek-chat",
    tools=[spawn_subworker],
    instructions="Delegate work with spawn_subworker and synthesize results.",
)

report = desk.run(lead, Job(input="Research top agent frameworks."))
```

## Compatibility Alias

`create_swarm_tool(...)` is equivalent to:

- same implementation and guardrails as `create_subworker_tool(...)`
- only tool name changes from `spawn_subworker` to `spawn_agent`

Use `create_subworker_tool` for new code.
