# Coding Agent Example

This example demonstrates a Python coding agent built with the Blackgeorge LLM agent framework. It uses Desk, Worker, Workforce, tools, pause/resume, workflow DSL, and collaboration features.

## Features

- **VectorMemoryStore**: Semantic search over project files
- **Tool timeouts/retries**: Resilient file operations
- **Channel**: Worker-to-worker messaging
- **Blackboard**: Shared state across workers
- **Swarm mode**: Dynamic handoffs between workers with `transfer_to_agent_tool`
- **Context compaction**: Proactive summarization with `max_context_messages`
- **Typed events**: Using `EventType` enum and typed payload classes
- **Custom exceptions**: `ToolExecutionError`, `ContextLimitError`, etc.

## Setup

- Set your API key:

```
export DEEPSEEK_API_KEY="..."
```

- Install the project in editable mode:

```
uv pip install -e .[dev]
```

## Run

```bash
# Interactive mode (default)
python examples/coding_agent/run.py

# Non-interactive with prompt
python examples/coding_agent/run.py --prompt "List files and read spec.txt"

# Swarm mode with dynamic handoffs
python examples/coding_agent/run.py --swarm --prompt "Review the calculator code"

# Disable streaming
python examples/coding_agent/run.py --no-stream --prompt "Read spec.txt"
```

### Flags

| Flag | Description |
|------|-------------|
| `--prompt TEXT` | Run non-interactively with the given prompt (auto-confirms tool actions) |
| `--swarm` | Use swarm mode instead of managed mode (dynamic worker handoffs) |
| `--no-stream` | Disable token streaming |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | (required) | API key for the LLM |
| `BLACKGEORGE_STREAM` | `1` | Set to `0` to disable streaming |
| `PRESERVE_EXAMPLE_CHANGES` | `0` | Set to `1` to keep file edits after run |

The script pauses for confirmations and user input when tools require it (interactive mode only). By default it restores any edits under `examples/coding_agent/project` after the run.

## Sample project

The agent edits files inside:

```
examples/coding_agent/project
```

## Tools

| Tool | Description | Features |
|------|-------------|----------|
| `list_files` | List project files | 5s timeout |
| `read_file` | Read file content | 5s timeout, 2 retries |
| `write_file` | Write file (confirmation required) | 10s timeout |
| `ask_user` | Prompt user for input | User input required |
| `search_docs` | Semantic search over files | Vector memory |
| `remember` | Save notes | Vector memory |
| `recall` | Recall notes by query | Vector memory |
| `channel_send` | Send a message to another worker | Channel |
| `channel_receive` | Receive channel messages | Channel |
| `blackboard_write` | Store shared state | Blackboard |
| `transfer_to_agent` | Hand off to another worker | Swarm mode |

## New Capabilities Demonstrated

### Swarm Mode
Workers can dynamically hand off execution using `transfer_to_agent_tool`. The workforce intercepts the handoff pending action and switches the active worker mid-run.

### Proactive Context Compaction
The example uses `max_context_messages=15` to proactively summarize conversation history before it grows too large, preventing context limit errors.

### Typed Events
Event handling uses the `EventType` enum and typed payload access:
```python
from blackgeorge import EventType

if event.type == EventType.TOOL_COMPLETED:
    cancelled = event.payload.get("cancelled", False)
```

### Custom Exceptions
Tool execution errors are caught with typed exceptions:
```python
from blackgeorge import ToolExecutionError

try:
    report = desk.run(workforce, job)
except ToolExecutionError as e:
    print(f"Tool {e.tool_name} failed: {e}")
```

## Output

- Events print to the console with typed event handling
- `WORKER_CONTEXT_SUMMARIZED` events show proactive compaction stats
- Tool timing includes cancellation status
- Blackboard state printed after run and after flow summary
- Channel messages printed after run
- Run data persists to `examples/coding_agent/.blackgeorge/blackgeorge.db`
- Vector memory at `examples/coding_agent/.blackgeorge/memory/`
