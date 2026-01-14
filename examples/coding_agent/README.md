# Coding Agent Example

This example demonstrates a Python coding agent built with the Blackgeorge LLM agent framework. It uses Desk, Worker, Workforce, tools, pause/resume, workflow DSL, and collaboration features.

## Features

- **VectorMemoryStore**: Semantic search over project files
- **Tool timeouts/retries**: Resilient file operations
- **Channel**: Worker-to-worker messaging
- **Blackboard**: Shared state across workers
- **New tools**: `search_docs`, `remember`, `recall`

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

```
python examples/coding_agent/run.py
```

The script pauses for confirmations and user input when tools require it.
By default it restores any edits under `examples/coding_agent/project` after the run.
Set `PRESERVE_EXAMPLE_CHANGES=1` to keep changes.
It streams tokens for non-tool steps and prints assistant messages as they are produced.
Set `BLACKGEORGE_STREAM=0` to disable streaming.

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

## Output

- Events print to the console
- Blackboard state printed after run and after flow summary
- Channel messages printed after run
- Tool completion events include a short result preview
- Run data persists to `examples/coding_agent/.blackgeorge/blackgeorge.db`
- Vector memory at `examples/coding_agent/.blackgeorge/memory/`
