# Coding Agent Example

This example demonstrates a small coding agent that uses Desk, Worker, Workforce, tools, pause/resume, and the workflow DSL.

## Setup

- Set your API key:

```
export DEEPSEEK_API_KEY="..."
```

- Install the project in editable mode:

```
uv pip install -e .
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

## Output

- Events print to the console.
- Run data persists to `examples/coding_agent/.blackgeorge/blackgeorge.db`.
