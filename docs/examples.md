# Examples

This repository includes a coding agent example under `examples/coding_agent`.

## What it demonstrates

- Desk, Worker, Workforce usage
- Tool calls and pause/resume
- Workflow steps with parallel execution
- Event streaming and logging

## Run the example

Set your API key for the selected model provider and run the script.

```
export DEEPSEEK_API_KEY="..."
python examples/coding_agent/run.py
```

The example stores run data at:

```
examples/coding_agent/.blackgeorge/blackgeorge.db
```

## Notes

- The example edits files inside `examples/coding_agent/project`.
- By default, edits are restored after the run completes.
- Set `PRESERVE_EXAMPLE_CHANGES=1` to keep changes.
- Set `BLACKGEORGE_STREAM=0` to disable streaming.
