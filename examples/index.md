# Examples

This repository includes a coding agent example under `examples/coding_agent`.

## What it demonstrates

- Desk, Worker, Workforce usage
- Tool calls and pause/resume
- Workflow steps with parallel execution
- Event streaming and logging
- Tool timeouts and retries
- Vector memory for semantic search and recall
- Channel and blackboard collaboration tools
- Tool result previews in event payloads

## Run the example

Set your API key for the selected model provider and run the script.

```text
export DEEPSEEK_API_KEY="..."
python examples/coding_agent/run.py
```

The example stores run data at:

```text
examples/coding_agent/.blackgeorge/blackgeorge.db
```

## Flow overview

The coding agent example is a managed workforce with a manager, coder, reviewer, and narrator. The manager selects the coder, the coder performs file operations with confirmation, and the reviewer summarizes changes. After the main run, a flow executes reviewer and narrator in parallel for a structured report and a human summary.

The flow job includes a `changed_files` list so the reviewer can ground its report in the actual modified files. Tool completion events include a short result preview so tool outputs are visible in logs without dumping full payloads.

## API surface used by the example

This example uses only public APIs and does not remove or replace any existing ones. The changes in this repository are additive and keep the original API surface intact.

Core runtime APIs:

- `Desk`, `Worker`, `Workforce`, `Job`, and `Report` drive the run lifecycle.
- `Workflow` nodes (`Step`, `Parallel`) orchestrate post-run summaries.

Tooling APIs:

- `tool` decorator registers callable tools with input validation.
- `ToolResult` carries tool output, errors, timeouts, and cancellation flags.
- `execute_tool` and `aexecute_tool` run tools with timeouts and retries.

Collaboration APIs:

- `Channel` and `Blackboard` provide messaging and shared state.
- `channel_send`, `channel_receive`, and `blackboard_write` are tool wrappers around those primitives.

Memory APIs:

- `VectorMemoryStore` stores file content and notes for semantic search.
- `search_docs`, `remember`, and `recall` are tools built on top of the memory store.

Event APIs:

- The event bus emits `tool.completed` with a short `result_preview` so tool outputs are visible in logs.

## Collaboration tools

Workers can communicate through channel tools and store shared state via blackboard tools. These tools are exposed as normal `Tool` instances so workers can call them like any other tool.

## Vector memory

File reads and notes are stored in the vector memory store to enable semantic search and recall. This allows the agent to look up recently read files and remembered decisions.

## Notes

- The example edits files inside `examples/coding_agent/project`.
- By default, edits are restored after the run completes.
- Set `PRESERVE_EXAMPLE_CHANGES=1` to keep changes.
- Set `BLACKGEORGE_STREAM=0` to disable streaming.
