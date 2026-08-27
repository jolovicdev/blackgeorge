# Storage

Run storage tracks the state of each run and stores events. Blackgeorge exposes this via the `RunStore` interface.

## RunStore interface

A run store supports:

- create_run(run_id, input_payload)
- update_run(run_id, status, output, output_json, state)
- get_run(run_id)
- list_runs(status, limit, offset)
- add_event(event)
- get_events(run_id)

## Listing runs

`list_runs` returns stored runs, most recently created first. Filter by status and page with `limit`/`offset`:

```python
from blackgeorge import Desk

desk = Desk(model="openai/gpt-5-nano")

# Everything, newest first
records = desk.run_store.list_runs()

# Failed runs only, one page of 20
failed = desk.run_store.list_runs(status="failed", limit=20, offset=0)

for record in failed:
    print(record.run_id, record.status, record.created_at)
```

Custom stores support the same queries by implementing the method. `limit` and `offset` must be non-negative.

## RunRecord

A `RunRecord` includes:

- run_id
- status
- input
- output
- output_json
- created_at
- updated_at
- state

## RunState

Run state stores enough data to resume a paused run. It includes:

- run_id, status, runner_type, runner_name
- job, messages, tool_calls
- pending_action
- metrics
- iteration
- payload

The payload field is used by workflows and workforces to store extra resume data.

## SQLiteRunStore

The default run store uses SQLite and writes to `.blackgeorge/blackgeorge.db` unless you override `storage_dir` or `run_store`.

- Inputs, outputs, and state are serialized as JSON.
- Events are stored in a separate table and returned in timestamp order.

## InMemoryRunStore

The in-memory store is useful for tests or ephemeral runs. It stores records and events in dictionaries and does not persist data.

## Custom stores

To build a custom run store, implement the `RunStore` interface and pass it to `Desk(run_store=...)`. Stores passed into `Desk` are caller-owned and remain open when the desk closes. The desk closes only the run and memory stores it creates itself.
