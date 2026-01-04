# Storage

Run storage tracks the state of each run and stores events. Blackgeorge exposes this via the `RunStore` interface.

## RunStore interface

A run store supports:

- create_run(run_id, input_payload)
- update_run(run_id, status, output, output_json, state)
- get_run(run_id)
- add_event(event)
- get_events(run_id)

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

To build a custom run store, implement the `RunStore` interface and pass it to `Desk(run_store=...)`.
