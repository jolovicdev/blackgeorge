# CLI

Blackgeorge ships a `blackgeorge` command for inspecting persisted runs and sessions without
writing Python. It reads the same SQLite database that `Desk` writes to.

```bash
blackgeorge --help
```

By default the CLI reads `.blackgeorge/blackgeorge.db`. Point `--db` at another file when you use a
custom `storage_dir`:

```bash
blackgeorge --db /path/to/blackgeorge.db runs list
```

If the database file does not exist, the CLI exits with status 1 and an error message.

## List runs

```bash
blackgeorge runs list
blackgeorge runs list --status failed --limit 10
```

Output is one run per line: `run_id  status  updated_at`, newest first. `--status` accepts
`completed`, `paused`, `failed`, or `running`.

## Show a run

```bash
blackgeorge runs show <run_id>
```

Prints the run as JSON: status, input, output, structured output, timestamps, and the stored
metrics (usage and cost) when the run has persisted state.

## List sessions

```bash
blackgeorge sessions list
blackgeorge sessions list --worker ChatBot --limit 10
```

Output is one session per line: `session_id  worker_name  updated_at`, recently updated first.

!!! note "Startup time"
    Importing the library loads LiteLLM, which fetches its model pricing map from the network.
    Set `LITELLM_LOCAL_MODEL_COST_MAP=True` to skip that fetch and use the local copy instead.
