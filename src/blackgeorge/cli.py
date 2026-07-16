import argparse
import json
import os
import sys
from typing import Any

from blackgeorge.core.serialization import to_json_value
from blackgeorge.store.base import RunRecord
from blackgeorge.store.sqlite import SQLiteRunStore
from blackgeorge.store.sqlite_session_store import SQLiteSessionStore

DEFAULT_DB = os.path.join(".blackgeorge", "blackgeorge.db")


def _run_payload(record: RunRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": record.run_id,
        "status": record.status,
        "input": record.input,
        "output": record.output,
        "output_json": record.output_json,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    }
    if record.state is not None and record.state.metrics:
        payload["metrics"] = to_json_value(record.state.metrics)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="blackgeorge", description="Inspect blackgeorge runs and sessions."
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help="path to the SQLite database (default: .blackgeorge/blackgeorge.db)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    runs = commands.add_parser("runs", help="list and inspect runs")
    runs_commands = runs.add_subparsers(dest="runs_command", required=True)
    runs_list = runs_commands.add_parser("list", help="list runs, newest first")
    runs_list.add_argument(
        "--status", choices=["completed", "paused", "failed", "running"], default=None
    )
    runs_list.add_argument("--limit", type=int, default=None)
    runs_show = runs_commands.add_parser("show", help="show one run as JSON")
    runs_show.add_argument("run_id")

    sessions = commands.add_parser("sessions", help="list sessions")
    sessions_commands = sessions.add_subparsers(dest="sessions_command", required=True)
    sessions_list = sessions_commands.add_parser(
        "list", help="list sessions, recently updated first"
    )
    sessions_list.add_argument("--worker", default=None)
    sessions_list.add_argument("--limit", type=int, default=None)

    args = parser.parse_args(argv)
    if not os.path.exists(args.db):
        print(f"error: no database at {args.db}", file=sys.stderr)
        return 1

    if args.command == "runs":
        store = SQLiteRunStore(args.db)
        try:
            if args.runs_command == "list":
                for run in store.list_runs(status=args.status, limit=args.limit):
                    print(f"{run.run_id}  {run.status}  {run.updated_at.isoformat()}")
                return 0
            record = store.get_run(args.run_id)
            if record is None:
                print(f"error: run not found: {args.run_id}", file=sys.stderr)
                return 1
            print(json.dumps(_run_payload(record), indent=2, default=str))
            return 0
        finally:
            store.close()

    session_store = SQLiteSessionStore(args.db)
    try:
        for session in session_store.list_sessions(worker_name=args.worker, limit=args.limit):
            print(f"{session.session_id}  {session.worker_name}  {session.updated_at.isoformat()}")
        return 0
    finally:
        session_store.close()


if __name__ == "__main__":
    sys.exit(main())
