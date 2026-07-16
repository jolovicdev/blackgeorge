import json
import sqlite3
import threading
from datetime import datetime
from typing import Any

from blackgeorge.core.event import Event
from blackgeorge.core.serialization import to_json_value
from blackgeorge.core.types import RunStatus
from blackgeorge.store.base import RunRecord, RunStore
from blackgeorge.store.state import RunState
from blackgeorge.utils import utc_now


def _serialize(value: Any) -> str:
    return json.dumps(to_json_value(value), ensure_ascii=True)


def _serialize_state(state: RunState | None) -> str | None:
    if state is None:
        return None
    return _serialize(state.model_dump(mode="json", warnings=False))


def _deserialize_state(payload: str | None) -> RunState | None:
    if payload is None:
        return None
    return RunState.model_validate(json.loads(payload))


def _deserialize_event(payload: str) -> Event:
    return Event.model_validate(json.loads(payload))


def _row_to_record(row: Any) -> RunRecord:
    input_payload = json.loads(row[2]) if row[2] else None
    output_json = json.loads(row[4]) if row[4] else None
    state = _deserialize_state(row[5])
    created_at = datetime.fromisoformat(row[6])
    updated_at = datetime.fromisoformat(row[7])
    return RunRecord(
        run_id=row[0],
        status=row[1],
        input=input_payload,
        output=row[3],
        output_json=output_json,
        created_at=created_at,
        updated_at=updated_at,
        state=state,
    )


class SQLiteRunStore(RunStore):
    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._closed = False
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    input TEXT,
                    output TEXT,
                    output_json TEXT,
                    state_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id)")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)"
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at)")

    def create_run(self, run_id: str, input_payload: Any) -> None:
        now = utc_now().isoformat()
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    """
                        INSERT INTO runs (
                            id, status, input, output, output_json, state_json,
                            created_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                    (
                        run_id,
                        "running",
                        _serialize(input_payload),
                        None,
                        None,
                        None,
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Run '{run_id}' already exists") from exc

    def update_run(
        self,
        run_id: str,
        status: RunStatus,
        output: str | None,
        output_json: Any | None,
        state: RunState | None,
    ) -> None:
        now = utc_now().isoformat()
        with self._lock, self._conn:
            self._conn.execute(
                """
                    UPDATE runs
                    SET status = ?, output = ?, output_json = ?, state_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                (
                    status,
                    output,
                    _serialize(output_json) if output_json is not None else None,
                    _serialize_state(state),
                    now,
                    run_id,
                ),
            )

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT id, status, input, output, output_json, state_json, created_at, updated_at
                FROM runs WHERE id = ?
                """,
                (run_id,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return _row_to_record(row)

    def list_runs(
        self,
        status: RunStatus | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[RunRecord]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        row_limit = -1 if limit is None else limit
        with self._lock:
            if status is not None:
                cursor = self._conn.execute(
                    """
                    SELECT id, status, input, output, output_json, state_json,
                           created_at, updated_at
                    FROM runs WHERE status = ?
                    ORDER BY created_at DESC, rowid DESC
                    LIMIT ? OFFSET ?
                    """,
                    (status, row_limit, offset),
                )
            else:
                cursor = self._conn.execute(
                    """
                    SELECT id, status, input, output, output_json, state_json,
                           created_at, updated_at
                    FROM runs
                    ORDER BY created_at DESC, rowid DESC
                    LIMIT ? OFFSET ?
                    """,
                    (row_limit, offset),
                )
            rows = cursor.fetchall()
        return [_row_to_record(row) for row in rows]

    def add_event(self, event: Event) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                    INSERT INTO events (id, run_id, type, payload, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                (
                    event.event_id,
                    event.run_id,
                    event.type,
                    _serialize(event.model_dump(mode="json", warnings=False)),
                    event.timestamp.isoformat(),
                ),
            )

    def get_events(self, run_id: str) -> list[Event]:
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT payload FROM events WHERE run_id = ? ORDER BY timestamp ASC, rowid ASC
                """,
                (run_id,),
            )
            rows = cursor.fetchall()
        return [_deserialize_event(row[0]) for row in rows]

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True

    def __enter__(self) -> "SQLiteRunStore":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
