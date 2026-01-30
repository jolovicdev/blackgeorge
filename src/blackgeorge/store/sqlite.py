import json
import sqlite3
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, cast

from pydantic import BaseModel

from blackgeorge.core.event import Event
from blackgeorge.core.types import RunStatus
from blackgeorge.store.base import RunRecord, RunStore
from blackgeorge.store.state import RunState
from blackgeorge.utils import utc_now


def _normalize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", warnings=False)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(cast(Any, value))
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    return value


def _serialize(value: Any) -> str:
    return json.dumps(_normalize(value), ensure_ascii=True)


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


class SQLiteRunStore(RunStore):
    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
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

    def _connect(self) -> sqlite3.Connection:
        return self._conn

    def create_run(self, run_id: str, input_payload: Any) -> None:
        now = utc_now().isoformat()
        with self._lock:
            conn = self._connect()
            with conn:
                conn.execute(
                    """
                    INSERT INTO runs (
                        id, status, input, output, output_json, state_json, created_at, updated_at
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

    def update_run(
        self,
        run_id: str,
        status: RunStatus,
        output: str | None,
        output_json: Any | None,
        state: RunState | None,
    ) -> None:
        now = utc_now().isoformat()
        with self._lock:
            conn = self._connect()
            with conn:
                conn.execute(
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
            conn = self._connect()
            cursor = conn.execute(
                """
                SELECT id, status, input, output, output_json, state_json, created_at, updated_at
                FROM runs WHERE id = ?
                """,
                (run_id,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        input_payload = json.loads(row[2]) if row[2] else None
        output_json = json.loads(row[4]) if row[4] else None
        state = _deserialize_state(row[5])
        created_at = datetime_from_iso(row[6])
        updated_at = datetime_from_iso(row[7])
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

    def add_event(self, event: Event) -> None:
        with self._lock:
            conn = self._connect()
            with conn:
                conn.execute(
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
            conn = self._connect()
            cursor = conn.execute(
                """
                SELECT payload FROM events WHERE run_id = ? ORDER BY timestamp ASC
                """,
                (run_id,),
            )
            rows = cursor.fetchall()
        return [_deserialize_event(row[0]) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def datetime_from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)
