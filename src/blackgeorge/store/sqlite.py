import json
import sqlite3
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
        return value.model_dump(mode="json")
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
    return _serialize(state.model_dump(mode="json"))


def _deserialize_state(payload: str | None) -> RunState | None:
    if payload is None:
        return None
    return RunState.model_validate(json.loads(payload))


def _deserialize_event(payload: str) -> Event:
    return Event.model_validate(json.loads(payload))


class SQLiteRunStore(RunStore):
    def __init__(self, path: str) -> None:
        self._path = path
        with sqlite3.connect(self._path) as conn:
            conn.execute(
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
            conn.execute(
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at)")
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def create_run(self, run_id: str, input_payload: Any) -> None:
        now = utc_now().isoformat()
        with self._connect() as conn:
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
            conn.commit()

    def update_run(
        self,
        run_id: str,
        status: RunStatus,
        output: str | None,
        output_json: Any | None,
        state: RunState | None,
    ) -> None:
        now = utc_now().isoformat()
        with self._connect() as conn:
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
            conn.commit()

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._connect() as conn:
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
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events (id, run_id, type, payload, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.run_id,
                    event.type,
                    _serialize(event.model_dump(mode="json")),
                    event.timestamp.isoformat(),
                ),
            )
            conn.commit()

    def get_events(self, run_id: str) -> list[Event]:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT payload FROM events WHERE run_id = ? ORDER BY timestamp ASC
                """,
                (run_id,),
            )
            rows = cursor.fetchall()
        return [_deserialize_event(row[0]) for row in rows]


def datetime_from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)
