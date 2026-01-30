import json
import sqlite3
import threading
from datetime import datetime
from typing import Any

from blackgeorge.core.message import Message
from blackgeorge.store.session_store import SessionRecord, SessionStore
from blackgeorge.utils import new_id, utc_now


def _serialize_message(message: Message) -> str:
    return message.model_dump_json()


def _deserialize_message(payload: str) -> Message:
    return Message.model_validate(json.loads(payload))


def _datetime_from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


class SQLiteSessionStore(SessionStore):
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    worker_name TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    message_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_worker ON sessions(worker_name)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_messages_session "
                "ON session_messages(session_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_messages_timestamp "
                "ON session_messages(timestamp)"
            )

    def _connect(self) -> sqlite3.Connection:
        return self._conn

    def create_session(
        self,
        session_id: str,
        worker_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = utc_now().isoformat()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=True)
        with self._lock:
            conn = self._connect()
            with conn:
                conn.execute(
                    """
                    INSERT INTO sessions (id, worker_name, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_id, worker_name, metadata_json, now, now),
                )

    def update_session(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = utc_now().isoformat()
        with self._lock:
            conn = self._connect()
            with conn:
                if metadata is not None:
                    conn.execute(
                        """
                        UPDATE sessions
                        SET metadata = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (json.dumps(metadata, ensure_ascii=True), now, session_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE sessions
                        SET updated_at = ?
                        WHERE id = ?
                        """,
                        (now, session_id),
                    )

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            conn = self._connect()
            cursor = conn.execute(
                """
                SELECT id, worker_name, metadata, created_at, updated_at
                FROM sessions WHERE id = ?
                """,
                (session_id,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return SessionRecord(
            session_id=row[0],
            worker_name=row[1],
            metadata=json.loads(row[2]),
            created_at=_datetime_from_iso(row[3]),
            updated_at=_datetime_from_iso(row[4]),
        )

    def list_sessions(
        self,
        worker_name: str | None = None,
        limit: int | None = None,
    ) -> list[SessionRecord]:
        with self._lock:
            conn = self._connect()
            if worker_name:
                cursor = conn.execute(
                    """
                    SELECT id, worker_name, metadata, created_at, updated_at
                    FROM sessions
                    WHERE worker_name = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (worker_name, limit or -1),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT id, worker_name, metadata, created_at, updated_at
                    FROM sessions
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (limit or -1,),
                )
            rows = cursor.fetchall()
        return [
            SessionRecord(
                session_id=row[0],
                worker_name=row[1],
                metadata=json.loads(row[2]),
                created_at=_datetime_from_iso(row[3]),
                updated_at=_datetime_from_iso(row[4]),
            )
            for row in rows
        ]

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            conn = self._connect()
            with conn:
                conn.execute("DELETE FROM session_messages WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    def add_messages(self, session_id: str, messages: list[Message]) -> None:
        now = utc_now().isoformat()
        with self._lock:
            conn = self._connect()
            with conn:
                for i, message in enumerate(messages):
                    message_id = f"{session_id}_{i}_{new_id()}"
                    conn.execute(
                        """
                        INSERT INTO session_messages (id, session_id, message_json, timestamp)
                        VALUES (?, ?, ?, ?)
                        """,
                        (message_id, session_id, _serialize_message(message), now),
                    )

    def get_messages(self, session_id: str) -> list[Message]:
        with self._lock:
            conn = self._connect()
            cursor = conn.execute(
                """
                SELECT message_json FROM session_messages
                WHERE session_id = ?
                ORDER BY timestamp ASC, rowid ASC
                """,
                (session_id,),
            )
            rows = cursor.fetchall()
        return [_deserialize_message(row[0]) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
