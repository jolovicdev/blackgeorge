import json
import sqlite3
import threading
from dataclasses import asdict, is_dataclass
from typing import Any

from pydantic import BaseModel

from blackgeorge.memory.base import MemoryScope, MemoryStore
from blackgeorge.utils import utc_now


class SQLiteMemoryStore(MemoryStore):
    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._closed = False
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(scope, key)
                )
                """
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True

    def __enter__(self) -> "SQLiteMemoryStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def write(self, key: str, value: Any, scope: MemoryScope) -> None:
        payload = json.dumps(self._normalize(value), ensure_ascii=True, default=str)
        now = utc_now().isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO memories (scope, key, value, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(scope, key)
                DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (scope, key, payload, now, now),
            )
            self._conn.commit()

    def read(self, key: str, scope: MemoryScope) -> Any | None:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT value FROM memories WHERE scope = ? AND key = ?",
                (scope, key),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def search(self, query: str, scope: MemoryScope) -> list[tuple[str, Any]]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT key, value FROM memories WHERE scope = ?",
                (scope,),
            )
            rows = cursor.fetchall()
        matches: list[tuple[str, Any]] = []
        for key, value in rows:
            if query in key or query in value:
                matches.append((key, json.loads(value)))
        return matches

    def reset(self, scope: MemoryScope) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM memories WHERE scope = ?", (scope,))
            self._conn.commit()

    def _normalize(self, value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json", warnings=False)
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        if isinstance(value, dict):
            return {key: self._normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._normalize(item) for item in value]
        if isinstance(value, tuple):
            return [self._normalize(item) for item in value]
        return value
