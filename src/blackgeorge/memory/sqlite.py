import json
import sqlite3
from typing import Any

from blackgeorge.memory.base import MemoryScope, MemoryStore
from blackgeorge.utils import utc_now


class SQLiteMemoryStore(MemoryStore):
    def __init__(self, path: str) -> None:
        self._path = path
        self._conn = sqlite3.connect(self._path)
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

    def write(self, key: str, value: Any, scope: MemoryScope) -> None:
        payload = json.dumps(value, ensure_ascii=True)
        now = utc_now().isoformat()
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
        cursor = self._conn.execute(
            "SELECT value FROM memories WHERE scope = ? AND key = ?",
            (scope, key),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def search(self, query: str, scope: MemoryScope) -> list[tuple[str, Any]]:
        cursor = self._conn.execute(
            "SELECT key, value FROM memories WHERE scope = ?",
            (scope,),
        )
        matches: list[tuple[str, Any]] = []
        for key, value in cursor.fetchall():
            if query in key or query in value:
                matches.append((key, json.loads(value)))
        return matches

    def reset(self, scope: MemoryScope) -> None:
        self._conn.execute("DELETE FROM memories WHERE scope = ?", (scope,))
        self._conn.commit()
