from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Any

from blackgeorge.utils import utc_now


@dataclass
class BlackboardEntry:
    key: str
    value: Any
    author: str
    created_at: datetime
    updated_at: datetime


BlackboardCallback = Callable[[str, Any, str], None]


class Blackboard:
    def __init__(self) -> None:
        self._data: dict[str, BlackboardEntry] = {}
        self._subscribers: dict[str, list[BlackboardCallback]] = {}
        self._global_subscribers: list[BlackboardCallback] = []
        self._lock = Lock()

    def write(self, key: str, value: Any, author: str) -> None:
        now = utc_now()
        callbacks: list[BlackboardCallback]
        global_callbacks: list[BlackboardCallback]
        with self._lock:
            if key in self._data:
                entry = self._data[key]
                self._data[key] = BlackboardEntry(
                    key=key,
                    value=value,
                    author=author,
                    created_at=entry.created_at,
                    updated_at=now,
                )
            else:
                self._data[key] = BlackboardEntry(
                    key=key,
                    value=value,
                    author=author,
                    created_at=now,
                    updated_at=now,
                )
            callbacks = list(self._subscribers.get(key, []))
            global_callbacks = list(self._global_subscribers)
        for callback in callbacks:
            callback(key, value, author)
        for callback in global_callbacks:
            callback(key, value, author)

    def read(self, key: str) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
        return entry.value if entry else None

    def read_entry(self, key: str) -> BlackboardEntry | None:
        with self._lock:
            return self._data.get(key)

    def exists(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    def all_entries(self) -> dict[str, BlackboardEntry]:
        with self._lock:
            return dict(self._data)

    def subscribe(self, key: str, callback: BlackboardCallback) -> None:
        with self._lock:
            if key not in self._subscribers:
                self._subscribers[key] = []
            self._subscribers[key].append(callback)

    def subscribe_all(self, callback: BlackboardCallback) -> None:
        with self._lock:
            self._global_subscribers.append(callback)

    def unsubscribe(self, key: str, callback: BlackboardCallback) -> None:
        with self._lock:
            if key in self._subscribers:
                self._subscribers[key] = [cb for cb in self._subscribers[key] if cb != callback]

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
