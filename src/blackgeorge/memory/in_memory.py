from typing import Any

from blackgeorge.memory.base import MemoryScope, MemoryStore


class InMemoryMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._data: dict[MemoryScope, dict[str, Any]] = {}

    def write(self, key: str, value: Any, scope: MemoryScope) -> None:
        self._data.setdefault(scope, {})[key] = value

    def read(self, key: str, scope: MemoryScope) -> Any | None:
        return self._data.get(scope, {}).get(key)

    def search(self, query: str, scope: MemoryScope) -> list[tuple[str, Any]]:
        matches: list[tuple[str, Any]] = []
        for key, value in self._data.get(scope, {}).items():
            if query in key or query in str(value):
                matches.append((key, value))
        return matches

    def reset(self, scope: MemoryScope) -> None:
        self._data[scope] = {}
