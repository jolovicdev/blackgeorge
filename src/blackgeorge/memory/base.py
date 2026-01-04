from typing import Any

MemoryScope = str


class MemoryStore:
    def write(self, key: str, value: Any, scope: MemoryScope) -> None:
        raise NotImplementedError

    def read(self, key: str, scope: MemoryScope) -> Any | None:
        raise NotImplementedError

    def search(self, query: str, scope: MemoryScope) -> list[tuple[str, Any]]:
        raise NotImplementedError

    def reset(self, scope: MemoryScope) -> None:
        raise NotImplementedError
