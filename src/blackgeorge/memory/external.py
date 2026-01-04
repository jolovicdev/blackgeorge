from blackgeorge.memory.base import MemoryStore


class ExternalMemoryStore(MemoryStore):
    def write(self, key: str, value: object, scope: str) -> None:
        raise NotImplementedError

    def read(self, key: str, scope: str) -> object | None:
        raise NotImplementedError

    def search(self, query: str, scope: str) -> list[tuple[str, object]]:
        raise NotImplementedError

    def reset(self, scope: str) -> None:
        raise NotImplementedError
