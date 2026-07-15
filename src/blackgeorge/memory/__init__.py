from typing import TYPE_CHECKING

from blackgeorge.memory.base import MemoryScope, MemoryStore
from blackgeorge.memory.external import ExternalMemoryStore
from blackgeorge.memory.in_memory import InMemoryMemoryStore
from blackgeorge.memory.sqlite import SQLiteMemoryStore

if TYPE_CHECKING:
    from blackgeorge.memory.vector import VectorMemoryStore

__all__ = [
    "ExternalMemoryStore",
    "InMemoryMemoryStore",
    "MemoryScope",
    "MemoryStore",
    "SQLiteMemoryStore",
    "VectorMemoryStore",
]


def __getattr__(name: str) -> object:
    if name != "VectorMemoryStore":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from blackgeorge.memory.vector import VectorMemoryStore

    return VectorMemoryStore
