from dataclasses import dataclass
from datetime import datetime
from typing import Any

from blackgeorge.core.message import Message


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    worker_name: str
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any]


class SessionStore:
    def create_session(
        self,
        session_id: str,
        worker_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    def update_session(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    def get_session(self, session_id: str) -> SessionRecord | None:
        raise NotImplementedError

    def list_sessions(
        self,
        worker_name: str | None = None,
        limit: int | None = None,
    ) -> list[SessionRecord]:
        raise NotImplementedError

    def delete_session(self, session_id: str) -> None:
        raise NotImplementedError

    def add_messages(self, session_id: str, messages: list[Message]) -> None:
        raise NotImplementedError

    def get_messages(self, session_id: str) -> list[Message]:
        raise NotImplementedError
