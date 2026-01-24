from typing import Any

from blackgeorge.core.message import Message
from blackgeorge.store.session_store import SessionRecord, SessionStore
from blackgeorge.utils import utc_now


class InMemorySessionStore(SessionStore):
    def __init__(self) -> None:
        self._sessions: dict[str, SessionRecord] = {}
        self._messages: dict[str, list[Message]] = {}

    def create_session(
        self,
        session_id: str,
        worker_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = utc_now()
        self._sessions[session_id] = SessionRecord(
            session_id=session_id,
            worker_name=worker_name,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )
        self._messages[session_id] = []

    def update_session(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        record = self._sessions.get(session_id)
        if record is None:
            return
        updated_metadata = metadata if metadata is not None else record.metadata
        self._sessions[session_id] = SessionRecord(
            session_id=record.session_id,
            worker_name=record.worker_name,
            metadata=updated_metadata,
            created_at=record.created_at,
            updated_at=utc_now(),
        )

    def get_session(self, session_id: str) -> SessionRecord | None:
        return self._sessions.get(session_id)

    def list_sessions(
        self,
        worker_name: str | None = None,
        limit: int | None = None,
    ) -> list[SessionRecord]:
        sessions = list(self._sessions.values())
        if worker_name:
            sessions = [s for s in sessions if s.worker_name == worker_name]
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        if limit:
            sessions = sessions[:limit]
        return sessions

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._messages.pop(session_id, None)

    def add_messages(self, session_id: str, messages: list[Message]) -> None:
        if session_id not in self._messages:
            self._messages[session_id] = []
        self._messages[session_id].extend(messages)

    def get_messages(self, session_id: str) -> list[Message]:
        return list(self._messages.get(session_id, []))
