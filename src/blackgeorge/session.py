from collections.abc import AsyncIterator, Iterator
from typing import Any

from pydantic import BaseModel, ConfigDict

from blackgeorge.core.event import Event
from blackgeorge.core.job import Job
from blackgeorge.core.message import Message
from blackgeorge.core.report import Report
from blackgeorge.desk import Desk
from blackgeorge.store.session_store import SessionStore
from blackgeorge.store.sqlite_session_store import SQLiteSessionStore
from blackgeorge.utils import new_id
from blackgeorge.worker import Worker


class WorkerSession(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    worker: Worker
    desk: Desk
    store: SessionStore

    @classmethod
    def start(
        cls,
        *,
        worker: Worker,
        desk: Desk,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "WorkerSession":
        store = SQLiteSessionStore(desk.db_path)
        session_id = session_id or new_id()

        store.create_session(
            session_id=session_id,
            worker_name=worker.name,
            metadata=metadata or {},
        )

        return cls(
            session_id=session_id,
            worker=worker,
            desk=desk,
            store=store,
        )

    @classmethod
    def resume(
        cls,
        *,
        session_id: str,
        worker: Worker,
        desk: Desk,
    ) -> "WorkerSession | None":
        store = SQLiteSessionStore(desk.db_path)

        record = store.get_session(session_id)
        if record is None or record.worker_name != worker.name:
            return None

        return cls(
            session_id=session_id,
            worker=worker,
            desk=desk,
            store=store,
        )

    def run(self, user_input: Any, *, stream: bool = False, **job_kwargs: Any) -> Report:
        messages = self.store.get_messages(self.session_id)
        initial_count = len(messages) if messages else 0
        job = Job(
            input=user_input,
            initial_messages=messages if messages else None,
            **job_kwargs,
        )

        report = self.desk.run(self.worker, job, stream=stream)

        if report.status == "completed":
            new_messages = self._extract_conversation_messages(report)
            self.store.add_messages(self.session_id, new_messages[initial_count:])
            self.store.update_session(self.session_id)

        return report

    async def arun(self, user_input: Any, *, stream: bool = False, **job_kwargs: Any) -> Report:
        messages = self.store.get_messages(self.session_id)
        initial_count = len(messages) if messages else 0
        job = Job(
            input=user_input,
            initial_messages=messages if messages else None,
            **job_kwargs,
        )

        report = await self.desk.arun(self.worker, job, stream=stream)

        if report.status == "completed":
            new_messages = self._extract_conversation_messages(report)
            self.store.add_messages(self.session_id, new_messages[initial_count:])
            self.store.update_session(self.session_id)

        return report

    def history(self) -> list[Message]:
        return self.store.get_messages(self.session_id)

    def stream_run(self, user_input: Any, **job_kwargs: Any) -> Iterator[Event]:
        """Run worker and yield events from the completed report.

        Note: Events are yielded after the run completes, not in real-time.
        For true streaming during execution, use the event bus directly.
        """
        messages = self.store.get_messages(self.session_id)
        initial_count = len(messages) if messages else 0
        job = Job(
            input=user_input,
            initial_messages=messages if messages else None,
            **job_kwargs,
        )

        report = self.desk.run(self.worker, job, stream=True)

        if report.status == "completed":
            new_messages = self._extract_conversation_messages(report)
            self.store.add_messages(self.session_id, new_messages[initial_count:])
            self.store.update_session(self.session_id)

        yield from report.events

    async def astream_run(self, user_input: Any, **job_kwargs: Any) -> AsyncIterator[Event]:
        """Run worker and yield events from the completed report.

        Note: Events are yielded after the run completes, not in real-time.
        For true streaming during execution, use the event bus directly.
        """
        messages = self.store.get_messages(self.session_id)
        initial_count = len(messages) if messages else 0
        job = Job(
            input=user_input,
            initial_messages=messages if messages else None,
            **job_kwargs,
        )

        report = await self.desk.arun(self.worker, job, stream=True)

        if report.status == "completed":
            new_messages = self._extract_conversation_messages(report)
            self.store.add_messages(self.session_id, new_messages[initial_count:])
            self.store.update_session(self.session_id)

        for event in report.events:
            yield event

    def close(self) -> None:
        self.store.delete_session(self.session_id)

    def _extract_conversation_messages(self, report: Report) -> list[Message]:
        messages: list[Message] = []

        for message in report.messages:
            if message.role in ("user", "assistant") or (
                message.role == "tool" and message.tool_call_id
            ):
                msg_dict = message.model_dump()
                if message.role == "assistant":
                    msg_dict.pop("reasoning_content", None)
                messages.append(Message(**msg_dict))

        return messages
