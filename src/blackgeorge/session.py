import asyncio
import queue
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

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
        session_id = session_id or new_id()
        session = cls.open(
            session_id=session_id,
            worker=worker,
            desk=desk,
            metadata=metadata,
            create=True,
        )
        if session is None:
            raise ValueError(f"Session '{session_id}' belongs to another worker")
        return session

    @classmethod
    def load(
        cls,
        *,
        session_id: str,
        worker: Worker,
        desk: Desk,
    ) -> "WorkerSession | None":
        return cls.open(
            session_id=session_id,
            worker=worker,
            desk=desk,
            create=False,
        )

    @classmethod
    def open(
        cls,
        *,
        session_id: str,
        worker: Worker,
        desk: Desk,
        metadata: dict[str, Any] | None = None,
        create: bool = True,
    ) -> "WorkerSession | None":
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        store = SQLiteSessionStore(desk.db_path)
        record = store.get_session(session_id)
        if record is not None and record.worker_name != worker.name:
            store.close()
            return None
        if record is None:
            if not create:
                store.close()
                return None
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

    def run(self, user_input: Any, *, stream: bool = False, **job_kwargs: Any) -> Report:
        job = self._build_job(user_input, job_kwargs)
        return self._persist_report(self.desk.run(self.worker, job, stream=stream))

    async def arun(self, user_input: Any, *, stream: bool = False, **job_kwargs: Any) -> Report:
        job = self._build_job(user_input, job_kwargs)
        return self._persist_report(await self.desk.arun(self.worker, job, stream=stream))

    def resume(
        self,
        report: Report,
        decision_or_input: Any,
        *,
        stream: bool | None = None,
    ) -> Report:
        return self._persist_report(self.desk.resume(report, decision_or_input, stream=stream))

    async def aresume(
        self,
        report: Report,
        decision_or_input: Any,
        *,
        stream: bool | None = None,
    ) -> Report:
        return self._persist_report(
            await self.desk.aresume(report, decision_or_input, stream=stream)
        )

    def history(self) -> list[Message]:
        return self.store.get_messages(self.session_id)

    def stream_run(self, user_input: Any, **job_kwargs: Any) -> Iterator[Event]:
        job = self._build_job(user_input, job_kwargs)
        run_id = new_id()
        stream_done = object()
        events: queue.Queue[Event | object] = queue.Queue()
        report: Report | None = None
        failure: Exception | None = None
        completed = False

        def handler(event: Event) -> None:
            if event.run_id == run_id:
                events.put(event)

        def run_job() -> None:
            nonlocal report, failure
            try:
                report = self.desk.run(self.worker, job, stream=True, run_id=run_id)
            except Exception as exc:
                failure = exc
            finally:
                events.put(stream_done)

        self.desk.event_bus.subscribe("*", handler)
        thread = threading.Thread(target=run_job, daemon=True)
        thread.start()

        try:
            while True:
                item = events.get()
                if item is stream_done:
                    completed = True
                    break
                yield cast(Event, item)
        finally:
            self.desk.event_bus.unsubscribe("*", handler)
            thread.join()
            if report is not None:
                self._persist_report(report)

        if not completed:
            return
        if failure is not None:
            raise failure
        if report is None:
            raise RuntimeError("Run did not produce a report")

    async def astream_run(self, user_input: Any, **job_kwargs: Any) -> AsyncIterator[Event]:
        job = self._build_job(user_input, job_kwargs)
        run_id = new_id()
        stream_done = object()
        events: asyncio.Queue[Event | object] = asyncio.Queue()
        completed = False

        def handler(event: Event) -> None:
            if event.run_id == run_id:
                events.put_nowait(event)

        self.desk.event_bus.subscribe("*", handler)
        run_task: asyncio.Task[Report] = asyncio.create_task(
            self.desk.arun(self.worker, job, stream=True, run_id=run_id)
        )
        run_task.add_done_callback(lambda _task: events.put_nowait(stream_done))

        try:
            while True:
                item = await events.get()
                if item is stream_done:
                    completed = True
                    break
                yield cast(Event, item)
        finally:
            self.desk.event_bus.unsubscribe("*", handler)
            if not run_task.done():
                await run_task
            if run_task.done() and not run_task.cancelled() and run_task.exception() is None:
                self._persist_report(run_task.result())

        if not completed:
            return
        if run_task.cancelled():
            raise asyncio.CancelledError
        exception = run_task.exception()
        if exception is not None:
            raise exception

    def close(self) -> None:
        self.store.close()

    def delete(self) -> None:
        try:
            self.store.delete_session(self.session_id)
        finally:
            self.store.close()

    def __enter__(self) -> "WorkerSession":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def _build_job(self, user_input: Any, job_kwargs: dict[str, Any]) -> Job:
        messages = self.store.get_messages(self.session_id)
        return Job(
            input=user_input,
            initial_messages=messages if messages else None,
            **job_kwargs,
        )

    def _persist_report(self, report: Report) -> Report:
        if report.status == "completed":
            new_messages = self._extract_conversation_messages(report)
            self.store.replace_messages(self.session_id, new_messages)
            self.store.update_session(self.session_id)
        return report

    def _extract_conversation_messages(self, report: Report) -> list[Message]:
        messages: list[Message] = []

        for message in report.messages:
            is_summary = message.role == "system" and message.metadata.get("summary") is True
            if (
                is_summary
                or message.role in ("user", "assistant")
                or (message.role == "tool" and message.tool_call_id)
            ):
                msg_dict = message.model_dump()
                if message.role == "assistant" and not message.tool_calls:
                    msg_dict.pop("reasoning_content", None)
                    msg_dict.pop("thinking_blocks", None)
                messages.append(Message(**msg_dict))

        return messages
