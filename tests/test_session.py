import sqlite3
import tempfile

import pytest

from blackgeorge.adapters.base import ModelResponse
from blackgeorge.core.message import Message
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.desk import Desk
from blackgeorge.store.in_memory import InMemoryRunStore
from blackgeorge.store.in_memory_session_store import InMemorySessionStore
from blackgeorge.store.session_store import SessionRecord
from blackgeorge.store.sqlite_session_store import SQLiteSessionStore
from blackgeorge.tools import tool
from blackgeorge.worker import Worker
from tests.utils import FakeAdapter


def test_session_multi_turn_conversation() -> None:
    responses = [
        ModelResponse(content="Hello!", tool_calls=[], usage={}, raw={}),
        ModelResponse(content="You said: hi again", tool_calls=[], usage={}, raw={}),
    ]
    adapter = FakeAdapter(responses)
    worker = Worker(name="Assistant", model="fake")
    desk = Desk(model="fake", adapter=adapter, run_store=InMemoryRunStore())

    session = desk.session(worker)

    report1 = session.run("hi")
    assert report1.status == "completed"
    assert report1.content == "Hello!"

    report2 = session.run("hi again")
    assert report2.status == "completed"
    assert report2.content == "You said: hi again"

    history = session.history()
    assert len(history) >= 4
    assert any(m.role == "user" and "hi" in m.content for m in history)
    assert any(m.role == "assistant" and "Hello" in m.content for m in history)


def test_session_memory_reuse() -> None:
    responses = [
        ModelResponse(content="I remember you", tool_calls=[], usage={}, raw={}),
        ModelResponse(content="You said: second message", tool_calls=[], usage={}, raw={}),
    ]
    adapter = FakeAdapter(responses)
    worker = Worker(name="Assistant", model="fake")
    store = InMemorySessionStore()
    desk = Desk(model="fake", adapter=adapter, run_store=InMemoryRunStore())

    session = desk.session(worker)
    session.store = store

    report1 = session.run("first")
    assert report1.status == "completed"

    report2 = session.run("second")
    assert report2.status == "completed"

    history = session.history()
    assert len(history) >= 4


def test_session_persistence_sqlite() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        responses = [
            ModelResponse(content="First response", tool_calls=[], usage={}, raw={}),
            ModelResponse(content="Second response", tool_calls=[], usage={}, raw={}),
            ModelResponse(content="Third response", tool_calls=[], usage={}, raw={}),
        ]

        worker = Worker(name="Assistant", model="fake")

        desk1 = Desk(
            model="fake",
            adapter=FakeAdapter(responses),
            run_store=InMemoryRunStore(),
            storage_dir=str(tmpdir),
        )

        session1 = desk1.session(worker)
        session_id = session1.session_id

        report1 = session1.run("hello")
        assert report1.status == "completed"
        assert report1.content == "First response"

        del session1
        del desk1

        desk2 = Desk(
            model="fake",
            adapter=FakeAdapter([ModelResponse(content="Fourth", tool_calls=[], usage={}, raw={})]),
            run_store=InMemoryRunStore(),
            storage_dir=str(tmpdir),
        )

        session2 = desk2.session(worker, session_id=session_id)
        assert session2 is not None

        history = session2.history()
        assert len(history) >= 2

        report2 = session2.run("second message")
        assert report2.status == "completed"
        assert report2.content == "Fourth"


def test_session_with_tools() -> None:
    @tool()
    def echo(text: str) -> str:
        return f"echoed: {text}"

    responses = [
        ModelResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "test"})],
            usage={},
            raw={},
        ),
        ModelResponse(content="Done", tool_calls=[], usage={}, raw={}),
    ]

    worker = Worker(name="Assistant", model="fake", tools=[echo])
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    session = desk.session(worker)

    report = session.run("use echo")
    assert report.status == "completed"
    assert len(report.tool_calls) == 1


def test_session_preserves_reasoning_content_for_tool_calls() -> None:
    @tool()
    def echo(text: str) -> str:
        return f"echoed: {text}"

    responses = [
        ModelResponse(
            content=None,
            reasoning_content="tool reasoning",
            tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "test"})],
            usage={},
            raw={},
        ),
        ModelResponse(content="Done", tool_calls=[], usage={}, raw={}),
    ]

    worker = Worker(name="Assistant", model="fake", tools=[echo])
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    session = desk.session(worker)

    report = session.run("use echo")
    assert report.status == "completed"

    history = session.history()
    tool_call_messages = [msg for msg in history if msg.role == "assistant" and msg.tool_calls]
    assert tool_call_messages
    assert tool_call_messages[0].reasoning_content == "tool reasoning"

    plain_messages = [msg for msg in history if msg.role == "assistant" and not msg.tool_calls]
    assert plain_messages
    assert all(msg.reasoning_content is None for msg in plain_messages)


def test_session_close() -> None:
    worker = Worker(name="Assistant", model="fake")
    desk = Desk(
        model="fake",
        adapter=FakeAdapter([ModelResponse(content="hi", tool_calls=[], usage={}, raw={})]),
        run_store=InMemoryRunStore(),
    )

    session = desk.session(worker)
    session_id = session.session_id
    session.run("hello")
    session.close()

    loaded = desk.session(worker, session_id=session_id)
    assert loaded is None


def test_session_list() -> None:
    worker1 = Worker(name="Assistant1", model="fake")
    worker2 = Worker(name="Assistant2", model="fake")

    with tempfile.TemporaryDirectory() as tmpdir:
        desk = Desk(
            model="fake",
            adapter=FakeAdapter([ModelResponse(content="hi", tool_calls=[], usage={}, raw={})]),
            run_store=InMemoryRunStore(),
            storage_dir=str(tmpdir),
        )
        store = SQLiteSessionStore(desk.db_path)

        desk.session(worker1)
        desk.session(worker2)
        desk.session(worker1)

        all_sessions = store.list_sessions()
        assert len(all_sessions) == 3

        worker1_sessions = store.list_sessions(worker_name="Assistant1")
        assert len(worker1_sessions) == 2


def test_session_worker_mismatch() -> None:
    worker1 = Worker(name="Assistant1", model="fake")
    worker2 = Worker(name="Assistant2", model="fake")
    desk = Desk(
        model="fake",
        adapter=FakeAdapter([ModelResponse(content="hi", tool_calls=[], usage={}, raw={})]),
        run_store=InMemoryRunStore(),
    )

    session = desk.session(worker1)
    loaded = desk.session(worker2, session_id=session.session_id)
    assert loaded is None


def test_session_initial_messages_job() -> None:
    from blackgeorge.core.job import Job

    messages = [
        Message(role="user", content="previous message"),
        Message(role="assistant", content="previous response"),
    ]

    job = Job(input="new", initial_messages=messages)

    assert job.initial_messages == messages
    assert len(job.initial_messages) == 2


@pytest.mark.asyncio
async def test_session_async() -> None:
    responses = [
        ModelResponse(content="Async hello!", tool_calls=[], usage={}, raw={}),
        ModelResponse(content="Async response 2", tool_calls=[], usage={}, raw={}),
    ]
    adapter = FakeAdapter(responses)
    worker = Worker(name="Assistant", model="fake")
    desk = Desk(model="fake", adapter=adapter, run_store=InMemoryRunStore())

    session = desk.session(worker)

    report1 = await session.arun("async hi")
    assert report1.status == "completed"
    assert report1.content == "Async hello!"

    report2 = await session.arun("async hi again")
    assert report2.status == "completed"
    assert report2.content == "Async response 2"

    history = session.history()
    assert len(history) >= 4


def test_in_memory_session_store() -> None:
    store = InMemorySessionStore()

    store.create_session("test-1", "worker1", {"key": "value"})

    record = store.get_session("test-1")
    assert record is not None
    assert record.session_id == "test-1"
    assert record.worker_name == "worker1"
    assert record.metadata == {"key": "value"}

    store.add_messages("test-1", [Message(role="user", content="test")])
    messages = store.get_messages("test-1")
    assert len(messages) == 1
    assert messages[0].content == "test"

    store.update_session("test-1", {"key": "updated"})
    updated = store.get_session("test-1")
    assert updated is not None
    assert updated.metadata == {"key": "updated"}

    sessions = store.list_sessions(worker_name="worker1")
    assert len(sessions) == 1

    store.delete_session("test-1")
    assert store.get_session("test-1") is None


def test_session_record_frozen() -> None:
    record = SessionRecord(
        session_id="test",
        worker_name="worker",
        created_at=None,
        updated_at=None,
        metadata={},
    )

    assert record.session_id == "test"
    assert record.worker_name == "worker"


def test_sqlite_session_message_order(tmp_path) -> None:
    store = SQLiteSessionStore(str(tmp_path / "sessions.db"))
    store.create_session("s1", "worker")
    store.add_messages(
        "s1",
        [
            Message(role="user", content="a"),
            Message(role="assistant", content="b"),
            Message(role="user", content="c"),
        ],
    )
    messages = store.get_messages("s1")
    assert [message.content for message in messages] == ["a", "b", "c"]
    with sqlite3.connect(str(tmp_path / "sessions.db")) as conn:
        row = conn.execute("SELECT count(*) FROM session_messages").fetchone()
    assert row is not None and row[0] == 3
