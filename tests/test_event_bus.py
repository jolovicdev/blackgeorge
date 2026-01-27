import asyncio
import json
import logging

from blackgeorge.core.event import Event
from blackgeorge.event_bus import EventBus
from blackgeorge.utils import new_id, utc_now


def test_eventbus_runs_async_handler() -> None:
    bus = EventBus()
    ran = {"value": False}

    async def handler(event: Event) -> None:
        ran["value"] = True

    bus.subscribe("x", handler)
    event = Event(
        event_id=new_id(),
        type="x",
        timestamp=utc_now(),
        run_id="r",
        source="test",
        payload={},
    )
    bus.emit(event)
    assert ran["value"] is True


async def test_eventbus_accepts_task_return() -> None:
    bus = EventBus()
    ran = {"value": False}

    async def do_work() -> None:
        ran["value"] = True

    def handler(event: Event) -> asyncio.Task[None]:
        return asyncio.create_task(do_work())

    bus.subscribe("x", handler)
    event = Event(
        event_id=new_id(),
        type="x",
        timestamp=utc_now(),
        run_id="r",
        source="test",
        payload={},
    )
    bus.emit(event)
    await asyncio.sleep(0)
    assert ran["value"] is True


async def test_eventbus_logs_async_handler_errors(caplog) -> None:
    bus = EventBus()

    async def handler(event: Event) -> None:
        raise ValueError("boom")

    bus.subscribe("x", handler)
    event = Event(
        event_id=new_id(),
        type="x",
        timestamp=utc_now(),
        run_id="r",
        source="test",
        payload={},
    )
    caplog.set_level(logging.ERROR, logger="blackgeorge.event_bus")
    bus.emit(event)
    await asyncio.sleep(0.01)
    messages = [record.getMessage() for record in caplog.records]
    assert messages
    payload = json.loads(messages[-1])
    assert payload["message"] == "event handler failed"
    assert payload["error_type"] == "ValueError"
