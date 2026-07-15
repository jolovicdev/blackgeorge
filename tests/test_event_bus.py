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


def test_eventbus_collects_async_handler_error_without_running_loop() -> None:
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

    bus.emit(event)

    errors = bus.get_errors()
    assert len(errors) == 1
    assert errors[0].event_type == "x"
    assert str(errors[0].handler_error) == "boom"


def test_eventbus_wildcard_subscription_receives_all_events() -> None:
    bus = EventBus()
    seen: list[str] = []

    def handler(event: Event) -> None:
        seen.append(event.type)

    bus.subscribe("*", handler)
    for event_type in ("x", "y"):
        bus.emit(
            Event(
                event_id=new_id(),
                type=event_type,
                timestamp=utc_now(),
                run_id="r",
                source="test",
                payload={},
            )
        )

    assert seen == ["x", "y"]


def test_eventbus_unsubscribe_handler() -> None:
    bus = EventBus()
    count = {"value": 0}

    def handler(event: Event) -> None:
        count["value"] += 1

    event = Event(
        event_id=new_id(),
        type="x",
        timestamp=utc_now(),
        run_id="r",
        source="test",
        payload={},
    )

    bus.subscribe("x", handler)
    bus.emit(event)
    bus.unsubscribe("x", handler)
    bus.emit(event)

    assert count["value"] == 1


def test_eventbus_collects_sync_handler_error_and_continues() -> None:
    bus = EventBus()
    seen: list[str] = []

    def failing_handler(event: Event) -> None:
        raise ValueError("boom")

    def working_handler(event: Event) -> None:
        seen.append(event.type)

    bus.subscribe("x", failing_handler)
    bus.subscribe("x", working_handler)
    event = Event(
        event_id=new_id(),
        type="x",
        timestamp=utc_now(),
        run_id="r",
        source="test",
        payload={},
    )

    bus.emit(event)

    assert seen == ["x"]
    errors = bus.get_errors()
    assert len(errors) == 1
    assert str(errors[0].handler_error) == "boom"


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


async def test_eventbus_aemit_wildcard_subscription_receives_event() -> None:
    bus = EventBus()
    seen: list[str] = []

    async def handler(event: Event) -> None:
        seen.append(event.type)

    bus.subscribe("*", handler)
    await bus.aemit(
        Event(
            event_id=new_id(),
            type="x",
            timestamp=utc_now(),
            run_id="r",
            source="test",
            payload={},
        )
    )

    assert seen == ["x"]


async def test_eventbus_aemit_collects_handler_error_and_continues() -> None:
    bus = EventBus()
    seen: list[str] = []

    async def failing_handler(event: Event) -> None:
        raise ValueError("boom")

    async def working_handler(event: Event) -> None:
        seen.append(event.type)

    bus.subscribe("x", failing_handler)
    bus.subscribe("x", working_handler)
    event = Event(
        event_id=new_id(),
        type="x",
        timestamp=utc_now(),
        run_id="r",
        source="test",
        payload={},
    )

    await bus.aemit(event)

    assert seen == ["x"]
    errors = bus.get_errors()
    assert len(errors) == 1
    assert str(errors[0].handler_error) == "boom"
