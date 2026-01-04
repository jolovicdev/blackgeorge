from collections.abc import Awaitable, Callable
from inspect import iscoroutinefunction
from typing import Any

from blackgeorge.core.event import Event

EventHandler = Callable[[Event], Any]


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def emit(self, event: Event) -> None:
        for handler in self._handlers.get(event.type, []):
            handler(event)

    async def aemit(self, event: Event) -> None:
        for handler in self._handlers.get(event.type, []):
            if iscoroutinefunction(handler):
                await handler(event)
            else:
                result = handler(event)
                if isinstance(result, Awaitable):
                    await result
