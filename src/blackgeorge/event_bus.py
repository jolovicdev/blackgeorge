import asyncio
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
            result = handler(event)
            if iscoroutinefunction(handler) or isinstance(result, Awaitable):
                self._run_async(result)

    def _run_async(self, awaitable: Awaitable[Any]) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            if isinstance(awaitable, asyncio.Future) and awaitable.get_loop().is_running():
                return
            asyncio.run(self._await(awaitable))
            return
        asyncio.ensure_future(awaitable, loop=loop)

    async def _await(self, awaitable: Awaitable[Any]) -> Any:
        return await awaitable

    async def aemit(self, event: Event) -> None:
        for handler in self._handlers.get(event.type, []):
            if iscoroutinefunction(handler):
                await handler(event)
            else:
                result = handler(event)
                if isinstance(result, Awaitable):
                    await result
