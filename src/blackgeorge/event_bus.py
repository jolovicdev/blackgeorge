import asyncio
from collections.abc import Awaitable, Callable
from inspect import iscoroutinefunction
from typing import Any

from blackgeorge.core.event import Event
from blackgeorge.logging import get_logger

EventHandler = Callable[[Event], Any]


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}
        self._pending: dict[asyncio.Future[Any], str | None] = {}
        self._logger = get_logger("blackgeorge.event_bus")

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def emit(self, event: Event) -> None:
        for handler in self._handlers.get(event.type, []):
            result = handler(event)
            if iscoroutinefunction(handler) or isinstance(result, Awaitable):
                self._run_async(result, event.type)

    def _run_async(self, awaitable: Awaitable[Any], event_type: str | None) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            if isinstance(awaitable, asyncio.Future) and awaitable.get_loop().is_running():
                self._track_task(awaitable, event_type)
                return
            asyncio.run(self._await(awaitable))
            return
        task = asyncio.ensure_future(awaitable, loop=loop)
        self._track_task(task, event_type)

    def _track_task(self, task: asyncio.Future[Any], event_type: str | None) -> None:
        self._pending[task] = event_type
        task.add_done_callback(self._handle_task)

    def _handle_task(self, task: asyncio.Future[Any]) -> None:
        event_type = self._pending.pop(task, None)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            payload = {"error": str(exc), "error_type": type(exc).__name__}
            if event_type is not None:
                payload["event_type"] = event_type
            self._logger.error("event handler failed", **payload)

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
