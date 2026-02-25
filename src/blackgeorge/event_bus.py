import asyncio
from collections.abc import Awaitable, Callable
from inspect import iscoroutinefunction
from typing import Any

from blackgeorge.core.event import Event
from blackgeorge.exceptions import EventHandlerError
from blackgeorge.logging import get_logger

EventHandler = Callable[[Event], Any]


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}
        self._pending: dict[asyncio.Future[Any], str | None] = {}
        self._errors: list[EventHandlerError] = []
        self._logger = get_logger("blackgeorge.event_bus")

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        handlers = self._handlers.get(event_type)
        if handlers is None:
            return
        self._handlers[event_type] = [
            registered for registered in handlers if registered != handler
        ]
        if not self._handlers[event_type]:
            self._handlers.pop(event_type, None)

    def emit(self, event: Event) -> None:
        handlers = list(self._handlers.get(event.type, []))
        handlers.extend(self._handlers.get("*", []))
        for handler in handlers:
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
            if isinstance(exc, Exception):
                error = EventHandlerError(event_type or "unknown", exc)
            else:
                error = EventHandlerError(event_type or "unknown", Exception(str(exc)))
            self._errors.append(error)
            payload = {"error": str(exc), "error_type": type(exc).__name__}
            if event_type is not None:
                payload["event_type"] = event_type
            self._logger.error("event handler failed", **payload)

    async def _await(self, awaitable: Awaitable[Any]) -> Any:
        return await awaitable

    async def aemit(self, event: Event) -> None:
        handlers = list(self._handlers.get(event.type, []))
        handlers.extend(self._handlers.get("*", []))
        for handler in handlers:
            if iscoroutinefunction(handler):
                await handler(event)
            else:
                result = handler(event)
                if isinstance(result, Awaitable):
                    await result

    def get_errors(self) -> list[EventHandlerError]:
        return list(self._errors)

    def clear_errors(self) -> None:
        self._errors.clear()

    async def await_pending(self, *, raise_on_error: bool = False) -> list[EventHandlerError]:
        loop = asyncio.get_running_loop()
        while True:
            pending = [
                future
                for future in list(self._pending)
                if not future.done() and future.get_loop() is loop
            ]
            if not pending:
                break
            await asyncio.gather(*pending, return_exceptions=True)
        if raise_on_error and self._errors:
            raise self._errors[0]
        return self.get_errors()
