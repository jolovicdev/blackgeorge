import asyncio
import concurrent.futures
import threading
from collections.abc import Coroutine
from typing import Any


def run_coroutine_in_thread[T](coro: Coroutine[Any, Any, T]) -> T:
    future: concurrent.futures.Future[T] = concurrent.futures.Future()

    def runner() -> None:
        try:
            result = asyncio.run(coro)
            future.set_result(result)
        except BaseException as exc:
            future.set_exception(exc)

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()
    return future.result()


def run_coroutine_sync[T](coro: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return run_coroutine_in_thread(coro)
