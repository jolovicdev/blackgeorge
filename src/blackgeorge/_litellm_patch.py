import asyncio
from typing import Any


def _apply_litellm_patches() -> None:
    try:
        import litellm
        from litellm.litellm_core_utils import logging_worker
    except ImportError:
        return

    litellm.suppress_debug_info = True

    original_ensure_queue = logging_worker.LoggingWorker._ensure_queue

    def patched_ensure_queue(self: Any) -> None:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        if self._queue is not None and self._bound_loop is not current_loop:
            while not self._queue.empty():
                try:
                    task = self._queue.get_nowait()
                    if task and "coroutine" in task:
                        task["coroutine"].close()
                except asyncio.QueueEmpty:
                    break

        original_ensure_queue(self)

    logging_worker.LoggingWorker._ensure_queue = patched_ensure_queue  # type: ignore[method-assign]


_apply_litellm_patches()
