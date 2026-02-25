import asyncio
from typing import Any

import pytest

from blackgeorge.adapters.base import BaseModelAdapter, ModelResponse
from blackgeorge.core.job import Job
from blackgeorge.desk import Desk
from blackgeorge.worker import Worker
from blackgeorge.workforce import Workforce


class SlowAdapter(BaseModelAdapter):
    def __init__(self, delay: float = 0.5, response: str = "worker result") -> None:
        self.delay = delay
        self.response = response
        self.call_count = 0
        self._responses: list[ModelResponse] = []

    def _get_response(self) -> ModelResponse:
        self.call_count += 1
        if self._responses:
            return self._responses.pop(0)
        return ModelResponse(content=self.response, tool_calls=[], usage={}, raw={})

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ModelResponse:
        return self._get_response()

    async def acomplete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ModelResponse:
        await asyncio.sleep(self.delay)
        return self._get_response()


class FailingOnSecondCallAdapter(BaseModelAdapter):
    def __init__(self, fail_after: int = 1, response: str = "worker result") -> None:
        self.fail_after = fail_after
        self.response = response
        self.call_count = 0

    def _get_response(self) -> ModelResponse:
        self.call_count += 1
        if self.call_count >= self.fail_after:
            raise RuntimeError("Simulated failure")
        return ModelResponse(content=self.response, tool_calls=[], usage={}, raw={})

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ModelResponse:
        return self._get_response()

    async def acomplete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ModelResponse:
        return self._get_response()


class ImmediateAdapter(BaseModelAdapter):
    def __init__(self, response: str = "worker result") -> None:
        self.response = response
        self.call_count = 0

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ModelResponse:
        self.call_count += 1
        return ModelResponse(content=self.response, tool_calls=[], usage={}, raw={})

    async def acomplete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ModelResponse:
        self.call_count += 1
        return ModelResponse(content=self.response, tool_calls=[], usage={}, raw={})


async def test_parallel_execution_completes_all_workers():
    adapter = SlowAdapter(delay=0.05, response="worker result")

    workers = [Worker(name=f"worker-{i}") for i in range(3)]
    workforce = Workforce(workers=workers, mode="collaborate")
    desk = Desk(model="test-model", adapter=adapter)

    job = Job(input="test")
    report = await desk.arun(workforce, job)

    assert report.status == "completed"
    assert adapter.call_count == 3


async def test_parallel_execution_with_worker_failure():
    adapter = FailingOnSecondCallAdapter(fail_after=2, response="worker result")

    workers = [Worker(name=f"worker-{i}") for i in range(3)]
    workforce = Workforce(workers=workers, mode="collaborate")
    desk = Desk(model="test-model", adapter=adapter)

    job = Job(input="test")

    report = await desk.arun(workforce, job)
    assert report.status == "failed"
    assert len(report.errors) > 0


async def test_external_cancellation_propagates():
    adapter = SlowAdapter(delay=1.0, response="worker result")

    workers = [Worker(name=f"worker-{i}") for i in range(3)]
    workforce = Workforce(workers=workers, mode="collaborate")
    desk = Desk(model="test-model", adapter=adapter)

    job = Job(input="test")

    async def run_with_timeout() -> None:
        await asyncio.wait_for(desk.arun(workforce, job), timeout=0.2)

    with pytest.raises(asyncio.TimeoutError):
        await run_with_timeout()


async def test_cancellation_cleanup():
    adapter = SlowAdapter(delay=0.5, response="worker result")

    workers = [Worker(name=f"worker-{i}") for i in range(3)]
    workforce = Workforce(workers=workers, mode="collaborate")
    desk = Desk(model="test-model", adapter=adapter)

    job = Job(input="test")

    task = asyncio.create_task(desk.arun(workforce, job))
    await asyncio.sleep(0.1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


async def test_parallel_execution_respects_max_iterations():
    adapter = ImmediateAdapter(response="worker result")

    workers = [Worker(name=f"worker-{i}") for i in range(3)]
    workforce = Workforce(workers=workers, mode="collaborate")
    desk = Desk(model="test-model", adapter=adapter, max_iterations=1)

    job = Job(input="test")
    report = await desk.arun(workforce, job)

    assert report.status == "completed"
