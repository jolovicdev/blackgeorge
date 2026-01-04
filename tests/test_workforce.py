from typing import Any

from blackgeorge.adapters.base import BaseModelAdapter, ModelResponse
from blackgeorge.core.job import Job
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.desk import Desk
from blackgeorge.store.in_memory import InMemoryRunStore
from blackgeorge.tools import tool
from blackgeorge.worker import Worker
from blackgeorge.workforce import Workforce
from tests.utils import FakeAdapter


class FailingAdapter(BaseModelAdapter):
    def __init__(self) -> None:
        self.calls = 0

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
    ) -> ModelResponse:
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(content="alpha", tool_calls=[], usage={}, raw={})
        raise RuntimeError("context length exceeded")

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
    ) -> ModelResponse:
        return self.complete(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stream_options=stream_options,
        )


class BrokenCompletions:
    def create(self, *args, **kwargs) -> object:
        raise RuntimeError("boom")


class BrokenChat:
    def __init__(self) -> None:
        self.completions = BrokenCompletions()


class BrokenClient:
    def __init__(self) -> None:
        self.chat = BrokenChat()


def test_workforce_collaborate_default_reducer() -> None:
    responses = [
        ModelResponse(content="alpha", tool_calls=[], usage={}, raw={}),
        ModelResponse(content="beta", tool_calls=[], usage={}, raw={}),
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker_a = Worker(name="A", model="fake")
    worker_b = Worker(name="B", model="fake")
    workforce = Workforce([worker_a, worker_b], mode="collaborate", name="team")
    report = desk.run(workforce, Job(input="work"))
    assert report.status == "completed"
    assert "[A]" in report.content
    assert "[B]" in report.content


def test_workforce_collaborate_failure_propagates() -> None:
    desk = Desk(
        model="fake",
        adapter=FailingAdapter(),
        run_store=InMemoryRunStore(),
        respect_context_window=False,
    )
    worker_a = Worker(name="A", model="fake")
    worker_b = Worker(name="B", model="fake")
    workforce = Workforce([worker_a, worker_b], mode="collaborate", name="team")
    report = desk.run(workforce, Job(input="work"))
    assert report.status == "failed"
    assert "[A]" in report.content


def test_workforce_managed_manager_failure(monkeypatch) -> None:
    import blackgeorge.worker_runner as worker_runner_module

    monkeypatch.setattr(
        worker_runner_module.instructor_clients,
        "get",
        lambda model, async_client: BrokenClient(),
    )
    desk = Desk(model="fake", adapter=FakeAdapter([]), run_store=InMemoryRunStore())
    manager = Worker(name="Manager", model="fake")
    worker = Worker(name="Worker", model="fake")
    workforce = Workforce([worker], mode="managed", name="team", manager=manager)
    report = desk.run(workforce, Job(input="work"))
    assert report.status == "failed"


def test_unregister_workforce_blocks_resume() -> None:
    @tool(requires_confirmation=True)
    def risky(action: str) -> str:
        return f"ok:{action}"

    responses = [
        ModelResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="risky", arguments={"action": "go"})],
            usage={},
            raw={},
        )
    ]
    desk = Desk(model="fake", adapter=FakeAdapter(responses), run_store=InMemoryRunStore())
    worker = Worker(name="Worker", tools=[risky], model="fake")
    workforce = Workforce([worker], mode="collaborate", name="team")
    report = desk.run(workforce, Job(input="work"))
    assert report.status == "paused"
    desk.unregister_workforce(workforce)
    resumed = desk.resume(report, True)
    assert resumed.status == "failed"
    assert "Workforce not registered" in resumed.errors
