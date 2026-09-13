from typing import Literal

from pydantic import BaseModel

from blackgeorge import Desk, Job, Worker
from blackgeorge.store.in_memory import InMemoryRunStore
from blackgeorge.workflow import Loop, Step
from blackgeorge.workflow.context import WorkflowContext
from tests.live.conftest import LIVE_MODEL


class Answer(BaseModel):
    city: str
    country: str


def _desk(stream: bool = False, structured_stream_mode: Literal["off", "preview"] = "off") -> Desk:
    return Desk(
        model=LIVE_MODEL,
        run_store=InMemoryRunStore(),
        max_tokens=200,
        stream=stream,
        structured_stream_mode=structured_stream_mode,
    )


def _capture_tokens(desk: Desk) -> list[str]:
    tokens: list[str] = []
    desk.event_bus.subscribe("stream.token", lambda event: tokens.append(event.payload["token"]))
    return tokens


def test_loop_stop_predicate_sees_outputs() -> None:
    worker = Worker(name="counter", instructions="Reply with a single word.")
    seen: list[int] = []

    def stop(context: WorkflowContext) -> bool:
        seen.append(len(context.outputs))
        return len(context.outputs) >= 2

    report = (
        _desk().flow([Loop([Step(worker)], stop=stop, max_iterations=4)]).run(Job(input="ping"))
    )
    assert report.status == "completed"
    assert seen == [1, 2]


def test_structured_job_is_metered() -> None:
    report = _desk().run(
        Worker(name="geo"), Job(input="Capital of France?", response_schema=Answer)
    )
    assert report.status == "completed"
    assert report.data == Answer(city="Paris", country="France")
    assert report.metrics["usage"]["total_tokens"] > 0
    assert report.metrics["cost_usd"] > 0
    assert any(event.type == "llm.completed" for event in report.events)


def test_structured_preview_streams_json() -> None:
    desk = _desk(stream=True, structured_stream_mode="preview")
    tokens = _capture_tokens(desk)
    report = desk.run(Worker(name="geo"), Job(input="Capital of France?", response_schema=Answer))
    assert report.status == "completed"
    assert report.data == Answer(city="Paris", country="France")
    assert Answer.model_validate_json("".join(tokens)) == report.data
    assert sum(1 for event in report.events if event.type == "llm.started") == 1


def test_flow_report_carries_usage_totals() -> None:
    a = Worker(name="a", instructions="Reply with one word.")
    b = Worker(name="b", instructions="Reply with one word.")
    report = _desk().flow([Step(a), Step(b)]).run(Job(input="Say hi"))
    assert report.status == "completed"
    assert report.metrics["usage"]["total_tokens"] > 0
    assert report.metrics["cost_usd"] > 0


def test_flow_applies_desk_preview_mode() -> None:
    desk = _desk(stream=True, structured_stream_mode="preview")
    tokens = _capture_tokens(desk)
    report = desk.flow([Step(Worker(name="geo"))]).run(
        Job(input="Capital of France?", response_schema=Answer)
    )
    assert report.status == "completed"
    assert tokens
