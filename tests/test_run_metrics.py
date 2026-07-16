import litellm
import pytest

from blackgeorge import Desk, Job, ScriptedAdapter, Worker, Workforce
from blackgeorge.adapters.base import ModelResponse
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.store.in_memory import InMemoryRunStore
from blackgeorge.tools import tool

MODEL = "deepseek/deepseek-v4-flash"
USAGE = {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}
INPUT_RATE = 1.4e-07
OUTPUT_RATE = 2.8e-07
TURN_COST = USAGE["prompt_tokens"] * INPUT_RATE + USAGE["completion_tokens"] * OUTPUT_RATE


@pytest.fixture(autouse=True)
def _pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        MODEL,
        {"input_cost_per_token": INPUT_RATE, "output_cost_per_token": OUTPUT_RATE},
    )


def _response(
    content: str | None = "ok", tool_calls: list[ToolCall] | None = None
) -> ModelResponse:
    return ModelResponse(content=content, tool_calls=tool_calls or [], usage=dict(USAGE), raw={})


def _desk(adapter: ScriptedAdapter, max_cost_usd: float | None = None) -> Desk:
    return Desk(
        model=MODEL,
        adapter=adapter,
        run_store=InMemoryRunStore(),
        max_cost_usd=max_cost_usd,
    )


@tool()
def ping() -> str:
    return "pong"


@tool(requires_confirmation=True)
def risky() -> str:
    return "confirmed"


def test_collaborate_report_has_run_totals() -> None:
    adapter = ScriptedAdapter([_response("a"), _response("b")])
    workforce = Workforce([Worker(name="W1"), Worker(name="W2")], mode="collaborate")
    report = _desk(adapter).run(workforce, Job(input="go"))
    assert report.status == "completed"
    assert report.metrics["cost_usd"] == pytest.approx(2 * TURN_COST)
    assert report.metrics["usage"]["prompt_tokens"] == 2000
    assert report.metrics["usage"]["completion_tokens"] == 1000
    assert report.metrics["usage"]["total_tokens"] == 3000


def test_sequential_collaborate_report_has_run_totals() -> None:
    adapter = ScriptedAdapter([_response("a"), _response("b")])
    workforce = Workforce(
        [Worker(name="W1", tools=[ping]), Worker(name="W2", tools=[ping])],
        mode="collaborate",
    )
    report = _desk(adapter).run(workforce, Job(input="go"))
    assert report.status == "completed"
    assert report.metrics["cost_usd"] == pytest.approx(2 * TURN_COST)
    assert report.metrics["usage"]["total_tokens"] == 3000


def test_managed_report_has_run_totals() -> None:
    adapter = ScriptedAdapter(
        [
            _response('{"worker": "Researcher", "reason": "best fit"}'),
            _response("answer"),
        ]
    )
    workforce = Workforce(
        [Worker(name="Manager"), Worker(name="Researcher")],
        mode="managed",
    )
    report = _desk(adapter).run(workforce, Job(input="go"))
    assert report.status == "completed"
    assert report.content == "answer"
    assert report.metrics["cost_usd"] == pytest.approx(TURN_COST)
    assert report.metrics["usage"]["total_tokens"] == 1500


def test_shared_budget_stops_second_worker() -> None:
    adapter = ScriptedAdapter([_response("a"), _response("unreachable")])
    workforce = Workforce(
        [Worker(name="W1", tools=[ping]), Worker(name="W2", tools=[ping])],
        mode="collaborate",
    )
    report = _desk(adapter, max_cost_usd=TURN_COST / 2).run(workforce, Job(input="go"))
    assert report.status == "failed"
    assert any("Cost budget exceeded" in error for error in report.errors)
    assert len(adapter.calls) == 1
    assert report.metrics["cost_usd"] == pytest.approx(TURN_COST)


def test_worker_usage_accumulates_across_turns() -> None:
    adapter = ScriptedAdapter(
        [
            _response(None, [ToolCall(id="1", name="ping", arguments={})]),
            _response("done"),
        ]
    )
    report = _desk(adapter).run(Worker(name="W", tools=[ping]), Job(input="go"))
    assert report.status == "completed"
    assert report.metrics["usage"]["total_tokens"] == 3000
    assert report.metrics["cost_usd"] == pytest.approx(2 * TURN_COST)


def test_resume_seeds_budget_from_state() -> None:
    adapter = ScriptedAdapter(
        [
            _response(None, [ToolCall(id="1", name="risky", arguments={})]),
            _response("unreachable"),
        ]
    )
    desk = _desk(adapter, max_cost_usd=TURN_COST / 2)
    report = desk.run(Worker(name="W", tools=[risky]), Job(input="go"))
    assert report.status == "paused"
    resumed = desk.resume(report, True)
    assert resumed.status == "failed"
    assert any("Cost budget exceeded" in error for error in resumed.errors)
    assert len(adapter.calls) == 1


def test_workforce_resume_preserves_totals() -> None:
    adapter = ScriptedAdapter(
        [
            _response("w1 done"),
            _response(None, [ToolCall(id="1", name="risky", arguments={})]),
            _response("w2 done"),
        ]
    )
    desk = _desk(adapter)
    workforce = Workforce(
        [Worker(name="W1", tools=[ping]), Worker(name="W2", tools=[risky])],
        mode="collaborate",
    )
    report = desk.run(workforce, Job(input="go"))
    assert report.status == "paused"
    resumed = desk.resume(report, True)
    assert resumed.status == "completed"
    assert resumed.metrics["cost_usd"] == pytest.approx(3 * TURN_COST)
    assert resumed.metrics["usage"]["total_tokens"] == 4500
    assert len(adapter.calls) == 3
