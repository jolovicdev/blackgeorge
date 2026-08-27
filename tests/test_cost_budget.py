import litellm
import pytest

from blackgeorge import Desk, Job, ScriptedAdapter, Worker
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
    content: str | None = None, tool_calls: list[ToolCall] | None = None
) -> ModelResponse:
    return ModelResponse(content=content, tool_calls=tool_calls or [], usage=dict(USAGE), raw={})


def _desk(adapter: ScriptedAdapter, max_cost_usd: float | None = None) -> Desk:
    return Desk(
        model=MODEL,
        adapter=adapter,
        run_store=InMemoryRunStore(),
        max_cost_usd=max_cost_usd,
    )


def test_cost_accumulates_in_metrics() -> None:
    adapter = ScriptedAdapter([_response("done")])
    report = _desk(adapter).run(Worker(name="W"), Job(input="hi"))
    assert report.status == "completed"
    assert report.metrics["usage"] == USAGE
    assert report.metrics["cost_usd"] == pytest.approx(TURN_COST)


def test_cost_budget_not_exceeded_completes() -> None:
    adapter = ScriptedAdapter([_response("done")])
    report = _desk(adapter, max_cost_usd=0.01).run(Worker(name="W"), Job(input="hi"))
    assert report.status == "completed"
    assert report.metrics["cost_usd"] == pytest.approx(TURN_COST)


def test_cost_budget_exceeded_fails_before_next_call() -> None:
    @tool()
    def ping() -> str:
        return "pong"

    adapter = ScriptedAdapter(
        [
            _response(None, [ToolCall(id="1", name="ping", arguments={})]),
            _response("unreachable"),
        ]
    )
    report = _desk(adapter, max_cost_usd=0.0001).run(
        Worker(name="W", tools=[ping]), Job(input="go")
    )
    assert report.status == "failed"
    assert any("Cost budget exceeded" in error for error in report.errors)
    assert report.tool_calls[0].result.content == "pong"
    assert len(adapter.calls) == 1
    assert report.metrics["cost_usd"] == pytest.approx(TURN_COST)


def test_cost_budget_zero_blocks_second_turn() -> None:
    @tool()
    def ping() -> str:
        return "pong"

    adapter = ScriptedAdapter(
        [
            _response(None, [ToolCall(id="1", name="ping", arguments={})]),
            _response("unreachable"),
        ]
    )
    report = _desk(adapter, max_cost_usd=0.0).run(Worker(name="W", tools=[ping]), Job(input="go"))
    assert report.status == "failed"
    assert any("Cost budget exceeded" in error for error in report.errors)
    assert len(adapter.calls) == 1


def test_cost_budget_unknown_model_never_exceeds() -> None:
    adapter = ScriptedAdapter([_response("done")])
    desk = Desk(
        model="unknown-model",
        adapter=adapter,
        run_store=InMemoryRunStore(),
        max_cost_usd=0.0,
    )
    report = desk.run(Worker(name="W"), Job(input="hi"))
    assert report.status == "completed"
    assert report.metrics["cost_usd"] == 0.0


def test_desk_rejects_negative_budget() -> None:
    adapter = ScriptedAdapter([])
    with pytest.raises(ValueError, match="max_cost_usd"):
        Desk(
            model=MODEL,
            adapter=adapter,
            run_store=InMemoryRunStore(),
            max_cost_usd=-0.5,
        )
