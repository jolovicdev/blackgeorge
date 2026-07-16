import pytest

from blackgeorge import (
    Desk,
    EvalCase,
    Job,
    ScriptedAdapter,
    Worker,
    aevaluate,
    evaluate,
)
from blackgeorge.adapters.base import ModelResponse
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.store.in_memory import InMemoryRunStore
from blackgeorge.tools import tool


def _response(
    content: str | None = "ok", tool_calls: list[ToolCall] | None = None
) -> ModelResponse:
    return ModelResponse(content=content, tool_calls=tool_calls or [], usage={}, raw={})


def _desk(adapter: ScriptedAdapter, **kwargs) -> Desk:
    return Desk(model="fake", adapter=adapter, run_store=InMemoryRunStore(), **kwargs)


def test_evaluate_passes_matching_case() -> None:
    adapter = ScriptedAdapter([_response("the answer is 4")])
    cases = [EvalCase(name="math", job=Job(input="2+2?"), contains=("4",))]
    results = evaluate(_desk(adapter), Worker(name="W"), cases)
    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].failures == ()
    assert results[0].report.content == "the answer is 4"


def test_evaluate_flags_missing_substring_and_failed_check() -> None:
    adapter = ScriptedAdapter([_response("no number here"), _response("still none")])
    cases = [
        EvalCase(name="contains", job=Job(input="2+2?"), contains=("4",)),
        EvalCase(
            name="check",
            job=Job(input="2+2?"),
            check=lambda report: bool(report.content and "4" in report.content),
        ),
    ]
    results = evaluate(_desk(adapter), Worker(name="W"), cases)
    assert results[0].passed is False
    assert results[0].failures == ("missing content substring: '4'",)
    assert results[1].passed is False
    assert results[1].failures == ("check returned False",)


def test_evaluate_flags_failed_run() -> None:
    @tool()
    def ping() -> str:
        return "pong"

    adapter = ScriptedAdapter([_response(None, [ToolCall(id="1", name="ping", arguments={})])])
    desk = _desk(adapter, max_iterations=1)
    cases = [EvalCase(name="loop", job=Job(input="go"))]
    results = evaluate(desk, Worker(name="W", tools=[ping]), cases)
    assert results[0].passed is False
    assert results[0].failures == ("status: expected completed, got failed",)


def test_evaluate_with_judge_scores() -> None:
    adapter = ScriptedAdapter(
        [
            _response("the answer is 4"),
            _response('{"score": 0.9, "reasoning": "correct"}'),
            _response("the answer is 5"),
            _response('{"score": 0.2, "reasoning": "wrong"}'),
        ]
    )
    cases = [
        EvalCase(name="good", job=Job(input="2+2?")),
        EvalCase(name="bad", job=Job(input="2+2?")),
    ]
    results = evaluate(
        _desk(adapter),
        Worker(name="W"),
        cases,
        judge=Worker(name="Judge"),
        rubric="1.0 if the answer is 4, else 0.0",
    )
    assert results[0].passed is True
    assert results[0].score == pytest.approx(0.9)
    assert results[1].passed is False
    assert results[1].score == pytest.approx(0.2)
    assert results[1].failures == ("score 0.20 below 0.70",)


def test_evaluate_judge_requires_rubric() -> None:
    adapter = ScriptedAdapter([])
    with pytest.raises(ValueError, match="judge and rubric"):
        evaluate(
            _desk(adapter),
            Worker(name="W"),
            [EvalCase(name="x", job=Job(input="hi"))],
            judge=Worker(name="Judge"),
        )


def test_evaluate_respects_min_score() -> None:
    adapter = ScriptedAdapter(
        [_response("close enough"), _response('{"score": 0.6, "reasoning": "partial"}')]
    )
    cases = [EvalCase(name="case", job=Job(input="hi"))]
    results = evaluate(
        _desk(adapter),
        Worker(name="W"),
        cases,
        judge=Worker(name="Judge"),
        rubric="rate the answer",
        min_score=0.5,
    )
    assert results[0].passed is True


async def test_aevaluate_async() -> None:
    adapter = ScriptedAdapter([_response("async ok")])
    cases = [EvalCase(name="async", job=Job(input="hi"), contains=("async",))]
    results = await aevaluate(_desk(adapter), Worker(name="W"), cases)
    assert results[0].passed is True
