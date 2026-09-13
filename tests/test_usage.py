from blackgeorge.core.report import Report
from blackgeorge.core.usage import (
    add_token_usage,
    record_turn,
    restore_totals,
    run_metrics,
    totals_from_metrics,
    with_run_metrics,
)


def test_add_token_usage_sums_known_numeric_keys_only() -> None:
    target = {"prompt_tokens": 1}
    add_token_usage(
        target,
        {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": True, "cached": 9},
    )
    assert target == {"prompt_tokens": 3, "completion_tokens": 3}


def test_record_turn_updates_metrics_and_totals() -> None:
    metrics: dict[str, object] = {}
    totals: dict[str, object] = {"cost_usd": 0.5, "total_tokens": 10}
    record_turn(metrics, totals, {"prompt_tokens": 4, "total_tokens": 6}, 0.25)
    assert metrics == {"cost_usd": 0.25, "usage": {"prompt_tokens": 4, "total_tokens": 6}}
    assert totals == {"cost_usd": 0.75, "total_tokens": 16, "prompt_tokens": 4}


def test_restore_totals_only_fills_empty_totals() -> None:
    totals: dict[str, object] = {}
    restore_totals(totals, {"cost_usd": 0.1})
    restore_totals(totals, {"cost_usd": 0.05})
    restore_totals(totals, "not a dict")
    assert totals == {"cost_usd": 0.1}


def test_totals_round_trip_through_metrics() -> None:
    totals = {"cost_usd": 0.1, "prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    assert totals_from_metrics(run_metrics(totals)) == totals
    assert run_metrics({}) == {"cost_usd": 0.0}


def test_with_run_metrics_keeps_unrelated_metrics() -> None:
    report = Report(run_id="r", status="completed", metrics={"context_summaries": 2})
    updated = with_run_metrics(report, {"cost_usd": 0.3, "total_tokens": 5})
    assert updated.metrics == {
        "context_summaries": 2,
        "cost_usd": 0.3,
        "usage": {"total_tokens": 5},
    }
