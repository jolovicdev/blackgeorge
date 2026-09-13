from typing import Any

from blackgeorge.core.report import Report

USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def add_token_usage(target: dict[str, Any], usage: dict[str, Any]) -> None:
    for key in USAGE_KEYS:
        value = usage.get(key)
        if _is_number(value):
            target[key] = target.get(key, 0) + value


def record_turn(
    metrics: dict[str, Any],
    totals: dict[str, Any],
    usage: dict[str, Any],
    cost_usd: float,
) -> None:
    metrics["cost_usd"] = metrics.get("cost_usd", 0.0) + cost_usd
    totals["cost_usd"] = totals.get("cost_usd", 0.0) + cost_usd
    add_token_usage(metrics.setdefault("usage", {}), usage)
    add_token_usage(totals, usage)


def totals_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    totals: dict[str, Any] = {}
    cost = metrics.get("cost_usd")
    if _is_number(cost):
        totals["cost_usd"] = cost
    usage = metrics.get("usage")
    if isinstance(usage, dict):
        add_token_usage(totals, usage)
    return totals


def restore_totals(totals: dict[str, Any], stored: Any) -> None:
    if isinstance(stored, dict):
        totals.update(stored)


def run_metrics(totals: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"cost_usd": totals.get("cost_usd", 0.0)}
    usage = {key: totals[key] for key in USAGE_KEYS if key in totals}
    if usage:
        metrics["usage"] = usage
    return metrics


def with_run_metrics(report: Report, totals: dict[str, Any]) -> Report:
    return report.model_copy(update={"metrics": {**report.metrics, **run_metrics(totals)}})
