from typing import Any

from pydantic import BaseModel

from blackgeorge.core.event import Event
from blackgeorge.core.job import Job
from blackgeorge.core.message import Message
from blackgeorge.core.report import Report
from blackgeorge.core.types import RunStatus
from blackgeorge.store.state import RunState
from blackgeorge.worker import Worker


def build_workforce_state(
    run_id: str,
    status: RunStatus,
    workforce_name: str,
    job: Job,
    worker_state: RunState,
    stage: str,
    payload: dict[str, Any] | None = None,
) -> RunState:
    return RunState(
        run_id=run_id,
        status=status,
        runner_type="workforce",
        runner_name=workforce_name,
        job=job,
        messages=worker_state.messages,
        tool_calls=worker_state.tool_calls,
        pending_action=worker_state.pending_action,
        metrics=worker_state.metrics,
        iteration=worker_state.iteration,
        payload={
            "stage": stage,
            "worker_state": worker_state.model_dump(mode="json"),
            **(payload or {}),
        },
    )


def select_worker_name(report: Report, workers: list[Worker]) -> str:
    worker_names = {worker.name for worker in workers}
    if isinstance(report.data, BaseModel) and hasattr(report.data, "worker"):
        candidate = report.data.worker
        if isinstance(candidate, str) and candidate in worker_names:
            return candidate
    if report.content:
        for worker in workers:
            if worker.name in report.content:
                return worker.name
    return workers[0].name


def find_worker(workers: list[Worker], name: str | None) -> Worker:
    if name is None:
        return workers[0]
    for worker in workers:
        if worker.name == name:
            return worker
    return workers[0]


def root_job(payload: dict[str, Any], fallback: Job) -> Job:
    raw = payload.get("root_job")
    if raw is None:
        return fallback
    return Job.model_validate(raw)


def default_reducer(
    reports: list[tuple[Worker, Report]],
    run_id: str,
    events: list[Event],
) -> Report:
    status: RunStatus = "completed"
    if any(report.status == "failed" for _, report in reports):
        status = "failed"
    return aggregate_reports(reports, run_id, events, status)


def merge_swarm_reports(
    reports: list[Report],
    final_report: Report,
) -> Report:
    if not reports:
        return final_report
    all_tool_calls = [call for rep in reports for call in rep.tool_calls]
    all_tool_calls.extend(final_report.tool_calls)
    all_errors = [error for rep in reports for error in rep.errors]
    all_errors.extend(final_report.errors)
    all_messages: list[Message] = [msg for rep in reports for msg in rep.messages]
    seen_ids: set[str] = set()
    unique_messages: list[Message] = []
    for msg in all_messages:
        msg_id = getattr(msg, "id", None) or getattr(msg, "tool_call_id", None)
        if msg_id and msg_id in seen_ids:
            continue
        if msg_id:
            seen_ids.add(msg_id)
        unique_messages.append(msg)
    merged_metrics: dict[str, Any] = {}
    for rep in reports:
        merged_metrics.update(rep.metrics)
    merged_metrics.update(final_report.metrics)
    return final_report.model_copy(
        update={
            "tool_calls": all_tool_calls,
            "errors": all_errors,
            "messages": unique_messages,
            "metrics": merged_metrics,
        }
    )


def aggregate_reports(
    reports: list[tuple[Worker, Report]],
    run_id: str,
    events: list[Event],
    status: RunStatus,
) -> Report:
    has_data = any(report.data is not None for _, report in reports)
    content_parts: list[str] = []
    data: Any | None = None
    if has_data:
        data = []
        for worker, report in reports:
            data.append({"worker": worker.name, "data": report.data, "content": report.content})
    for worker, report in reports:
        content_parts.append(f"[{worker.name}] {report.content or ''}")
    return Report(
        run_id=run_id,
        status=status,
        content="\n\n".join(content_parts),
        data=data,
        messages=[message for _, report in reports for message in report.messages],
        tool_calls=[call for _, report in reports for call in report.tool_calls],
        metrics={},
        events=events,
        pending_action=None,
        errors=[error for _, report in reports for error in report.errors],
    )
