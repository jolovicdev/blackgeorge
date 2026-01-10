from datetime import datetime
from typing import Any

from pydantic import BaseModel

from blackgeorge.core.report import Report
from blackgeorge.core.types import RunStatus


class WorkerResponse(BaseModel):
    name: str
    model: str | None
    instructions: str | None
    tools: list[str]
    memory_scope: str


class WorkforceResponse(BaseModel):
    name: str
    workers: list[str]
    mode: str
    manager: str | None


class ReportResponse(BaseModel):
    run_id: str
    status: RunStatus
    content: str | None
    data: Any | None
    pending_action: dict[str, Any] | None
    metrics: dict[str, Any]
    errors: list[str]
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]

    @classmethod
    def from_report(cls, report: Report) -> "ReportResponse":
        return cls(
            run_id=report.run_id,
            status=report.status,
            content=report.content,
            data=report.data,
            pending_action=report.pending_action.model_dump(mode="json")
            if report.pending_action
            else None,
            metrics=report.metrics,
            errors=report.errors,
            messages=[m.model_dump(mode="json") for m in report.messages],
            tool_calls=[tc.model_dump(mode="json") for tc in report.tool_calls],
        )


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    runner_type: str | None
    runner_name: str | None
    input: dict[str, Any]
    output: str | None
    output_json: Any | None
    created_at: datetime | None
    updated_at: datetime | None
    is_paused: bool


class EventResponse(BaseModel):
    event_id: str
    type: str
    timestamp: datetime
    source: str | None
    payload: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    version: str


class SuccessResponse(BaseModel):
    success: bool
    message: str
