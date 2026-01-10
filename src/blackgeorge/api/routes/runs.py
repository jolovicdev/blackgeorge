from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from blackgeorge.api.dependencies import get_desk
from blackgeorge.api.exceptions import RunNotFoundError
from blackgeorge.api.models.requests import JobCreateRequest, ResumeRequest
from blackgeorge.api.models.responses import ReportResponse
from blackgeorge.core.job import Job
from blackgeorge.core.report import Report
from blackgeorge.desk import Desk

router = APIRouter()


def _resolve_resume_input(request: ResumeRequest) -> Any:
    if request.decision is not None:
        return request.decision
    return request.input


@router.post("/worker/{name}", response_model=ReportResponse)
async def run_worker(
    name: str, request: JobCreateRequest, desk: Desk = Depends(get_desk)
) -> ReportResponse:
    worker = desk._workers.get(name)
    if not worker:
        raise HTTPException(status_code=404, detail=f"Worker '{name}' not found")

    job = Job(
        input=request.input,
        expected_output=request.expected_output,
        response_schema=request.response_schema,
        constraints=request.constraints,
        metadata=request.metadata,
    )

    report = desk.run(worker, job, stream=request.stream)
    return ReportResponse.from_report(report)


@router.post("/workforce/{name}", response_model=ReportResponse)
async def run_workforce(
    name: str, request: JobCreateRequest, desk: Desk = Depends(get_desk)
) -> ReportResponse:
    workforce = desk._workforces.get(name)
    if not workforce:
        raise HTTPException(status_code=404, detail=f"Workforce '{name}' not found")

    job = Job(
        input=request.input,
        expected_output=request.expected_output,
        response_schema=request.response_schema,
        constraints=request.constraints,
        metadata=request.metadata,
    )

    report = desk.run(workforce, job, stream=request.stream)
    return ReportResponse.from_report(report)


@router.post("/{run_id}/resume", response_model=ReportResponse)
async def resume_run(
    run_id: str, request: ResumeRequest, desk: Desk = Depends(get_desk)
) -> ReportResponse:
    record = desk.run_store.get_run(run_id)
    if not record:
        raise RunNotFoundError(run_id)

    if record.status != "paused":
        raise HTTPException(
            status_code=400, detail=f"Run '{run_id}' is not paused (status: {record.status})"
        )

    input_value = _resolve_resume_input(request)

    report = Report(
        run_id=run_id,
        status=record.status,
        content=record.output,
        data=record.output_json,
        pending_action=None,
        metrics={},
        errors=[],
        messages=[],
        tool_calls=[],
        events=[],
    )

    updated_report = desk.resume(report, input_value)
    return ReportResponse.from_report(updated_report)
