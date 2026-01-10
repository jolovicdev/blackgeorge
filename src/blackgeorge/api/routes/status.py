from fastapi import APIRouter, Depends, Query

from blackgeorge.api.dependencies import get_desk
from blackgeorge.api.exceptions import RunNotFoundError
from blackgeorge.api.models.responses import EventResponse, RunStatusResponse
from blackgeorge.desk import Desk

router = APIRouter()


@router.get("/{run_id}", response_model=RunStatusResponse)
async def get_run(run_id: str, desk: Desk = Depends(get_desk)) -> RunStatusResponse:
    record = desk.run_store.get_run(run_id)
    if not record:
        raise RunNotFoundError(run_id)

    return RunStatusResponse(
        run_id=record.run_id,
        status=record.status,
        runner_type=record.state.runner_type if record.state else None,
        runner_name=record.state.runner_name if record.state else None,
        input=record.input,
        output=record.output,
        output_json=record.output_json,
        created_at=record.created_at,
        updated_at=record.updated_at,
        is_paused=record.status == "paused",
    )


@router.get("", response_model=list[RunStatusResponse])
async def list_runs(
    desk: Desk = Depends(get_desk),
    status: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[RunStatusResponse]:
    runs = desk.run_store.list_runs()

    if status:
        runs = [r for r in runs if r.status == status]

    runs = runs[offset : offset + limit]

    return [
        RunStatusResponse(
            run_id=r.run_id,
            status=r.status,
            runner_type=r.state.runner_type if r.state else None,
            runner_name=r.state.runner_name if r.state else None,
            input=r.input,
            output=r.output,
            output_json=r.output_json,
            created_at=r.created_at,
            updated_at=r.updated_at,
            is_paused=r.status == "paused",
        )
        for r in runs
    ]


@router.get("/{run_id}/events", response_model=list[EventResponse])
async def get_run_events(run_id: str, desk: Desk = Depends(get_desk)) -> list[EventResponse]:
    record = desk.run_store.get_run(run_id)
    if not record:
        raise RunNotFoundError(run_id)

    events = desk.run_store.get_events(run_id)

    return [
        EventResponse(
            event_id=e.event_id,
            type=e.type,
            timestamp=e.timestamp,
            source=e.source,
            payload=e.payload,
        )
        for e in events
    ]
