from fastapi import APIRouter, Depends, HTTPException

from blackgeorge.api.dependencies import get_desk
from blackgeorge.api.exceptions import WorkforceNotFoundError
from blackgeorge.api.models.requests import WorkforceCreateRequest
from blackgeorge.api.models.responses import SuccessResponse, WorkforceResponse
from blackgeorge.desk import Desk
from blackgeorge.worker import Worker
from blackgeorge.workforce import Workforce

router = APIRouter()


def _find_workers(desk: Desk, worker_names: list[str]) -> list[Worker]:
    workers = []
    for name in worker_names:
        worker = desk._workers.get(name)
        if not worker:
            raise HTTPException(status_code=404, detail=f"Worker '{name}' not found")
        workers.append(worker)
    return workers


@router.post("", response_model=WorkforceResponse, status_code=201)
async def create_workforce(
    request: WorkforceCreateRequest, desk: Desk = Depends(get_desk)
) -> WorkforceResponse:
    if request.name in desk._workforces:
        raise HTTPException(status_code=409, detail=f"Workforce '{request.name}' already exists")

    workers = _find_workers(desk, request.workers)

    manager_worker = None
    if request.mode == "managed" and request.manager:
        manager_worker = desk._workers.get(request.manager)
        if not manager_worker:
            raise HTTPException(
                status_code=404, detail=f"Manager worker '{request.manager}' not found"
            )

    workforce = Workforce(
        workers=workers,
        mode=request.mode,
        name=request.name,
        manager=manager_worker,
    )
    desk.register_workforce(workforce)

    return WorkforceResponse(
        name=workforce.name,
        workers=[w.name for w in workforce.workers],
        mode=workforce.mode,
        manager=workforce.manager.name if workforce.manager else None,
    )


@router.get("", response_model=list[WorkforceResponse])
async def list_workforces(desk: Desk = Depends(get_desk)) -> list[WorkforceResponse]:
    return [
        WorkforceResponse(
            name=w.name,
            workers=[worker.name for worker in w.workers],
            mode=w.mode,
            manager=w.manager.name if w.manager else None,
        )
        for w in desk._workforces.values()
    ]


@router.get("/{name}", response_model=WorkforceResponse)
async def get_workforce(name: str, desk: Desk = Depends(get_desk)) -> WorkforceResponse:
    workforce = desk._workforces.get(name)
    if not workforce:
        raise WorkforceNotFoundError(name)

    return WorkforceResponse(
        name=workforce.name,
        workers=[w.name for w in workforce.workers],
        mode=workforce.mode,
        manager=workforce.manager.name if workforce.manager else None,
    )


@router.delete("/{name}", response_model=SuccessResponse)
async def delete_workforce(name: str, desk: Desk = Depends(get_desk)) -> SuccessResponse:
    if name not in desk._workforces:
        raise WorkforceNotFoundError(name)

    desk.unregister_workforce(name)
    return SuccessResponse(success=True, message=f"Workforce '{name}' deleted")
