from fastapi import APIRouter, Depends, HTTPException

from blackgeorge.api.dependencies import get_desk
from blackgeorge.api.exceptions import WorkerNotFoundError
from blackgeorge.api.models.requests import WorkerCreateRequest
from blackgeorge.api.models.responses import SuccessResponse, WorkerResponse
from blackgeorge.desk import Desk
from blackgeorge.worker import Worker

router = APIRouter()


@router.post("", response_model=WorkerResponse, status_code=201)
async def create_worker(
    request: WorkerCreateRequest, desk: Desk = Depends(get_desk)
) -> WorkerResponse:
    if request.name in desk._workers:
        raise HTTPException(status_code=409, detail=f"Worker '{request.name}' already exists")

    worker = Worker(
        name=request.name,
        model=request.model,
        instructions=request.instructions,
        tools=[],
        memory_scope=request.memory_scope,
    )
    desk.register_worker(worker)

    return WorkerResponse(
        name=worker.name,
        model=worker.model,
        instructions=worker.instructions,
        tools=[t.name for t in worker.tools()],
        memory_scope=worker.memory_scope,
    )


@router.get("", response_model=list[WorkerResponse])
async def list_workers(desk: Desk = Depends(get_desk)) -> list[WorkerResponse]:
    return [
        WorkerResponse(
            name=w.name,
            model=w.model,
            instructions=w.instructions,
            tools=[t.name for t in w.tools()],
            memory_scope=w.memory_scope,
        )
        for w in desk._workers.values()
    ]


@router.get("/{name}", response_model=WorkerResponse)
async def get_worker(name: str, desk: Desk = Depends(get_desk)) -> WorkerResponse:
    worker = desk._workers.get(name)
    if not worker:
        raise WorkerNotFoundError(name)

    return WorkerResponse(
        name=worker.name,
        model=worker.model,
        instructions=worker.instructions,
        tools=[t.name for t in worker.tools()],
        memory_scope=worker.memory_scope,
    )


@router.delete("/{name}", response_model=SuccessResponse)
async def delete_worker(name: str, desk: Desk = Depends(get_desk)) -> SuccessResponse:
    if name not in desk._workers:
        raise WorkerNotFoundError(name)

    desk.unregister_worker(name)
    return SuccessResponse(success=True, message=f"Worker '{name}' deleted")
