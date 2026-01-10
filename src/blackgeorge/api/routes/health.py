from fastapi import APIRouter

from blackgeorge.api.config import get_config
from blackgeorge.api.models.responses import HealthResponse

router = APIRouter()


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    config = get_config()
    return HealthResponse(status="healthy", version=config.version)
