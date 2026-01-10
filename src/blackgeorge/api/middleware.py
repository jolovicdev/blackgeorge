from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request, Response

from blackgeorge.logging import get_logger

logger = get_logger("blackgeorge.api")


def register_middleware(app: Any) -> None:
    @app.middleware("http")
    async def log_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        logger.info(
            "Incoming request",
            method=request.method,
            path=request.url.path,
            client=str(request.client),
        )

        response = await call_next(request)

        logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
        )

        return response
