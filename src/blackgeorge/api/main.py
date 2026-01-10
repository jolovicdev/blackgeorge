from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from blackgeorge.api.config import get_config
from blackgeorge.api.exceptions import APIException
from blackgeorge.api.middleware import register_middleware
from blackgeorge.api.routes import health, runs, status, workers, workforces


def create_app() -> FastAPI:
    config = get_config()

    app = FastAPI(
        title=config.title,
        description="REST API for the Blackgeorge agentic framework. **Experimental - subject to change.**",
        version=config.version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_middleware(app)

    @app.exception_handler(APIException)
    async def api_exception_handler(_: Request, exc: APIException) -> Response:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "errors": exc.errors},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(_: Request, exc: Exception) -> Response:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(workers.router, prefix="/api/v1/workers", tags=["Workers"])
    app.include_router(workforces.router, prefix="/api/v1/workforces", tags=["Workforces"])
    app.include_router(runs.router, prefix="/api/v1/runs", tags=["Runs"])
    app.include_router(status.router, prefix="/api/v1/runs", tags=["Status"])

    return app
