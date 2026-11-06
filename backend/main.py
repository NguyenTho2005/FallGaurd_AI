"""FastAPI application factory for FallGuard AI.

    uvicorn backend.main:app --host 0.0.0.0 --port 8000

Startup and shutdown are handled through the lifespan protocol, so the event store, the
model and the pose estimator are released deterministically.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai.config import Settings, get_settings
from ai.logging_config import configure_logging, get_logger
from backend.api.routes import router
from backend.services.inference_service import InferenceService

logger = get_logger(__name__)

DESCRIPTION = """
FallGuard AI -- prototype fall-event detection from camera pose data.

**Pipeline:** frame -> MediaPipe Pose (33 landmarks) -> preprocessing -> temporal
sequence -> LSTM/BiLSTM -> event confirmation -> API -> dashboard -> Telegram alert.

**Privacy:** raw video is processed in memory and never stored. Only skeleton landmarks
and event metadata are persisted. No face or identity recognition is implemented.

**This is not a medical device.** FallGuard AI flags *possible* fall events for human
verification; it does not diagnose injuries or replace emergency services.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the inference service on startup and release it on shutdown."""
    settings: Settings = app.state.settings
    configure_logging(settings)
    logger.info("Starting %s v%s (environment=%s)", settings.app_name, settings.version,
                settings.environment)

    service = InferenceService(settings)
    app.state.service = service

    model = service.pipeline.model_info()
    logger.info(
        "Model ready: backend=%s loaded=%s sequence_length=%d feature_dim=%d path=%s",
        model.backend, model.loaded, model.sequence_length, model.feature_dim, model.path,
    )
    if not model.loaded:
        logger.error("Model is NOT loaded; /predict will fall back to the mock backend")
    if not service.pipeline.pose_available:
        logger.error(
            "Pose estimator is NOT available; run scripts/download_mediapipe_model.py. "
            "POST /predict will still accept landmark arrays directly."
        )
    logger.info(
        "Event store: %s | Telegram: %s",
        settings.db_path, service.alerter.status()["reason"],
    )

    if settings.enable_monitor:
        service.start_monitor("camera")

    try:
        yield
    finally:
        logger.info("Shutting down %s", settings.app_name)
        service.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application."""
    settings = settings or get_settings()
    configure_logging(settings)

    app = FastAPI(
        title=settings.api_title,
        version=settings.version,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.service = None

    # The dashboard is served separately by Streamlit, so a permissive CORS policy here
    # only affects local tooling. Bind to specific origins in production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Log the failure with a traceback and return a 500 without leaking internals."""
        logger.error(
            "Unhandled error while serving %s %s", request.method, request.url.path,
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "internal server error; see the server log for details"},
        )

    @app.get("/", tags=["system"], summary="Service index")
    async def index() -> dict[str, object]:
        return {
            "service": settings.app_name,
            "version": settings.version,
            "docs": "/docs",
            "endpoints": [
                "GET /health",
                "GET /status",
                "POST /predict",
                "GET /events",
                "GET /events/{event_id}",
                "PATCH /events/{event_id}",
                "GET /pose/latest",
            ],
            "disclaimer": (
                "Prototype fall-event detection. Not a medical device and not a diagnosis."
            ),
        }

    app.include_router(router)
    return app


app = create_app()

__all__ = ["app", "create_app", "lifespan"]


if __name__ == "__main__":  # pragma: no cover - manual launch helper
    import uvicorn

    logging.getLogger("uvicorn").setLevel(logging.INFO)
    uvicorn.run(
        "backend.main:app",
        host=get_settings().api_host,
        port=get_settings().api_port,
        reload=False,
    )
