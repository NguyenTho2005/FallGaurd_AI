"""API route definitions and shared FastAPI dependencies."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status

from ai.logging_config import get_logger
from backend.api.schemas import (
    ErrorResponse,
    EventListResponse,
    EventResponse,
    EventStatusUpdate,
    HealthResponse,
    PoseResponse,
    PredictRequest,
    PredictResponse,
    StatusResponse,
)
from backend.services.inference_service import InferenceService

logger = get_logger(__name__)

router = APIRouter()


def get_service(request: Request) -> InferenceService:
    """Resolve the shared :class:`InferenceService` from application state."""
    service = getattr(request.app.state, "service", None)
    if service is None:  # pragma: no cover - only if startup failed
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="inference service is not initialised",
        )
    return service


ServiceDep = Annotated[InferenceService, Depends(get_service)]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# health & status
# --------------------------------------------------------------------------- #
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    tags=["system"],
)
def health(service: ServiceDep) -> HealthResponse:
    """Report that the process is up. Deliberately does no I/O, so it never fails slowly."""
    settings = service._settings  # noqa: SLF001 - the service owns its settings
    degraded = not service.pipeline.pose_available
    return HealthResponse(
        status="degraded" if degraded else "ok",
        service=settings.app_name,
        version=settings.version,
        environment=settings.environment,
        timestamp=_now_iso(),
    )


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="System status for the dashboard",
    tags=["system"],
)
def system_status(service: ServiceDep) -> StatusResponse:
    """Full runtime status: model, camera, config, pipeline counters and alert state."""
    payload = service.status()
    payload["privacy"] = {
        "retention": "skeleton landmarks and event metadata only",
        "raw_video_stored": "never",
        "identity_recognition": "not implemented",
        "disclaimer": (
            "FallGuard AI is a prototype that detects possible fall events. "
            "It is not a medical device and does not diagnose."
        ),
    }
    return StatusResponse(**payload)


# --------------------------------------------------------------------------- #
# prediction
# --------------------------------------------------------------------------- #
@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={422: {"model": ErrorResponse}},
    summary="Score the current frame or the live buffer",
    tags=["inference"],
)
def predict(payload: PredictRequest, service: ServiceDep) -> PredictResponse:
    """Run one frame (or the buffered window) through the pipeline.

    Supply ``landmarks`` to score a specific frame; omit it to score whatever the camera
    monitor has buffered. Raw video is never accepted.
    """
    frame_index = service.pipeline.stats.frames_processed
    if payload.landmarks is not None:
        try:
            outcome = service.process_landmarks(
                np.asarray(payload.landmarks, dtype=np.float32),
                timestamp=None,
                frame_index=frame_index,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
    else:
        if not service.pipeline.buffer.is_ready:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "no landmarks were supplied and the live buffer is not ready "
                    f"({len(service.pipeline.buffer)}/"
                    f"{service.pipeline.buffer.sequence_length} frames). Send a "
                    "'landmarks' array, or enable the camera monitor."
                ),
            )
        # Re-score the buffered window without consuming a new frame.
        window = service.pipeline.buffer.window()
        if window is None:  # pragma: no cover - guarded by is_ready
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="the live buffer is not ready"
            )
        probability = service.pipeline.predictor.predict_proba(window)
        outcome = _outcome_from_buffer(service, probability)

    label = outcome.prediction if outcome.prediction in {"FALL", "NORMAL"} else "UNKNOWN"
    message = ""
    if outcome.event is not None:
        message = f"Fall event confirmed and stored: {outcome.event.event_id}"
    elif not outcome.sequence_ready:
        message = "buffering frames; the sequence window is not full yet"
    elif label == "FALL":
        message = "above threshold but awaiting temporal confirmation"

    return PredictResponse(
        prediction=label,
        confidence=round(float(outcome.confidence), 6),
        fall_probability=round(float(outcome.fall_probability), 6),
        timestamp=payload.timestamp or _now_iso(),
        camera_id=payload.camera_id or outcome.camera_id,
        pose_detected=outcome.pose_detected,
        sequence_ready=outcome.sequence_ready,
        event_confirmed=outcome.event is not None,
        event_id=outcome.event.event_id if outcome.event else None,
        model_backend=service.pipeline.predictor.backend,
        message=message,
    )


def _outcome_from_buffer(service: InferenceService, probability: float):
    """Build a :class:`PredictionOutcome` for a buffer re-score.

    The buffer path is a *read*: feeding the probability back into the confirmation state
    machine would advance it a second time per request, so the detector is deliberately
    left untouched here.
    """
    from ai.inference.pipeline import PredictionOutcome

    threshold = service._settings.confidence_threshold  # noqa: SLF001
    return PredictionOutcome(
        frame_index=service.pipeline.stats.frames_processed,
        timestamp=_now_iso(),
        pose_detected=service.pipeline.last_pose is not None,
        sequence_ready=True,
        fall_probability=probability,
        prediction="FALL" if probability >= threshold else "NORMAL",
        above_threshold=probability >= threshold,
        event=None,
        confidence=probability,
        camera_id=service._settings.camera_id,  # noqa: SLF001
    )


# --------------------------------------------------------------------------- #
# events
# --------------------------------------------------------------------------- #
@router.get(
    "/events",
    response_model=EventListResponse,
    summary="List confirmed fall events",
    tags=["events"],
)
def list_events(
    service: ServiceDep,
    limit: Annotated[int, Query(ge=1, le=1000)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    camera_id: Annotated[str | None, Query(max_length=64)] = None,
    event_status: Annotated[str | None, Query(alias="status", max_length=32)] = None,
) -> EventListResponse:
    """Return stored events, newest first."""
    try:
        rows = service.repository.list(
            limit=limit, offset=offset, camera_id=camera_id, status=event_status
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    total = service.repository.count(camera_id=camera_id, status=event_status)
    return EventListResponse(
        total=total,
        limit=limit,
        offset=offset,
        events=[EventResponse(**row) for row in rows],
    )


@router.get(
    "/events/{event_id}",
    response_model=EventResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Fetch one fall event",
    tags=["events"],
)
def get_event(
    service: ServiceDep,
    event_id: Annotated[str, Path(min_length=1, max_length=64)],
) -> EventResponse:
    """Return a single event, or 404 when the id is unknown."""
    row = service.repository.get(event_id)
    if row is None:
        logger.warning("Requested unknown event_id=%s", event_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"event '{event_id}' not found"
        )
    return EventResponse(**row)


@router.patch(
    "/events/{event_id}",
    response_model=EventResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Update an event's status",
    tags=["events"],
)
def update_event(
    payload: EventStatusUpdate,
    service: ServiceDep,
    event_id: Annotated[str, Path(min_length=1, max_length=64)],
) -> EventResponse:
    """Acknowledge an event or mark it as a false positive."""
    if not service.repository.update_status(
        event_id, payload.status, note=payload.note
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"event '{event_id}' not found"
        )
    row = service.repository.get(event_id)
    if row is None:  # pragma: no cover - deleted concurrently
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"event '{event_id}' not found"
        )
    return EventResponse(**row)


# --------------------------------------------------------------------------- #
# pose
# --------------------------------------------------------------------------- #
@router.get(
    "/pose/latest",
    response_model=PoseResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Latest skeleton landmarks (no video)",
    tags=["inference"],
)
def latest_pose(service: ServiceDep) -> PoseResponse:
    """Return the most recent 33 landmarks so the dashboard can draw a skeleton.

    This endpoint exists so the dashboard can show *skeletons instead of footage*, which
    is the default privacy-preserving visualisation.
    """
    payload: dict[str, Any] | None = service.latest_pose()
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="no pose has been detected yet; the camera monitor may be disabled",
        )
    return PoseResponse(**payload)
