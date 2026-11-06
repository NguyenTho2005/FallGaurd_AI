"""Pydantic request/response schemas for the FallGuard API.

The schemas are the API's contract, so they are explicit about the two privacy-relevant
facts: a prediction request carries **landmarks, never video**, and a response says which
model backend produced it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai.pose.landmarks import POSE_LANDMARK_COUNT, POSE_LANDMARK_WIDTH

Prediction = Literal["FALL", "NORMAL", "UNKNOWN"]


class HealthResponse(BaseModel):
    """Liveness probe payload."""

    status: Literal["ok", "degraded"] = Field(description="Service liveness")
    service: str
    version: str
    environment: str
    timestamp: str


class ModelStatus(BaseModel):
    backend: str
    loaded: bool
    path: str | None = None
    sequence_length: int
    feature_dim: int
    detail: str = ""


class CameraStatus(BaseModel):
    id: str
    source: str
    connected: bool
    monitor_enabled: bool


class TelegramStatus(BaseModel):
    enabled: bool
    configured: bool
    dry_run: bool
    reason: str
    api_base: str


class PipelineStatus(BaseModel):
    frames_processed: int
    poses_detected: int
    pose_dropouts: int
    pose_detection_rate: float
    windows_scored: int
    events_confirmed: int
    uptime_seconds: float
    buffered_frames: int
    sequence_length: int
    sequence_ready: bool
    missing_ratio: float
    pose_estimator_available: bool


class StatusResponse(BaseModel):
    """Full system status for the dashboard."""

    system: str
    app: dict[str, Any]
    model: ModelStatus
    camera: CameraStatus
    config: dict[str, Any]
    telegram: TelegramStatus
    alerts: dict[str, int]
    events: dict[str, Any]
    pipeline: PipelineStatus
    detector: dict[str, Any]
    last_prediction: dict[str, Any] | None = None
    privacy: dict[str, str] = Field(default_factory=dict)


class PredictRequest(BaseModel):
    """Ask for a prediction.

    Either supply ``landmarks`` for the current frame(s), or omit it to score whatever the
    live monitor has buffered. Video frames are never accepted.
    """

    model_config = ConfigDict(extra="forbid")

    landmarks: list[list[float]] | None = Field(
        default=None,
        description=(
            f"A single frame of {POSE_LANDMARK_COUNT} landmarks, each "
            f"[x, y, z, visibility] in [0, 1]. Omit to score the live buffer."
        ),
    )
    camera_id: str | None = Field(default=None, max_length=64)
    timestamp: str | None = Field(default=None, max_length=64)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("landmarks")
    @classmethod
    def _validate_landmarks(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        if value is None:
            return None
        if len(value) != POSE_LANDMARK_COUNT:
            raise ValueError(
                f"expected {POSE_LANDMARK_COUNT} landmarks, got {len(value)}"
            )
        for index, landmark in enumerate(value):
            if len(landmark) != POSE_LANDMARK_WIDTH:
                raise ValueError(
                    f"landmark {index} must have {POSE_LANDMARK_WIDTH} values "
                    f"[x, y, z, visibility], got {len(landmark)}"
                )
        return value


class PredictResponse(BaseModel):
    """The structured result of one prediction request."""

    prediction: Prediction
    confidence: float = Field(ge=0.0, le=1.0)
    fall_probability: float = Field(ge=0.0, le=1.0)
    timestamp: str
    camera_id: str
    pose_detected: bool
    sequence_ready: bool
    event_confirmed: bool
    event_id: str | None = None
    model_backend: str
    message: str = ""


class EventResponse(BaseModel):
    """A stored fall event."""

    event_id: str
    timestamp: str
    camera_id: str
    prediction: str
    confidence: float
    peak_confidence: float
    status: str
    window_frames: int
    source: str
    notified: bool
    note: str
    created_at: str


class EventListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    events: list[EventResponse]


class EventStatusUpdate(BaseModel):
    """Acknowledge or mark an event as a false positive."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["detected", "alerted", "acknowledged", "false_positive"]
    note: str | None = Field(default=None, max_length=500)


class PoseResponse(BaseModel):
    """Latest skeleton, for privacy-preserving visualisation."""

    camera_id: str
    landmarks: list[list[float]]
    connections: list[list[int]]
    frame_index: int
    privacy_note: str


class ErrorResponse(BaseModel):
    detail: str
