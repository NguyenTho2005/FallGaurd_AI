"""End-to-end inference pipeline.

    frame -> pose -> features -> sequence window -> model -> event confirmation

:class:`FallDetectionPipeline` owns one instance of every stage, so there is exactly one
place where the dataflow is wired. It is deliberately free of any web framework
dependency: the FastAPI service and the CLI demo both drive this same class, which means
a demo run and a production run execute identical inference code.

Thread safety
-------------
The pipeline holds mutable state (the ring buffer, the detector). Access is guarded by a
single re-entrant lock, so the API can accept ``POST /predict`` while the optional camera
monitor thread is pushing frames.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ai.config import Settings, get_settings
from ai.inference.fall_detector import DetectionResult, EventStatus, FallEvent, FallDetector
from ai.inference.sequence_buffer import SequenceBuffer
from ai.logging_config import get_logger
from ai.models.predictor import FallPredictor, ModelInfo, create_predictor
from ai.pose.landmarks import POSE_CONNECTIONS, PoseFrame
from ai.pose.pose_estimator import PoseEstimator
from ai.preprocessing.pose_preprocessor import PosePreprocessor

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PredictionOutcome:
    """Everything produced by processing one frame."""

    frame_index: int
    timestamp: float
    pose_detected: bool
    sequence_ready: bool
    fall_probability: float
    prediction: str
    above_threshold: bool
    event: FallEvent | None
    confidence: float
    camera_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.prediction,
            "confidence": round(self.confidence, 6),
            "fall_probability": round(self.fall_probability, 6),
            "pose_detected": self.pose_detected,
            "sequence_ready": self.sequence_ready,
            "event_confirmed": self.event is not None,
            "event_id": self.event.event_id if self.event else None,
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "camera_id": self.camera_id,
        }


@dataclass(slots=True)
class PipelineStats:
    """Operational counters, surfaced through ``GET /status``."""

    frames_processed: int = 0
    poses_detected: int = 0
    pose_dropouts: int = 0
    windows_scored: int = 0
    events_confirmed: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def pose_detection_rate(self) -> float:
        if self.frames_processed == 0:
            return 0.0
        return self.poses_detected / self.frames_processed

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_processed": self.frames_processed,
            "poses_detected": self.poses_detected,
            "pose_dropouts": self.pose_dropouts,
            "pose_detection_rate": round(self.pose_detection_rate, 4),
            "windows_scored": self.windows_scored,
            "events_confirmed": self.events_confirmed,
            "uptime_seconds": round(time.monotonic() - self.started_at, 2),
        }


class FallDetectionPipeline:
    """Wire the pose, preprocessing, sequence, model and event-confirmation stages."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        pose_estimator: PoseEstimator | None = None,
        predictor: FallPredictor | None = None,
        camera_id: str | None = None,
        on_event: Callable[[FallEvent], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        import threading

        self._settings = settings or get_settings()
        self._lock = threading.RLock()
        self._clock = clock or time.monotonic

        self.preprocessor = PosePreprocessor.from_settings(self._settings)
        self.pose_estimator = pose_estimator
        self.predictor = predictor or create_predictor(self._settings)
        self.camera_id = camera_id or self._settings.camera_id
        self.buffer = SequenceBuffer(
            sequence_length=self.predictor.sequence_length,
            feature_dim=self.predictor.feature_dim,
            max_missing_ratio=self._settings.max_missing_ratio,
            clock=self._clock,
        )
        self.detector = FallDetector(self._settings, camera_id=self.camera_id)
        self.stats = PipelineStats()

        self._on_event = on_event
        self._last_pose: PoseFrame | None = None
        self._last_outcome: PredictionOutcome | None = None
        self._pose_failures = 0

        if self.pose_estimator is None:
            self._try_init_pose()
        else:
            logger.info("Pipeline initialised with an externally supplied pose estimator")

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def _try_init_pose(self) -> None:
        """Initialise MediaPipe, degrading gracefully when the asset is unavailable."""
        try:
            self.pose_estimator = PoseEstimator(self._settings)
        except Exception as exc:
            self.pose_estimator = None
            self._pose_failures += 1
            logger.error(
                "Pose estimator unavailable (%s). The pipeline can still score landmark "
                "arrays supplied directly (e.g. POST /predict).",
                exc,
            )

    @property
    def pose_available(self) -> bool:
        return self.pose_estimator is not None and self.pose_estimator.is_available

    def close(self) -> None:
        if self.pose_estimator is not None:
            self.pose_estimator.close()
            self.pose_estimator = None
        logger.info("Pipeline closed after %d frames", self.stats.frames_processed)

    def __enter__(self) -> "FallDetectionPipeline":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # main entry point
    # ------------------------------------------------------------------ #
    def process_frame(
        self, frame: np.ndarray, *, timestamp: float | None = None, frame_index: int = 0
    ) -> PredictionOutcome:
        """Run one camera frame through the whole pipeline."""
        pose_frame: PoseFrame | None = None
        if self.pose_estimator is not None and self.pose_estimator.is_available:
            pose_frame = self.pose_estimator.process(
                frame, timestamp=timestamp, frame_index=frame_index
            )
        else:
            logger.debug("No pose estimator available; frame %d skipped", frame_index)
        return self.process_pose(pose_frame, frame_index=frame_index)

    def process_landmarks(
        self, landmarks: np.ndarray | list | None, *, timestamp: float | None = None,
        frame_index: int = 0,
    ) -> PredictionOutcome:
        """Run a directly supplied ``(33, 4)`` landmark array through the pipeline.

        Used by ``POST /predict`` and by tests. ``None`` is treated exactly like a frame
        in which no pose was found.
        """
        pose_frame: PoseFrame | None = None
        if landmarks is not None:
            array = np.asarray(landmarks, dtype=np.float32)
            if array.shape == (33, 4):
                # A frame of NaNs is not a pose. Accepting it would report
                # ``pose_detected = True`` while the preprocessor produced a gap, which
                # makes the whole detection rate meaningless.
                if np.isfinite(array).all():
                    pose_frame = PoseFrame(
                        landmarks=array,
                        timestamp=timestamp if timestamp is not None else self._clock(),
                        frame_index=frame_index,
                    )
                else:
                    logger.warning(
                        "Rejecting a landmark array with %d non-finite values at frame %d",
                        int((~np.isfinite(array)).sum()),
                        frame_index,
                    )
            else:
                logger.warning(
                    "Ignoring landmark array with shape %s; expected (33, 4)", (array.shape,)
                )
        return self.process_pose(pose_frame, frame_index=frame_index)

    def process_pose(
        self, pose_frame: PoseFrame | None, *, frame_index: int = 0
    ) -> PredictionOutcome:
        """Score an already-extracted pose (``None`` means "no pose in this frame")."""
        with self._lock:
            return self._process_pose_locked(pose_frame, frame_index)

    def _process_pose_locked(
        self, pose_frame: PoseFrame | None, frame_index: int
    ) -> PredictionOutcome:
        self.stats.frames_processed += 1
        timestamp = pose_frame.timestamp if pose_frame is not None else self._clock()

        if pose_frame is not None:
            self.stats.poses_detected += 1
            self._last_pose = pose_frame
            features = self.preprocessor.transform(pose_frame.landmarks)
            self.buffer.push(features, timestamp=timestamp)
        else:
            self.stats.pose_dropouts += 1
            self.buffer.push_zeros(timestamp=timestamp)

        window = self.buffer.window()
        if window is None:
            result = self.detector.update(0.0, has_signal=False)
            outcome = PredictionOutcome(
                frame_index=frame_index,
                timestamp=timestamp,
                pose_detected=pose_frame is not None,
                sequence_ready=False,
                fall_probability=0.0,
                prediction="NORMAL",
                above_threshold=False,
                event=None,
                confidence=0.0,
                camera_id=self.camera_id,
            )
            del result
            self._last_outcome = outcome
            return outcome

        self.stats.windows_scored += 1
        probability = self.predictor.predict_proba(window)
        detection = self.detector.update(probability, has_signal=True)
        if detection.event is not None:
            self.stats.events_confirmed += 1

        outcome = PredictionOutcome(
            frame_index=frame_index,
            timestamp=timestamp,
            pose_detected=pose_frame is not None,
            sequence_ready=True,
            fall_probability=probability,
            prediction=detection.label,
            above_threshold=detection.above_threshold,
            event=detection.event,
            confidence=probability,
            camera_id=self.camera_id,
        )
        self._last_outcome = outcome
        if detection.event is not None and self._on_event is not None:
            self._notify(detection.event)
        return outcome

    def _notify(self, event: FallEvent) -> None:
        try:
            self._on_event(event)
        except Exception:
            logger.error(
                "Event callback raised for event_id=%s; the event itself is unaffected",
                event.event_id,
                exc_info=True,
            )

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    @property
    def last_outcome(self) -> PredictionOutcome | None:
        return self._last_outcome

    @property
    def last_pose(self) -> PoseFrame | None:
        return self._last_pose

    def model_info(self) -> ModelInfo:
        return self.predictor.info()

    def current_pose(self) -> list[list[float]] | None:
        """Latest 33x4 landmark array as plain lists (for the ``/pose/latest`` route)."""
        with self._lock:
            if self._last_pose is None:
                return None
            return self._last_pose.landmarks.astype(float).tolist()

    def status(self) -> dict[str, Any]:
        """Aggregate runtime status for ``GET /status``."""
        with self._lock:
            outcome = self._last_outcome
            return {
                "pipeline": {
                    **self.stats.to_dict(),
                    "buffered_frames": len(self.buffer),
                    "sequence_length": self.buffer.sequence_length,
                    "sequence_ready": self.buffer.is_ready,
                    "missing_ratio": round(self.buffer.missing_ratio, 4),
                    "pose_estimator_available": self.pose_available,
                },
                "detector": {
                    **self.detector.stats.to_dict(),
                    "confidence_threshold": self.detector.threshold,
                    "min_fall_frames": self.detector.min_fall_frames,
                    "cooldown_seconds": self.detector.cooldown_seconds,
                    "in_cooldown": self.detector.in_cooldown,
                    "cooldown_remaining_seconds": round(self.detector.cooldown_remaining(), 2),
                    "consecutive_above_threshold": self.detector.consecutive_frames,
                },
                "last_prediction": outcome.to_dict() if outcome is not None else None,
            }

    def reset(self) -> None:
        """Clear the buffer and detector state (used between demo/test runs)."""
        with self._lock:
            self.buffer.reset()
            self.detector.reset()
            self._last_pose = None
            self._last_outcome = None

    @property
    def connections(self) -> tuple[tuple[int, int], ...]:
        return POSE_CONNECTIONS

    @staticmethod
    def event_status_name(status: EventStatus) -> str:
        return status.value
