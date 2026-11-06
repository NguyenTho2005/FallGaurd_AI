"""Application service layer.

:class:`InferenceService` is the single place where the AI pipeline, the event store and
the alert channel are combined. Both the API routes and the optional camera monitor thread
call into it, so "confirm an event" always means the same sequence of operations:

    1. the detector confirms a fall
    2. the event is **persisted first**
    3. the alert is attempted, and its outcome is recorded on the event

Persisting before alerting is deliberate: a Telegram outage can delay a notification but
must never lose the event.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

import numpy as np

from ai.config import Settings, get_settings
from ai.inference.fall_detector import EventStatus, FallEvent
from ai.inference.pipeline import FallDetectionPipeline, PredictionOutcome
from ai.logging_config import get_logger
from ai.pose.landmarks import POSE_CONNECTIONS
from backend.alerts.telegram import AlertResult, TelegramAlertSender, build_sender
from backend.database.repository import EventRepository

logger = get_logger(__name__)


class InferenceService:
    """Own the pipeline, the event store and the alert channel for one process."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        pipeline: FallDetectionPipeline | None = None,
        repository: EventRepository | None = None,
        alerter: TelegramAlertSender | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.repository = repository or EventRepository(self._settings.db_path)
        self.alerter = alerter or build_sender(self._settings)
        self.pipeline = pipeline or FallDetectionPipeline(
            self._settings, on_event=self.handle_event
        )
        self.monitor: "CameraMonitor | None" = None
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self.alerts_sent = 0
        self.alerts_failed = 0

    # ------------------------------------------------------------------ #
    # event handling
    # ------------------------------------------------------------------ #
    def handle_event(self, event: FallEvent) -> dict[str, Any]:
        """Persist a confirmed fall, then alert. Never raises."""
        try:
            self.repository.add(event)
        except Exception:
            logger.error("Failed to persist event %s", event.event_id, exc_info=True)
            return {"persisted": False, "alert": {"sent": False, "error": "persistence failed"}}

        alert: AlertResult
        try:
            alert = self.alerter.send_fall_alert(event)
        except Exception:
            logger.error("Alert dispatch raised for event %s", event.event_id, exc_info=True)
            alert = AlertResult(sent=False, dry_run=False, message="", error="alert dispatch failed")

        if alert.sent:
            self.alerts_sent += 1
            self.repository.update_status(event.event_id, EventStatus.ALERTED, notified=True)
            logger.warning(
                "FALL ALERT SENT: event_id=%s camera=%s confidence=%.3f",
                event.event_id, event.camera_id, event.confidence,
            )
        else:
            self.alerts_failed += 1
            # `alerted` would be a lie, so the event stays `detected` and the reason is recorded.
            self.repository.update_status(
                event.event_id, EventStatus.DETECTED, notified=False, note=alert.error or ""
            )
            logger.warning(
                "FALL ALERT NOT SENT (event still recorded): event_id=%s dry_run=%s error=%s",
                event.event_id, alert.dry_run, alert.error,
            )
        return {"persisted": True, "alert": alert.to_dict()}

    # ------------------------------------------------------------------ #
    # inference passthrough
    # ------------------------------------------------------------------ #
    def process_frame(
        self, frame: np.ndarray, *, timestamp: float | None = None, frame_index: int = 0
    ) -> PredictionOutcome:
        return self.pipeline.process_frame(frame, timestamp=timestamp, frame_index=frame_index)

    def process_landmarks(
        self, landmarks: np.ndarray | list | None, *, timestamp: float | None = None,
        frame_index: int = 0,
    ) -> PredictionOutcome:
        return self.pipeline.process_landmarks(
            landmarks, timestamp=timestamp, frame_index=frame_index
        )

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        settings = self._settings
        model_info = self.pipeline.model_info()
        return {
            "system": "online",
            "app": {"name": settings.app_name, "version": settings.version,
                    "environment": settings.environment},
            "model": {
                **model_info.to_dict(),
                "architecture": self._metadata_architecture(),
                "checkpoint_present": bool(settings.model_path.is_file()),
            },
            "camera": {
                "id": settings.camera_id,
                "source": settings.camera_source,
                "monitor_enabled": settings.enable_monitor,
                "connected": self._camera_connected(),
            },
            "config": {
                "sequence_length": settings.sequence_length,
                "confidence_threshold": settings.confidence_threshold,
                "min_fall_frames": settings.min_fall_frames,
                "cooldown_seconds": settings.cooldown_seconds,
                "max_missing_ratio": settings.max_missing_ratio,
                "store_raw_video": settings.store_raw_video,
            },
            "telegram": self.alerter.status(),
            "alerts": {"sent": self.alerts_sent, "failed": self.alerts_failed},
            "events": self.repository.summary(),
            **self.pipeline.status(),
        }

    def _camera_connected(self) -> bool:
        if self.monitor is not None and self.monitor.is_running:
            return self.monitor.source_connected
        # Without a monitor thread the camera state is unknown rather than "connected".
        return False

    def _metadata_architecture(self) -> str | None:
        metadata = self.pipeline.predictor.metadata
        value = metadata.get("architecture")
        return str(value) if value else None

    def latest_pose(self) -> dict[str, Any] | None:
        """Latest skeleton for the dashboard. Landmarks only -- never pixels."""
        landmarks = self.pipeline.current_pose()
        if landmarks is None:
            return None
        return {
            "camera_id": self._settings.camera_id,
            "landmarks": landmarks,
            "connections": [list(pair) for pair in POSE_CONNECTIONS],
            "frame_index": self.pipeline.last_outcome.frame_index
            if self.pipeline.last_outcome
            else 0,
            "privacy_note": (
                "Skeleton landmarks only. FallGuard AI does not retain or serve raw video."
            ),
        }

    # ------------------------------------------------------------------ #
    # optional camera monitor
    # ------------------------------------------------------------------ #
    def start_monitor(self, source_kind: str = "camera", path: str | None = None) -> None:
        """Start the background camera monitor (only when enabled in configuration)."""
        if not self._settings.enable_monitor:
            logger.warning(
                "Monitor start requested but FALLGUARD_ENABLE_MONITOR is false; ignoring"
            )
            return
        with self._lock:
            if self.monitor is not None and self.monitor.is_running:
                logger.warning("Monitor is already running")
                return
            self.monitor = CameraMonitor(
                self, source_kind=source_kind, path=path, settings=self._settings
            )
            self.monitor.start()
        logger.info("Camera monitor started (source=%s)", source_kind)

    def stop_monitor(self) -> None:
        with self._lock:
            monitor = self.monitor
        if monitor is not None:
            monitor.stop()
        logger.info("Camera monitor stopped")

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self.stop_monitor()
        try:
            self.pipeline.close()
        except Exception:  # pragma: no cover - defensive
            logger.warning("Error closing the pipeline", exc_info=True)
        try:
            self.alerter.close()
        except Exception:  # pragma: no cover - defensive
            logger.warning("Error closing the alerter", exc_info=True)
        try:
            self.repository.close()
        except Exception:  # pragma: no cover - defensive
            logger.warning("Error closing the event store", exc_info=True)
        logger.info("Inference service closed")

    def __enter__(self) -> "InferenceService":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class CameraMonitor:
    """Background thread that pumps frames from a source into the pipeline.

    Off by default (``FALLGUARD_ENABLE_MONITOR``). It exists so a deployment can run the
    whole system as one process; the MVP API is happy without it.
    """

    def __init__(
        self,
        service: InferenceService,
        *,
        source_kind: str = "camera",
        path: str | None = None,
        settings: Settings | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self._service = service
        self._settings = settings or get_settings()
        self._source_kind = source_kind
        self._path = path
        self._on_error = on_error
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._source: Any | None = None
        self.frames = 0
        self.errors = 0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def source_connected(self) -> bool:
        return bool(getattr(self._source, "is_connected", False))

    def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="fallguard-camera-monitor", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        if self._source is not None:
            try:
                self._source.close()
            except Exception:  # pragma: no cover - defensive
                logger.warning("Error closing the monitor source", exc_info=True)
            self._source = None

    def _run(self) -> None:  # pragma: no cover - exercised only with a live source
        from ai.sources.video_source import build_source

        try:
            self._source = build_source(
                self._source_kind, settings=self._settings, path=self._path
            )
        except Exception as exc:
            self.errors += 1
            logger.error("Monitor could not open source %r: %s", self._source_kind, exc)
            if self._on_error:
                self._on_error(exc)
            return

        logger.info("Monitor thread started (source=%s)", self._source_kind)
        try:
            while not self._stop.is_set():
                try:
                    source_frame = self._source.read()
                except Exception as exc:
                    self.errors += 1
                    logger.error("Monitor source raised on read: %s", exc, exc_info=True)
                    if self._on_error:
                        self._on_error(exc)
                    break
                if source_frame is None:
                    if self._source_kind == "camera":
                        self._stop.wait(self._settings.target_fps / 4.0)
                        continue
                    logger.info("Monitor source %r reached end of stream", self._source_kind)
                    break
                try:
                    self._service.process_frame(
                        source_frame.frame,
                        timestamp=source_frame.timestamp,
                        frame_index=source_frame.index,
                    )
                    self.frames += 1
                except Exception as exc:
                    self.errors += 1
                    logger.error("Monitor failed to process a frame: %s", exc, exc_info=True)
        finally:
            logger.info(
                "Monitor thread finished after %d frames (%d errors)", self.frames, self.errors
            )
