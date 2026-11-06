"""MediaPipe Pose wrapper -- the only module that talks to MediaPipe.

Responsibilities
----------------
* Load the ``pose_landmarker`` task asset (legacy ``mp.solutions`` no longer exists
  in mediapipe 1.x, so the Tasks API is used).
* Convert a BGR frame into a :class:`~ai.pose.landmarks.PoseFrame`, or return ``None``
  when no person is visible.
* Never raise on a single bad frame: an unreadable frame, a landmark with NaN
  coordinates or a detector internal error is logged at DEBUG/WARNING level and turned
  into "no pose" so the calling pipeline keeps running.
* Release the native resources deterministically via the context-manager protocol.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

from ai.config import Settings, get_settings
from ai.logging_config import get_logger
from ai.pose.landmarks import POSE_LANDMARK_COUNT, POSE_LANDMARK_WIDTH, PoseFrame

logger = get_logger(__name__)

# MediaPipe / TensorFlow write a lot of noise to stderr before logging is configured.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


class PoseModelUnavailableError(RuntimeError):
    """Raised when the MediaPipe pose asset cannot be found or loaded."""


def pose_asset_is_available(settings: Settings | None = None) -> bool:
    """Return ``True`` when the pose landmarker asset exists on disk."""
    settings = settings or get_settings()
    return Path(settings.pose_model_path).is_file()


class PoseEstimator:
    """Extract 33 pose landmarks from BGR frames.

    Parameters
    ----------
    settings:
        Configuration; defaults to the process-wide settings.
    static_image_mode:
        When ``True`` each frame is processed independently. When ``False`` (default)
        the landmarker runs in ``VIDEO`` mode, which enables temporal tracking and
        requires monotonically increasing timestamps.
    """

    def __init__(self, settings: Settings | None = None, *, static_image_mode: bool = False) -> None:
        self._settings = settings or get_settings()
        self._static_image_mode = static_image_mode
        self._lock = threading.Lock()
        self._landmarker: Any | None = None
        self._last_timestamp_ms: int = -1
        self.frames_processed = 0
        self.poses_detected = 0
        self.detector_errors = 0
        self._closed = False
        self._ensure_model()

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def _ensure_model(self) -> None:
        model_path = Path(self._settings.pose_model_path)
        if not model_path.is_file():
            raise PoseModelUnavailableError(
                f"MediaPipe pose asset not found at '{model_path}'. "
                "Run: python scripts/download_mediapipe_model.py"
            )
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise PoseModelUnavailableError(
                "mediapipe is not installed. Install dependencies with: "
                "pip install -r requirements.txt"
            ) from exc

        self._mp = mp
        self._vision = vision
        running_mode = (
            vision.RunningMode.IMAGE if self._static_image_mode else vision.RunningMode.VIDEO
        )
        # mediapipe renamed `min_pose_tracking_confidence` to `min_tracking_confidence`
        # between releases, so the keyword is resolved against the actual signature
        # rather than hard-coded.
        import inspect

        parameters = inspect.signature(vision.PoseLandmarkerOptions.__init__).parameters
        tracking_keyword = (
            "min_pose_tracking_confidence"
            if "min_pose_tracking_confidence" in parameters
            else "min_tracking_confidence"
        )
        options_kwargs: dict[str, Any] = {
            "base_options": BaseOptions(model_asset_path=str(model_path)),
            "running_mode": running_mode,
            "num_poses": self._settings.pose_num_poses,
            "min_pose_detection_confidence": self._settings.pose_min_detection_confidence,
            "min_pose_presence_confidence": self._settings.pose_min_presence_confidence,
            "output_segmentation_masks": False,
        }
        if tracking_keyword in parameters:
            options_kwargs[tracking_keyword] = self._settings.pose_min_tracking_confidence
        options = vision.PoseLandmarkerOptions(**options_kwargs)
        try:
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
        except Exception as exc:  # pragma: no cover - native failure path
            raise PoseModelUnavailableError(
                f"Failed to initialise MediaPipe PoseLandmarker from '{model_path}': {exc}"
            ) from exc
        logger.info(
            "MediaPipe Pose landmarker loaded (model=%s, mode=%s, num_poses=%d)",
            model_path.name,
            "IMAGE" if self._static_image_mode else "VIDEO",
            self._settings.pose_num_poses,
        )

    def close(self) -> None:
        """Release the native landmarker. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        landmarker, self._landmarker = self._landmarker, None
        if landmarker is None:
            return
        try:
            landmarker.close()
        except Exception:  # pragma: no cover - defensive
            logger.warning("Error while closing MediaPipe landmarker", exc_info=True)
        else:
            logger.debug("MediaPipe landmarker closed")

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - GC safety net
        try:
            self.close()
        except Exception:
            logger.debug("PoseEstimator.__del__ failed", exc_info=True)

    # ------------------------------------------------------------------ #
    # properties
    # ------------------------------------------------------------------ #
    @property
    def is_available(self) -> bool:
        """``True`` while the native landmarker is loaded."""
        return self._landmarker is not None and not self._closed

    @property
    def detection_rate(self) -> float:
        """Share of processed frames in which a pose was found."""
        if self.frames_processed == 0:
            return 0.0
        return self.poses_detected / self.frames_processed

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def process(self, frame: np.ndarray, *, timestamp: float | None = None,
                frame_index: int = 0) -> PoseFrame | None:
        """Extract landmarks from a BGR frame.

        Returns ``None`` when no pose is detected or the frame cannot be processed.
        This method does not raise for per-frame problems.
        """
        if self._landmarker is None or self._closed:
            logger.error("PoseEstimator.process called after close()")
            return None
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.warning("PoseEstimator received an empty or invalid frame; skipping")
            return None
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            logger.warning("PoseEstimator received a frame with shape %s; skipping", (frame.shape,))
            return None

        self.frames_processed += 1
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        height, width = frame.shape[:2]
        try:
            import cv2  # local import: keeps module import cheap for non-video callers
            rgb = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2RGB)
        except Exception:  # pragma: no cover - cv2 is a hard dependency
            logger.warning("BGR->RGB conversion failed; treating channels as RGB", exc_info=True)
            rgb = frame[:, :, :3]

        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

        with self._lock:
            try:
                if self._static_image_mode:
                    result = self._landmarker.detect(image)
                else:
                    timestamp_ms = self._next_timestamp_ms(timestamp)
                    result = self._landmarker.detect_for_video(image, timestamp_ms)
            except Exception:
                self.detector_errors += 1
                logger.warning(
                    "MediaPipe inference failed on frame %d; treating as no-pose", frame_index,
                    exc_info=True,
                )
                return None

        pose_frame = self._to_pose_frame(result, timestamp, frame_index, width, height)
        if pose_frame is None:
            logger.debug("No pose detected on frame %d", frame_index)
        else:
            self.poses_detected += 1
        return pose_frame

    def _next_timestamp_ms(self, timestamp: float | None) -> int:
        """MediaPipe VIDEO mode requires strictly increasing millisecond timestamps."""
        if timestamp is None:
            step = 1.0 / max(self._settings.target_fps, 1.0)
            timestamp = (self._last_timestamp_ms + 1) / 1000.0 + step
        timestamp_ms = int(round(timestamp * 1000.0))
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def _to_pose_frame(
        self,
        result: Any,
        timestamp: float | None,
        frame_index: int,
        width: int,
        height: int,
    ) -> PoseFrame | None:
        landmark_sets = getattr(result, "pose_landmarks", None)
        if not landmark_sets:
            return None
        best = landmark_sets[0]
        if best is None or len(best) != POSE_LANDMARK_COUNT:
            logger.warning(
                "MediaPipe returned %s landmarks, expected %d; skipping frame",
                0 if best is None else len(best),
                POSE_LANDMARK_COUNT,
            )
            return None
        array = np.empty((POSE_LANDMARK_COUNT, POSE_LANDMARK_WIDTH), dtype=np.float32)
        for index, landmark in enumerate(best):
            visibility = float(getattr(landmark, "visibility", 0.0) or 0.0)
            # A landmark can be reported with NaN/inf coordinates by the detector when
            # the joint is outside the frame; neutralise it instead of propagating.
            x = _safe_float(getattr(landmark, "x", 0.0))
            y = _safe_float(getattr(landmark, "y", 0.0))
            z = _safe_float(getattr(landmark, "z", 0.0))
            if not np.isfinite([x, y, z]).all():
                x, y, z, visibility = 0.0, 0.0, 0.0, 0.0
            array[index] = (x, y, z, min(max(visibility, 0.0), 1.0))
        if not np.isfinite(array).all():  # pragma: no cover - defensive
            logger.warning("Non-finite values in landmark array; skipping frame")
            return None
        resolved_ts = timestamp if timestamp is not None else float(frame_index)
        try:
            return PoseFrame(
                landmarks=array,
                timestamp=float(resolved_ts),
                frame_index=int(frame_index),
                width=int(width),
                height=int(height),
            )
        except ValueError:  # pragma: no cover - shape already validated
            logger.warning("Failed to build PoseFrame for frame %d", frame_index, exc_info=True)
            return None


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if np.isfinite(result) else 0.0
