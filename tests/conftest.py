"""Shared pytest fixtures.

Two rules the whole suite depends on:

1. **No GPU and no network.** TensorFlow is imported through the ``tensorflow-cpu``
   wheel, and every HTTP call is mocked.
2. **No trained checkpoint required.** Tests run with
   ``FALLGUARD_MODEL_BACKEND=mock`` so the suite is meaningful on a fresh clone. The
   Keras tests build tiny models in-process instead of loading one from disk.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Configure the environment before any application module reads settings.
os.environ.setdefault("FALLGUARD_ENVIRONMENT", "test")
os.environ.setdefault("FALLGUARD_MODEL_BACKEND", "mock")
os.environ.setdefault("FALLGUARD_LOG_LEVEL", "WARNING")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

from ai.config import Settings, get_settings, reload_settings  # noqa: E402
from ai.logging_config import configure_logging, silence_external_loggers  # noqa: E402
from ai.pose.landmarks import (  # noqa: E402
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from ai.preprocessing.pose_preprocessor import FEATURE_DIM  # noqa: E402

# A person standing upright, and the same person lying on the floor. Both are expressed
# as normalised image coordinates and are used throughout the suite.
STANDING: dict[int, tuple[float, float]] = {
    NOSE: (0.500, 0.130),
    LEFT_SHOULDER: (0.440, 0.270),
    RIGHT_SHOULDER: (0.560, 0.270),
    LEFT_ELBOW: (0.420, 0.410),
    RIGHT_ELBOW: (0.580, 0.410),
    LEFT_WRIST: (0.410, 0.550),
    RIGHT_WRIST: (0.590, 0.550),
    LEFT_HIP: (0.465, 0.545),
    RIGHT_HIP: (0.535, 0.545),
    LEFT_KNEE: (0.470, 0.710),
    RIGHT_KNEE: (0.530, 0.710),
    LEFT_ANKLE: (0.475, 0.875),
    RIGHT_ANKLE: (0.525, 0.875),
}
ON_FLOOR: dict[int, tuple[float, float]] = {
    NOSE: (0.150, 0.770),
    LEFT_SHOULDER: (0.265, 0.795),
    RIGHT_SHOULDER: (0.305, 0.860),
    LEFT_ELBOW: (0.360, 0.690),
    RIGHT_ELBOW: (0.400, 0.955),
    LEFT_WRIST: (0.460, 0.640),
    RIGHT_WRIST: (0.505, 0.985),
    LEFT_HIP: (0.500, 0.845),
    RIGHT_HIP: (0.545, 0.885),
    LEFT_KNEE: (0.680, 0.855),
    RIGHT_KNEE: (0.700, 0.960),
    LEFT_ANKLE: (0.850, 0.855),
    RIGHT_ANKLE: (0.870, 1.010),
}
#: A forward bend: the hip is low but the torso is still vertical. The case a naive
#: "hip dropped" rule would get wrong.
BENDING: dict[int, tuple[float, float]] = {
    NOSE: (0.500, 0.400),
    LEFT_SHOULDER: (0.450, 0.550),
    RIGHT_SHOULDER: (0.550, 0.550),
    LEFT_ELBOW: (0.420, 0.660),
    RIGHT_ELBOW: (0.580, 0.660),
    LEFT_WRIST: (0.400, 0.740),
    RIGHT_WRIST: (0.600, 0.740),
    LEFT_HIP: (0.465, 0.700),
    RIGHT_HIP: (0.535, 0.700),
    LEFT_KNEE: (0.470, 0.800),
    RIGHT_KNEE: (0.530, 0.800),
    LEFT_ANKLE: (0.475, 0.920),
    RIGHT_ANKLE: (0.525, 0.920),
}


#: Where landmarks absent from a pose mapping are placed. Facial and foot landmarks are
#: irrelevant to fall detection, but they still take part in the bounding box, so they
#: need a defined position -- and that position must transform with the body, otherwise
#: invariance tests measure the fixture rather than the code.
DEFAULT_POINT: tuple[float, float] = (0.5, 0.5)


def landmark_array(
    points: dict[int, tuple[float, float]],
    *,
    visibility: float = 1.0,
    z: float = 0.0,
) -> np.ndarray:
    """Build a ``(33, 4)`` landmark array from a sparse coordinate mapping."""
    array = np.zeros((33, 4), dtype=np.float32)
    for index in range(33):
        x, y = points.get(index, DEFAULT_POINT)
        array[index] = (x, y, z, visibility)
    return array


def transform_points(
    points: dict[int, tuple[float, float]],
    *,
    scale: float = 1.0,
    dx: float = 0.0,
    dy: float = 0.0,
    rotation: float = 0.0,
) -> dict[int, tuple[float, float]]:
    """Apply a similarity transform (scale, translate, rotate) to a coordinate mapping.

    The transform is applied to the **complete** 33-landmark set, filling in
    :data:`DEFAULT_POINT` first, so every landmark moves consistently and an invariance
    assertion really tests the normaliser.
    """
    cos_t, sin_t = np.cos(rotation), np.sin(rotation)
    complete = {index: points.get(index, DEFAULT_POINT) for index in range(33)}
    result: dict[int, tuple[float, float]] = {}
    for index, (x, y) in complete.items():
        sx, sy = x * scale, y * scale
        result[index] = (sx * cos_t - sy * sin_t + dx, sx * sin_t + sy * cos_t + dy)
    return result


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session", autouse=True)
def _session_logging() -> Iterator[None]:
    configure_logging()
    silence_external_loggers()
    yield


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Isolated settings rooted in a temporary directory."""
    get_settings.cache_clear()
    config = Settings(
        environment="test",
        model_backend="mock",
        db_path=tmp_path / "events.db",
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        samples_dir=tmp_path / "samples",
        model_path=tmp_path / "models" / "fallguard_lstm.keras",
        model_metadata_path=tmp_path / "models" / "fallguard_lstm.meta.json",
        log_level="WARNING",
        telegram_dry_run=True,
    )
    config.ensure_directories()
    yield config
    get_settings.cache_clear()


@pytest.fixture
def standing_landmarks() -> np.ndarray:
    return landmark_array(STANDING)


@pytest.fixture
def on_floor_landmarks() -> np.ndarray:
    return landmark_array(ON_FLOOR)


@pytest.fixture
def preprocessor(settings: Settings):
    from ai.preprocessing.pose_preprocessor import PosePreprocessor

    return PosePreprocessor.from_settings(settings)


@pytest.fixture
def standing_window(preprocessor, settings: Settings) -> np.ndarray:
    features = preprocessor.transform(landmark_array(STANDING))
    return np.stack([features] * settings.sequence_length)


@pytest.fixture
def on_floor_window(preprocessor, settings: Settings) -> np.ndarray:
    features = preprocessor.transform(landmark_array(ON_FLOOR))
    return np.stack([features] * settings.sequence_length)


class FakeClock:
    """Manually advanced monotonic clock, so cooldown tests never sleep."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def repository(settings: Settings):
    from backend.database.repository import EventRepository

    store = EventRepository(settings.db_path)
    yield store
    store.close()


@pytest.fixture
def make_event():
    from ai.inference.fall_detector import EventStatus, FallEvent

    def _factory(
        event_id: str = "evt_test_0001",
        *,
        confidence: float = 0.95,
        camera_id: str = "camera_01",
        status: EventStatus = EventStatus.DETECTED,
    ) -> Any:
        return FallEvent(
            event_id=event_id,
            timestamp="2026-01-01T00:00:00Z",
            camera_id=camera_id,
            prediction="FALL",
            confidence=confidence,
            peak_confidence=confidence,
            window_frames=8,
            status=status,
        )

    return _factory


class StubPoseEstimator:
    """Replays a fixed list of poses in place of MediaPipe.

    Lets the integration tests exercise the whole pipeline without loading the native
    landmarker, so they stay fast and GPU-free.
    """

    def __init__(self, poses: list[np.ndarray | None]) -> None:
        self._poses = list(poses)
        self.is_available = True
        self.index = 0
        self.closed = False
        self.frames_processed = 0
        self.poses_detected = 0

    def process(self, frame: np.ndarray, *, timestamp: float | None = None,
                frame_index: int = 0):
        from ai.pose.landmarks import PoseFrame

        self.frames_processed += 1
        if self.index >= len(self._poses):
            return None
        landmarks = self._poses[self.index]
        self.index += 1
        if landmarks is None:
            return None
        self.poses_detected += 1
        return PoseFrame(
            landmarks=np.asarray(landmarks, dtype=np.float32),
            timestamp=timestamp if timestamp is not None else float(frame_index),
            frame_index=frame_index,
        )

    def close(self) -> None:
        self.closed = True
        self.is_available = False

    def __enter__(self) -> "StubPoseEstimator":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


@pytest.fixture
def stub_pose_estimator():
    return StubPoseEstimator


@pytest.fixture
def feature_dim() -> int:
    return FEATURE_DIM


@pytest.fixture
def make_pipeline(settings: Settings):
    """Factory for a pipeline wired to a stub pose estimator and no MediaPipe."""
    from ai.inference.pipeline import FallDetectionPipeline

    created: list[FallDetectionPipeline] = []

    def _factory(poses: list[np.ndarray | None], on_event: Any = None,
                 **overrides: Any) -> FallDetectionPipeline:
        for key, value in overrides.items():
            setattr(settings, key, value)
        estimator = StubPoseEstimator(poses)
        pipeline = FallDetectionPipeline(
            settings, pose_estimator=estimator, on_event=on_event
        )
        created.append(pipeline)
        return pipeline

    yield _factory
    for pipeline in created:
        pipeline.close()
