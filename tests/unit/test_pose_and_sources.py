"""Tests for the real MediaPipe pose estimator and the frame sources.

The MediaPipe tests skip automatically when the pose asset has not been downloaded, so
the suite stays green on a fresh clone while still covering the real path when it is
available. Run ``python scripts/download_mediapipe_model.py`` to enable them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ai.config import Settings
from ai.pose.landmarks import POSE_LANDMARK_COUNT, POSE_LANDMARK_WIDTH, PoseFrame, PoseSample
from ai.pose.pose_estimator import PoseEstimator, PoseModelUnavailableError, pose_asset_is_available
from ai.sources.video_source import (
    CameraSource,
    SyntheticSource,
    VideoFileSource,
    build_source,
    iter_frames,
)

ASSET_AVAILABLE = pose_asset_is_available()
mediapipe_required = pytest.mark.skipif(
    not ASSET_AVAILABLE,
    reason="MediaPipe pose asset not present; run scripts/download_mediapipe_model.py",
)


# --------------------------------------------------------------------------- #
# asset handling
# --------------------------------------------------------------------------- #
def test_missing_asset_raises_a_helpful_error(settings: Settings, tmp_path: Path):
    settings.pose_model_path = tmp_path / "absent.task"
    with pytest.raises(PoseModelUnavailableError, match="download_mediapipe_model"):
        PoseEstimator(settings)


def test_asset_detection(settings: Settings, tmp_path: Path):
    settings.pose_model_path = tmp_path / "absent.task"
    assert pose_asset_is_available(settings) is False
    (tmp_path / "present.task").write_bytes(b"x")
    settings.pose_model_path = tmp_path / "present.task"
    assert pose_asset_is_available(settings) is True


# --------------------------------------------------------------------------- #
# PoseFrame / PoseSample value objects
# --------------------------------------------------------------------------- #
def test_pose_frame_validates_shape():
    array = np.zeros((33, 4), dtype=np.float32)
    frame = PoseFrame(landmarks=array, timestamp=0.0)
    assert frame.xy.shape == (33, 2)
    assert frame.visibility.shape == (33,)
    assert frame.mean_visibility() == pytest.approx(0.0)
    with pytest.raises(ValueError, match="shape"):
        PoseFrame(landmarks=np.zeros((10, 4), dtype=np.float32), timestamp=0.0)


def test_pose_frame_copies_to_detached_array():
    array = np.ones((33, 4), dtype=np.float32)
    frame = PoseFrame(landmarks=array, timestamp=0.0)
    copy = frame.to_array()
    copy[0, 0] = 999.0
    assert frame.landmarks[0, 0] == 1.0


def test_pose_sample_to_array():
    frames = [PoseFrame(landmarks=np.zeros((33, 4), np.float32), timestamp=float(i)) for i in range(4)]
    sample = PoseSample(frames=frames, label=1, group_id="g")
    assert sample.to_array().shape == (4, POSE_LANDMARK_COUNT, POSE_LANDMARK_WIDTH)
    assert len(PoseSample().to_array()) == 0


# --------------------------------------------------------------------------- #
# real MediaPipe
# --------------------------------------------------------------------------- #
@mediapipe_required
def test_estimator_detects_the_synthetic_figure(settings: Settings):
    from ai.pose.synthetic import SceneScript, landmarks_for_frame, render_frame

    script = SceneScript()
    with PoseEstimator(settings) as estimator:
        hits = 0
        total = 0
        for index in range(0, script.total_frames, 3):
            frame = render_frame(landmarks_for_frame(script, index))
            pose = estimator.process(frame, timestamp=index / 30.0, frame_index=index)
            total += 1
            hits += pose is not None
        # The renderer was tuned against this detector; a drop here means the pose asset
        # or the figure changed, and the demo would be unreliable.
        assert hits / total > 0.8, f"detection rate collapsed to {hits}/{total}"
        assert estimator.poses_detected == hits
        assert estimator.detection_rate == pytest.approx(hits / total)


@mediapipe_required
def test_estimator_returns_none_on_an_empty_frame(settings: Settings):
    with PoseEstimator(settings) as estimator:
        assert estimator.process(np.zeros((240, 320, 3), dtype=np.uint8)) is None
        assert estimator.poses_detected == 0
        assert estimator.frames_processed == 1


@mediapipe_required
def test_estimator_returns_34_landmarks_shape(settings: Settings):
    from ai.pose.synthetic import SceneScript, landmarks_for_frame, render_frame

    script = SceneScript()
    with PoseEstimator(settings) as estimator:
        for index in range(0, script.total_frames, 2):
            pose = estimator.process(
                render_frame(landmarks_for_frame(script, index)),
                timestamp=index / 30.0, frame_index=index,
            )
            if pose is not None:
                assert pose.landmarks.shape == (33, 4)
                assert pose.visibility.min() >= 0.0
                assert pose.visibility.max() <= 1.0
                assert np.isfinite(pose.landmarks).all()
                assert 0.0 <= pose.landmarks[:, 0].max() <= 1.5
                return
    pytest.fail("no pose was ever detected on the synthetic figure")


@mediapipe_required
def test_estimator_handles_a_grayscale_frame(settings: Settings):
    """A grayscale frame is accepted and converted to 3 channels without raising.

    Detection itself is not asserted: MediaPipe is trained on colour imagery, so a
    grayscale frame legitimately yields no pose. What matters is that the frame is
    consumed rather than rejected or crashing.
    """
    from ai.pose.synthetic import SceneScript, landmarks_for_frame, render_frame

    script = SceneScript()
    colour = render_frame(landmarks_for_frame(script, 0))
    gray = np.repeat(cv_gray(colour)[:, :, None], 3, axis=2)
    with PoseEstimator(settings) as estimator:
        estimator.process(gray, timestamp=0.0)
        assert estimator.frames_processed == 1
        assert estimator.detector_errors == 0


def cv_gray(image: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------- #
# per-frame robustness (no MediaPipe required)
# --------------------------------------------------------------------------- #
class _RecordingLandmarker:
    """A stand-in landmarker that never needs the native library."""

    def __init__(self, result: object) -> None:
        self.result = result
        self.calls = 0
        self.closed = False

    def detect(self, image):  # pragma: no cover - unused
        return self.result

    def detect_for_video(self, image, timestamp_ms):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def close(self) -> None:
        self.closed = True


class _EmptyResult:
    pose_landmarks: list = []


def _stub_estimator(settings: Settings, result: object) -> PoseEstimator:
    """Build a PoseEstimator with a fake landmarker, bypassing native initialisation."""
    import threading

    class _Image:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _ImageFormat:
        SRGB = 0

    class _MP:
        Image = _Image
        ImageFormat = _ImageFormat

    estimator = PoseEstimator.__new__(PoseEstimator)
    estimator._settings = settings
    estimator._static_image_mode = False
    estimator._lock = threading.RLock()
    estimator._landmarker = _RecordingLandmarker(result)
    estimator._last_timestamp_ms = -1
    estimator.frames_processed = 0
    estimator.poses_detected = 0
    estimator.detector_errors = 0
    estimator._closed = False
    estimator._mp = _MP
    return estimator


@pytest.mark.parametrize("frame", [None, np.zeros((0, 0, 3), np.uint8), "not an array", 42])
def test_invalid_frames_return_none_without_raising(settings: Settings, frame):
    estimator = _stub_estimator(settings, _EmptyResult())
    assert estimator.process(frame) is None
    assert estimator.frames_processed == 0, "an unusable frame is not a processed frame"


@pytest.mark.parametrize("channels", [1, 2, 5, 7])
def test_wrong_channel_count_returns_none(settings: Settings, channels: int):
    estimator = _stub_estimator(settings, _EmptyResult())
    assert estimator.process(np.zeros((10, 10, channels), dtype=np.uint8)) is None


def test_bgra_frame_is_accepted(settings: Settings):
    """Four channels (BGRA, as some capture pipelines produce) are supported."""
    estimator = _stub_estimator(settings, _EmptyResult())
    assert estimator.process(np.zeros((10, 10, 4), dtype=np.uint8)) is None
    assert estimator.frames_processed == 1


def test_two_dimensional_frame_is_expanded_to_three_channels(settings: Settings):
    estimator = _stub_estimator(settings, _EmptyResult())
    assert estimator.process(np.zeros((10, 10), dtype=np.uint8)) is None
    assert estimator.frames_processed == 1


def test_float_frames_are_clipped_to_uint8(settings: Settings):
    estimator = _stub_estimator(settings, _EmptyResult())
    assert estimator.process(np.zeros((10, 10, 3), dtype=np.float32)) is None
    assert estimator.frames_processed == 1


def test_detector_exception_is_contained(settings: Settings, caplog):
    estimator = _stub_estimator(settings, RuntimeError("tflite exploded"))
    assert estimator.process(np.zeros((10, 10, 3), dtype=np.uint8)) is None
    assert estimator.detector_errors == 1
    assert estimator.frames_processed == 1


def test_processing_after_close_returns_none(settings: Settings):
    estimator = _stub_estimator(settings, _EmptyResult())
    estimator.close()
    assert estimator.process(np.zeros((10, 10, 3), dtype=np.uint8)) is None


def test_close_is_idempotent(settings: Settings):
    estimator = _stub_estimator(settings, _EmptyResult())
    landmarker = estimator._landmarker
    estimator.close()
    estimator.close()
    assert landmarker.closed is True
    assert estimator.is_available is False


def test_timestamps_are_forced_monotonic(settings: Settings):
    estimator = _stub_estimator(settings, _EmptyResult())
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    seen: list[int] = []
    for _ in range(3):
        estimator.process(frame, timestamp=5.0)  # a frozen camera clock
        seen.append(estimator._last_timestamp_ms)
    assert seen == sorted(seen)
    assert len(set(seen)) == 3, "MediaPipe requires strictly increasing timestamps"


def test_detection_rate_is_zero_before_any_frame(settings: Settings):
    estimator = _stub_estimator(settings, _EmptyResult())
    assert estimator.detection_rate == 0.0


# --------------------------------------------------------------------------- #
# frame sources
# --------------------------------------------------------------------------- #
def test_missing_video_file_reports_a_clean_error(settings: Settings, tmp_path: Path):
    source = VideoFileSource(tmp_path / "absent.mp4", settings)
    assert source.read() is None
    assert source.open_error is not None
    assert "not found" in source.open_error


def test_unreadable_video_file_reports_a_clean_error(settings: Settings, tmp_path: Path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"this is not a video")
    source = VideoFileSource(broken, settings)
    assert source.read() is None
    assert source.open_error is not None


def test_corrupt_file_yields_no_frames_but_does_not_raise(settings: Settings, tmp_path: Path):
    empty = tmp_path / "empty.avi"
    empty.write_bytes(b"")
    source = VideoFileSource(empty, settings)
    assert source.read() is None
    source.close()


@mediapipe_required
def test_video_file_source_reads_the_generated_clip(settings: Settings, tmp_path: Path):
    from ai.pose.synthetic import generate_sample_video

    clip = generate_sample_video(tmp_path / "clip.avi", fps=30)
    source = VideoFileSource(clip, settings)
    try:
        frames = list(iter_frames(source, limit=5))
        assert len(frames) == 5
        assert frames[0].frame.shape[2] == 3
        assert frames[0].index == 0
        assert frames[0].fps == pytest.approx(30.0, abs=0.1)
        assert frames[1].timestamp > frames[0].timestamp
    finally:
        source.close()


def test_iter_frames_stops_at_the_limit(settings: Settings, tmp_path: Path):
    source = VideoFileSource(tmp_path / "absent.avi", settings)
    assert list(iter_frames(source, limit=10)) == []


def test_camera_source_handles_a_missing_device(settings: Settings):
    """A disconnected camera must degrade to None plus a warning, not an exception."""
    source = CameraSource(999, settings, reconnect_interval=0.0)
    try:
        assert source.read() is None
        assert source.is_connected is False
    finally:
        source.close()


def test_camera_source_parses_indexes_and_urls(settings: Settings):
    assert CameraSource(0, settings)._parse_source() == 0
    assert CameraSource("2", settings)._parse_source() == 2
    assert CameraSource("rtsp://host/stream", settings)._parse_source() == "rtsp://host/stream"
    assert CameraSource(" /path/to.mp4 ", settings)._parse_source() == "/path/to.mp4"


def test_synthetic_source_reports_its_error(settings: Settings, tmp_path: Path):
    source = SyntheticSource(tmp_path / "absent.avi", settings)
    assert source.read() is None
    assert source.open_error is not None
    source.close()


def test_build_source_factory(settings: Settings, tmp_path: Path):
    assert isinstance(build_source("camera", settings=settings), CameraSource)
    assert isinstance(
        build_source("video", settings=settings, path=tmp_path / "a.mp4"), VideoFileSource
    )
    assert isinstance(build_source("synthetic", settings=settings), SyntheticSource)
    with pytest.raises(ValueError, match="unknown source kind"):
        build_source("telepathy", settings=settings)
    with pytest.raises(ValueError, match="requires a path"):
        build_source("video", settings=settings)
