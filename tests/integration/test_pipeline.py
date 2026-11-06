"""Integration test: the full pipeline from frame to confirmed event.

    frame -> pose -> preprocessing -> sequence -> model -> event confirmation

A stub pose estimator replaces MediaPipe so the test is fast and GPU-free, but every
other stage is the production code path.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai.inference.fall_detector import EventStatus, FallEvent
from ai.inference.pipeline import FallDetectionPipeline
from ai.preprocessing.pose_preprocessor import FEATURE_DIM, PRESENT_FEATURE_INDEX
from tests.conftest import ON_FLOOR, STANDING, landmark_array

pytestmark = pytest.mark.integration


def test_standing_frames_never_confirm_a_fall(make_pipeline, settings):
    poses = [landmark_array(STANDING) for _ in range(settings.sequence_length + 20)]
    pipeline = make_pipeline(poses, min_fall_frames=3, cooldown_seconds=0.0)
    for index in range(len(poses)):
        outcome = pipeline.process_pose(_pose(poses[index], index))
        assert outcome.event is None
        assert outcome.prediction == "NORMAL"
    assert pipeline.stats.events_confirmed == 0
    assert pipeline.stats.poses_detected == len(poses)


def test_lying_frames_confirm_exactly_one_event(make_pipeline, settings):
    poses = [landmark_array(ON_FLOOR) for _ in range(settings.sequence_length + 15)]
    events: list[FallEvent] = []
    pipeline = make_pipeline(
        poses, on_event=events.append, min_fall_frames=3, cooldown_seconds=30.0
    )
    for index in range(len(poses)):
        pipeline.process_pose(_pose(poses[index], index))
    assert len(events) == 1, "a sustained fall must produce one event, not one per frame"
    assert events[0].prediction == "FALL"
    assert events[0].confidence > 0.0
    assert pipeline.stats.events_confirmed == 1


def test_standing_then_falling_produces_one_event(make_pipeline, settings):
    """The realistic sequence: normal activity, then a fall."""
    standing = [landmark_array(STANDING)] * (settings.sequence_length + 5)
    lying = [landmark_array(ON_FLOOR)] * 20
    poses = standing + lying
    events: list[FallEvent] = []
    pipeline = make_pipeline(
        poses, on_event=events.append, min_fall_frames=3, cooldown_seconds=60.0
    )
    outcomes = [
        pipeline.process_pose(_pose(pose, index)) for index, pose in enumerate(poses)
    ]
    assert len(events) == 1
    confirmation_frame = next(
        index for index, outcome in enumerate(outcomes) if outcome.event is not None
    )
    assert confirmation_frame >= len(standing), "no event while the person is standing"
    assert outcomes[confirmation_frame].fall_probability > outcomes[0].fall_probability


def test_pose_dropouts_do_not_produce_events(make_pipeline, settings):
    """A camera pointed at an empty room must stay silent."""
    poses = [None] * (settings.sequence_length + 30)
    events: list[FallEvent] = []
    pipeline = make_pipeline(poses, on_event=events.append, min_fall_frames=3)
    for index in range(len(poses)):
        outcome = pipeline.process_pose(None, frame_index=index)
        assert outcome.pose_detected is False
        assert outcome.event is None
    assert events == []
    assert pipeline.stats.pose_dropouts == len(poses)


def test_dropouts_are_kept_in_the_window(make_pipeline, settings):
    """A gap must occupy a slot; skipping it would compress real time."""
    poses = [None] * 5 + [landmark_array(STANDING)] * settings.sequence_length
    pipeline = make_pipeline(poses, max_missing_ratio=0.9)
    for index, pose in enumerate(poses):
        pipeline.process_pose(_pose(pose, index) if pose is not None else None,
                              frame_index=index)
    window = pipeline.buffer.window()
    assert window is not None
    assert window.shape == (settings.sequence_length, FEATURE_DIM)
    # The five gap frames have since been evicted by the standing frames.
    assert window[-1, PRESENT_FEATURE_INDEX] == 1.0
    assert pipeline.stats.pose_dropouts == 5


def test_interleaved_dropouts_do_not_reset_a_real_fall(make_pipeline, settings):
    poses = [landmark_array(ON_FLOOR)] * (settings.sequence_length + 5) + [None] * 10
    events: list[FallEvent] = []
    pipeline = make_pipeline(
        poses, on_event=events.append, min_fall_frames=3, cooldown_seconds=60.0,
        max_missing_ratio=0.9,
    )
    for index, pose in enumerate(poses):
        pipeline.process_pose(_pose(pose, index) if pose is not None else None, frame_index=index)
    assert len(events) == 1, "a brief dropout must not cancel an ongoing fall"


def test_process_frame_runs_the_stub_estimator(make_pipeline, settings):
    poses = [landmark_array(STANDING)] * 5
    pipeline = make_pipeline(poses)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    for index in range(5):
        outcome = pipeline.process_frame(frame, timestamp=index / 30.0, frame_index=index)
        assert outcome.pose_detected is True
    assert pipeline.pose_estimator.frames_processed == 5
    assert pipeline.pose_estimator.poses_detected == 5


def test_process_frame_without_a_pose_estimator(make_pipeline, settings):
    poses: list = []
    pipeline = make_pipeline(poses)
    pipeline.pose_estimator.close()  # simulate a disconnected/failed MediaPipe
    outcome = pipeline.process_frame(np.zeros((48, 64, 3), dtype=np.uint8))
    assert outcome.pose_detected is False
    assert outcome.sequence_ready is False
    assert pipeline.pose_available is False


def test_process_landmarks_accepts_a_raw_array(make_pipeline, settings):
    poses = [landmark_array(ON_FLOOR)] * (settings.sequence_length + 5)
    pipeline = make_pipeline(poses, min_fall_frames=2)
    events: list[FallEvent] = []
    pipeline._on_event = events.append
    for index in range(settings.sequence_length + 4):
        outcome = pipeline.process_landmarks(landmark_array(ON_FLOOR), frame_index=index)
    assert outcome.pose_detected is True
    assert len(events) == 1


def test_process_landmarks_rejects_a_malformed_array(make_pipeline, settings):
    pipeline = make_pipeline([None])
    outcome = pipeline.process_landmarks(np.zeros((5, 4), dtype=np.float32))
    assert outcome.pose_detected is False


def test_status_reflects_the_run(make_pipeline, settings):
    poses = [landmark_array(STANDING)] * (settings.sequence_length + 5)
    pipeline = make_pipeline(poses)
    for index in range(settings.sequence_length + 2):
        pipeline.process_pose(_pose(poses[index], index))
    status = pipeline.status()
    assert status["pipeline"]["frames_processed"] == settings.sequence_length + 2
    assert status["pipeline"]["sequence_ready"] is True
    assert status["pipeline"]["missing_ratio"] == 0.0
    assert status["last_prediction"]["prediction"] == "NORMAL"
    assert status["detector"]["confidence_threshold"] == settings.confidence_threshold
    assert status["detector"]["min_fall_frames"] == settings.min_fall_frames


def test_current_pose_is_exposed_for_the_dashboard(make_pipeline, settings):
    pipeline = make_pipeline([landmark_array(STANDING)] * 2)
    assert pipeline.current_pose() is None
    pipeline.process_pose(_pose(landmark_array(STANDING), 0))
    landmarks = pipeline.current_pose()
    assert landmarks is not None
    assert np.asarray(landmarks).shape == (33, 4)


def test_reset_clears_pipeline_state(make_pipeline, settings):
    poses = [landmark_array(ON_FLOOR)] * (settings.sequence_length + 5)
    pipeline = make_pipeline(poses, min_fall_frames=2)
    for index in range(settings.sequence_length + 2):
        pipeline.process_pose(_pose(poses[index], index))
    assert pipeline.stats.events_confirmed == 1
    pipeline.reset()
    # Inference state is cleared...
    assert len(pipeline.buffer) == 0
    assert pipeline.detector.consecutive_frames == 0
    assert pipeline.last_outcome is None
    assert pipeline.last_pose is None
    # ...while the lifetime counters persist, so /status keeps a meaningful uptime count.
    assert pipeline.stats.events_confirmed == 1
    assert pipeline.stats.frames_processed == settings.sequence_length + 2


def test_a_failing_event_callback_does_not_break_inference(make_pipeline, settings):
    """Persistence/alerting is downstream of detection; a fault there must not stop it."""
    poses = [landmark_array(ON_FLOOR)] * (settings.sequence_length + 5)
    calls: list[str] = []

    def explode(event: FallEvent) -> None:
        calls.append(event.event_id)
        raise RuntimeError("callback failure")

    pipeline = make_pipeline(poses, on_event=explode, min_fall_frames=2)
    outcomes = [
        pipeline.process_pose(_pose(poses[index], index))
        for index in range(settings.sequence_length + 2)
    ]
    assert calls, "the callback should have been attempted"
    confirmed = [outcome for outcome in outcomes if outcome.event is not None]
    assert confirmed, "the confirmed event must survive a failing callback"
    assert confirmed[0].event.event_id in calls
    assert pipeline.stats.frames_processed == settings.sequence_length + 2


def test_pipeline_survives_a_closed_estimator(make_pipeline, settings):
    poses = [landmark_array(STANDING)] * 3
    pipeline = make_pipeline(poses)
    pipeline.pose_estimator.close()
    pipeline.pose_estimator.process = lambda *args, **kwargs: None  # type: ignore[method-assign]
    for index in range(3):
        pipeline.process_frame(np.zeros((10, 10, 3), dtype=np.uint8), frame_index=index)
    assert pipeline.stats.frames_processed == 3


def test_event_metadata_is_complete(make_pipeline, settings):
    poses = [landmark_array(ON_FLOOR)] * (settings.sequence_length + 5)
    events: list[FallEvent] = []
    pipeline = make_pipeline(poses, on_event=events.append, min_fall_frames=2)
    for index in range(settings.sequence_length + 2):
        pipeline.process_pose(_pose(poses[index], index))
    event = events[0]
    assert event.status is EventStatus.DETECTED
    assert event.camera_id == settings.camera_id
    assert event.timestamp.endswith("Z")
    assert event.peak_confidence >= event.confidence
    payload = event.to_dict()
    assert payload["event_id"] and payload["timestamp"] and payload["camera_id"]


def _pose(landmarks: np.ndarray, index: int):
    from ai.pose.landmarks import PoseFrame

    return PoseFrame(landmarks=landmarks, timestamp=index / 30.0, frame_index=index)


def test_pipeline_can_run_without_media_pip(settings):
    """A missing pose asset must degrade, not crash: landmarks can still be posted."""
    pipeline = FallDetectionPipeline.__new__(FallDetectionPipeline)
    assert hasattr(FallDetectionPipeline, "process_landmarks")
    del pipeline


def test_non_finite_landmarks_are_treated_as_no_pose(make_pipeline, settings):
    """A frame of NaNs is not a detection; reporting it as one corrupts the stats."""
    pipeline = make_pipeline([None])
    broken = np.full((33, 4), np.nan, dtype=np.float32)
    outcome = pipeline.process_landmarks(broken)
    assert outcome.pose_detected is False
    assert pipeline.stats.pose_dropouts == 1
    assert pipeline.stats.poses_detected == 0


def test_partially_non_finite_landmarks_are_rejected(make_pipeline, settings):
    pipeline = make_pipeline([None])
    broken = landmark_array(STANDING)
    broken[11, 0] = np.inf
    outcome = pipeline.process_landmarks(broken)
    assert outcome.pose_detected is False
