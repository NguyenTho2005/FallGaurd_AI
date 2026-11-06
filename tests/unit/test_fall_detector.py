"""Unit tests for fall event confirmation.

Every test drives a :class:`FakeClock`, so the cooldown behaviour is verified without
sleeping and without flakiness.
"""

from __future__ import annotations

import pytest

from ai.config import Settings
from ai.inference.fall_detector import DetectionResult, EventStatus, FallDetector


@pytest.fixture
def detector(settings: Settings, fake_clock):
    settings.confidence_threshold = 0.8
    settings.min_fall_frames = 3
    settings.cooldown_seconds = 30.0
    return FallDetector(settings, clock=fake_clock, timestamp_fn=lambda: "2026-01-01T00:00:00Z")


def feed(detector: FallDetector, probability: float, times: int,
         *, has_signal: bool = True) -> list[DetectionResult]:
    return [detector.update(probability, has_signal=has_signal) for _ in range(times)]


# --------------------------------------------------------------------------- #
# threshold
# --------------------------------------------------------------------------- #
def test_below_threshold_never_confirms(detector: FallDetector):
    results = feed(detector, 0.79, 20)
    assert not any(result.confirmed for result in results)
    assert all(not result.above_threshold for result in results)
    assert detector.stats.events_confirmed == 0


def test_threshold_is_inclusive(detector: FallDetector):
    results = feed(detector, 0.8, 3)
    assert results[0].above_threshold
    assert results[2].confirmed


def test_a_single_spike_does_not_confirm(detector: FallDetector):
    """The core false-positive guard: one noisy frame must never page a human."""
    results = feed(detector, 0.99, 1)
    assert not results[0].confirmed
    assert results[0].consecutive_frames == 1
    assert detector.stats.events_confirmed == 0


def test_min_fall_frames_is_respected(detector: FallDetector):
    """With MIN_FALL_FRAMES=3, the third consecutive window is the first that fires."""
    assert not detector.update(0.95).confirmed
    assert not detector.update(0.95).confirmed
    result = detector.update(0.95)
    assert result.confirmed
    assert result.event is not None
    assert result.event.window_frames == 3


def test_interrupted_evidence_resets_the_run(detector: FallDetector):
    feed(detector, 0.95, 2)
    assert detector.consecutive_frames == 2
    detector.update(0.10)
    assert detector.consecutive_frames == 0
    feed(detector, 0.95, 2)
    assert not detector.in_cooldown
    assert detector.consecutive_frames == 2, "the run must restart from zero"
    assert detector.update(0.95).confirmed


# --------------------------------------------------------------------------- #
# cooldown
# --------------------------------------------------------------------------- #
def test_cooldown_suppresses_immediate_repeat_alerts(detector: FallDetector):
    feed(detector, 0.95, 3)
    assert detector.stats.events_confirmed == 1

    feed(detector, 0.95, 10)
    assert detector.stats.events_confirmed == 1, "one physical fall must not alert 11 times"
    assert detector.in_cooldown

    results = feed(detector, 0.95, 3)
    assert all(result.in_cooldown for result in results)


def test_cooldown_expires_and_detection_resumes(detector: FallDetector, fake_clock):
    feed(detector, 0.95, 3)
    assert detector.in_cooldown
    assert detector.cooldown_remaining() == pytest.approx(30.0)

    fake_clock.advance(29.0)
    assert detector.in_cooldown
    fake_clock.advance(1.5)
    assert not detector.in_cooldown
    assert detector.cooldown_remaining() == 0.0

    results = feed(detector, 0.95, 3)
    assert results[-1].confirmed
    assert detector.stats.events_confirmed == 2


def test_run_resets_after_a_confirmed_event(detector: FallDetector, fake_clock):
    """Evidence gathered before the event must not carry over into the next one.

    Otherwise a second fall shortly after the first would be confirmed on fewer frames
    than the policy requires.
    """
    feed(detector, 0.95, 3)
    assert detector.stats.events_confirmed == 1
    assert detector.consecutive_frames == 0

    fake_clock.advance(31.0)
    assert not detector.update(0.95).confirmed
    assert not detector.update(0.95).confirmed
    assert detector.update(0.95).confirmed
    assert detector.stats.events_confirmed == 2


def test_zero_cooldown_allows_back_to_back_events(settings: Settings, fake_clock):
    settings.min_fall_frames = 2
    settings.cooldown_seconds = 0.0
    detector = FallDetector(settings, clock=fake_clock)
    feed(detector, 0.95, 2)
    assert detector.stats.events_confirmed == 1
    results = feed(detector, 0.95, 2)
    assert results[-1].confirmed
    assert detector.stats.events_confirmed == 2


# --------------------------------------------------------------------------- #
# missing signal
# --------------------------------------------------------------------------- #
def test_windows_without_signal_do_not_accumulate_evidence(detector: FallDetector):
    feed(detector, 0.99, 2)
    assert detector.consecutive_frames == 2
    detector.update(0.0, has_signal=False)
    assert detector.consecutive_frames == 0, "absence of evidence is not evidence"
    results = feed(detector, 0.99, 2)
    assert not results[-1].confirmed


def test_windows_without_signal_are_counted(detector: FallDetector):
    feed(detector, 0.0, 4, has_signal=False)
    assert detector.stats.windows_skipped_no_signal == 4
    assert detector.stats.windows_evaluated == 4


def test_missing_signal_reports_not_ready(detector: FallDetector):
    result = detector.update(0.0, has_signal=False)
    assert not result.sequence_ready
    assert result.label == "NORMAL"


# --------------------------------------------------------------------------- #
# event payload
# --------------------------------------------------------------------------- #
def test_confirmed_event_payload(detector: FallDetector):
    detector.update(0.99)
    detector.update(0.99)
    result = detector.update(0.93)
    assert result.event is not None
    event = result.event
    assert event.prediction == "FALL"
    assert event.confidence == pytest.approx(0.93)
    assert event.peak_confidence == pytest.approx(0.99)
    assert event.status is EventStatus.DETECTED
    assert event.camera_id == detector.camera_id
    assert len(event.event_id) >= 16
    payload = event.to_dict()
    assert payload["status"] == "detected"
    assert payload["peak_confidence"] == pytest.approx(0.99)


def test_event_ids_are_unique(detector: FallDetector, fake_clock):
    seen: list[str] = []
    for _ in range(3):
        result = feed(detector, 0.95, 3)[-1]
        assert result.event is not None
        seen.append(result.event.event_id)
        fake_clock.advance(31.0)
    assert len(set(seen)) == 3, "each confirmed event needs a distinct id"
    assert detector.stats.events_confirmed == 3


def test_out_of_range_probability_is_clamped(detector: FallDetector):
    result = detector.update(1.5)
    assert result.probability == pytest.approx(1.0)
    result = detector.update(-0.2)
    assert result.probability == pytest.approx(0.0)


def test_camera_id_comes_from_settings(detector: FallDetector, settings: Settings):
    settings.camera_id = "hallway_02"
    custom = FallDetector(settings, camera_id="hallway_02")
    custom.update(0.99)
    custom.update(0.99)
    assert custom.update(0.99).event.camera_id == "hallway_02"


def test_reset_clears_state_and_cooldown(detector: FallDetector, fake_clock):
    feed(detector, 0.95, 3)
    detector.reset()
    assert detector.stats.events_confirmed == 0
    assert detector.consecutive_frames == 0
    assert not detector.in_cooldown


def test_stats_reported_for_status(detector: FallDetector):
    feed(detector, 0.95, 2)
    feed(detector, 0.10, 1)
    feed(detector, 0.95, 3)
    stats = detector.stats.to_dict()
    assert stats["windows_evaluated"] == 6
    assert stats["above_threshold"] == 5
    assert stats["events_confirmed"] == 1
