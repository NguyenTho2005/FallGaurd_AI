"""Unit tests for the dashboard's skeleton rendering and for logging configuration."""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from pathlib import Path

import numpy as np
import pytest

from ai.config import Settings
from ai.logging_config import (
    LOGGER_ROOT,
    ConsoleFormatter,
    JsonFormatter,
    SecretRedactionFilter,
    configure_logging,
    get_logger,
    mask_secrets,
    silence_external_loggers,
)
from frontend.skeleton import (
    COLORS,
    KEY_LANDMARKS,
    landmarks_to_canvas,
    skeleton_summary,
)
from tests.conftest import STANDING, landmark_array


# --------------------------------------------------------------------------- #
# skeleton rendering
# --------------------------------------------------------------------------- #
def test_landmarks_are_mapped_to_canvas_pixels():
    points, segments = landmarks_to_canvas(landmark_array(STANDING), width=640, height=480)
    assert len(points) == 33
    assert all(len(point) == 2 for point in points)
    from ai.pose.landmarks import POSE_CONNECTIONS

    assert len(segments) == len(POSE_CONNECTIONS)
    assert segments, "a standing figure must produce skeleton segments"
    for x1, y1, x2, y2 in segments:
        assert 0.0 <= x1 <= 640.0 and 0.0 <= y1 <= 480.0
        assert 0.0 <= x2 <= 640.0 and 0.0 <= y2 <= 480.0


def test_low_visibility_joints_are_moved_off_canvas():
    landmarks = landmark_array(STANDING)
    landmarks[15, 3] = 0.0  # the left wrist
    from ai.pose.landmarks import POSE_CONNECTIONS

    points, segments = landmarks_to_canvas(landmarks, min_visibility=0.3)
    assert points[15] == [-1.0, -1.0]
    # The left wrist is an endpoint of exactly one connection, so one segment is dropped.
    touching_wrist = sum(1 for a, b in POSE_CONNECTIONS if 15 in (a, b))
    assert touching_wrist == 1
    assert len(segments) == len(POSE_CONNECTIONS) - touching_wrist


def test_no_segments_are_drawn_when_nothing_is_visible():
    landmarks = landmark_array(STANDING)
    landmarks[:, 3] = 0.0
    points, segments = landmarks_to_canvas(landmarks, min_visibility=0.3)
    assert segments == []
    assert all(point == [-1.0, -1.0] for point in points)


def test_nan_coordinates_do_not_crash_the_renderer():
    landmarks = landmark_array(STANDING)
    landmarks[11, 0] = np.nan
    points, _ = landmarks_to_canvas(landmarks, min_visibility=0.3)
    assert points[11] == [-1.0, -1.0]


def test_wrong_landmark_count_is_rejected():
    with pytest.raises(ValueError, match="expected"):
        landmarks_to_canvas(np.zeros((10, 4), dtype=np.float32))


def test_skeleton_summary_counts_detected_joints():
    summary = skeleton_summary(landmark_array(STANDING))
    assert summary["landmarks"] == 33
    assert summary["detected_joints"] == 33
    assert summary["mean_visibility"] == pytest.approx(1.0)


def test_skeleton_summary_of_a_partial_pose():
    landmarks = landmark_array(STANDING)
    landmarks[10:33, 3] = 0.0
    summary = skeleton_summary(landmarks)
    assert summary["detected_joints"] == 10


def test_every_state_has_a_colour():
    for state in ("normal", "fall", "unknown"):
        assert state in COLORS


def test_key_landmarks_are_within_range():
    assert all(0 <= index < 33 for index in KEY_LANDMARKS)


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def test_get_logger_namespaces_under_fallguard():
    assert get_logger("pose").name == f"{LOGGER_ROOT}.pose"
    assert get_logger("fallguard.pose").name == "fallguard.pose"


def test_mask_secrets_redacts_the_environment_value(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abcdef123456")
    assert "abcdef123456" not in mask_secrets("using abcdef123456 now")


def test_json_formatter_emits_one_object_per_record():
    formatter = JsonFormatter()
    record = logging.LogRecord("fallguard.test", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    payload = json.loads(formatter.format(record))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "fallguard.test"
    assert payload["message"] == "hello world"
    assert "timestamp" in payload


def test_json_formatter_includes_exception_text():
    formatter = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys as _sys

        record = logging.LogRecord(
            "fallguard.test", logging.ERROR, __file__, 1, "failed", (), _sys.exc_info()
        )
    payload = json.loads(formatter.format(record))
    assert "ValueError: boom" in payload["exception"]


def test_console_formatter_is_readable():
    formatter = ConsoleFormatter()
    record = logging.LogRecord("fallguard.test", logging.WARNING, __file__, 1, "careful", (), None)
    text = formatter.format(record)
    assert "WARNING" in text and "careful" in text and "fallguard.test" in text


def test_redaction_filter_handles_dict_args():
    record = logging.LogRecord(
        "fallguard.test", logging.INFO, __file__, 1, "%(token)s",
        ({"token": "1234567890:AAHaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},), None,
    )
    SecretRedactionFilter().filter(record)
    assert "AAHaaaaaaaa" not in record.getMessage()


def test_configure_logging_is_idempotent(settings: Settings, capsys):
    settings.log_file = None
    settings.log_level = "INFO"
    first = configure_logging(settings, force=True)
    count = len(first.handlers)
    second = configure_logging(settings)
    assert second is first
    assert len(second.handlers) == count


def test_configure_logging_writes_a_file(settings: Settings, tmp_path: Path):
    log_path = tmp_path / "logs" / "fallguard.log"
    settings.log_file = str(log_path)
    settings.log_level = "DEBUG"
    try:
        logger = configure_logging(settings, force=True)
        assert any(
            isinstance(handler, logging.handlers.RotatingFileHandler)
            for handler in logger.handlers
        )
        get_logger("test").error("recorded message")
        for handler in logger.handlers:
            handler.flush()
        assert log_path.is_file()
        assert "recorded message" in log_path.read_text(encoding="utf-8")
    finally:
        logger = configure_logging(settings, force=True)
        logger.setLevel(logging.WARNING)
        settings.log_file = None


def test_third_party_loggers_are_quietened(settings: Settings):
    logger = configure_logging(settings, force=True)
    try:
        for name in ("mediapipe", "absl", "tensorflow"):
            assert logging.getLogger(name).level >= logging.ERROR
    finally:
        configure_logging(settings, force=True)


def test_silence_external_loggers_helper():
    silence_external_loggers(("some.noisy.library",))
    assert logging.getLogger("some.noisy.library").level == logging.ERROR


def test_stream_handler_writes_to_stderr(settings: Settings):
    """Log records go to stderr, keeping stdout clean for CLI output."""
    logger = configure_logging(settings, force=True)
    try:
        stream_handlers = [
            handler for handler in logger.handlers
            if isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.FileHandler)
        ]
        assert stream_handlers
        assert all(handler.stream is sys.stderr for handler in stream_handlers)
    finally:
        configure_logging(settings, force=True)
