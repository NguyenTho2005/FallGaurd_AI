"""Unit tests for the sequence ring buffer."""

from __future__ import annotations

import numpy as np
import pytest

from ai.inference.sequence_buffer import SequenceBuffer
from ai.preprocessing.pose_preprocessor import FEATURE_DIM, PRESENT_FEATURE_INDEX


def make_features(value: float = 1.0, feature_dim: int = FEATURE_DIM) -> np.ndarray:
    vector = np.full(feature_dim, value, dtype=np.float32)
    vector[PRESENT_FEATURE_INDEX] = 1.0
    return vector


def test_buffer_is_not_ready_until_full():
    buffer = SequenceBuffer(sequence_length=30)
    for _ in range(29):
        buffer.push(make_features())
    assert not buffer.is_ready
    assert buffer.window() is None
    assert len(buffer) == 29

    buffer.push(make_features())
    assert buffer.is_ready
    window = buffer.window()
    assert window is not None
    assert window.shape == (30, FEATURE_DIM)


def test_window_preserves_order():
    buffer = SequenceBuffer(sequence_length=4)
    for index in range(4):
        vector = np.zeros(FEATURE_DIM, dtype=np.float32)
        vector[0] = float(index)
        vector[PRESENT_FEATURE_INDEX] = 1.0
        buffer.push(vector)
    window = buffer.window()
    assert window is not None
    assert list(window[:, 0]) == [0.0, 1.0, 2.0, 3.0]


def test_window_slides_and_evicts_the_oldest_frame():
    buffer = SequenceBuffer(sequence_length=3)
    for index in range(5):
        vector = np.zeros(FEATURE_DIM, dtype=np.float32)
        vector[0] = float(index)
        vector[PRESENT_FEATURE_INDEX] = 1.0
        buffer.push(vector)
    window = buffer.window()
    assert window is not None
    assert list(window[:, 0]) == [2.0, 3.0, 4.0]
    assert len(buffer) == 3
    assert buffer.frames_pushed == 5


def test_push_infers_presence_from_the_feature_vector():
    buffer = SequenceBuffer(sequence_length=2)
    buffer.push(make_features())
    buffer.push(np.zeros(FEATURE_DIM, dtype=np.float32))
    assert buffer.poses_in_buffer == 1
    assert buffer.missing_ratio == pytest.approx(0.5)


def test_gaps_are_preserved_rather_than_skipped():
    """A dropout must occupy a slot, otherwise the window silently compresses time."""
    buffer = SequenceBuffer(sequence_length=3)
    buffer.push(make_features(1.0))
    buffer.push_zeros()
    buffer.push(make_features(3.0))
    window = buffer.window()
    assert window is not None
    assert window[1].tolist() == [0.0] * FEATURE_DIM
    assert window[1, PRESENT_FEATURE_INDEX] == 0.0
    assert window[0, 0] == 1.0 and window[2, 0] == 3.0


def test_too_many_gaps_makes_the_window_unready():
    buffer = SequenceBuffer(sequence_length=4, max_missing_ratio=0.5)
    buffer.push(make_features())
    buffer.push_zeros()
    buffer.push_zeros()
    buffer.push(make_features())
    assert len(buffer) == 4
    assert buffer.is_ready, "exactly 50% missing is the limit and must be allowed"
    buffer.push_zeros()  # window slides: [zero, zero, present, zero] -> 75% missing
    assert buffer.missing_ratio == pytest.approx(0.75)
    assert not buffer.is_ready
    assert buffer.window() is None


def test_max_missing_ratio_zero_requires_a_dense_window():
    buffer = SequenceBuffer(sequence_length=3, max_missing_ratio=0.0)
    buffer.push(make_features())
    buffer.push(make_features())
    buffer.push_zeros()
    assert not buffer.is_ready
    # Three consecutive present frames evict the gap entirely.
    buffer.push(make_features())
    buffer.push(make_features())
    buffer.push(make_features())
    assert buffer.missing_ratio == 0.0
    assert buffer.is_ready


def test_presence_can_be_overridden_explicitly():
    buffer = SequenceBuffer(sequence_length=2)
    buffer.push(make_features(), present=False)
    assert buffer.poses_in_buffer == 0
    buffer.push(make_features(), present=True)
    assert buffer.poses_in_buffer == 1
    assert buffer.is_ready


def test_reset_clears_the_buffer():
    buffer = SequenceBuffer(sequence_length=2)
    buffer.push(make_features())
    buffer.push(make_features())
    assert buffer.is_ready
    buffer.reset()
    assert len(buffer) == 0
    assert buffer.poses_in_buffer == 0
    assert not buffer.is_ready


def test_latest_and_last_timestamp():
    buffer = SequenceBuffer(sequence_length=3, clock=lambda: 7.5)
    assert buffer.latest() is None
    assert buffer.last_timestamp is None
    buffer.push(make_features(2.0), timestamp=1.0)
    buffer.push(make_features(3.0), timestamp=2.0)
    latest = buffer.latest()
    assert latest is not None
    assert latest[0] == pytest.approx(3.0)
    assert buffer.last_timestamp == pytest.approx(2.0)
    assert buffer.window_duration() == pytest.approx(1.0)


def test_clock_is_used_when_no_timestamp_is_given():
    ticks = iter([1.0, 2.0, 3.0])
    buffer = SequenceBuffer(sequence_length=2, clock=lambda: next(ticks))
    buffer.push(make_features())
    buffer.push(make_features())
    assert buffer.last_timestamp == pytest.approx(2.0)


def test_wrong_feature_width_is_rejected():
    buffer = SequenceBuffer(sequence_length=2, feature_dim=10)
    with pytest.raises(ValueError, match="expected a feature vector of length 10"):
        buffer.push(np.zeros(9, dtype=np.float32))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sequence_length": 1},
        {"sequence_length": 0},
        {"sequence_length": 5, "feature_dim": 0},
        {"sequence_length": 5, "max_missing_ratio": 1.5},
        {"sequence_length": 5, "max_missing_ratio": -0.1},
    ],
)
def test_invalid_construction_is_rejected(kwargs: dict):
    with pytest.raises(ValueError):
        SequenceBuffer(**kwargs)


def test_present_flags_are_returned_oldest_first():
    buffer = SequenceBuffer(sequence_length=3)
    buffer.push(make_features())
    buffer.push_zeros()
    buffer.push(make_features())
    assert buffer.present_flags().tolist() == [True, False, True]


def test_eviction_keeps_the_pose_counter_accurate():
    """After eviction the counter must reflect the buffer, not the lifetime total."""
    buffer = SequenceBuffer(sequence_length=2)
    buffer.push(make_features())
    buffer.push(make_features())
    assert buffer.poses_in_buffer == 2
    buffer.push_zeros()  # evicts a present frame
    assert buffer.poses_in_buffer == 1
    assert buffer.missing_ratio == pytest.approx(0.5)
    buffer.push_zeros()  # evicts the other present frame
    assert buffer.poses_in_buffer == 0
