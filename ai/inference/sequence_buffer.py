"""Ring buffer that turns a stream of per-frame features into a model input window.

Design notes
------------
* **Gaps are preserved, not closed.** A frame with no pose is stored as a zero vector
  with ``present = 0``. Skipping it would compress real time and hide dropouts, which
  would both distort the temporal dynamics and let a short burst of good frames look
  like a long window.
* **A window is only "ready"** once it holds ``sequence_length`` frames *and* the share
  of missing frames stays within ``max_missing_ratio``. Otherwise the model would be
  asked to score a window that is mostly absence.
* The buffer is deliberately dependency-free and clock-injectable so it can be unit
  tested without sleeping.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque

import numpy as np

from ai.preprocessing.pose_preprocessor import FEATURE_DIM, PRESENT_FEATURE_INDEX

DEFAULT_CLOCK: Callable[[], float] = time.monotonic


@dataclass(frozen=True, slots=True)
class BufferedFrame:
    """One entry of the buffer."""

    features: np.ndarray
    timestamp: float
    present: bool


class SequenceBuffer:
    """Fixed-length sliding window over per-frame feature vectors.

    Parameters
    ----------
    sequence_length:
        Number of frames in the window handed to the model.
    feature_dim:
        Width of one feature vector. Defaults to :data:`FEATURE_DIM`.
    max_missing_ratio:
        Largest tolerated fraction of pose-less frames in a window that is still
        allowed to be scored.
    clock:
        Monotonic time source, injected by tests.
    """

    def __init__(
        self,
        sequence_length: int,
        feature_dim: int = FEATURE_DIM,
        *,
        max_missing_ratio: float = 0.5,
        clock: Callable[[], float] = DEFAULT_CLOCK,
    ) -> None:
        if sequence_length < 2:
            raise ValueError(f"sequence_length must be >= 2, got {sequence_length}")
        if feature_dim < 1:
            raise ValueError(f"feature_dim must be >= 1, got {feature_dim}")
        if not 0.0 <= max_missing_ratio <= 1.0:
            raise ValueError(f"max_missing_ratio must be in [0, 1], got {max_missing_ratio}")
        self.sequence_length = int(sequence_length)
        self.feature_dim = int(feature_dim)
        self.max_missing_ratio = float(max_missing_ratio)
        self._clock = clock
        self._buffer: Deque[BufferedFrame] = deque(maxlen=self.sequence_length)
        self.frames_pushed = 0
        self.poses_in_buffer = 0

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """``True`` when a full, sufficiently dense window can be scored."""
        if len(self._buffer) < self.sequence_length:
            return False
        return self.missing_ratio <= self.max_missing_ratio

    @property
    def missing_ratio(self) -> float:
        """Share of pose-less frames currently in the buffer."""
        if not self._buffer:
            return 1.0
        return 1.0 - (self.poses_in_buffer / len(self._buffer))

    @property
    def last_timestamp(self) -> float | None:
        return self._buffer[-1].timestamp if self._buffer else None

    # ------------------------------------------------------------------ #
    def push(
        self,
        features: np.ndarray,
        *,
        present: bool | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Append one frame, evicting the oldest when the buffer is full.

        ``present`` defaults to the presence flag carried in the feature vector, so a
        caller that uses :class:`~ai.preprocessing.pose_preprocessor.PosePreprocessor`
        does not have to pass it explicitly.
        """
        vector = np.asarray(features, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.feature_dim:
            raise ValueError(
                f"expected a feature vector of length {self.feature_dim}, got {vector.shape[0]}"
            )
        if present is None:
            present = bool(vector[PRESENT_FEATURE_INDEX] > 0.5) if (
                self.feature_dim > PRESENT_FEATURE_INDEX
            ) else True
        if self._buffer and self._buffer.maxlen is not None and len(self._buffer) == self._buffer.maxlen:
            evicted = self._buffer[0]
            if evicted.present:
                self.poses_in_buffer -= 1
        self._buffer.append(
            BufferedFrame(
                features=vector,
                timestamp=self._clock() if timestamp is None else float(timestamp),
                present=bool(present),
            )
        )
        if present:
            self.poses_in_buffer += 1
        self.frames_pushed += 1

    def push_zeros(self, *, timestamp: float | None = None) -> None:
        """Append an explicit "no pose in this frame" entry."""
        self.push(np.zeros(self.feature_dim, dtype=np.float32), present=False, timestamp=timestamp)

    # ------------------------------------------------------------------ #
    def window(self) -> np.ndarray | None:
        """Return the ``(sequence_length, feature_dim)`` window, or ``None``."""
        if not self.is_ready:
            return None
        return np.stack([entry.features for entry in self._buffer], axis=0).astype(np.float32)

    def latest(self) -> np.ndarray | None:
        """The most recent feature vector, or ``None`` when the buffer is empty."""
        return self._buffer[-1].features.copy() if self._buffer else None

    def present_flags(self) -> np.ndarray:
        """``(len(buffer),)`` boolean array of presence flags, oldest first."""
        return np.array([entry.present for entry in self._buffer], dtype=bool)

    def window_duration(self) -> float:
        """Seconds spanned by the current buffer (``0.0`` with fewer than two frames)."""
        if len(self._buffer) < 2:
            return 0.0
        return float(self._buffer[-1].timestamp - self._buffer[0].timestamp)

    def reset(self) -> None:
        """Drop every buffered frame."""
        self._buffer.clear()
        self.poses_in_buffer = 0

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"SequenceBuffer(len={len(self)}/{self.sequence_length}, "
            f"feature_dim={self.feature_dim}, missing_ratio={self.missing_ratio:.2f})"
        )
