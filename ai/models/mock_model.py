"""A deterministic, dependency-free stand-in for the trained LSTM.

Why this exists
---------------
The API, the event-confirmation logic and the Streamlit dashboard must be runnable and
testable *before* a checkpoint exists -- and on machines with no GPU. Rather than skip
the model, FallGuard substitutes a transparent kinematic heuristic that scores the
12 kinematic descriptors the preprocessor already computes.

**This is not a machine-learning model and produces no benchmark numbers.** It exists to
exercise the software pipeline end to end. The backend reports which backend is active in
``/status`` and the CLI prints a warning, so a demo can never be mistaken for a trained
result.

Heuristic
---------
Two cues dominate real fall detection from skeleton data:

1. the hip has dropped towards the ankles (a standing hip sits ~2.5 torso lengths above
   the feet; a person on the floor sits at ~0), and
2. the torso has rotated away from vertical.

The score is a soft-min of those two normalised cues, kept at 0.0 while the body is
upright so ordinary activity produces no false alarms.
"""

from __future__ import annotations

import numpy as np

from ai.logging_config import get_logger
from ai.preprocessing.pose_preprocessor import (
    FEATURE_DIM,
    KINEMATIC_FEATURE_NAMES,
    KINEMATIC_FEATURE_OFFSET,
    PRESENT_FEATURE_INDEX,
)

logger = get_logger(__name__)

#: Index of each kinematic feature inside the feature vector, resolved by name so a
#: layout change cannot silently shift the meaning of a magic number.
_KINE_INDEX = {name: i for i, name in enumerate(KINEMATIC_FEATURE_NAMES)}

#: A standing person's hip sits roughly this many torso lengths above the ankles
#: (hip-to-ankle ~0.40 of body height / shoulder-to-hip ~0.28 of body height ~ 1.4).
_HIP_HEIGHT_STANDING = 1.5
#: A person lying on the floor is at roughly this value.
_HIP_HEIGHT_GROUND = 0.25
#: Torso lean (degrees) considered fully fallen / fully upright.
_LEAN_GROUND_DEG = 80.0
_LEAN_STANDING_DEG = 25.0


class MockFallModel:
    """Deterministic fall scorer with the same interface as the Keras models."""

    def __init__(self, sequence_length: int = 30, feature_dim: int = FEATURE_DIM) -> None:
        if sequence_length < 2:
            raise ValueError(f"sequence_length must be >= 2, got {sequence_length}")
        if feature_dim != FEATURE_DIM:
            raise ValueError(
                f"MockFallModel expects feature_dim={FEATURE_DIM}, got {feature_dim}"
            )
        self.sequence_length = int(sequence_length)
        self.feature_dim = int(feature_dim)
        self.name = "mock_kinematic_heuristic"

    # -- Keras-compatible introspection ---------------------------------- #
    @property
    def input_shape(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return (
            (None, self.sequence_length, self.feature_dim),
            (None, self.sequence_length, self.feature_dim),
        )

    @property
    def output_shape(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return ((None, 1), (None, 1))

    def count_params(self) -> int:
        """Zero: there are no learned weights here, and pretending otherwise is a lie."""
        return 0

    # -- scoring ----------------------------------------------------------- #
    def score_window(self, window: np.ndarray) -> float:
        """Fall probability for a single ``(sequence_length, feature_dim)`` window."""
        array = np.asarray(window, dtype=np.float32)
        if array.ndim == 3:
            array = array[0]
        if array.ndim != 2:
            raise ValueError(f"expected a 2-D window, got shape {array.shape}")
        if array.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected feature_dim={self.feature_dim}, got {array.shape[1]}"
            )
        present = array[:, PRESENT_FEATURE_INDEX] > 0.5
        if not present.any():
            return 0.0
        frames = array[present]

        hip_height = frames[
            :, KINEMATIC_FEATURE_OFFSET + _KINE_INDEX["hip_height_above_ankles"]
        ]
        lean = frames[:, KINEMATIC_FEATURE_OFFSET + _KINE_INDEX["torso_lean_deg"]]

        # Frame-level cues in [0, 1]: 0 = upright, 1 = on the floor.
        drop = np.clip(
            (_HIP_HEIGHT_STANDING - hip_height) / (_HIP_HEIGHT_STANDING - _HIP_HEIGHT_GROUND),
            0.0,
            1.0,
        )
        tilt = np.clip((lean - _LEAN_STANDING_DEG) / (_LEAN_GROUND_DEG - _LEAN_STANDING_DEG), 0.0, 1.0)

        # A frame counts as fall-like only when *both* cues fire: a person bending over
        # has a low hip but stays vertical, and leaning back on a chair tilts but keeps
        # the hip high.
        frame_cue = np.minimum(drop, tilt)
        if frame_cue.size == 0:
            return 0.0
        return float(frame_cue.max())

    def predict(self, window: np.ndarray, verbose: int = 0) -> np.ndarray:
        """Keras-compatible prediction.

        Accepts a single ``(sequence_length, feature_dim)`` window or a
        ``(batch, sequence_length, feature_dim)`` batch and always returns ``(N, 1)``,
        exactly like ``keras.Model.predict``. Batch support matters: the evaluation
        pipeline scores a whole test split in one call.
        """
        del verbose
        return self.predict_proba(window)

    def predict_proba(self, window: np.ndarray) -> np.ndarray:
        """Keras-compatible batch prediction -> ``(N, 1)`` array of probabilities."""
        array = np.asarray(window, dtype=np.float32)
        if array.ndim == 2:
            array = array[None, ...]
        if array.ndim != 3:
            raise ValueError(f"expected a 3-D batch, got shape {array.shape}")
        return np.array(
            [[self.score_window(item)] for item in array], dtype=np.float32
        )

    def __call__(self, window: np.ndarray, training: bool = False) -> np.ndarray:
        del training
        return self.predict_proba(window)

    def summary(self) -> str:
        return (
            "MockFallModel (deterministic kinematic heuristic, 0 learned parameters).\n"
            "This backend is for pipeline testing and demos only -- it is NOT a trained "
            "model and its output is not a benchmark result."
        )
