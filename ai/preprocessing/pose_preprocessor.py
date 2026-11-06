"""Feature engineering: normalised pose -> one model-ready feature vector.

The feature layout is versioned (:data:`FEATURE_SPEC_VERSION`) because a model
checkpoint is only valid for the exact layout it was trained on. The
:class:`PosePreprocessor` writes a JSON sidecar next to every checkpoint, and
:class:`~ai.models.predictor.FallPredictor` refuses to run on a mismatch.

Layout (``FEATURE_DIM = 145``)
-----------------------------
===========  =====  ==========================================================
Block        Size   Content
===========  =====  ==========================================================
``xyz``      99     33 normalised landmarks x (x, y, z)
``vis``      33     per-landmark visibility, zero for missing joints
``kine``     12     torso lean, hip drop, joint angles, bounding box, separations
``present``  1      1.0 when a usable pose was extracted, 0.0 otherwise
===========  =====  ==========================================================

The 12 kinematic descriptors are what let a compact LSTM separate a controlled
descent from ordinary movement; raw coordinates alone are largely redundant once
the pose has been centred, rotated and scaled.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from ai.config import Settings, get_settings
from ai.logging_config import get_logger
from ai.pose.landmarks import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    POSE_LANDMARK_COUNT,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from ai.preprocessing.pose_normalizer import NormalizedPose, PoseNormalizer

logger = get_logger(__name__)

FEATURE_SPEC_VERSION: Final[str] = "1.0.0"

XYZ_BLOCK_SIZE: Final[int] = POSE_LANDMARK_COUNT * 3
VISIBILITY_BLOCK_SIZE: Final[int] = POSE_LANDMARK_COUNT
KINEMATIC_FEATURE_COUNT: Final[int] = 12
PRESENT_FEATURE_COUNT: Final[int] = 1

FEATURE_DIM: Final[int] = (
    XYZ_BLOCK_SIZE
    + VISIBILITY_BLOCK_SIZE
    + KINEMATIC_FEATURE_COUNT
    + PRESENT_FEATURE_COUNT
)

#: Index of the presence flag, the last element of every feature vector.
PRESENT_FEATURE_INDEX: Final[int] = FEATURE_DIM - 1

#: Index of the first kinematic descriptor inside a feature vector.
KINEMATIC_FEATURE_OFFSET: Final[int] = XYZ_BLOCK_SIZE + VISIBILITY_BLOCK_SIZE

KINEMATIC_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "torso_lean_deg",
    "hip_height_above_ankles",
    "shoulder_height_above_hip",
    "bbox_aspect_ratio",
    "bbox_center_y",
    "bbox_height",
    "elbow_angle_left",
    "elbow_angle_right",
    "knee_angle_left",
    "knee_angle_right",
    "wrist_separation",
    "ankle_separation",
)

FEATURE_NAMES: Final[tuple[str, ...]] = (
    tuple(f"lm{i}_{axis}" for i in range(POSE_LANDMARK_COUNT) for axis in ("x", "y", "z"))
    + tuple(f"lm{i}_visibility" for i in range(POSE_LANDMARK_COUNT))
    + KINEMATIC_FEATURE_NAMES
    + ("pose_present",)
)

if len(FEATURE_NAMES) != FEATURE_DIM:  # pragma: no cover - guards against layout drift
    raise RuntimeError(
        f"feature name count ({len(FEATURE_NAMES)}) does not match FEATURE_DIM ({FEATURE_DIM})"
    )

_XYZ_SLICE = slice(0, XYZ_BLOCK_SIZE)
_VIS_SLICE = slice(XYZ_BLOCK_SIZE, XYZ_BLOCK_SIZE + VISIBILITY_BLOCK_SIZE)
_KINE_SLICE = slice(
    XYZ_BLOCK_SIZE + VISIBILITY_BLOCK_SIZE,
    XYZ_BLOCK_SIZE + VISIBILITY_BLOCK_SIZE + KINEMATIC_FEATURE_COUNT,
)


def _angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at ``b`` of the triangle ``a-b-c``, in degrees.

    Returns ``0.0`` when the triangle is degenerate, which keeps the feature finite.
    """
    first = a[:2] - b[:2]
    second = c[:2] - b[:2]
    norm_product = float(np.linalg.norm(first) * np.linalg.norm(second))
    if norm_product < 1e-8:
        return 0.0
    cosine = float(np.dot(first, second) / norm_product)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def compute_kinematic_features(pose: NormalizedPose) -> np.ndarray:
    """Compute the 12 kinematic descriptors for a normalised pose.

    All values are expressed in *torso lengths*, so they are independent of subject
    size and camera distance.

    The descriptors are computed from the **unrotated** view
    (:attr:`NormalizedPose.unaligned`). This matters: forcing the torso vertical would
    make ``torso_lean_deg`` identically zero and would flip the sign of the hip height,
    destroying exactly the two cues that separate a fall from ordinary movement.
    """
    coords = pose.unaligned
    features = np.zeros(KINEMATIC_FEATURE_COUNT, dtype=np.float32)

    left_shoulder, right_shoulder = coords[LEFT_SHOULDER], coords[RIGHT_SHOULDER]
    left_hip, right_hip = coords[LEFT_HIP], coords[RIGHT_HIP]
    left_ankle, right_ankle = coords[LEFT_ANKLE], coords[RIGHT_ANKLE]
    left_wrist, right_wrist = coords[LEFT_WRIST], coords[RIGHT_WRIST]

    shoulder_mid = (left_shoulder + right_shoulder) * 0.5
    hip_mid = (left_hip + right_hip) * 0.5
    ankle_mid = (left_ankle + right_ankle) * 0.5

    # 0. Torso lean: 0 deg = perfectly upright, 90 deg = lying on its side/back.
    # Only the horizontal component counts; using the full 2-D norm would floor the
    # result at 45 deg for a perfectly vertical torso.
    torso = shoulder_mid - hip_mid
    features[0] = float(
        np.degrees(np.arctan2(abs(float(torso[0])), max(abs(float(torso[1])), 1e-6)))
    )

    # 1-2. Vertical separation in torso lengths. The hip-to-ankle distance collapses
    #      towards zero when a person is on the floor -- the single strongest cue.
    features[1] = float(ankle_mid[1] - hip_mid[1])
    features[2] = float(hip_mid[1] - shoulder_mid[1])

    x_min, y_min, x_max, y_max = pose.bounding_box()
    width = x_max - x_min
    height = y_max - y_min
    # 3-5. Bounding box shape: a lying body is wide and flat, a standing one is tall.
    features[3] = float(width / height) if height > 1e-6 else 0.0
    features[4] = float((y_min + y_max) * 0.5)
    features[5] = float(height)

    # 6-9. Joint angles: arms and legs folding are the mechanics of a fall.
    features[6] = _angle_degrees(coords[LEFT_SHOULDER], coords[LEFT_ELBOW], coords[LEFT_WRIST])
    features[7] = _angle_degrees(coords[RIGHT_SHOULDER], coords[RIGHT_ELBOW], coords[RIGHT_WRIST])
    features[8] = _angle_degrees(coords[LEFT_HIP], coords[LEFT_KNEE], coords[LEFT_ANKLE])
    features[9] = _angle_degrees(coords[RIGHT_HIP], coords[RIGHT_KNEE], coords[RIGHT_ANKLE])

    # 10-11. Hand/foot separation: a spread-eagle fall differs from a compact one.
    features[10] = float(np.linalg.norm(left_wrist[:2] - right_wrist[:2]))
    features[11] = float(np.linalg.norm(left_ankle[:2] - right_ankle[:2]))

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def build_feature_vector(pose: NormalizedPose) -> np.ndarray:
    """Assemble the full feature vector for a normalised pose."""
    vector = np.zeros(FEATURE_DIM, dtype=np.float32)
    vector[_XYZ_SLICE] = pose.coordinates.reshape(-1)
    vector[_VIS_SLICE] = pose.valid.astype(np.float32)
    vector[_KINE_SLICE] = compute_kinematic_features(pose)
    vector[PRESENT_FEATURE_INDEX] = 1.0
    return vector


def zero_feature_vector() -> np.ndarray:
    """The vector used for a frame in which no pose could be extracted.

    Everything is zero, including the presence flag, so the model and the event
    detector can both tell a real pose from a gap.
    """
    return np.zeros(FEATURE_DIM, dtype=np.float32)


class PosePreprocessor:
    """Convert a ``(33, 4)`` landmark array into a ``FEATURE_DIM`` feature vector.

    This is *the* preprocessing entry point. Both
    :mod:`ai.training.dataset` and :mod:`ai.inference.pipeline` use this class, so
    training-time and inference-time feature construction cannot diverge.

    The class is stateless and safe to share between threads.
    """

    def __init__(
        self,
        normalizer: PoseNormalizer | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.normalizer = normalizer or PoseNormalizer.from_settings(self._settings)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "PosePreprocessor":
        settings = settings or get_settings()
        return cls(PoseNormalizer.from_settings(settings), settings=settings)

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FEATURE_NAMES

    def transform(self, landmarks: np.ndarray | None) -> np.ndarray:
        """Return the feature vector for one frame.

        Never raises for a bad frame: unusable input yields the zero vector with
        ``present = 0``, which the sequence buffer preserves as a visible gap.
        """
        if landmarks is None:
            return zero_feature_vector()
        try:
            pose = self.normalizer.normalize(landmarks)
        except ValueError:
            logger.warning("Preprocessor received a malformed landmark array", exc_info=True)
            return zero_feature_vector()
        if pose is None:
            return zero_feature_vector()
        return build_feature_vector(pose)

    def transform_sequence(self, frames: np.ndarray) -> np.ndarray:
        """Vectorised convenience wrapper for a ``(T, 33, 4)`` array of landmarks.

        Returns ``(T, FEATURE_DIM)``. Frames with no usable pose become zero rows.
        """
        array = np.asarray(frames, dtype=np.float32)
        if array.ndim == 2:
            array = array[None, ...]
        if array.ndim != 3 or array.shape[1:] != (POSE_LANDMARK_COUNT, 4):
            raise ValueError(
                f"expected a (T, {POSE_LANDMARK_COUNT}, 4) array, got {array.shape}"
            )
        return np.stack([self.transform(frame) for frame in array], axis=0)

    def describe(self) -> dict[str, object]:
        """Machine-readable description of the feature layout, for checkpoint metadata."""
        return {
            "feature_spec_version": FEATURE_SPEC_VERSION,
            "feature_dim": FEATURE_DIM,
            "feature_names": list(FEATURE_NAMES),
            "min_landmark_visibility": self.normalizer.min_visibility,
            "rotation_normalization": self.normalizer.rotate,
            "required_landmark_indices": list(self.normalizer.required_indices),
            "present_feature_index": PRESENT_FEATURE_INDEX,
        }


__all__ = [
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "FEATURE_SPEC_VERSION",
    "KINEMATIC_FEATURE_NAMES",
    "PRESENT_FEATURE_INDEX",
    "PosePreprocessor",
    "build_feature_vector",
    "compute_kinematic_features",
    "zero_feature_vector",
]
