"""Pose normalisation: translation, rotation and scale invariance.

The functions here turn raw MediaPipe landmark coordinates into a canonical body
representation. They are pure numpy, stateless, and are used *identically* by the
training pipeline and by live inference -- the training pipeline imports this exact
module rather than reimplementing it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai.config import Settings
from ai.logging_config import get_logger
from ai.pose.landmarks import (
    LEFT_HIP,
    LEFT_SHOULDER,
    POSE_LANDMARK_COUNT,
    RIGHT_HIP,
    RIGHT_SHOULDER,
)

logger = get_logger(__name__)

#: Below this torso length (in normalised image units) a pose is considered degenerate.
MIN_TORSO_LENGTH = 1e-3
#: Below this axis separation the torso direction is ambiguous.
MIN_TORSO_SPAN = 1e-4


@dataclass(frozen=True, slots=True)
class NormalizedPose:
    """A canonically normalised pose.

    Attributes
    ----------
    coordinates:
        ``(33, 3)`` array: hip-centred, torso-aligned, torso-length-scaled ``xyz``.
        Invalid landmarks are exactly zero. Because the torso axis is forced vertical,
        this view is invariant to camera tilt and subject lean -- which is what the
        LSTM coordinate block wants, but it *destroys* the absolute fall cues, so the
        kinematic features are computed from :attr:`unaligned` instead.
    unaligned:
        ``(33, 3)`` array: hip-centred and torso-length-scaled but **not** rotated.
        The body keeps its true orientation relative to the image, so "is the torso
        horizontal" and "has the hip dropped towards the feet" remain measurable.
    valid:
        ``(33,)`` boolean mask, ``True`` where the landmark survived normalisation.
    scale:
        The original torso length in image-normalised units, before scaling. Useful
        for diagnostics and for the ``bbox`` features.
    center:
        The original hip midpoint in image-normalised ``(x, y)``.
    """

    coordinates: np.ndarray
    unaligned: np.ndarray
    valid: np.ndarray
    scale: float
    center: tuple[float, float]

    @property
    def is_valid(self) -> bool:
        return bool(self.valid.any())

    def bounding_box(self, unaligned: bool = True) -> tuple[float, float, float, float]:
        """``(x_min, y_min, x_max, y_max)`` of the valid landmarks.

        Defaults to the unrotated view, which is the one with a meaningful aspect ratio
        (a lying body is wide and flat; a standing one is tall and narrow).
        """
        source = self.unaligned if unaligned else self.coordinates
        points = source[self.valid][:, :2]
        if points.size == 0:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        )


class PoseNormalizer:
    """Normalise a ``(33, 4)`` landmark array into hip-centred, torso-scaled space.

    Pipeline
    --------
    1. **Visibility mask** -- landmarks with ``visibility < min_visibility`` are marked
       invalid. Among the required landmarks the two hips and two shoulders are
       repaired from the other pair before the frame is accepted.
    2. **Translate** -- subtract the hip midpoint (shoulder midpoint as a fallback).
    3. **Rotate** -- rotate about the centre so the shoulder->hip axis points straight
       up, which removes camera tilt and subject lean from the coordinate frame.
       ``z`` is scaled by the same cosine so the 3D shape stays consistent.
    4. **Scale** -- divide by the torso length, so a child and an adult, or a subject
       at 2 m and at 5 m, produce comparable features.
    """

    def __init__(
        self,
        *,
        min_visibility: float = 0.5,
        rotate: bool = True,
        required_indices: tuple[int, ...] = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP),
    ) -> None:
        if not 0.0 <= min_visibility <= 1.0:
            raise ValueError(f"min_visibility must be in [0, 1], got {min_visibility}")
        self.min_visibility = float(min_visibility)
        self.rotate = bool(rotate)
        self.required_indices = tuple(required_indices)
        for index in self.required_indices:
            if not 0 <= index < POSE_LANDMARK_COUNT:
                raise ValueError(f"required landmark index out of range: {index}")

    @classmethod
    def from_settings(cls, settings: Settings) -> "PoseNormalizer":
        return cls(
            min_visibility=settings.min_landmark_visibility,
            rotate=settings.rotation_normalization,
            required_indices=tuple(settings.required_landmark_indices),
        )

    # ------------------------------------------------------------------ #
    def normalize(self, landmarks: np.ndarray) -> NormalizedPose | None:
        """Normalise one frame.

        Returns ``None`` when the frame carries no usable body reference, which the
        caller translates into "pose not available for this frame".
        """
        array = np.asarray(landmarks, dtype=np.float32)
        if array.shape != (POSE_LANDMARK_COUNT, 4):
            raise ValueError(
                f"expected a ({POSE_LANDMARK_COUNT}, 4) landmark array, got {array.shape}"
            )

        valid = self._validity_mask(array)
        xy = array[:, :2].copy()
        z = array[:, 2].copy()

        if not np.isfinite(xy).all() or not np.isfinite(z).all():
            logger.debug("Non-finite landmark coordinates; marking frame invalid")
            return None

        reference = self._repair_reference(array, valid)
        if reference is None:
            logger.debug("No usable torso reference in frame; rejecting")
            return None
        shoulder_mid, hip_mid = reference

        center = np.asarray(hip_mid, dtype=np.float32)
        torso = np.asarray(shoulder_mid, dtype=np.float32) - center
        scale = float(np.linalg.norm(torso))
        # A missing hip (or shoulder) collapses the torso vector. Rather than throw the
        # frame away, fall back to the diagonal of the bounding box of the valid body
        # landmarks, which is still a stable subject-size reference.
        if scale < MIN_TORSO_LENGTH:
            scale = self._bounding_box_diagonal(xy, valid)
        if scale < MIN_TORSO_LENGTH:
            logger.debug("Degenerate body scale %.6f; rejecting frame", scale)
            return None

        # Angle of the torso axis away from "straight up", measured in image space
        # (y grows downwards). Rotating by -theta maps it back onto the vertical axis,
        # which cancels both camera tilt and transient body lean.
        theta = float(np.arctan2(torso[0], -torso[1]))
        if self.rotate and abs(torso[0]) > MIN_TORSO_SPAN and abs(torso[1]) > MIN_TORSO_SPAN:
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rotation = np.asarray(
                [[cos_t, sin_t, 0.0], [-sin_t, cos_t, 0.0], [0.0, 0.0, cos_t]],
                dtype=np.float32,
            )
        else:
            rotation = np.eye(3, dtype=np.float32)

        centered = np.concatenate(
            [xy - center, z[:, None]], axis=1
        )  # (33, 3) hip-centred in image space

        # View A: centred + scaled only. Keeps the body's true orientation in the image,
        # which is what the kinematic (fall) descriptors are derived from.
        unaligned = centered / scale
        # View B: additionally rotated so the torso axis is vertical. Invariant to camera
        # tilt, which is what the raw coordinate block fed to the LSTM needs.
        normalized = (centered @ rotation.T) / scale

        # Re-apply the mask after transformation so invalid joints stay exactly zero.
        unaligned[~valid] = 0.0
        normalized[~valid] = 0.0
        unaligned = np.nan_to_num(unaligned, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return NormalizedPose(
            coordinates=normalized,
            unaligned=unaligned,
            valid=valid,
            scale=scale,
            center=(float(center[0]), float(center[1])),
        )

    # ------------------------------------------------------------------ #
    def _validity_mask(self, array: np.ndarray) -> np.ndarray:
        visibility = array[:, 3]
        valid = np.isfinite(visibility) & (visibility >= self.min_visibility)
        valid &= np.isfinite(array[:, 0]) & np.isfinite(array[:, 1])
        return valid

    @staticmethod
    def _bounding_box_diagonal(xy: np.ndarray, valid: np.ndarray) -> float:
        """Diagonal of the bounding box of the valid landmarks (a scale reference)."""
        points = xy[valid]
        if points.shape[0] < 2:
            return 0.0
        extent = points.max(axis=0) - points.min(axis=0)
        return float(np.linalg.norm(extent))

    def _repair_reference(
        self, array: np.ndarray, valid: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(shoulder_mid, hip_mid)`` in image coordinates, or ``None``.

        Falls back to the other pair when one pair is unusable, and to a single
        landmark when only one of each pair is valid.
        """
        xy = array[:, :2]
        shoulder_mid = self._midpoint(xy, valid, LEFT_SHOULDER, RIGHT_SHOULDER)
        hip_mid = self._midpoint(xy, valid, LEFT_HIP, RIGHT_HIP)
        if shoulder_mid is not None and hip_mid is not None:
            return shoulder_mid, hip_mid
        if shoulder_mid is not None or hip_mid is not None:
            available = shoulder_mid if shoulder_mid is not None else hip_mid
            return available, available  # degenerate, but caught by the scale guard
        return None

    @staticmethod
    def _midpoint(
        xy: np.ndarray, valid: np.ndarray, left: int, right: int
    ) -> np.ndarray | None:
        """Midpoint of a landmark pair, using whichever landmark is valid."""
        points = [xy[index] for index in (left, right) if valid[index]]
        if not points:
            return None
        return np.mean(np.stack(points, axis=0), axis=0).astype(np.float32)
