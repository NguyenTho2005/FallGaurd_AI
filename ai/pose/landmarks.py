"""MediaPipe Pose landmark constants and pose record types.

The 33 MediaPipe Pose landmarks are referenced by index everywhere in FallGuard AI.
This module is the single place those indices are named, so a landmark can never be
mis-identified in one module and correctly identified in another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Final

import numpy as np

#: Number of landmarks produced by MediaPipe Pose.
POSE_LANDMARK_COUNT: Final[int] = 33

#: Number of values stored per landmark: ``[x, y, z, visibility]``.
POSE_LANDMARK_WIDTH: Final[int] = 4

#: A training window counts as a fall when at least this share of its frames are
#: fall-annotated. 0.5 is the natural midpoint for a binary per-frame annotation.
FALL_LABEL_WINDOW_RATIO: Final[float] = 0.5


class PoseLandmark(IntEnum):
    """Indices of the 33 MediaPipe Pose landmarks."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


#: Landmark triplets used as the anatomical reference for normalisation.
LEFT_SHOULDER: Final[int] = int(PoseLandmark.LEFT_SHOULDER)
RIGHT_SHOULDER: Final[int] = int(PoseLandmark.RIGHT_SHOULDER)
LEFT_HIP: Final[int] = int(PoseLandmark.LEFT_HIP)
RIGHT_HIP: Final[int] = int(PoseLandmark.RIGHT_HIP)
LEFT_ELBOW: Final[int] = int(PoseLandmark.LEFT_ELBOW)
RIGHT_ELBOW: Final[int] = int(PoseLandmark.RIGHT_ELBOW)
LEFT_WRIST: Final[int] = int(PoseLandmark.LEFT_WRIST)
RIGHT_WRIST: Final[int] = int(PoseLandmark.RIGHT_WRIST)
LEFT_KNEE: Final[int] = int(PoseLandmark.LEFT_KNEE)
RIGHT_KNEE: Final[int] = int(PoseLandmark.RIGHT_KNEE)
LEFT_ANKLE: Final[int] = int(PoseLandmark.LEFT_ANKLE)
RIGHT_ANKLE: Final[int] = int(PoseLandmark.RIGHT_ANKLE)
LEFT_HEEL: Final[int] = int(PoseLandmark.LEFT_HEEL)
RIGHT_HEEL: Final[int] = int(PoseLandmark.RIGHT_HEEL)
LEFT_FOOT_INDEX: Final[int] = int(PoseLandmark.LEFT_FOOT_INDEX)
RIGHT_FOOT_INDEX: Final[int] = int(PoseLandmark.RIGHT_FOOT_INDEX)
NOSE: Final[int] = int(PoseLandmark.NOSE)

#: Skeleton connections, expressed as landmark index pairs. Used by the dashboard
#: and by the privacy-friendly skeleton renderer.
POSE_CONNECTIONS: Final[tuple[tuple[int, int], ...]] = (
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL),
    (LEFT_ANKLE, LEFT_FOOT_INDEX),
    (RIGHT_ANKLE, RIGHT_HEEL),
    (RIGHT_ANKLE, RIGHT_FOOT_INDEX),
    (NOSE, LEFT_SHOULDER),
    (NOSE, RIGHT_SHOULDER),
)

#: Landmarks that describe the torso/limbs. Facial landmarks add noise and, per the
#: privacy requirement, are excluded from the modelling feature set.
BODY_LANDMARK_INDICES: Final[tuple[int, ...]] = tuple(
    i for i in range(POSE_LANDMARK_COUNT) if i >= LEFT_SHOULDER
)


@dataclass(frozen=True, slots=True)
class PoseFrame:
    """A single frame of pose data.

    Attributes
    ----------
    landmarks:
        ``(33, 4)`` float array of ``[x, y, z, visibility]``. ``x``/``y`` are
        normalised image coordinates in ``[0, 1]``; ``z`` is relative depth with the
        hip midpoint as origin; ``visibility`` is a confidence in ``[0, 1]``.
    timestamp:
        Monotonic capture time in seconds.
    frame_index:
        Index of the frame in the source stream (0-based).
    """

    landmarks: np.ndarray
    timestamp: float
    frame_index: int = 0
    width: int = 0
    height: int = 0

    def __post_init__(self) -> None:
        array = np.asarray(self.landmarks, dtype=np.float32)
        if array.shape != (POSE_LANDMARK_COUNT, POSE_LANDMARK_WIDTH):
            raise ValueError(
                f"landmarks must have shape ({POSE_LANDMARK_COUNT}, {POSE_LANDMARK_WIDTH}), "
                f"got {array.shape}"
            )
        object.__setattr__(self, "landmarks", array)

    @property
    def xy(self) -> np.ndarray:
        """The ``(33, 2)`` view of the landmark coordinates."""
        return self.landmarks[:, :2]

    @property
    def visibility(self) -> np.ndarray:
        """The ``(33,)`` visibility/confidence values."""
        return self.landmarks[:, 3]

    def mean_visibility(self) -> float:
        return float(self.visibility.mean())

    def to_array(self) -> np.ndarray:
        """Return a detached copy of the landmark array."""
        return self.landmarks.copy()


@dataclass(frozen=True, slots=True)
class PoseSample:
    """A contiguous run of pose frames extracted from one recording.

    ``group_id`` identifies the source video / subject and is what the training pipeline
    uses to split data, which is how frame-level leakage is prevented.

    ``frame_labels`` carries a *per-frame* binary annotation when the source provides
    one. A fall recording is not entirely a fall: the seconds before the fall are normal
    activity, and labelling them ``fall`` teaches the model to fire on ordinary standing.
    When present, a training window is labelled by the share of fall frames it contains
    (:data:`FALL_LABEL_WINDOW_RATIO`); when absent the recording-level ``label`` is used.
    """

    frames: list[PoseFrame] = field(default_factory=list)
    label: int = 0
    group_id: str = "unknown"
    source: str = ""
    frame_labels: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.frame_labels is not None:
            values = np.asarray(self.frame_labels).reshape(-1)
            if values.shape[0] != len(self.frames):
                raise ValueError(
                    f"frame_labels has {values.shape[0]} entries but the sample has "
                    f"{len(self.frames)} frames"
                )
            object.__setattr__(self, "frame_labels", values.astype(np.int64))

    def __len__(self) -> int:
        return len(self.frames)

    def to_array(self) -> np.ndarray:
        """``(T, 33, 4)`` array of raw landmarks."""
        if not self.frames:
            return np.zeros((0, POSE_LANDMARK_COUNT, POSE_LANDMARK_WIDTH), dtype=np.float32)
        return np.stack([frame.landmarks for frame in self.frames], axis=0)


__all__ = [
    "BODY_LANDMARK_INDICES",
    "FALL_LABEL_WINDOW_RATIO",
    "POSE_CONNECTIONS",
    "POSE_LANDMARK_COUNT",
    "POSE_LANDMARK_WIDTH",
    "PoseFrame",
    "PoseLandmark",
    "PoseSample",
]
