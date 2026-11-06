"""Dataset assembly and leakage-free splitting.

Two input formats are supported, and both keep the notion of a **group**:

``.npy`` landmark files
    ``(T, 33, 4)`` raw MediaPipe landmarks saved by
    ``scripts/generate_sample_data.py``. The file's relative path is the group id, so
    every window derived from it is tied to that group.

video files
    Videos are decoded with MediaPipe on the fly (:func:`extract_samples_from_videos`).

Splitting
---------
A window is a *slice* of a longer recording. Splitting those slices at random would put
near-identical frames from the same recording on both sides of the train/test boundary
and inflate every metric. :func:`group_split` therefore assigns whole **groups** to
splits, deterministically, and keeps each split class-balanced where the data allows.
``tests/unit/test_dataset.py`` asserts that no group appears in two splits.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np

from ai.config import Settings, get_settings
from ai.logging_config import get_logger
from ai.pose.landmarks import (
    FALL_LABEL_WINDOW_RATIO,
    POSE_LANDMARK_COUNT,
    POSE_LANDMARK_WIDTH,
    PoseFrame,
    PoseSample,
)
from ai.preprocessing.pose_preprocessor import FEATURE_DIM, PosePreprocessor

logger = get_logger(__name__)

SplitName = Literal["train", "val", "test"]

#: Directory / file-name conventions that map a path to a binary label.
FALL_TOKENS = ("fall", "falling", "fallen", "positive", "1")
NON_FALL_TOKENS = (
    "non_fall",
    "nonfall",
    "non-fall",
    "normal",
    "adl",
    "activities",
    "negative",
    "0",
    "sit",
    "walk",
    "stand",
    "lying",
)


@dataclass(slots=True)
class WindowedDataset:
    """Sliding windows ready for a Keras model.

    Attributes
    ----------
    x: ``(N, sequence_length, FEATURE_DIM)`` float32
    y: ``(N,)`` float32 in ``{0.0, 1.0}``
    groups: ``(N,)`` object array of group ids (source recording / subject)
    """

    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    sequence_length: int
    feature_dim: int
    label_names: dict[int, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def positive_rate(self) -> float:
        if len(self) == 0:
            return 0.0
        return float(self.y.mean())

    def subset(self, mask: Sequence[bool] | np.ndarray) -> "WindowedDataset":
        index = np.asarray(mask, dtype=bool)
        return WindowedDataset(
            x=self.x[index],
            y=self.y[index],
            groups=self.groups[index],
            sequence_length=self.sequence_length,
            feature_dim=self.feature_dim,
            label_names=dict(self.label_names),
        )

    def summary(self) -> dict[str, object]:
        unique, counts = np.unique(self.groups, return_counts=True)
        return {
            "windows": len(self),
            "sequence_length": self.sequence_length,
            "feature_dim": self.feature_dim,
            "positive_rate": round(self.positive_rate, 4),
            "groups": int(unique.size),
            "windows_per_group_min": int(counts.min()) if counts.size else 0,
            "windows_per_group_max": int(counts.max()) if counts.size else 0,
        }


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def label_for_path(path: Path, root: Path) -> int | None:
    """Infer a binary label from a path, or ``None`` when it is ambiguous."""
    relative = path.relative_to(root) if path.is_relative_to(root) else path
    text = "/".join(relative.parts).lower()
    # Check the negative patterns first: "non_fall" contains "fall".
    for token in NON_FALL_TOKENS:
        if token in text:
            return 0
    for token in FALL_TOKENS:
        if token in text:
            return 1
    return None


def load_npy_samples(
    directory: str | Path, *, min_frames: int = 2
) -> list[PoseSample]:
    """Load every ``*.npy`` landmark file under ``directory`` into a :class:`PoseSample`.

    A ``*.json`` sidecar next to the file may supply ``label``, ``group_id`` and
    ``source``; otherwise they are inferred from the path.
    """
    root = Path(directory)
    if not root.is_dir():
        logger.warning("Dataset directory %s does not exist", root)
        return []
    samples: list[PoseSample] = []
    for npy_path in sorted(root.rglob("*.npy")):
        try:
            array = np.load(npy_path)
        except Exception:
            logger.warning("Could not read %s; skipping", npy_path, exc_info=True)
            continue
        if array.ndim != 3 or array.shape[1:] != (POSE_LANDMARK_COUNT, POSE_LANDMARK_WIDTH):
            logger.warning(
                "Skipping %s: expected (T, %d, %d), got %s",
                npy_path.name, POSE_LANDMARK_COUNT, POSE_LANDMARK_WIDTH, (array.shape,),
            )
            continue
        if array.shape[0] < min_frames:
            logger.warning("Skipping %s: only %d frames", npy_path.name, array.shape[0])
            continue

        metadata = _read_sidecar(npy_path)
        label = metadata.get("label")
        if label is None:
            label = label_for_path(npy_path, root)
        if label is None:
            logger.warning(
                "Skipping %s: cannot infer a label from the path and no sidecar provides one",
                npy_path.name,
            )
            continue
        group = str(metadata.get("group_id") or npy_path.relative_to(root).with_suffix("").as_posix())
        frames = [
            PoseFrame(landmarks=array[i].astype(np.float32), timestamp=i / 30.0, frame_index=i)
            for i in range(array.shape[0])
        ]
        frame_labels = metadata.get("frame_labels")
        samples.append(
            PoseSample(
                frames=frames,
                label=int(label),
                group_id=group,
                source=str(metadata.get("source") or npy_path.stem),
                frame_labels=(
                    np.asarray(frame_labels, dtype=np.int64)
                    if isinstance(frame_labels, list)
                    else None
                ),
            )
        )
    logger.info("Loaded %d landmark samples from %s", len(samples), root)
    return samples


def _read_sidecar(npy_path: Path) -> dict[str, object]:
    sidecar = npy_path.with_suffix(".json")
    if not sidecar.is_file():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read sidecar %s", sidecar, exc_info=True)
        return {}


def extract_samples_from_videos(
    video_paths: Iterable[Path],
    labels: dict[str, int] | None = None,
    *,
    settings: Settings | None = None,
    stride: int = 1,
    progress: bool = False,
) -> list[PoseSample]:
    """Decode videos and extract MediaPipe landmarks for each frame.

    The pose estimator is imported and constructed here rather than at module import so
    that the rest of the training package stays usable without the MediaPipe asset.
    """
    from ai.pose.pose_estimator import PoseEstimator
    from ai.sources.video_source import VideoFileSource

    settings = settings or get_settings()
    samples: list[PoseSample] = []
    try:
        estimator = PoseEstimator(settings)
    except Exception as exc:
        logger.error("Cannot extract poses from video: %s", exc)
        return samples

    try:
        for video_path in video_paths:
            label = 1 if labels is None else int(labels.get(video_path.name, 0))
            source = VideoFileSource(video_path, settings)
            frames: list[PoseFrame] = []
            index = 0
            while True:
                source_frame = source.read()
                if source_frame is None:
                    break
                if index % stride == 0:
                    pose = estimator.process(
                        source_frame.frame,
                        timestamp=source_frame.timestamp,
                        frame_index=source_frame.index,
                    )
                    if pose is not None:
                        frames.append(pose)
                index += 1
            source.close()
            if len(frames) < 2:
                logger.warning("Video %s yielded %d usable frames; skipping", video_path.name, len(frames))
                continue
            samples.append(
                PoseSample(
                    frames=frames,
                    label=label,
                    group_id=video_path.stem,
                    source=str(video_path.name),
                )
            )
            if progress:
                logger.info("Extracted %d pose frames from %s", len(frames), video_path.name)
    finally:
        estimator.close()
    return samples


# --------------------------------------------------------------------------- #
# windowing
# --------------------------------------------------------------------------- #
def build_windows(
    samples: Sequence[PoseSample],
    preprocessor: PosePreprocessor,
    *,
    sequence_length: int | None = None,
    stride: int | None = None,
    fall_label_ratio: float = FALL_LABEL_WINDOW_RATIO,
) -> WindowedDataset:
    """Slide a window over every sample and convert it to model inputs.

    Pose-less frames are *kept* as zero vectors with ``present = 0`` so a dropout inside a
    recording is represented rather than silently closed up.

    Window labels come from :attr:`PoseSample.frame_labels` when present (a window is a
    fall when at least ``fall_label_ratio`` of its frames are fall-annotated), and from
    the recording-level :attr:`PoseSample.label` otherwise.
    """
    settings = get_settings()
    sequence_length = int(sequence_length or settings.sequence_length)
    # An explicit stride of 0 is a caller error, not a request for the default.
    stride = int(max(1, sequence_length // 3)) if stride is None else int(stride)
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    chunks: list[np.ndarray] = []
    labels: list[float] = []
    groups: list[str] = []
    for sample in samples:
        features = preprocessor.transform_sequence(sample.to_array())
        total = features.shape[0]
        if total < sequence_length:
            logger.warning(
                "Sample %s has %d frames, fewer than the %d-frame window; skipping",
                sample.group_id, total, sequence_length,
            )
            continue
        for start in range(0, total - sequence_length + 1, stride):
            end = start + sequence_length
            if sample.frame_labels is not None:
                window_labels = sample.frame_labels[start:end]
                label = float(window_labels.mean() >= fall_label_ratio)
            else:
                label = float(sample.label)
            chunks.append(features[start:end])
            labels.append(label)
            groups.append(sample.group_id)

    if not chunks:
        return WindowedDataset(
            x=np.zeros((0, sequence_length, preprocessor.feature_dim), dtype=np.float32),
            y=np.zeros((0,), dtype=np.float32),
            groups=np.zeros((0,), dtype=object),
            sequence_length=sequence_length,
            feature_dim=preprocessor.feature_dim,
            label_names={0: "non_fall", 1: "fall"},
        )

    return WindowedDataset(
        x=np.stack(chunks, axis=0).astype(np.float32),
        y=np.asarray(labels, dtype=np.float32),
        groups=np.asarray(groups, dtype=object),
        sequence_length=sequence_length,
        feature_dim=preprocessor.feature_dim,
        label_names={0: "non_fall", 1: "fall"},
    )


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def unique_groups(groups: np.ndarray) -> list[str]:
    """Sorted unique group ids, for logging and report provenance."""
    return sorted({str(group) for group in np.asarray(groups, dtype=object)})


def _group_key(group: str, seed: int) -> int:
    """Stable per-group pseudo-random key, so a split is reproducible across runs."""
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def group_split(
    groups: np.ndarray,
    labels: np.ndarray,
    *,
    test_split: float = 0.2,
    val_split: float = 0.2,
    seed: int = 20240917,
) -> dict[str, np.ndarray]:
    """Assign whole groups to ``train`` / ``val`` / ``test``.

    The split is a deterministic function of ``(seed, group_id)``, so re-running the
    pipeline reproduces the same partition -- and no group can straddle two splits,
    which is what prevents frame-level leakage.

    Class balance is approached greedily: groups are visited in a stable order and each
    is assigned to the split that is furthest below its target size *and* still needs the
    sample's label. This is deterministic, unlike random shuffling.
    """
    if test_split < 0.0 or val_split < 0.0 or test_split + val_split >= 1.0:
        raise ValueError(
            f"splits must be non-negative and sum to < 1.0, got test={test_split}, val={val_split}"
        )
    groups = np.asarray(groups, dtype=object)
    labels = np.asarray(labels)
    if groups.shape[0] != labels.shape[0]:
        raise ValueError("groups and labels must have the same length")

    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[str(group)].append(index)

    if not by_group:
        return {
            name: np.zeros(0, dtype=bool) for name in ("train", "val", "test")
        }

    total = len(groups)
    targets = {
        "test": test_split * total,
        "val": val_split * total,
        "train": (1.0 - test_split - val_split) * total,
    }
    filled = {"test": 0, "val": 0, "train": 0}
    assigned: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    ordered = sorted(by_group, key=lambda g: (_group_key(g, seed), g))
    for group in ordered:
        indices = by_group[group]
        group_labels = {int(round(float(labels[i]))) for i in indices}
        if len(group_labels) > 1:
            logger.warning(
                "Group %s contains both classes %s; assigning it as a unit and noting the "
                "conflict", group, sorted(group_labels),
            )
        label = max(group_labels) if group_labels else 0
        # Prefer a split that still needs this label, then the emptiest relative deficit.
        candidates = [
            name
            for name in ("train", "val", "test")
            if targets[name] > 0.0 or name == "train"
        ]
        need = [
            name
            for name in candidates
            if not _split_has_label(assigned, labels, by_group, name, label)
        ] or candidates
        chosen = max(need, key=lambda name: (targets[name] - filled[name]) / max(targets[name], 1.0))
        assigned[chosen].append(group)
        filled[chosen] += len(indices)

    masks: dict[str, np.ndarray] = {}
    group_to_split = {
        group: name for name, members in assigned.items() for group in members
    }
    for name in ("train", "val", "test"):
        mask = np.array(
            [group_to_split[str(group)] == name for group in groups], dtype=bool
        )
        masks[name] = mask
        logger.info(
            "Split %-5s windows=%4d groups=%2d positive_rate=%.3f",
            name, int(mask.sum()), len(assigned[name]),
            float(labels[mask].mean()) if mask.any() else 0.0,
        )
    return masks


def _split_has_label(
    assigned: dict[str, list[str]],
    labels: np.ndarray,
    by_group: dict[str, list[int]],
    split: str,
    label: int,
) -> bool:
    for group in assigned[split]:
        for index in by_group[group]:
            if int(round(float(labels[index]))) == label:
                return True
    return False


def assert_no_group_leakage(
    dataset: WindowedDataset, masks: dict[str, np.ndarray]
) -> None:
    """Raise if any group appears in more than one split.

    This is a hard invariant, not a heuristic, so it is enforced both in the trainer and
    in the unit tests.
    """
    seen: dict[str, str] = {}
    for name, mask in masks.items():
        for group in np.unique(dataset.groups[mask]):
            key = str(group)
            if key in seen and seen[key] != name:
                raise AssertionError(
                    f"Data leakage: group '{key}' appears in both the "
                    f"'{seen[key]}' and '{name}' splits"
                )
            seen[key] = name


def split_dataset(
    dataset: WindowedDataset, settings: Settings | None = None
) -> dict[str, WindowedDataset]:
    """Convenience wrapper: split and materialise per-split datasets."""
    settings = settings or get_settings()
    masks = group_split(
        dataset.groups,
        dataset.y,
        test_split=settings.test_split,
        val_split=settings.validation_split,
        seed=settings.random_seed,
    )
    assert_no_group_leakage(dataset, masks)
    return {name: dataset.subset(mask) for name, mask in masks.items()}


def class_weights(labels: np.ndarray) -> dict[int, float]:
    """Inverse-frequency weights, so an imbalanced dataset does not collapse to one class."""
    values = np.asarray(labels)
    if values.size == 0:
        return {}
    total = values.size
    weights: dict[int, float] = {}
    for value in (0, 1):
        count = int((values == value).sum())
        weights[value] = float(total / (2.0 * count)) if count else 1.0
    logger.info("Computed class weights: %s", {k: round(v, 3) for k, v in weights.items()})
    return weights


def stack_or_empty(
    datasets: Sequence[WindowedDataset], feature_dim: int = FEATURE_DIM
) -> np.ndarray:
    """Concatenate datasets, returning an empty array when all are empty."""
    parts = [item.x for item in datasets if len(item)]
    if not parts:
        return np.zeros((0, 1, feature_dim), dtype=np.float32)
    return np.concatenate(parts, axis=0)
