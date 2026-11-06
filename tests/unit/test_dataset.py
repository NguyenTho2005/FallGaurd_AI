"""Unit tests for dataset assembly, group-aware splitting and leakage prevention.

The leakage tests are the important ones: window-level random shuffling is the classic
way to produce a beautiful, meaningless accuracy number.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ai.pose.landmarks import PoseFrame, PoseSample
from ai.preprocessing.pose_preprocessor import FEATURE_DIM, PosePreprocessor
from ai.training.dataset import (
    WindowedDataset,
    assert_no_group_leakage,
    build_windows,
    class_weights,
    group_split,
    label_for_path,
    load_npy_samples,
    split_dataset,
    unique_groups,
)
from tests.conftest import ON_FLOOR, STANDING, landmark_array


def make_sample(
    group_id: str,
    label: int,
    *,
    frames: int = 40,
    frame_labels: np.ndarray | None = None,
) -> PoseSample:
    return PoseSample(
        frames=[
            PoseFrame(
                landmarks=landmark_array(STANDING if label == 0 else ON_FLOOR),
                timestamp=index / 30.0,
                frame_index=index,
            )
            for index in range(frames)
        ],
        label=label,
        group_id=group_id,
        source=f"{group_id}.avi",
        frame_labels=frame_labels,
    )


@pytest.fixture
def balanced_dataset(preprocessor: PosePreprocessor) -> WindowedDataset:
    samples = [make_sample(f"fall_{i:02d}", 1) for i in range(6)]
    samples += [make_sample(f"normal_{i:02d}", 0) for i in range(6)]
    return build_windows(samples, preprocessor, sequence_length=10, stride=5)


# --------------------------------------------------------------------------- #
# windowing
# --------------------------------------------------------------------------- #
def test_windows_have_the_expected_shape(balanced_dataset: WindowedDataset):
    assert balanced_dataset.x.ndim == 3
    assert balanced_dataset.x.shape[1:] == (10, FEATURE_DIM)
    assert balanced_dataset.y.shape == (len(balanced_dataset),)
    assert balanced_dataset.groups.shape == (len(balanced_dataset),)


def test_every_window_keeps_its_group(balanced_dataset: WindowedDataset):
    for index, group in enumerate(balanced_dataset.groups):
        assert str(group).startswith(("fall_", "normal_"))
        assert balanced_dataset.y[index] == float(str(group).startswith("fall_"))


def test_stride_controls_window_density(preprocessor: PosePreprocessor):
    samples = [make_sample("fall_00", 1, frames=40)]
    dense = build_windows(samples, preprocessor, sequence_length=10, stride=1)
    sparse = build_windows(samples, preprocessor, sequence_length=10, stride=10)
    assert len(dense) == 31
    assert len(sparse) == 4


def test_short_recordings_are_skipped(preprocessor: PosePreprocessor):
    samples = [make_sample("tiny", 1, frames=5)]
    dataset = build_windows(samples, preprocessor, sequence_length=30)
    assert len(dataset) == 0
    assert dataset.x.shape == (0, 30, FEATURE_DIM)


def test_invalid_stride_is_rejected(preprocessor: PosePreprocessor):
    with pytest.raises(ValueError, match="stride"):
        build_windows([make_sample("g", 1)], preprocessor, sequence_length=5, stride=0)


def test_temporal_labels_decide_the_window_label(preprocessor: PosePreprocessor):
    """The seconds before a fall are normal activity and must be labelled as such."""
    frames = 40
    frame_labels = np.zeros(frames, dtype=np.int64)
    frame_labels[30:] = 1  # the fall occupies the last 10 frames
    sample = make_sample("fall_00", 1, frames=frames, frame_labels=frame_labels)
    dataset = build_windows([sample], preprocessor, sequence_length=10, stride=10)
    assert len(dataset) > 1
    assert dataset.y[0] == 0.0, "a window entirely before the fall is normal"
    assert dataset.y[-1] == 1.0, "a window inside the fall is a fall"


def test_recording_level_label_is_used_without_temporal_annotations(
    preprocessor: PosePreprocessor,
):
    dataset = build_windows([make_sample("fall_00", 1)], preprocessor, sequence_length=10)
    assert set(dataset.y.tolist()) == {1.0}


def test_frame_labels_length_must_match():
    with pytest.raises(ValueError, match="frame_labels"):
        PoseSample(
            frames=[PoseFrame(landmarks=landmark_array(STANDING), timestamp=0.0)],
            label=1,
            frame_labels=np.zeros(5, dtype=np.int64),
        )


def test_dataset_summary(balanced_dataset: WindowedDataset):
    summary = balanced_dataset.summary()
    assert summary["windows"] == len(balanced_dataset)
    assert summary["groups"] == 12
    assert summary["feature_dim"] == FEATURE_DIM
    assert 0.0 < summary["positive_rate"] < 1.0


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def test_no_group_appears_in_two_splits(balanced_dataset: WindowedDataset):
    masks = group_split(
        balanced_dataset.groups, balanced_dataset.y, test_split=0.25, val_split=0.25
    )
    assert_no_group_leakage(balanced_dataset, masks)

    seen: dict[str, str] = {}
    for name, mask in masks.items():
        for group in np.unique(balanced_dataset.groups[mask]):
            key = str(group)
            assert key not in seen or seen[key] == name, f"{key} leaked into {name}"
            seen[key] = name


def test_split_covers_every_window_exactly_once(balanced_dataset: WindowedDataset):
    masks = group_split(
        balanced_dataset.groups, balanced_dataset.y, test_split=0.25, val_split=0.25
    )
    stacked = np.stack([masks["train"], masks["val"], masks["test"]])
    assert stacked.sum(axis=0).max() == 1
    assert stacked.sum() == len(balanced_dataset)


def test_split_is_deterministic(balanced_dataset: WindowedDataset):
    first = group_split(balanced_dataset.groups, balanced_dataset.y, seed=1234)
    second = group_split(balanced_dataset.groups, balanced_dataset.y, seed=1234)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_different_seeds_can_produce_different_splits(balanced_dataset: WindowedDataset):
    first = group_split(balanced_dataset.groups, balanced_dataset.y, seed=1)
    second = group_split(balanced_dataset.groups, balanced_dataset.y, seed=999)
    assert any(
        not np.array_equal(first[name], second[name]) for name in first
    ) or len(unique_groups(balanced_dataset.groups)) <= 3


def test_both_classes_reach_the_training_split(balanced_dataset: WindowedDataset):
    masks = group_split(balanced_dataset.groups, balanced_dataset.y, test_split=0.25, val_split=0.25)
    train_labels = balanced_dataset.y[masks["train"]]
    assert set(train_labels.tolist()) == {0.0, 1.0}


def test_leakage_assertion_catches_a_violation(balanced_dataset: WindowedDataset):
    masks = group_split(balanced_dataset.groups, balanced_dataset.y, test_split=0.25, val_split=0.25)
    broken = {name: mask.copy() for name, mask in masks.items()}
    train_group = np.unique(balanced_dataset.groups[broken["train"]])[0]
    broken["test"][balanced_dataset.groups == train_group] = True
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_group_leakage(balanced_dataset, broken)


def test_random_window_level_split_would_leak(balanced_dataset: WindowedDataset):
    """Documents the failure mode the group split exists to prevent.

    A random split puts frames from one recording on both sides; the guard catches it.
    """
    rng = np.random.default_rng(0)
    random_masks = rng.random(len(balanced_dataset)) < 0.75
    broken = {"train": random_masks, "val": ~random_masks & (rng.random(len(balanced_dataset)) < 0.5),
              "test": np.zeros(len(balanced_dataset), dtype=bool)}
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_group_leakage(balanced_dataset, broken)


@pytest.mark.parametrize(
    "test_split,val_split", [(-0.1, 0.2), (0.6, 0.6), (0.5, 0.5)]
)
def test_invalid_split_ratios_are_rejected(test_split: float, val_split: float):
    with pytest.raises(ValueError):
        group_split(np.array(["a", "b"], dtype=object), np.array([0.0, 1.0]),
                    test_split=test_split, val_split=val_split)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        group_split(np.array(["a", "b"], dtype=object), np.array([0.0]))


def test_empty_input_yields_empty_masks():
    masks = group_split(np.zeros(0, dtype=object), np.zeros(0))
    assert all(mask.size == 0 for mask in masks.values())


def test_split_dataset_materialises_per_split_datasets(
    balanced_dataset: WindowedDataset, settings
):
    splits = split_dataset(balanced_dataset, settings)
    assert set(splits) == {"train", "val", "test"}
    total = sum(len(item) for item in splits.values())
    assert total == len(balanced_dataset)
    for item in splits.values():
        assert item.x.shape[1:] == (balanced_dataset.sequence_length, FEATURE_DIM)


# --------------------------------------------------------------------------- #
# class weights
# --------------------------------------------------------------------------- #
def test_class_weights_balance_an_imbalanced_set():
    weights = class_weights(np.array([0.0] * 90 + [1.0] * 10))
    assert weights[1] > weights[0]
    assert weights[1] / weights[0] == pytest.approx(9.0, rel=0.1)


def test_class_weights_of_a_balanced_set_are_equal():
    assert class_weights(np.array([0.0, 1.0]))[0] == pytest.approx(1.0)
    assert class_weights(np.array([0.0, 1.0]))[1] == pytest.approx(1.0)


def test_class_weights_of_empty_input_is_empty():
    assert class_weights(np.zeros(0)) == {}


# --------------------------------------------------------------------------- #
# loading from disk
# --------------------------------------------------------------------------- #
def test_label_inference_from_path(tmp_path: Path):
    assert label_for_path(tmp_path / "fall" / "a.npy", tmp_path) == 1
    assert label_for_path(tmp_path / "non_fall" / "a.npy", tmp_path) == 0
    assert label_for_path(tmp_path / "normal" / "a.npy", tmp_path) == 0
    assert label_for_path(tmp_path / "unlabelled" / "a.npy", tmp_path) is None


def test_load_npy_samples_reads_arrays_and_sidecars(tmp_path: Path):
    directory = tmp_path / "fall"
    directory.mkdir()
    landmarks = np.stack([landmark_array(ON_FLOOR)] * 50)
    np.save(directory / "clip.npy", landmarks)
    (directory / "clip.json").write_text(
        json.dumps({"label": 1, "group_id": "subject01_fall01", "source": "clip.avi"}),
        encoding="utf-8",
    )
    samples = load_npy_samples(tmp_path)
    assert len(samples) == 1
    assert samples[0].label == 1
    assert samples[0].group_id == "subject01_fall01"
    assert len(samples[0]) == 50


def test_load_npy_samples_infers_label_from_the_folder(tmp_path: Path):
    directory = tmp_path / "non_fall"
    directory.mkdir()
    np.save(directory / "walk.npy", np.stack([landmark_array(STANDING)] * 30))
    samples = load_npy_samples(tmp_path)
    assert len(samples) == 1
    assert samples[0].label == 0
    assert samples[0].group_id == "non_fall/walk"


def test_load_npy_samples_reads_temporal_labels(tmp_path: Path):
    directory = tmp_path / "fall"
    directory.mkdir()
    np.save(directory / "clip.npy", np.stack([landmark_array(ON_FLOOR)] * 30))
    (directory / "clip.json").write_text(
        json.dumps({"label": 1, "frame_labels": [0] * 20 + [1] * 10}), encoding="utf-8"
    )
    samples = load_npy_samples(tmp_path)
    assert samples[0].frame_labels is not None
    assert samples[0].frame_labels.sum() == 10


def test_load_npy_samples_skips_bad_shapes(tmp_path: Path):
    (tmp_path / "fall").mkdir()
    np.save(tmp_path / "fall" / "good.npy", np.stack([landmark_array(STANDING)] * 30))
    np.save(tmp_path / "fall" / "bad.npy", np.zeros((10, 5, 4), dtype=np.float32))
    # One frame is below the default min_frames=2, so it cannot form a sequence.
    np.save(tmp_path / "fall" / "short.npy", np.zeros((1, 33, 4), dtype=np.float32))
    samples = load_npy_samples(tmp_path)
    assert len(samples) == 1
    assert samples[0].group_id.endswith("good")


def test_load_npy_samples_skips_ambiguous_paths(tmp_path: Path):
    directory = tmp_path / "clips"
    directory.mkdir()
    np.save(directory / "unknown.npy", np.stack([landmark_array(STANDING)] * 30))
    assert load_npy_samples(tmp_path) == []


def test_load_npy_samples_tolerates_a_missing_directory(tmp_path: Path):
    assert load_npy_samples(tmp_path / "absent") == []


def test_load_npy_samples_tolerates_a_corrupt_sidecar(tmp_path: Path):
    directory = tmp_path / "fall"
    directory.mkdir()
    np.save(directory / "clip.npy", np.stack([landmark_array(ON_FLOOR)] * 30))
    (directory / "clip.json").write_text("{broken", encoding="utf-8")
    samples = load_npy_samples(tmp_path)
    assert len(samples) == 1
    assert samples[0].label == 1, "the folder name still supplies the label"


def test_end_to_end_from_disk_to_leak_free_splits(tmp_path: Path, preprocessor, settings):
    (tmp_path / "fall").mkdir()
    (tmp_path / "non_fall").mkdir()
    for index in range(4):
        np.save(tmp_path / "fall" / f"f{index}.npy", np.stack([landmark_array(ON_FLOOR)] * 40))
        np.save(tmp_path / "non_fall" / f"n{index}.npy", np.stack([landmark_array(STANDING)] * 40))
    samples = load_npy_samples(tmp_path)
    dataset = build_windows(samples, preprocessor, sequence_length=10, stride=5)
    assert len(dataset) > 0
    splits = split_dataset(dataset, settings)
    train_groups = set(str(g) for g in splits["train"].groups)
    test_groups = set(str(g) for g in splits["test"].groups)
    assert train_groups.isdisjoint(test_groups)
