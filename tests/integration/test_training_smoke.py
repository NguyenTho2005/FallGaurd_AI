"""End-to-end training smoke test.

Exercises the full training path -- load ``.npy`` landmarks, window them, split by group,
train, checkpoint, write the metadata sidecar, reload and score -- on a small synthetic
in-memory dataset. No video decoding, no real dataset, no GPU.

Marked ``slow`` because it trains a real Keras model, and therefore deselected by default:
run it with ``pytest -m slow``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ai.config import Settings
from ai.pose.landmarks import PoseFrame, PoseSample
from ai.preprocessing.pose_preprocessor import FEATURE_DIM, PosePreprocessor
from ai.training.dataset import build_windows, load_npy_samples, split_dataset
from ai.training.trainer import InsufficientDataError, train_model
from tests.conftest import ON_FLOOR, STANDING, landmark_array

pytestmark = [pytest.mark.slow, pytest.mark.model, pytest.mark.integration]

SEQUENCE_LENGTH = 8


def _write_dataset(root: Path, fall_groups: int = 4, normal_groups: int = 4) -> None:
    (root / "fall").mkdir(parents=True, exist_ok=True)
    (root / "non_fall").mkdir(parents=True, exist_ok=True)
    for index in range(fall_groups):
        frames = 40
        labels = [0] * (frames - 12) + [1] * 12
        np.save(root / "fall" / f"f{index}.npy",
                np.stack([landmark_array(STANDING if not l else ON_FLOOR)
                          for l in labels]))
        (root / "fall" / f"f{index}.json").write_text(
            json.dumps({"label": 1, "group_id": f"fall_group_{index}",
                        "frame_labels": labels}), encoding="utf-8"
        )
    for index in range(normal_groups):
        np.save(root / "non_fall" / f"n{index}.npy",
                np.stack([landmark_array(STANDING) for _ in range(40)]))
        (root / "non_fall" / f"n{index}.json").write_text(
            json.dumps({"label": 0, "group_id": f"normal_group_{index}"}), encoding="utf-8"
        )


@pytest.fixture
def dataset(settings: Settings, tmp_path: Path):
    _write_dataset(tmp_path / "data")
    samples = load_npy_samples(tmp_path / "data")
    preprocessor = PosePreprocessor.from_settings(settings)
    return build_windows(
        samples, preprocessor, sequence_length=SEQUENCE_LENGTH, stride=4
    )


def test_dataset_contains_both_classes(dataset):
    assert len(dataset) > 0
    assert set(dataset.y.tolist()) == {0.0, 1.0}
    assert dataset.x.shape[2] == FEATURE_DIM


def test_train_checkpoint_and_reload(settings: Settings, dataset, tmp_path: Path):
    settings.sequence_length = SEQUENCE_LENGTH
    settings.epochs = 3
    settings.batch_size = 8
    settings.test_split = 0.25
    settings.validation_split = 0.25
    output = tmp_path / "models" / "smoke.keras"

    result = train_model(
        dataset, architecture="lstm", settings=settings, model_path=output, verbose=0
    )
    assert output.is_file()
    assert result.metadata_path is not None and result.metadata_path.is_file()
    assert result.model.count_params() > 0
    assert "loss" in result.history and result.history["loss"]
    assert set(result.splits) == {"train", "val", "test"}

    # No group may appear in two splits.
    splits = split_dataset(dataset, settings)
    train_groups = {str(g) for g in splits["train"].groups}
    test_groups = {str(g) for g in splits["test"].groups}
    assert train_groups.isdisjoint(test_groups)


def test_metadata_sidecar_records_the_serving_contract(settings: Settings, dataset,
                                                       tmp_path: Path):
    settings.sequence_length = SEQUENCE_LENGTH
    settings.epochs = 2
    settings.batch_size = 8
    output = tmp_path / "models" / "meta.keras"
    result = train_model(dataset, settings=settings, model_path=output, verbose=0)

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["sequence_length"] == SEQUENCE_LENGTH
    assert metadata["feature_dim"] == FEATURE_DIM
    assert metadata["architecture"] == "lstm"
    assert metadata["feature_spec_version"]
    assert metadata["parameters"] > 0
    assert metadata["hyperparameters"]["epochs"] == 2
    assert metadata["serving"]["confidence_threshold"] == settings.confidence_threshold
    assert metadata["created_utc"].endswith("Z")
    assert metadata["environment"]["python"]


def test_saved_model_can_be_served_again(settings: Settings, dataset, tmp_path: Path):
    """The checkpoint must load back through the production predictor path."""
    from ai.models.predictor import FallPredictor

    settings.sequence_length = SEQUENCE_LENGTH
    settings.epochs = 2
    settings.batch_size = 8
    settings.model_backend = "keras"
    settings.model_path = tmp_path / "models" / "serve.keras"
    settings.model_metadata_path = tmp_path / "models" / "serve.meta.json"
    train_model(dataset, settings=settings, model_path=settings.model_path, verbose=0)

    predictor = FallPredictor(settings)
    assert predictor.is_loaded
    assert predictor.backend == "keras"
    assert predictor.sequence_length == SEQUENCE_LENGTH
    probability = predictor.predict_proba(np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), np.float32))
    assert 0.0 <= probability <= 1.0


def test_trained_model_produces_measured_metrics(settings: Settings, dataset, tmp_path: Path):
    """Metrics are recorded because they were computed -- and are marked as synthetic."""
    settings.sequence_length = SEQUENCE_LENGTH
    settings.epochs = 3
    settings.batch_size = 8
    settings.test_split = 0.25
    settings.validation_split = 0.25
    result = train_model(
        dataset, settings=settings, model_path=tmp_path / "models" / "m.keras", verbose=0
    )
    assert result.evaluation is not None
    for key in ("accuracy", "precision", "recall", "f1", "confusion_matrix", "n_samples"):
        assert key in result.evaluation
    assert 0.0 <= result.evaluation["accuracy"] <= 1.0
    assert result.evaluation["n_samples"] > 0


def test_bilstm_trains_too(settings: Settings, dataset, tmp_path: Path):
    settings.sequence_length = SEQUENCE_LENGTH
    settings.epochs = 2
    settings.batch_size = 8
    result = train_model(
        dataset, architecture="bilstm", settings=settings,
        model_path=tmp_path / "models" / "bi.keras", verbose=0,
    )
    assert result.metadata_path.is_file()
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["architecture"] == "bilstm"


def test_empty_dataset_is_rejected_with_guidance(settings: Settings, tmp_path: Path):
    preprocessor = PosePreprocessor.from_settings(settings)
    empty = build_windows([], preprocessor, sequence_length=8)
    with pytest.raises(InsufficientDataError, match="empty"):
        train_model(empty, settings=settings, model_path=tmp_path / "x.keras", verbose=0)


def test_single_group_dataset_reports_that_no_metrics_were_produced(settings: Settings,
                                                                   tmp_path: Path):
    """One recording cannot fill three splits.

    The run must still produce a usable checkpoint, but it must say plainly that no
    evaluation happened rather than reporting an empty or invented score.
    """
    settings.sequence_length = 4
    settings.epochs = 1
    settings.batch_size = 4
    sample = PoseSample(
        frames=[
            PoseFrame(landmarks=landmark_array(STANDING), timestamp=float(i), frame_index=i)
            for i in range(20)
        ],
        label=1,
        group_id="only_group",
    )
    preprocessor = PosePreprocessor.from_settings(settings)
    dataset = build_windows([sample], preprocessor, sequence_length=4)
    assert len(dataset) > 0

    result = train_model(
        dataset, settings=settings, model_path=tmp_path / "models" / "single.keras", verbose=0
    )
    assert result.model_path.is_file()
    assert result.evaluation is None, "no test data means no metrics, not zero metrics"
    assert any("no test metrics" in message for message in result.messages)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["evaluation"] is None
