"""Unit tests for model construction, checkpointing, contract validation and inference.

These build tiny Keras models in-process instead of loading a checkpoint, so the suite
needs no trained artefact and no GPU.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ai.config import Settings
from ai.models.lstm import build_bilstm, build_lstm, build_model, model_input_spec
from ai.models.mock_model import MockFallModel
from ai.models.predictor import (
    FallPredictor,
    ModelContractError,
    ModelUnavailableError,
    create_predictor,
)
from ai.preprocessing.pose_preprocessor import (
    FEATURE_DIM,
    FEATURE_SPEC_VERSION,
    PRESENT_FEATURE_INDEX,
)

keras = pytest.importorskip("keras", reason="TensorFlow/Keras is required for model tests")

TINY_SEQUENCE = 6
TINY_FEATURES = 8


# --------------------------------------------------------------------------- #
# architecture
# --------------------------------------------------------------------------- #
@pytest.mark.model
def test_build_lstm_has_the_expected_signature():
    model = build_lstm(sequence_length=TINY_SEQUENCE, feature_dim=TINY_FEATURES, units=8,
                       settings=Settings(sequence_length=TINY_SEQUENCE))
    sequence_length, feature_dim = model_input_spec(model)
    assert (sequence_length, feature_dim) == (TINY_SEQUENCE, TINY_FEATURES)
    assert model.output_shape[-1] == 1
    assert model.count_params() > 0
    assert "fall_probability" in [layer.name for layer in model.layers]


@pytest.mark.model
def test_build_bilstm_has_the_expected_signature():
    model = build_bilstm(sequence_length=TINY_SEQUENCE, feature_dim=TINY_FEATURES, units=8,
                         settings=Settings(sequence_length=TINY_SEQUENCE))
    sequence_length, feature_dim = model_input_spec(model)
    assert (sequence_length, feature_dim) == (TINY_SEQUENCE, TINY_FEATURES)
    assert any(isinstance(layer, keras.layers.Bidirectional) for layer in model.layers)


@pytest.mark.model
def test_bilstm_is_larger_than_lstm():
    settings = Settings(sequence_length=TINY_SEQUENCE)
    lstm = build_lstm(sequence_length=TINY_SEQUENCE, feature_dim=TINY_FEATURES, units=8,
                      settings=settings)
    bilstm = build_bilstm(sequence_length=TINY_SEQUENCE, feature_dim=TINY_FEATURES, units=8,
                          settings=settings)
    assert bilstm.count_params() > lstm.count_params()


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="unknown architecture"):
        build_model("transformer")


@pytest.mark.model
def test_untrained_model_outputs_a_probability():
    model = build_lstm(sequence_length=TINY_SEQUENCE, feature_dim=TINY_FEATURES, units=8,
                       settings=Settings(sequence_length=TINY_SEQUENCE))
    output = model.predict(np.zeros((2, TINY_SEQUENCE, TINY_FEATURES)), verbose=0)
    assert output.shape == (2, 1)
    assert np.all((output >= 0.0) & (output <= 1.0))


# --------------------------------------------------------------------------- #
# checkpoint round trip
# --------------------------------------------------------------------------- #
@pytest.mark.model
def test_checkpoint_round_trip(tmp_path, settings: Settings):
    settings.sequence_length = TINY_SEQUENCE
    settings.lstm_units = 8
    path = tmp_path / "model.keras"
    model = build_lstm(sequence_length=TINY_SEQUENCE, feature_dim=TINY_FEATURES, units=8,
                       settings=settings)
    window = np.zeros((1, TINY_SEQUENCE, TINY_FEATURES), dtype=np.float32)
    before = float(model.predict(window, verbose=0)[0][0])
    model.save(path)
    assert path.is_file()

    restored = keras.models.load_model(path)
    after = float(restored.predict(window, verbose=0)[0][0])
    assert after == pytest.approx(before, abs=1e-5)


# --------------------------------------------------------------------------- #
# predictor
# --------------------------------------------------------------------------- #
def test_missing_checkpoint_raises_with_a_helpful_message(settings: Settings, tmp_path):
    settings.model_backend = "keras"
    settings.model_path = tmp_path / "absent.keras"
    with pytest.raises(ModelUnavailableError, match="not found"):
        FallPredictor(settings)


def test_create_predictor_falls_back_to_mock_when_the_checkpoint_is_missing(settings: Settings):
    settings.model_backend = "keras"
    settings.model_path = "/nonexistent/fallguard.keras"
    predictor = create_predictor(settings)
    assert predictor.backend == "mock"
    assert predictor.is_loaded
    assert "not a trained model" in predictor.info().detail


def test_create_predictor_can_refuse_to_fall_back(settings: Settings):
    settings.model_backend = "keras"
    settings.model_path = "/nonexistent/fallguard.keras"
    with pytest.raises(ModelUnavailableError):
        create_predictor(settings, allow_missing=False)


def test_mock_predictor_reports_its_own_shape(settings: Settings):
    settings.sequence_length = 12
    predictor = create_predictor(settings)
    assert predictor.sequence_length == 12
    assert predictor.feature_dim == FEATURE_DIM
    info = predictor.info().to_dict()
    assert info["backend"] == "mock"
    assert info["loaded"] is True
    assert info["path"] is None


def test_predict_proba_validates_the_window_shape(settings: Settings):
    predictor = create_predictor(settings)
    bad = np.zeros((settings.sequence_length, FEATURE_DIM + 3), dtype=np.float32)
    assert predictor.predict_proba(bad) == 0.0, "a shape mismatch must not raise"
    assert np.isfinite(predictor.predict_proba(np.zeros((1, 1, 1), dtype=np.float32)))


def test_predict_proba_handles_a_batched_window(settings: Settings):
    predictor = create_predictor(settings)
    window = np.zeros((settings.sequence_length, FEATURE_DIM), dtype=np.float32)
    single = predictor.predict_proba(window)
    reference = float(np.asarray(predictor.model.predict_proba(window[None, ...])).reshape(-1)[0])
    assert single == pytest.approx(reference, abs=1e-6)
    # A 3-D batch must be scored exactly like the equivalent 2-D window.
    batched = predictor.predict_proba(np.stack([window, window]))
    assert isinstance(batched, float)
    assert batched == pytest.approx(reference, abs=1e-6)


def test_predict_applies_the_threshold(settings: Settings):
    predictor = create_predictor(settings)
    settings.confidence_threshold = 0.5
    window = np.zeros((settings.sequence_length, FEATURE_DIM), dtype=np.float32)
    label, probability = predictor.predict(window, threshold=0.5)
    assert label == ("FALL" if probability >= 0.5 else "NORMAL")


def test_model_errors_degrade_to_zero_instead_of_raising(settings: Settings):
    class Exploding:
        def predict(self, batch, verbose=0):
            raise RuntimeError("kernel failure")

    predictor = FallPredictor(settings, model=Exploding(), backend="injected")
    assert predictor.predict_proba(np.zeros((1, 1, 1), dtype=np.float32)) == 0.0


def test_non_finite_model_output_is_neutralised(settings: Settings):
    class Nonsense:
        def predict(self, batch, verbose=0):
            return np.array([[float("nan")]])

    predictor = FallPredictor(settings, model=Nonsense(), backend="injected")
    window = np.zeros((settings.sequence_length, FEATURE_DIM), dtype=np.float32)
    assert predictor.predict_proba(window) == 0.0


def test_missing_metadata_sidecar_is_tolerated(settings: Settings, tmp_path):
    settings.model_backend = "keras"
    settings.model_path = tmp_path / "model.keras"
    settings.model_metadata_path = tmp_path / "absent.meta.json"
    model = build_lstm(sequence_length=settings.sequence_length, feature_dim=FEATURE_DIM,
                       units=8, settings=settings)
    model.save(settings.model_path)
    predictor = FallPredictor(settings)
    assert predictor.is_loaded
    assert predictor.metadata == {}


# --------------------------------------------------------------------------- #
# serving contract
# --------------------------------------------------------------------------- #
@pytest.mark.model
def test_sequence_length_mismatch_is_rejected(settings: Settings, tmp_path):
    settings.model_backend = "keras"
    settings.model_path = tmp_path / "model.keras"
    settings.model_metadata_path = tmp_path / "model.meta.json"
    model = build_lstm(sequence_length=settings.sequence_length + 5, feature_dim=FEATURE_DIM,
                       units=8, settings=settings)
    model.save(settings.model_path)
    settings.model_metadata_path.write_text(
        json.dumps({"sequence_length": settings.sequence_length + 5}), encoding="utf-8"
    )
    with pytest.raises(ModelContractError, match="SEQUENCE_LENGTH"):
        FallPredictor(settings)


@pytest.mark.model
def test_feature_spec_version_mismatch_is_rejected(settings: Settings, tmp_path):
    settings.model_backend = "keras"
    settings.model_path = tmp_path / "model.keras"
    settings.model_metadata_path = tmp_path / "model.meta.json"
    model = build_lstm(sequence_length=settings.sequence_length, feature_dim=FEATURE_DIM,
                       units=8, settings=settings)
    model.save(settings.model_path)
    settings.model_metadata_path.write_text(
        json.dumps(
            {
                "sequence_length": settings.sequence_length,
                "feature_dim": FEATURE_DIM,
                "feature_spec_version": "0.0.1-ancient",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ModelContractError, match="feature spec"):
        FallPredictor(settings)


@pytest.mark.model
def test_matching_contract_is_accepted(settings: Settings, tmp_path):
    settings.model_backend = "keras"
    settings.model_path = tmp_path / "model.keras"
    settings.model_metadata_path = tmp_path / "model.meta.json"
    model = build_lstm(sequence_length=settings.sequence_length, feature_dim=FEATURE_DIM,
                       units=8, settings=settings)
    model.save(settings.model_path)
    settings.model_metadata_path.write_text(
        json.dumps(
            {
                "sequence_length": settings.sequence_length,
                "feature_dim": FEATURE_DIM,
                "feature_spec_version": FEATURE_SPEC_VERSION,
            }
        ),
        encoding="utf-8",
    )
    predictor = FallPredictor(settings)
    assert predictor.sequence_length == settings.sequence_length
    assert predictor.feature_dim == FEATURE_DIM


def test_corrupt_metadata_is_tolerated(settings: Settings, tmp_path):
    settings.model_backend = "keras"
    settings.model_path = tmp_path / "model.keras"
    settings.model_metadata_path = tmp_path / "model.meta.json"
    model = build_lstm(sequence_length=settings.sequence_length, feature_dim=FEATURE_DIM,
                       units=8, settings=settings)
    model.save(settings.model_path)
    settings.model_metadata_path.write_text("{not json", encoding="utf-8")
    predictor = FallPredictor(settings)
    assert predictor.is_loaded


# --------------------------------------------------------------------------- #
# mock model
# --------------------------------------------------------------------------- #
def _kine_vector(lean_deg: float, hip_height: float) -> np.ndarray:
    from ai.preprocessing.pose_preprocessor import (
        KINEMATIC_FEATURE_COUNT,
        KINEMATIC_FEATURE_NAMES,
        KINEMATIC_FEATURE_OFFSET,
    )

    vector = np.zeros(FEATURE_DIM, dtype=np.float32)
    vector[PRESENT_FEATURE_INDEX] = 1.0
    names = {name: index for index, name in enumerate(KINEMATIC_FEATURE_NAMES)}
    vector[KINEMATIC_FEATURE_OFFSET + names["torso_lean_deg"]] = lean_deg
    vector[KINEMATIC_FEATURE_OFFSET + names["hip_height_above_ankles"]] = hip_height
    del KINEMATIC_FEATURE_COUNT
    return vector


def test_mock_model_is_deterministic():
    model = MockFallModel(sequence_length=4)
    window = np.stack([_kine_vector(80.0, 0.05)] * 4)
    assert model.score_window(window) == model.score_window(window)


def test_mock_model_scores_a_lying_pose_highly():
    model = MockFallModel(sequence_length=4)
    lying = np.stack([_kine_vector(85.0, 0.05)] * 4)
    standing = np.stack([_kine_vector(2.0, 1.6)] * 4)
    assert model.score_window(lying) > 0.8
    assert model.score_window(standing) == 0.0


def test_mock_model_requires_both_cues():
    """A low hip with a vertical torso is a bend, not a fall."""
    model = MockFallModel(sequence_length=4)
    bend = np.stack([_kine_vector(5.0, 0.2)] * 4)
    lean_only = np.stack([_kine_vector(85.0, 1.5)] * 4)
    assert model.score_window(bend) == 0.0
    assert model.score_window(lean_only) == 0.0


def test_mock_model_ignores_gap_frames():
    model = MockFallModel(sequence_length=4)
    window = np.stack([np.zeros(FEATURE_DIM, dtype=np.float32)] * 4)
    assert model.score_window(window) == 0.0


def test_mock_model_validates_its_input():
    model = MockFallModel(sequence_length=4, feature_dim=FEATURE_DIM)
    with pytest.raises(ValueError):
        model.score_window(np.zeros((4, FEATURE_DIM + 1), dtype=np.float32))
    with pytest.raises(ValueError):
        MockFallModel(sequence_length=1)


def test_mock_model_advertises_no_learned_parameters():
    """It must not pretend to be a trained model."""
    model = MockFallModel(sequence_length=4)
    assert model.count_params() == 0
    assert "NOT a trained model" in model.summary()


def test_mock_model_keras_compatible_api():
    model = MockFallModel(sequence_length=4, feature_dim=FEATURE_DIM)
    window = np.stack([_kine_vector(80.0, 0.05)] * 4)
    assert model.predict(window).shape == (1, 1)
    assert model.predict_proba(np.stack([window, window])).shape == (2, 1)
    assert model(np.stack([window, window])).shape == (2, 1)
    assert model.input_shape[0][1:] == (4, FEATURE_DIM)
