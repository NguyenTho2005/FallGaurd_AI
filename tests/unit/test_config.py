"""Unit tests for configuration loading, validation and defaults."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ai.config import ENV_PREFIX, PACKAGE_ROOT, PROJECT_ROOT, Settings, get_settings, reload_settings
from ai.preprocessing.pose_preprocessor import FEATURE_DIM


def test_defaults_match_the_documented_assumptions():
    settings = Settings()
    assert settings.sequence_length == 30
    assert settings.confidence_threshold == 0.80
    assert settings.min_fall_frames == 8
    assert settings.cooldown_seconds == 30.0
    assert settings.min_landmark_visibility == 0.5
    assert settings.rotation_normalization is True
    assert settings.max_missing_ratio == 0.5
    assert settings.camera_id == "camera_01"
    assert settings.store_raw_video is False
    assert settings.enable_monitor is False
    assert settings.telegram_dry_run is True


def test_paths_are_anchored_to_the_project():
    settings = Settings()
    assert settings.project_root == PROJECT_ROOT
    assert PACKAGE_ROOT in settings.pose_model_path.parents
    assert settings.models_dir == PROJECT_ROOT / "models"
    assert settings.db_path == PROJECT_ROOT / "data" / "fallguard.db"


def test_telegram_is_not_configured_by_default():
    settings = Settings()
    assert settings.telegram_configured is False


def test_telegram_configured_when_both_values_are_present():
    settings = Settings(telegram_bot_token="abc", telegram_chat_id="123")
    assert settings.telegram_configured is True


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_telegram_values_become_none(blank: str):
    settings = Settings(telegram_bot_token=blank, telegram_chat_id=blank)
    assert settings.telegram_bot_token is None
    assert settings.telegram_chat_id is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("sequence_length", 1),
        ("sequence_length", 1000),
        ("confidence_threshold", 1.5),
        ("confidence_threshold", -0.1),
        ("min_fall_frames", 0),
        ("cooldown_seconds", -1.0),
        ("min_landmark_visibility", 2.0),
        ("max_missing_ratio", 1.5),
        ("batch_size", 0),
        ("epochs", 0),
        ("learning_rate", 0.0),
        ("environment", "productionn"),
        ("log_level", "CHATTY"),
        ("model_backend", "onnx"),
    ],
)
def test_out_of_range_values_are_rejected(field: str, value: object):
    with pytest.raises(ValidationError):
        Settings(**{field: value})


def test_environment_variables_override_defaults(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(f"{ENV_PREFIX}SEQUENCE_LENGTH", "45")
    monkeypatch.setenv(f"{ENV_PREFIX}CONFIDENCE_THRESHOLD", "0.55")
    monkeypatch.setenv(f"{ENV_PREFIX}MIN_FALL_FRAMES", "3")
    monkeypatch.setenv(f"{ENV_PREFIX}COOLDOWN_SECONDS", "12.5")
    monkeypatch.setenv(f"{ENV_PREFIX}CAMERA_ID", "kitchen_02")
    monkeypatch.setenv(f"{ENV_PREFIX}ROTATION_NORMALIZATION", "false")
    settings = Settings()
    assert settings.sequence_length == 45
    assert settings.confidence_threshold == pytest.approx(0.55)
    assert settings.min_fall_frames == 3
    assert settings.cooldown_seconds == pytest.approx(12.5)
    assert settings.camera_id == "kitchen_02"
    assert settings.rotation_normalization is False


def test_path_environment_variables_are_expanded(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(f"{ENV_PREFIX}MODEL_PATH", str(tmp_path / "custom.keras"))
    monkeypatch.setenv(f"{ENV_PREFIX}DB_PATH", str(tmp_path / "custom.db"))
    settings = Settings()
    assert settings.model_path == tmp_path / "custom.keras"
    assert settings.db_path == tmp_path / "custom.db"


def test_unprefixed_variables_are_ignored(monkeypatch):
    """A stray ``SEQUENCE_LENGTH`` in the shell must not silently change the model."""
    monkeypatch.setenv("SEQUENCE_LENGTH", "999")
    assert Settings().sequence_length == 30


def test_log_level_is_normalised_to_upper_case():
    assert Settings(log_level="debug").log_level == "DEBUG"
    assert Settings(log_level="warning").log_level == "WARNING"


def test_ensure_directories_creates_everything(settings: Settings, tmp_path: Path):
    settings.ensure_directories()
    for path in (
        settings.data_dir, settings.models_dir, settings.reports_dir, settings.samples_dir,
        settings.raw_dir, settings.processed_dir, settings.db_path.parent,
    ):
        assert path.is_dir(), f"{path} was not created"


def test_get_settings_is_a_singleton():
    assert get_settings() is get_settings()


def test_reload_settings_rebuilds_the_singleton(monkeypatch):
    monkeypatch.setenv(f"{ENV_PREFIX}CAMERA_ID", "first")
    first = reload_settings()
    assert first.camera_id == "first"
    monkeypatch.setenv(f"{ENV_PREFIX}CAMERA_ID", "second")
    second = reload_settings()
    assert second.camera_id == "second"
    assert second is not first


def test_split_fractions_must_leave_room_for_training():
    """Leave no room for training is a configuration error, caught at construction."""
    with pytest.raises(ValidationError, match="no windows would"):
        Settings(validation_split=0.6, test_split=0.5)
    with pytest.raises(ValidationError):
        Settings(validation_split=0.5, test_split=0.5)


def test_feature_dim_is_independent_of_configuration():
    """Feature width is a code-level contract; no setting may change it."""
    settings = Settings(sequence_length=10, min_landmark_visibility=0.9)
    from ai.preprocessing.pose_preprocessor import PosePreprocessor

    assert PosePreprocessor.from_settings(settings).feature_dim == FEATURE_DIM


def test_privacy_flag_defaults_to_storing_nothing():
    assert Settings().store_raw_video is False


def test_env_file_can_be_redirected(tmp_path: Path, monkeypatch):
    env_file = tmp_path / "custom.env"
    env_file.write_text("FALLGUARD_CAMERA_ID=from_file\n", encoding="utf-8")
    monkeypatch.setenv("FALLGUARD_ENV_FILE", str(env_file))
    assert Settings(_env_file=str(env_file)).camera_id == "from_file"


def test_extra_environment_variables_are_ignored(monkeypatch):
    monkeypatch.setenv(f"{ENV_PREFIX}TOTALLY_UNKNOWN", "1")
    assert Settings().sequence_length == 30


def test_settings_survives_a_stress_of_unusual_but_valid_values():
    settings = Settings(
        sequence_length=2,
        confidence_threshold=0.0,
        min_fall_frames=1,
        cooldown_seconds=0.0,
        max_missing_ratio=0.0,
        target_fps=1.0,
    )
    assert settings.sequence_length == 2
    assert settings.confidence_threshold == 0.0
