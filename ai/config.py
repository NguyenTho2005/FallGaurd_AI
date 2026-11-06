"""Central configuration for FallGuard AI.

Every tunable in the system is defined exactly once, here, and is overridable through
environment variables (optionally loaded from a ``.env`` file) using the ``FALLGUARD_``
prefix.  No other module is allowed to read an environment variable for a tunable
value: this is what keeps training-time and inference-time behaviour identical.

Example
-------
    FALLGUARD_SEQUENCE_LENGTH=45 FALLGUARD_CONFIDENCE_THRESHOLD=0.75 uvicorn backend.main:app
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

ENV_PREFIX = "FALLGUARD_"


class Settings(BaseSettings):
    """Application settings, populated from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=os.environ.get("FALLGUARD_ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- general ---------------------------------------------------------- #
    app_name: str = "FallGuard AI"
    version: str = "0.1.0"
    environment: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    log_file: str | None = None
    log_json: bool = False

    # -- filesystem ------------------------------------------------------- #
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    db_path: Path = PROJECT_ROOT / "data" / "fallguard.db"
    samples_dir: Path = PROJECT_ROOT / "data" / "samples"

    # -- pose estimation -------------------------------------------------- #
    pose_model_path: Path = PACKAGE_ROOT / "pose" / "mediapipe_models" / "pose_landmarker_lite.task"
    pose_num_poses: int = Field(default=1, ge=1, le=10)
    pose_min_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    pose_min_presence_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    pose_min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # -- preprocessing ---------------------------------------------------- #
    #: Landmarks below this visibility are treated as missing.
    min_landmark_visibility: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Rotate the body so the torso axis is vertical (cancels camera tilt).
    rotation_normalization: bool = True
    #: Zero-based indices of the body landmarks that must be valid to accept a frame.
    required_landmark_indices: tuple[int, ...] = (11, 12, 23, 24)

    # -- temporal sequence ------------------------------------------------ #
    sequence_length: int = Field(default=30, ge=2, le=600)
    #: Highest tolerated share of pose-less frames inside a ready window.
    max_missing_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Target processing rate for live sources; used to synthesise frame timestamps.
    target_fps: float = Field(default=30.0, gt=0.0)

    # -- model ------------------------------------------------------------ #
    model_backend: Literal["keras", "mock"] = "keras"
    model_path: Path = PROJECT_ROOT / "models" / "fallguard_lstm.keras"
    model_metadata_path: Path = PROJECT_ROOT / "models" / "fallguard_lstm.meta.json"
    lstm_units: int = Field(default=64, ge=8, le=512)
    bilstm_units: int = Field(default=64, ge=8, le=512)
    dropout_rate: float = Field(default=0.3, ge=0.0, lt=1.0)

    # -- fall event confirmation ------------------------------------------ #
    confidence_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    min_fall_frames: int = Field(default=8, ge=1, le=300)
    cooldown_seconds: float = Field(default=30.0, ge=0.0)

    # -- camera / source -------------------------------------------------- #
    camera_id: str = "camera_01"
    camera_source: str = "0"  # webcam index, RTSP URL, or a video file path
    frame_width: int = Field(default=640, ge=64, le=4096)
    frame_height: int = Field(default=480, ge=64, le=4096)

    # -- api -------------------------------------------------------------- #
    api_host: str = "0.0.0.0"  # noqa: S104 - container/host binding is intentional
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_title: str = "FallGuard AI API"
    #: Run the camera monitor thread inside the API process.
    enable_monitor: bool = False
    #: Never write frames or videos to disk. Enforced by the code paths, kept as
    #: an explicit, auditable flag.
    store_raw_video: bool = False

    # -- alerts ----------------------------------------------------------- #
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    telegram_enabled: bool = True
    telegram_api_base: str = "https://api.telegram.org"
    telegram_timeout_seconds: float = Field(default=10.0, gt=0.0)
    telegram_max_retries: int = Field(default=3, ge=0, le=10)
    #: With no token configured, alerts are rendered and logged but not sent.
    telegram_dry_run: bool = True

    # -- training --------------------------------------------------------- #
    batch_size: int = Field(default=32, ge=1, le=1024)
    epochs: int = Field(default=20, ge=1, le=1000)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    validation_split: float = Field(default=0.2, gt=0.0, lt=1.0)
    test_split: float = Field(default=0.2, gt=0.0, lt=1.0)
    early_stopping_patience: int = Field(default=5, ge=1, le=100)
    random_seed: int = Field(default=20240917, ge=0)

    # -- validation ------------------------------------------------------- #
    @field_validator("db_path", "model_path", "model_metadata_path", "pose_model_path",
                     "data_dir", "models_dir", "reports_dir", "samples_dir", mode="after")
    @classmethod
    def _expand(cls, value: Path) -> Path:
        return Path(value).expanduser()

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}:
            raise ValueError(f"invalid log level: {value!r}")
        return level

    @field_validator("telegram_bot_token", "telegram_chat_id", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _check_split_fractions(self) -> "Settings":
        if self.test_split + self.validation_split >= 1.0:
            raise ValueError(
                "test_split + validation_split must be < 1.0, otherwise no windows would "
                f"be left for training (got {self.test_split} + {self.validation_split})"
            )
        return self

    # -- derived helpers -------------------------------------------------- #
    @property
    def telegram_configured(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def alerts_dir(self) -> Path:
        return self.reports_dir / "alerts"

    def ensure_directories(self) -> None:
        """Create the runtime directories the application writes into."""
        for path in (
            self.data_dir,
            self.models_dir,
            self.reports_dir,
            self.samples_dir,
            self.raw_dir,
            self.processed_dir,
            self.alerts_dir,
            self.db_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    settings = Settings()
    settings.ensure_directories()
    return settings


def reload_settings() -> Settings:
    """Clear the cache and rebuild settings (used by tests and the demo script)."""
    get_settings.cache_clear()
    return get_settings()
