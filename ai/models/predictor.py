"""Checkpoint loading, model/serving-contract validation and single-window inference.

:class:`FallPredictor` is the only component allowed to turn a feature window into a fall
probability. It owns three responsibilities:

1. **Loading** a Keras checkpoint (or substituting :class:`MockFallModel`), with clear
   errors when the file is missing or unreadable.
2. **Contract validation** against a JSON metadata sidecar. A checkpoint trained with a
   different feature layout or a different sequence length is *rejected*, because
   silently scoring with mismatched inputs produces meaningless probabilities.
3. **Inference**, with the TensorFlow logger silenced and exceptions converted into a
   logged error plus a neutral 0.0 -- a model failure must not crash the camera loop.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ai.config import Settings, get_settings
from ai.logging_config import get_logger
from ai.preprocessing.pose_preprocessor import FEATURE_DIM, FEATURE_SPEC_VERSION

logger = get_logger(__name__)

# Silence the TensorFlow/oneDNN banner that otherwise floods a fresh console.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


class ModelUnavailableError(RuntimeError):
    """Raised when no usable model can be loaded."""


class ModelContractError(RuntimeError):
    """Raised when a checkpoint's feature contract does not match the runtime."""


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Everything a caller needs to know about the active model."""

    backend: str
    loaded: bool
    path: Path | None
    sequence_length: int
    feature_dim: int
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "loaded": self.loaded,
            "path": str(self.path) if self.path else None,
            "sequence_length": self.sequence_length,
            "feature_dim": self.feature_dim,
            "detail": self.detail,
        }


class FallPredictor:
    """Score a feature window and report the model's serving contract."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model: Any | None = None,
        backend: str | None = None,
        model_path: Path | str | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._backend = (backend or self._settings.model_backend).lower()
        self._model_path = Path(model_path) if model_path else Path(self._settings.model_path)
        self._load_error: str | None = None
        self._model: Any | None = None
        self._sequence_length = int(self._settings.sequence_length)
        self._feature_dim = int(FEATURE_DIM)
        self._metadata: dict[str, Any] = {}

        if model is not None:
            self._model = model
            self._backend = self._backend if self._backend != "keras" else "injected"
        elif self._backend == "mock":
            self._install_mock()
        else:
            self._load_keras()

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def _install_mock(self) -> None:
        from ai.models.mock_model import MockFallModel

        self._model = MockFallModel(
            sequence_length=self._sequence_length, feature_dim=FEATURE_DIM
        )
        logger.warning(
            "MODEL_BACKEND=mock: using the deterministic kinematic heuristic. "
            "This is NOT a trained model; results are for pipeline testing only."
        )

    def _load_keras(self) -> None:
        if not self._model_path.is_file():
            self._load_error = (
                f"Model checkpoint not found at '{self._model_path}'. "
                "Train one with: python scripts/train.py --synthetic"
            )
            logger.error(self._load_error)
            raise ModelUnavailableError(self._load_error)
        try:
            import keras
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ModelUnavailableError(
                "TensorFlow/Keras is not installed; install requirements.txt"
            ) from exc
        try:
            logger.info("Loading model checkpoint from %s", self._model_path)
            self._model = keras.models.load_model(self._model_path)
        except Exception as exc:
            self._load_error = f"Failed to load model from '{self._model_path}': {exc}"
            logger.error(self._load_error)
            raise ModelUnavailableError(self._load_error) from exc
        logger.info("Model loaded: %s", self._model_path)
        self._read_metadata()
        self._validate_contract()

    def _read_metadata(self) -> None:
        metadata_path = Path(self._settings.model_metadata_path)
        if not metadata_path.is_file():
            logger.warning(
                "No metadata sidecar at '%s'. Falling back to the architecture's own "
                "input shape; a mismatch with the serving config will be detected from it.",
                metadata_path,
            )
            return
        try:
            self._metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            logger.info("Loaded model metadata from %s", metadata_path)
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read model metadata from %s", metadata_path, exc_info=True)

    def _validate_contract(self) -> None:
        """Cross-check the checkpoint against the configured serving contract."""
        from ai.models.lstm import model_input_spec

        arch_sequence, arch_feature = model_input_spec(self._model)
        self._sequence_length = arch_sequence
        self._feature_dim = arch_feature

        meta_feature = self._metadata.get("feature_dim")
        meta_sequence = self._metadata.get("sequence_length")
        meta_spec_version = self._metadata.get("feature_spec_version")

        if meta_spec_version and meta_spec_version != FEATURE_SPEC_VERSION:
            raise ModelContractError(
                f"Checkpoint was trained with feature spec v{meta_spec_version} but this build "
                f"provides v{FEATURE_SPEC_VERSION}. Re-train the model or install the matching "
                "code version."
            )
        if meta_feature is not None and int(meta_feature) != self._feature_dim:
            raise ModelContractError(
                f"Checkpoint feature_dim={meta_feature} does not match the architecture "
                f"input width {self._feature_dim}."
            )
        if meta_sequence is not None and int(meta_sequence) != self._sequence_length:
            raise ModelContractError(
                f"Checkpoint sequence_length={meta_sequence} does not match the architecture "
                f"input length {self._sequence_length}."
            )
        if self._sequence_length != int(self._settings.sequence_length):
            raise ModelContractError(
                f"FALLGUARD_SEQUENCE_LENGTH={self._settings.sequence_length} but the checkpoint "
                f"expects {self._sequence_length} frames. Align the two before serving."
            )
        if self._feature_dim != FEATURE_DIM:
            raise ModelContractError(
                f"Checkpoint expects {self._feature_dim} input features but this build produces "
                f"{FEATURE_DIM}. Re-train the model."
            )
        logger.info(
            "Model contract verified: sequence_length=%d feature_dim=%d",
            self._sequence_length,
            self._feature_dim,
        )

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self) -> Any:
        if self._model is None:
            raise ModelUnavailableError(self._load_error or "model is not loaded")
        return self._model

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def info(self) -> ModelInfo:
        detail = ""
        if self._backend == "mock":
            detail = "deterministic kinematic heuristic, not a trained model"
        elif self._load_error:
            detail = self._load_error
        return ModelInfo(
            backend=self._backend,
            loaded=self.is_loaded,
            path=self._model_path if self._backend not in {"mock", "injected"} else None,
            sequence_length=self._sequence_length,
            feature_dim=self._feature_dim,
            detail=detail,
        )

    def predict_proba(self, window: np.ndarray) -> float:
        """Return the fall probability for one ``(sequence_length, feature_dim)`` window.

        Returns ``0.0`` on any inference failure, after logging the error, so a broken
        model degrades to "no fall" rather than taking the process down.
        """
        array = np.asarray(window, dtype=np.float32)
        if array.ndim == 2:
            batch = array[None, ...]
        elif array.ndim == 3:
            batch = array
        else:
            logger.error("predict_proba got a %d-D array; expected 2-D or 3-D", array.ndim)
            return 0.0

        if batch.shape[1:] != (self._sequence_length, self._feature_dim):
            logger.error(
                "predict_proba received a window of shape %s but the model expects %s",
                batch.shape[1:],
                (self._sequence_length, self._feature_dim),
            )
            return 0.0

        try:
            prediction = self.model.predict(batch, verbose=0)
        except Exception:
            logger.error("Model inference failed; treating the window as no-fall", exc_info=True)
            return 0.0

        probability = float(np.asarray(prediction).reshape(-1)[0])
        if not np.isfinite(probability):
            logger.error("Model returned a non-finite probability (%s); using 0.0", probability)
            return 0.0
        return float(np.clip(probability, 0.0, 1.0))

    def predict(self, window: np.ndarray, threshold: float | None = None) -> tuple[str, float]:
        """Return ``("FALL" | "NORMAL", probability)`` for one window."""
        probability = self.predict_proba(window)
        limit = self._settings.confidence_threshold if threshold is None else float(threshold)
        label = "FALL" if probability >= limit else "NORMAL"
        return label, probability


def create_predictor(
    settings: Settings | None = None,
    *,
    allow_missing: bool = True,
    fallback_backend: str = "mock",
) -> FallPredictor:
    """Build a predictor, degrading to the mock backend instead of raising.

    Used by the API and the demo so a missing checkpoint degrades gracefully. The
    degradation is logged loudly and surfaced through ``/status``.
    """
    settings = settings or get_settings()
    try:
        return FallPredictor(settings)
    except (ModelUnavailableError, ModelContractError) as exc:
        if not allow_missing:
            raise
        logger.warning(
            "Falling back to the %s model backend because the configured backend failed to "
            "load: %s",
            fallback_backend,
            exc,
        )
        return FallPredictor(settings, backend=fallback_backend)
