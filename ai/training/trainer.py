"""Training loop, checkpointing and metadata sidecar.

    dataset -> preprocessing -> group split -> model -> training -> checkpoint -> evaluation

The trainer writes two files next to each other:

``<name>.keras``
    The Keras weights.
``<name>.meta.json``
    The serving contract: feature spec version, feature dimension, sequence length,
    architecture, class balance and the metrics actually measured on the held-out split.

The sidecar is what lets :class:`~ai.models.predictor.FallPredictor` refuse to serve a
checkpoint whose input layout does not match the running code. Without it, a stale
checkpoint would silently produce meaningless probabilities.
"""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ai.config import Settings, get_settings
from ai.logging_config import get_logger
from ai.models.lstm import Architecture, build_model, model_input_spec
from ai.preprocessing.pose_preprocessor import (
    FEATURE_SPEC_VERSION,
    PosePreprocessor,
)
from ai.training.dataset import WindowedDataset, class_weights, split_dataset

logger = get_logger(__name__)


@dataclass(slots=True)
class TrainingResult:
    """Outcome of a training run."""

    model: Any
    model_path: Path | None
    metadata_path: Path | None
    history: dict[str, list[float]]
    dataset_summary: dict[str, Any]
    splits: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] | None = None
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path) if self.model_path else None,
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
            "dataset": self.dataset_summary,
            "splits": self.splits,
            "evaluation": self.evaluation,
            "final_metrics": {
                key: (value[-1] if value else None) for key, value in self.history.items()
            },
            "messages": self.messages,
        }


class InsufficientDataError(RuntimeError):
    """Raised when there is not enough data to train (or to split) at all."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def train_model(
    dataset: WindowedDataset,
    *,
    architecture: Architecture = "lstm",
    settings: Settings | None = None,
    model_path: str | Path | None = None,
    evaluate: bool = True,
    epochs: int | None = None,
    verbose: int = 1,
) -> TrainingResult:
    """Train a fall detector on a windowed dataset and checkpoint it.

    Parameters
    ----------
    dataset:
        Windows produced by :func:`ai.training.dataset.build_windows`.
    architecture:
        ``lstm`` or ``bilstm``.
    model_path:
        Where to write the checkpoint. Defaults to ``settings.model_path``.
    evaluate:
        Score the held-out test split and record the metrics in the metadata sidecar.
    """
    settings = settings or get_settings()
    if len(dataset) == 0:
        raise InsufficientDataError(
            "the training set is empty. Load or generate data first "
            "(see scripts/generate_sample_data.py)."
        )

    splits = split_dataset(dataset, settings)
    train_set, val_set, test_set = splits["train"], splits["val"], splits["test"]
    logger.info(
        "Split sizes -- train=%d val=%d test=%d (sequence_length=%d, feature_dim=%d)",
        len(train_set), len(val_set), len(test_set), dataset.sequence_length, dataset.feature_dim,
    )
    if len(train_set) < 2:
        raise InsufficientDataError(
            f"only {len(train_set)} training windows across "
            f"{np.unique(train_set.groups).size} group(s). Add more recordings; a model "
            "trained on a single recording cannot generalise."
        )

    messages: list[str] = []
    if np.unique(train_set.y).size < 2:
        messages.append(
            "WARNING: the training split contains a single class. The model can only learn "
            "a constant output; metrics will be meaningless."
        )
        logger.warning(messages[-1])

    keras = _require_keras()
    np.random.seed(settings.random_seed)
    try:
        import random as _random

        _random.seed(settings.random_seed)
    except Exception:  # pragma: no cover - defensive
        logger.debug("Could not seed the stdlib RNG", exc_info=True)
    try:
        import tensorflow as tf

        tf.random.set_seed(settings.random_seed)
        tf.config.experimental.enable_op_determinism()
    except Exception:  # pragma: no cover - optional determinism
        logger.debug("TensorFlow determinism could not be enabled", exc_info=True)

    model = build_model(
        architecture,
        sequence_length=dataset.sequence_length,
        feature_dim=dataset.feature_dim,
        settings=settings,
    )

    has_validation = len(val_set) > 0
    callbacks = _build_callbacks(keras, settings, has_validation=has_validation)
    if not has_validation:
        logger.warning(
            "The validation split is empty; training without validation, early stopping or "
            "learning-rate scheduling."
        )
    weights = class_weights(train_set.y)
    history_object = model.fit(
        train_set.x,
        train_set.y,
        validation_data=(val_set.x, val_set.y) if len(val_set) else None,
        epochs=int(epochs or settings.epochs),
        batch_size=min(settings.batch_size, max(1, len(train_set))),
        class_weight=weights or None,
        callbacks=callbacks,
        verbose=verbose,
    )
    history = {key: [float(value) for value in values] for key, values in history_object.history.items()}

    evaluation: dict[str, Any] | None = None
    if evaluate and len(test_set):
        evaluation = _evaluate_split(model, test_set, settings)
        logger.info(
            "Test split -- accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f (n=%d)",
            evaluation["accuracy"], evaluation["precision"], evaluation["recall"],
            evaluation["f1"], evaluation["n_samples"],
        )
    elif evaluate:
        messages.append(
            "The test split is empty, so no test metrics were computed. Nothing is reported "
            "rather than reporting a placeholder."
        )
        logger.warning(messages[-1])

    destination = Path(model_path) if model_path else Path(settings.model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(destination)
    logger.info("Saved model checkpoint to %s", destination)

    metadata_path = destination.with_suffix(".meta.json")
    metadata = _build_metadata(
        model=model,
        dataset=dataset,
        splits=splits,
        settings=settings,
        architecture=architecture,
        history=history,
        evaluation=evaluation,
        messages=messages,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved serving metadata to %s", metadata_path)

    return TrainingResult(
        model=model,
        model_path=destination,
        metadata_path=metadata_path,
        history=history,
        dataset_summary=dataset.summary(),
        splits={name: item.summary() for name, item in splits.items()},
        evaluation=evaluation,
        messages=messages,
    )


def _build_callbacks(keras: Any, settings: Settings, *, has_validation: bool) -> list:
    """Build the training callbacks.

    ``ReduceLROnPlateau`` is only installed when a validation split exists: monitoring
    ``val_loss`` without validation data makes Keras warn on every epoch and never fire.
    """
    monitor = "val_loss" if has_validation else "loss"
    callbacks = []
    try:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=settings.early_stopping_patience,
                restore_best_weights=True,
            )
        )
    except Exception:  # pragma: no cover - defensive
        logger.warning("Could not install the EarlyStopping callback", exc_info=True)
    if has_validation:
        try:
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
            )
        except Exception:  # pragma: no cover - defensive
            logger.warning("Could not install the ReduceLROnPlateau callback", exc_info=True)
    return callbacks


def _evaluate_split(model: Any, test_set: WindowedDataset, settings: Settings) -> dict[str, Any]:
    from ai.evaluation.evaluate import compute_metrics

    probabilities = np.asarray(model.predict(test_set.x, verbose=0)).reshape(-1)
    threshold = float(settings.confidence_threshold)
    predictions = (probabilities >= threshold).astype(int)
    report = compute_metrics(test_set.y, predictions, threshold=threshold)
    return report.to_dict()


def _build_metadata(
    *,
    model: Any,
    dataset: WindowedDataset,
    splits: dict[str, WindowedDataset],
    settings: Settings,
    architecture: Architecture,
    history: dict[str, list[float]],
    evaluation: dict[str, Any] | None,
    messages: Sequence[str],
) -> dict[str, Any]:
    preprocessor = PosePreprocessor.from_settings(settings)
    sequence_length, feature_dim = model_input_spec(model)
    is_synthetic = any(
        str(group).lower().startswith("synthetic") for group in np.unique(dataset.groups)
    )
    return {
        "created_utc": _utc_now(),
        "app_version": settings.version,
        "architecture": architecture,
        "sequence_length": int(sequence_length),
        "feature_dim": int(feature_dim),
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "feature_spec": preprocessor.describe(),
        "parameters": int(model.count_params()),
        "dataset": {
            **dataset.summary(),
            "synthetic": bool(is_synthetic),
            "groups": sorted(str(group) for group in np.unique(dataset.groups))[:200],
        },
        "splits": {name: item.summary() for name, item in splits.items()},
        "hyperparameters": {
            "epochs": settings.epochs,
            "batch_size": settings.batch_size,
            "learning_rate": settings.learning_rate,
            "dropout_rate": settings.dropout_rate,
            "test_split": settings.test_split,
            "validation_split": settings.validation_split,
            "random_seed": settings.random_seed,
        },
        "serving": {
            "confidence_threshold": settings.confidence_threshold,
            "min_fall_frames": settings.min_fall_frames,
            "cooldown_seconds": settings.cooldown_seconds,
            "max_missing_ratio": settings.max_missing_ratio,
        },
        "history": history,
        "evaluation": evaluation,
        "messages": list(messages),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tensorflow": _tensorflow_version(),
        },
    }


def _tensorflow_version() -> str:
    try:
        import tensorflow as tf

        return str(tf.__version__)
    except Exception:  # pragma: no cover - defensive
        return "unavailable"


def _require_keras():
    try:
        import keras
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "TensorFlow/Keras is required to train. Install requirements.txt"
        ) from exc
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    return keras
