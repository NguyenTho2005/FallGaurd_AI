"""Keras sequence models for fall detection.

Two architectures are provided, both consuming ``(batch, sequence_length, FEATURE_DIM)``
and emitting a single fall probability via a sigmoid head:

``build_lstm``
    A single unidirectional LSTM. Cheap, causal and the MVP baseline.
``build_bilstm``
    A bidirectional LSTM. Reads the whole window in both directions, which helps because
    the *aftermath* of a fall (lying still) is as informative as the descent, but costs
    roughly twice the parameters and cannot be streamed frame-by-frame.

Design choices
--------------
* ``return_sequences=False`` on the recurrent stack: the classifier only needs a window
  summary, and dropping the time axis before the dense head keeps the model small.
* ``BatchNormalization`` on the input is deliberate. Feature blocks have very different
  scales (normalised coordinates in [-2, 2], angles in degrees, ratios in [0, 5]) and an
  LSTM is sensitive to input scale.
* The output layer is a single unit with a sigmoid activation so the value is directly
  usable as a confidence by the event-confirmation logic.
"""

from __future__ import annotations

from typing import Literal

from ai.config import Settings, get_settings
from ai.logging_config import get_logger
from ai.preprocessing.pose_preprocessor import FEATURE_DIM

logger = get_logger(__name__)

Architecture = Literal["lstm", "bilstm"]


def _require_keras():
    """Import Keras lazily so that non-ML parts of the project stay cheap to import."""
    try:
        import keras
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "TensorFlow/Keras is required to build or train models. "
            "Install it with: pip install -r requirements.txt"
        ) from exc
    return keras


def build_lstm(
    *,
    sequence_length: int | None = None,
    feature_dim: int = FEATURE_DIM,
    units: int | None = None,
    dropout_rate: float | None = None,
    settings: Settings | None = None,
) -> "object":
    """Build the baseline single-layer LSTM fall classifier."""
    settings = settings or get_settings()
    keras = _require_keras()
    sequence_length = int(sequence_length or settings.sequence_length)
    units = int(units or settings.lstm_units)
    dropout_rate = float(dropout_rate if dropout_rate is not None else settings.dropout_rate)

    keras_layers = keras.layers
    inputs = keras_layers.Input(shape=(sequence_length, feature_dim), name="sequence")
    x = keras_layers.BatchNormalization(name="input_norm")(inputs)
    x = keras_layers.LSTM(units, return_sequences=False, name="lstm")(x)
    x = keras_layers.Dropout(dropout_rate, name="dropout")(x)
    x = keras_layers.Dense(32, activation="relu", name="dense_1")(x)
    x = keras_layers.Dropout(dropout_rate * 0.5, name="dropout_1")(x)
    outputs = keras_layers.Dense(1, activation="sigmoid", name="fall_probability")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="fallguard_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=settings.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", "precision", "recall"],
    )
    logger.info(
        "Built LSTM model: sequence_length=%d feature_dim=%d units=%d params=%d",
        sequence_length,
        feature_dim,
        units,
        model.count_params(),
    )
    return model


def build_bilstm(
    *,
    sequence_length: int | None = None,
    feature_dim: int = FEATURE_DIM,
    units: int | None = None,
    dropout_rate: float | None = None,
    settings: Settings | None = None,
) -> "object":
    """Build the bidirectional LSTM variant (optional per the specification)."""
    settings = settings or get_settings()
    keras = _require_keras()
    sequence_length = int(sequence_length or settings.sequence_length)
    units = int(units or settings.bilstm_units)
    dropout_rate = float(dropout_rate if dropout_rate is not None else settings.dropout_rate)

    keras_layers = keras.layers
    inputs = keras_layers.Input(shape=(sequence_length, feature_dim), name="sequence")
    x = keras_layers.BatchNormalization(name="input_norm")(inputs)
    x = keras_layers.Bidirectional(
        keras_layers.LSTM(units, return_sequences=False), name="bilstm"
    )(x)
    x = keras_layers.Dropout(dropout_rate, name="dropout")(x)
    x = keras_layers.Dense(32, activation="relu", name="dense_1")(x)
    x = keras_layers.Dropout(dropout_rate * 0.5, name="dropout_1")(x)
    outputs = keras_layers.Dense(1, activation="sigmoid", name="fall_probability")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="fallguard_bilstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=settings.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", "precision", "recall"],
    )
    logger.info(
        "Built BiLSTM model: sequence_length=%d feature_dim=%d units=%d params=%d",
        sequence_length,
        feature_dim,
        units,
        model.count_params(),
    )
    return model


MODEL_BUILDERS = {"lstm": build_lstm, "bilstm": build_bilstm}


def build_model(
    architecture: Architecture = "lstm",
    *,
    sequence_length: int | None = None,
    feature_dim: int = FEATURE_DIM,
    settings: Settings | None = None,
    **kwargs: object,
) -> "object":
    """Build a model by architecture name (``lstm`` or ``bilstm``)."""
    key = architecture.strip().lower()
    if key not in MODEL_BUILDERS:
        raise ValueError(
            f"unknown architecture {architecture!r}; expected one of {sorted(MODEL_BUILDERS)}"
        )
    return MODEL_BUILDERS[key](
        sequence_length=sequence_length, feature_dim=feature_dim, settings=settings, **kwargs
    )


def model_input_spec(model: "object") -> tuple[int, int]:
    """Return ``(sequence_length, feature_dim)`` from a built or loaded Keras model."""
    shape = getattr(getattr(model, "input_shape", None), "__getitem__", None)
    input_shape = model.input_shape if shape is not None else None
    if isinstance(input_shape, list):  # pragma: no cover - multi-input models
        input_shape = input_shape[0]
    if input_shape is None or len(input_shape) != 3:  # pragma: no cover - defensive
        raise ValueError(f"cannot determine input shape from model: {input_shape!r}")
    return int(input_shape[1]), int(input_shape[2])
