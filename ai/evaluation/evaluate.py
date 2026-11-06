"""Model evaluation: accuracy, precision, recall, F1 and a confusion matrix.

Two rules govern this module:

1. **Metrics are only ever computed from arrays that were actually predicted.** There is
   no default value, no placeholder and no synthetic "expected" number. If there is
   nothing to evaluate, the function reports that fact and writes nothing.
2. **Every artefact is labelled with its provenance.** A report produced from synthetic
   data carries ``dataset_kind="synthetic"`` and a ``SYNTHETIC`` prefix, so a smoke-test
   number can never be quoted as a benchmark.

Outputs (in ``reports/``):

* ``metrics.json`` -- accuracy, precision, recall, F1, support, confusion matrix
* ``classification_report.txt`` -- the scikit-learn per-class report
* ``confusion_matrix.png`` -- the rendered matrix
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ai.config import Settings, get_settings
from ai.logging_config import get_logger

logger = get_logger(__name__)

CLASS_NAMES = ("non_fall", "fall")

#: Marks a report as derived from synthetic/rendered data rather than real recordings.
SYNTHETIC_MARKER = "SYNTHETIC"


@dataclass(slots=True)
class EvaluationReport:
    """Metrics for one evaluation run."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    support: list[int]
    n_samples: int
    threshold: float
    dataset_kind: str = "real"
    positive_rate: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metrics"] = {
            "accuracy": round(self.accuracy, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
        }
        return payload

    @property
    def is_synthetic(self) -> bool:
        return self.dataset_kind.lower().startswith("synthetic")


class NoSamplesError(ValueError):
    """Raised when an evaluation is requested with nothing to evaluate."""


def compute_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    threshold: float = 0.5,
    dataset_kind: str = "real",
    notes: Sequence[str] = (),
) -> EvaluationReport:
    """Compute the full metric set for one set of predictions.

    Raises
    ------
    NoSamplesError
        If either array is empty. Returning zeros here would be indistinguishable from a
        genuinely terrible model, which is exactly the kind of fabricated result this
        project must not produce.
    """
    from sklearn.metrics import confusion_matrix

    true_array = np.asarray(y_true).reshape(-1)
    pred_array = np.asarray(y_pred).reshape(-1)
    if true_array.size == 0 or pred_array.size == 0:
        raise NoSamplesError(
            "refusing to compute metrics on an empty set: there is nothing to evaluate"
        )
    if true_array.size != pred_array.size:
        raise ValueError(
            f"y_true has {true_array.size} elements but y_pred has {pred_array.size}"
        )
    labels = np.unique(np.concatenate([true_array, pred_array, np.array([0, 1])]))
    labels = np.array([0, 1]) if set(labels.tolist()) <= {0, 1} else labels

    matrix = confusion_matrix(true_array, pred_array, labels=labels)
    accuracy = float((true_array == pred_array).mean())

    true_positive = int(matrix[1, 1]) if matrix.shape == (2, 2) else 0
    false_positive = int(matrix[0, 1]) if matrix.shape == (2, 2) else 0
    false_negative = int(matrix[1, 0]) if matrix.shape == (2, 2) else 0

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )

    report = EvaluationReport(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=matrix.astype(int).tolist(),
        support=[int(matrix[0].sum()), int(matrix[1].sum())] if matrix.shape == (2, 2) else matrix.sum(axis=1).tolist(),
        n_samples=int(true_array.size),
        threshold=float(threshold),
        dataset_kind=dataset_kind,
        positive_rate=float(true_array.mean()),
        notes=list(notes),
    )
    if report.is_synthetic:
        report.notes.insert(
            0,
            "Metrics computed on SYNTHETIC rendered data. They describe the software "
            "pipeline only and are NOT a benchmark of fall-detection performance.",
        )
    return report


def classification_report_text(
    y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray
) -> str:
    """The scikit-learn per-class text report."""
    from sklearn.metrics import classification_report

    return classification_report(
        np.asarray(y_true).reshape(-1),
        np.asarray(y_pred).reshape(-1),
        labels=[0, 1],
        target_names=list(CLASS_NAMES),
        zero_division=0,
        digits=4,
    )


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    output_path: Path,
    *,
    title: str = "FallGuard AI -- confusion matrix",
    notes: Sequence[str] = (),
) -> Path:
    """Render a confusion matrix to PNG.

    Uses the ``Agg`` backend so it works on a headless server.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    array = np.asarray(matrix, dtype=int)
    figure, axes = plt.subplots(figsize=(5.2, 4.4))
    image = axes.imshow(array, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    axes.set_xticks(range(array.shape[1]), labels=list(CLASS_NAMES))
    axes.set_yticks(range(array.shape[0]), labels=list(CLASS_NAMES))
    axes.set_xlabel("Predicted")
    axes.set_ylabel("True")
    axes.set_title(title, fontsize=10)
    threshold = array.max() / 2.0 if array.size else 0.5
    for row in range(array.shape[0]):
        for column in range(array.shape[1]):
            axes.text(
                column, row, str(array[row, column]),
                ha="center", va="center",
                color="white" if array[row, column] > threshold else "black",
            )
    caption = "\n".join(notes)
    if caption:
        figure.text(0.5, -0.02, caption, ha="center", va="top", fontsize=7, wrap=True)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return output_path


def write_report(
    report: EvaluationReport,
    output_dir: str | Path | None = None,
    *,
    settings: Settings | None = None,
    stem: str | None = None,
    y_true: Sequence[int] | np.ndarray | None = None,
    y_pred: Sequence[int] | np.ndarray | None = None,
) -> dict[str, Path]:
    """Write ``metrics.json``, ``classification_report.txt`` and ``confusion_matrix.png``.

    The file stem is prefixed with ``SYNTHETIC_`` when the report came from rendered data.
    Pass ``y_true``/``y_pred`` to also emit the scikit-learn per-class text report.
    """
    settings = settings or get_settings()
    directory = Path(output_dir) if output_dir is not None else settings.reports_dir
    directory.mkdir(parents=True, exist_ok=True)
    base = stem or ("SYNTHETIC_evaluation" if report.is_synthetic else "evaluation")

    metrics_path = directory / f"{base}_metrics.json"
    report_path = directory / f"{base}_classification_report.txt"
    matrix_path = directory / f"{base}_confusion_matrix.png"

    metrics_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    lines = [
        f"FallGuard AI -- {base}",
        f"dataset_kind : {report.dataset_kind}",
        f"samples      : {report.n_samples}",
        f"positive rate: {report.positive_rate:.4f}",
        f"threshold    : {report.threshold}",
        "",
        f"accuracy  : {report.accuracy:.4f}",
        f"precision : {report.precision:.4f}   (positive class = 'fall')",
        f"recall    : {report.recall:.4f}   (positive class = 'fall')",
        f"f1        : {report.f1:.4f}   (positive class = 'fall')",
        "",
        "confusion matrix (rows = true, columns = predicted, order = "
        f"{list(CLASS_NAMES)}):",
    ]
    lines.extend(
        "  " + "  ".join(f"{value:>6d}" for value in row) for row in report.confusion_matrix
    )
    if y_true is not None and y_pred is not None:
        lines += ["", "classification report:", "", classification_report_text(y_true, y_pred)]
    if report.notes:
        lines += ["", "notes:"] + [f"  - {note}" for note in report.notes]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    title = "FallGuard AI -- confusion matrix"
    notes = list(report.notes)
    if report.is_synthetic:
        title = f"[{SYNTHETIC_MARKER}] {title}"
    plot_confusion_matrix(report.confusion_matrix, matrix_path, title=title, notes=notes)

    written = {"metrics": metrics_path, "report": report_path, "confusion_matrix": matrix_path}
    logger.info("Evaluation report written to %s", directory)
    for name, path in written.items():
        logger.debug("  %-16s %s", name, path)
    return written


def evaluate_model(
    model: Any,
    x_test: np.ndarray,
    y_true: np.ndarray,
    *,
    output_dir: str | Path | None = None,
    threshold: float | None = None,
    settings: Settings | None = None,
    dataset_kind: str = "real",
    stem: str | None = None,
    notes: Sequence[str] = (),
) -> tuple[EvaluationReport, dict[str, Path]]:
    """Predict with ``model`` and write a full report. Returns the report and the paths."""
    settings = settings or get_settings()
    if x_test.shape[0] == 0:
        raise NoSamplesError("x_test is empty: refusing to report metrics for no samples")
    limit = float(threshold if threshold is not None else settings.confidence_threshold)
    probabilities = np.asarray(model.predict(x_test, verbose=0)).reshape(-1)
    predictions = (probabilities >= limit).astype(int)
    report = compute_metrics(
        y_true, predictions, threshold=limit, dataset_kind=dataset_kind, notes=notes
    )
    paths = write_report(
        report, output_dir, settings=settings, stem=stem, y_true=y_true, y_pred=predictions
    )
    return report, paths
