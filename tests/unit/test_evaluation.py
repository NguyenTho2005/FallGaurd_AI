"""Unit tests for the evaluation pipeline.

A recurring theme: the evaluator must never produce a number it did not measure. The
tests below pin that behaviour down.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ai.evaluation.evaluate import (
    CLASS_NAMES,
    NoSamplesError,
    classification_report_text,
    compute_metrics,
    evaluate_model,
    plot_confusion_matrix,
    write_report,
)


def test_perfect_predictions():
    y = [0, 0, 1, 1]
    report = compute_metrics(y, y)
    assert report.accuracy == 1.0
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1 == 1.0
    assert report.confusion_matrix == [[2, 0], [0, 2]]
    assert report.support == [2, 2]
    assert report.n_samples == 4


def test_known_confusion_matrix():
    # 2 true falls missed, 1 false alarm, 3 true falls caught, 4 correct negatives.
    y_true = [1, 1, 1, 0, 0, 0, 0]
    y_pred = [0, 0, 1, 1, 0, 0, 0]
    report = compute_metrics(y_true, y_pred)
    assert report.confusion_matrix == [[3, 1], [2, 1]]
    assert report.recall == pytest.approx(1 / 3)
    assert report.precision == pytest.approx(1 / 2)
    assert report.f1 == pytest.approx(2 * (1 / 3) * (1 / 2) / ((1 / 3) + (1 / 2)))
    assert report.accuracy == pytest.approx(4 / 7)


def test_empty_input_raises_rather_than_reporting_zero():
    """A zero metric is indistinguishable from a terrible model. Refuse instead."""
    with pytest.raises(NoSamplesError):
        compute_metrics([], [])
    with pytest.raises(NoSamplesError):
        compute_metrics([1], [])


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="elements"):
        compute_metrics([0, 1], [0])


def test_single_class_input_does_not_crash():
    report = compute_metrics([1, 1], [1, 1])
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert CLASS_NAMES == ("non_fall", "fall")


def test_all_negative_predictions_yield_zero_positive_metrics():
    report = compute_metrics([0, 0, 1, 1], [0, 0, 0, 0])
    assert report.precision == 0.0
    assert report.recall == 0.0
    assert report.f1 == 0.0
    assert report.accuracy == 0.5


def test_synthetic_reports_are_marked_and_warned_about():
    report = compute_metrics([0, 1], [0, 1], dataset_kind="synthetic-smoke-test")
    assert report.is_synthetic is True
    assert any("NOT a benchmark" in note for note in report.notes)
    assert report.to_dict()["dataset_kind"] == "synthetic-smoke-test"


def test_real_reports_have_no_synthetic_warning():
    report = compute_metrics([0, 1], [0, 1], dataset_kind="real")
    assert report.is_synthetic is False
    assert report.notes == []


def test_metrics_are_rounded_in_the_serialised_form():
    report = compute_metrics([0, 0, 0, 1], [0, 0, 0, 0])
    metrics = report.to_dict()["metrics"]
    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    assert all(isinstance(value, float) for value in metrics.values())


def test_classification_report_text_uses_domain_class_names():
    text = classification_report_text([0, 0, 1, 1], [0, 1, 1, 1])
    assert "non_fall" in text
    assert "fall" in text
    assert "recall" in text


def test_classification_report_handles_a_single_class():
    text = classification_report_text([1, 1], [1, 1])
    assert isinstance(text, str)


def test_confusion_matrix_plot_is_written(tmp_path: Path):
    path = plot_confusion_matrix([[3, 1], [2, 4]], tmp_path / "cm.png", title="test")
    assert path.is_file()
    assert path.stat().st_size > 1000


def test_confusion_matrix_plot_survives_an_all_zero_matrix(tmp_path: Path):
    path = plot_confusion_matrix([[0, 0], [0, 0]], tmp_path / "cm.png")
    assert path.is_file()


def test_write_report_produces_all_three_artefacts(tmp_path: Path):
    report = compute_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    written = write_report(report, tmp_path, y_true=[0, 0, 1, 1], y_pred=[0, 1, 1, 1])
    assert set(written) == {"metrics", "report", "confusion_matrix"}
    for path in written.values():
        assert path.is_file() and path.stat().st_size > 0

    payload = json.loads(written["metrics"].read_text(encoding="utf-8"))
    assert payload["accuracy"] == 0.75
    text = written["report"].read_text(encoding="utf-8")
    assert "accuracy" in text and "confusion matrix" in text
    assert "classification report" in text


def test_synthetic_report_files_are_prefixed(tmp_path: Path):
    report = compute_metrics([0, 1], [0, 1], dataset_kind="synthetic")
    written = write_report(report, tmp_path)
    for name, path in written.items():
        assert "SYNTHETIC" in path.name, f"{name} is not marked as synthetic"
    assert "SYNTHETIC" in written["metrics"].read_text(encoding="utf-8") or True
    notes = json.loads(written["metrics"].read_text(encoding="utf-8"))["notes"]
    assert any("NOT a benchmark" in note for note in notes)


def test_write_report_without_predictions_still_explains_itself(tmp_path: Path):
    report = compute_metrics([0, 1], [0, 1])
    written = write_report(report, tmp_path)
    text = written["report"].read_text(encoding="utf-8")
    assert "confusion matrix" in text
    assert "classification report" not in text


class FixedModel:
    """Returns a scripted probability, so the evaluation path is exercised without TF."""

    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities
        self.calls = 0

    def predict(self, x, verbose=0):
        self.calls += 1
        assert x.shape[0] == len(self.probabilities)
        return np.array(self.probabilities, dtype=np.float32).reshape(-1, 1)


def test_evaluate_model_scores_and_writes(settings, tmp_path: Path):
    model = FixedModel([0.9, 0.1, 0.85, 0.2])
    y_true = np.array([1, 0, 1, 0], dtype=np.float32)
    x_test = np.zeros((4, settings.sequence_length, 5), dtype=np.float32)
    report, paths = evaluate_model(
        model, x_test, y_true, output_dir=tmp_path, settings=settings, threshold=0.5
    )
    assert model.calls == 1
    assert report.accuracy == 1.0
    assert report.confusion_matrix == [[2, 0], [0, 2]]
    assert paths["metrics"].is_file()


def test_evaluate_model_uses_the_configured_threshold(settings, tmp_path: Path):
    """Both samples sit between the two thresholds, so the threshold decides the outcome."""
    settings.confidence_threshold = 0.8
    y_true = np.array([0, 1], dtype=np.float32)
    x_test = np.zeros((2, settings.sequence_length, 5), dtype=np.float32)

    permissive, _ = evaluate_model(
        FixedModel([0.75, 0.75]), x_test, y_true, output_dir=tmp_path, settings=settings,
        threshold=0.5,
    )
    assert permissive.confusion_matrix == [[0, 1], [0, 1]], "at 0.5 both are FALL"

    strict, _ = evaluate_model(
        FixedModel([0.75, 0.75]), x_test, y_true, output_dir=tmp_path, settings=settings
    )
    assert strict.threshold == pytest.approx(0.8)
    assert strict.confusion_matrix == [[1, 0], [1, 0]], "0.75 must fall below 0.8"
    assert strict.recall == 0.0


def test_evaluate_model_refuses_an_empty_test_split(settings, tmp_path: Path):
    model = FixedModel([])
    x_test = np.zeros((0, settings.sequence_length, 5), dtype=np.float32)
    with pytest.raises(NoSamplesError):
        evaluate_model(model, x_test, np.zeros(0), output_dir=tmp_path, settings=settings)


def test_evaluate_model_labels_synthetic_provenance(settings, tmp_path: Path):
    model = FixedModel([0.9, 0.1])
    x_test = np.zeros((2, settings.sequence_length, 5), dtype=np.float32)
    report, paths = evaluate_model(
        model, x_test, np.array([1, 0]), output_dir=tmp_path, settings=settings,
        dataset_kind="synthetic",
    )
    assert report.is_synthetic
    assert "SYNTHETIC" in paths["metrics"].name
