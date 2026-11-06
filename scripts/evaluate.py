#!/usr/bin/env python
"""Evaluate a trained checkpoint and write a report to ``reports/``.

    python scripts/evaluate.py                       # evaluate models/fallguard_lstm.keras
    python scripts/evaluate.py --model models/other.keras --data data/processed
    python scripts/evaluate.py --mock                # score the mock backend instead

Outputs
-------
``reports/evaluation_metrics.json``
``reports/evaluation_classification_report.txt``
``reports/evaluation_confusion_matrix.png``

For a dataset under ``data/processed/synthetic`` the stem is prefixed ``SYNTHETIC_`` and
every artefact carries a note saying the numbers describe the software pipeline only.

The script **refuses to report metrics on an empty set**. If the test split has no
samples, or the data has only one class, it says so and exits non-zero rather than
printing a number that would be mistaken for a result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ai.config import get_settings  # noqa: E402
from ai.evaluation.evaluate import NoSamplesError, evaluate_model  # noqa: E402
from ai.logging_config import configure_logging, get_logger  # noqa: E402
from ai.preprocessing.pose_preprocessor import PosePreprocessor  # noqa: E402
from ai.training.dataset import (  # noqa: E402
    build_windows,
    load_npy_samples,
    split_dataset,
    unique_groups,
)

logger = get_logger(__name__)


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=None, help="checkpoint to evaluate")
    parser.add_argument("--data", default=None, help="directory of .npy landmark files")
    parser.add_argument("--architecture", choices=("lstm", "bilstm"), default="lstm")
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None,
                        help="decision threshold (default: FALLGUARD_CONFIDENCE_THRESHOLD)")
    parser.add_argument("--mock", action="store_true",
                        help="evaluate the deterministic mock backend instead of Keras")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    configure_logging()
    if args.sequence_length:
        settings.sequence_length = args.sequence_length

    data_dir = Path(args.data) if args.data else Path(settings.processed_dir)
    # Provenance is derived from the data, not from the path: the default directory
    # (data/processed) legitimately contains both real and synthetic subdirectories, so a
    # path check would silently mislabel synthetic results as real benchmarks.
    is_synthetic = any("synthetic" in sample.group_id.lower() for sample in load_npy_samples(data_dir))

    if args.mock:
        from ai.models.mock_model import MockFallModel

        model = MockFallModel(sequence_length=settings.sequence_length)
        kind = "synthetic-mock-heuristic"
        print("=" * 78)
        print("  Evaluating the MOCK BACKEND -- a kinematic heuristic, not a trained model.")
        print("  Its output is not a benchmark of fall-detection performance.")
        print("=" * 78)
    else:
        from ai.models.predictor import ModelContractError, ModelUnavailableError, FallPredictor

        model_path = Path(args.model) if args.model else Path(settings.model_path)
        settings.model_path = model_path
        try:
            predictor = FallPredictor(settings)
        except (ModelUnavailableError, ModelContractError) as exc:
            logger.error("Cannot evaluate: %s", exc)
            logger.error("Train one first: python scripts/train.py --synthetic")
            return 1
        model = predictor.model
        kind = "synthetic-smoke-test" if is_synthetic else "real"
        if is_synthetic:
            print("=" * 78)
            print("  SYNTHETIC DATA -- metrics below describe the software pipeline only")
            print("  and are NOT a benchmark of fall-detection performance.")
            print("=" * 78)

    preprocessor = PosePreprocessor.from_settings(settings)
    samples = load_npy_samples(data_dir)
    if not samples:
        logger.error("No landmark files found under '%s'", data_dir)
        return 1
    dataset = build_windows(
        samples, preprocessor, sequence_length=settings.sequence_length, stride=args.stride
    )
    if len(dataset) == 0:
        logger.error(
            "No windows produced. Every recording must exceed the %d-frame window.",
            settings.sequence_length,
        )
        return 1

    splits = split_dataset(dataset, settings)
    test_set = splits["test"]
    print(json.dumps({name: item.summary() for name, item in splits.items()}, indent=2))

    if len(test_set) == 0:
        logger.error(
            "The test split is empty (no group was assigned to it). Refusing to report "
            "metrics for zero samples -- add more recordings or adjust the split ratios."
        )
        return 2
    present = set(int(value) for value in test_set.y)
    if present != {0, 1}:
        logger.error(
            "The test split contains only the classes %s, so precision/recall/F1 for the "
            "positive class are undefined. Refusing to report a misleading number. "
            "Add recordings of both classes.", sorted(present),
        )
        return 2

    notes = [
        f"sequence_length={dataset.sequence_length}, feature_dim={dataset.feature_dim}",
        f"groups={sorted(str(group) for group in unique_groups(test_set.groups))}",
    ]
    if is_synthetic:
        notes.insert(
            0,
            "SYNTHETIC rendered data. These numbers validate the software pipeline only "
            "and are not a benchmark of fall-detection performance on real people.",
        )

    try:
        report, paths = evaluate_model(
            model,
            test_set.x,
            test_set.y,
            output_dir=args.output_dir or settings.reports_dir,
            threshold=args.threshold,
            settings=settings,
            dataset_kind=kind,
            notes=notes,
        )
    except NoSamplesError as exc:
        logger.error("Evaluation aborted: %s", exc)
        return 2

    print("-" * 78)
    print(f"samples        : {report.n_samples}  (positive rate {report.positive_rate:.3f})")
    print(f"threshold      : {report.threshold}")
    print(f"accuracy       : {report.accuracy:.4f}")
    print(f"precision      : {report.precision:.4f}")
    print(f"recall         : {report.recall:.4f}")
    print(f"f1             : {report.f1:.4f}")
    print(f"confusion      : {report.confusion_matrix}")
    for name, path in paths.items():
        print(f"{name:15s}: {path}")
    if report.is_synthetic:
        print("Reminder: SYNTHETIC result -- not a benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
