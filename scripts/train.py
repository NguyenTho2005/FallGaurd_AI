#!/usr/bin/env python
"""Train the fall-detection model.

    # 1. smoke test on synthetic data (no real dataset needed)
    python scripts/train.py --synthetic

    # 2. train on a real dataset
    python scripts/train.py --data data/processed

``--data`` points at a directory of ``.npy`` landmark files, ideally laid out as::

    data/processed/
    ├── fall/       subject01_fall_01.npy + .json
    ├── non_fall/   subject01_walk_01.npy + .json
    └── ...

A ``.json`` sidecar may provide ``label``, ``group_id``, ``source`` and ``frame_labels``
(per-frame 0/1 temporal annotation). Without a sidecar the label is inferred from the
folder name and the group is the file's relative path.

**Data leakage:** splits are computed over whole groups (a video or a subject), never over
windows, so near-identical frames from one recording cannot land on both sides of the
train/test boundary.

**No metrics are invented.** If the test split is empty, the run reports that fact and
writes no numbers.
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
from ai.logging_config import configure_logging, get_logger  # noqa: E402
from ai.preprocessing.pose_preprocessor import PosePreprocessor  # noqa: E402
from ai.training.dataset import (  # noqa: E402
    WindowedDataset,
    build_windows,
    load_npy_samples,
)
from ai.training.trainer import InsufficientDataError, train_model  # noqa: E402

logger = get_logger(__name__)


def collect_dataset(data_dir: Path, sequence_length: int, stride: int | None) -> WindowedDataset:
    """Load ``.npy`` landmark files and window them."""
    settings = get_settings()
    preprocessor = PosePreprocessor.from_settings(settings)
    samples = load_npy_samples(data_dir)
    if not samples:
        raise InsufficientDataError(
            f"no usable landmark files were found under '{data_dir}'. Expected "
            "*.npy files of shape (T, 33, 4). Run scripts/generate_sample_data.py --synthetic "
            "for a smoke test, or point --data at your own dataset."
        )
    groups = {sample.group_id for sample in samples}
    logger.info(
        "Loaded %d recording(s) across %d group(s)", len(samples), len(groups)
    )
    return build_windows(
        samples, preprocessor, sequence_length=sequence_length, stride=stride
    )


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data", default=None, help="directory of .npy landmark files")
    source.add_argument(
        "--synthetic", action="store_true",
        help="train on data/processed/synthetic (a pipeline smoke test, not a benchmark)",
    )
    parser.add_argument("--architecture", choices=("lstm", "bilstm"), default="lstm")
    parser.add_argument("--output", default=None, help="checkpoint path")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None, help="window stride in frames")
    parser.add_argument("--no-evaluate", action="store_true", help="skip test-split scoring")
    parser.add_argument("--quiet", action="store_true", help="verbose=0 for Keras")
    args = parser.parse_args()

    configure_logging()
    if args.sequence_length:
        settings.sequence_length = args.sequence_length
    if args.batch_size:
        settings.batch_size = args.batch_size

    if args.synthetic:
        data_dir = Path(settings.processed_dir) / "synthetic"
        if not data_dir.is_dir():
            print(
                "No synthetic data found. Run:\n"
                "  python scripts/generate_sample_data.py",
                file=sys.stderr,
            )
            return 1
        print("=" * 78)
        print("  TRAINING ON SYNTHETIC DATA -- pipeline smoke test only.")
        print("  Any metric produced here is NOT a benchmark of fall detection.")
        print("=" * 78)
    else:
        data_dir = Path(args.data) if args.data else Path(settings.processed_dir)

    try:
        dataset = collect_dataset(data_dir, settings.sequence_length, args.stride)
    except InsufficientDataError as exc:
        logger.error("%s", exc)
        return 1

    print(json.dumps(dataset.summary(), indent=2))
    if len(dataset) == 0:
        logger.error(
            "No windows were produced. Check that each recording is longer than the "
            "%d-frame sequence length.", settings.sequence_length
        )
        return 1

    try:
        result = train_model(
            dataset,
            architecture=args.architecture,
            settings=settings,
            model_path=args.output or Path(settings.model_path),
            evaluate=not args.no_evaluate,
            epochs=args.epochs,
            verbose=0 if args.quiet else 1,
        )
    except InsufficientDataError as exc:
        logger.error("Training aborted: %s", exc)
        return 1

    print("-" * 78)
    print(f"checkpoint : {result.model_path}")
    print(f"metadata   : {result.metadata_path}")
    print(f"splits     : " + ", ".join(
        f"{name}={item['windows']}" for name, item in result.splits.items()
    ))
    if result.evaluation:
        evaluation = result.evaluation
        print(
            f"test split : accuracy={evaluation['accuracy']:.4f} "
            f"precision={evaluation['precision']:.4f} recall={evaluation['recall']:.4f} "
            f"f1={evaluation['f1']:.4f} (n={evaluation['n_samples']})"
        )
        if evaluation.get("dataset_kind", "").startswith("synthetic"):
            print("             ^^ SYNTHETIC -- not a benchmark result.")
    else:
        print("test split : not evaluated (no metrics are reported rather than guessed)")
    for message in result.messages:
        print(f"note       : {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
