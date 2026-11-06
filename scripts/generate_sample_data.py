#!/usr/bin/env python
"""Generate the synthetic sample clip and landmark datasets.

    python scripts/generate_sample_data.py                 # clip + landmark datasets
    python scripts/generate_sample_data.py --clips-only    # just the demo video
    python scripts/generate_sample_data.py --check-detection

What is produced
----------------
``data/samples/synthetic_room.avi``
    A short clip of a fictional figure that stands, walks, then falls. Fed through the
    **real** MediaPipe estimator so the demo exercises the whole pipeline.

``data/processed/synthetic/<label>/<group>.npy``
    ``(T, 33, 4)`` landmark arrays extracted from generated clips, plus a JSON sidecar.
    Several clips with different timings/positions are produced so the group-aware split
    has something real to work with.

.. warning::

   Everything this script writes is **SYNTHETIC**. It exists to exercise the software
   pipeline without a camera or a licensed dataset. A model trained on it is a smoke-test
   artefact, and any metric derived from it is labelled ``SYNTHETIC_`` and is **not** a
   benchmark. See ``docs/model.md`` for how to train on a real dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ai.config import get_settings  # noqa: E402
from ai.logging_config import configure_logging, get_logger  # noqa: E402
from ai.pose.landmarks import PoseFrame, PoseSample  # noqa: E402
from ai.pose.synthetic import (  # noqa: E402
    SYNTHETIC_BANNER,
    SceneScript,
    generate_sample_video,
    render_frame,
)
from ai.preprocessing.pose_preprocessor import PosePreprocessor  # noqa: E402

logger = get_logger(__name__)

#: Variants differ in timing, fall speed and starting position, so the generated groups
#: are not near-duplicates of one another.
#:
#: There are deliberately *many* of them. Splits are assigned per recording, so with only
#: a handful of fall clips the training split receives almost no positive windows and the
#: model cannot learn anything. Group count is the binding constraint on a smoke test, not
#: window count.
VARIANTS: tuple[dict[str, object], ...] = (
    {"name": "fall_slow", "idle": 60, "walk": 30, "pause": 20, "fall": 45, "down": 60,
     "offset_x": 0.0},
    {"name": "fall_fast", "idle": 40, "walk": 60, "pause": 15, "fall": 20, "down": 40,
     "offset_x": 0.04},
    {"name": "fall_left", "idle": 50, "walk": 45, "pause": 25, "fall": 30, "down": 50,
     "offset_x": -0.05},
    {"name": "fall_right", "idle": 55, "walk": 35, "pause": 20, "fall": 35, "down": 55,
     "offset_x": 0.05},
    {"name": "fall_early", "idle": 25, "walk": 25, "pause": 10, "fall": 25, "down": 70,
     "offset_x": -0.02},
    {"name": "fall_late", "idle": 90, "walk": 50, "pause": 35, "fall": 40, "down": 45,
     "offset_x": 0.02},
    {"name": "fall_gradual", "idle": 45, "walk": 40, "pause": 20, "fall": 55, "down": 60,
     "offset_x": 0.06},
    {"name": "fall_tumble", "idle": 35, "walk": 70, "pause": 30, "fall": 18, "down": 65,
     "offset_x": -0.06},
    {"name": "fall_after_walk", "idle": 30, "walk": 100, "pause": 20, "fall": 32, "down": 58,
     "offset_x": 0.03},
    {"name": "fall_mid", "idle": 65, "walk": 55, "pause": 18, "fall": 28, "down": 62,
     "offset_x": -0.01},
)

#: Normal-activity clips, so the dataset is balanced and the model sees both classes.
ACTIVITY_VARIANTS: tuple[dict[str, object], ...] = (
    {"name": "normal_idle", "idle": 90, "walk": 40, "pause": 40, "fall": 0, "down": 50,
     "offset_x": 0.0},
    {"name": "normal_walk", "idle": 50, "walk": 90, "pause": 40, "fall": 0, "down": 50,
     "offset_x": 0.03},
    {"name": "normal_shift", "idle": 70, "walk": 60, "pause": 50, "fall": 0, "down": 40,
     "offset_x": -0.04},
    {"name": "normal_pace", "idle": 60, "walk": 75, "pause": 30, "fall": 0, "down": 60,
     "offset_x": 0.05},
    {"name": "normal_slow", "idle": 80, "walk": 50, "pause": 60, "fall": 0, "down": 45,
     "offset_x": -0.03},
    {"name": "normal_brisk", "idle": 40, "walk": 100, "pause": 35, "fall": 0, "down": 55,
     "offset_x": 0.01},
    {"name": "normal_centre", "idle": 75, "walk": 65, "pause": 45, "fall": 0, "down": 50,
     "offset_x": 0.0},
    {"name": "normal_edge", "idle": 55, "walk": 80, "pause": 40, "fall": 0, "down": 65,
     "offset_x": -0.07},
)


def _script_for(variant: dict[str, object]) -> SceneScript:
    if int(variant["fall"]) == 0:
        # No fall: pad the timeline with idle frames so the group length stays comparable.
        return SceneScript(
            segments=(
                ("idle", int(variant["idle"])),
                ("walk", int(variant["walk"])),
                ("idle", int(variant["pause"])),
                ("idle", int(variant["down"])),
            )
        )
    return SceneScript(
        segments=(
            ("idle", int(variant["idle"])),
            ("walk", int(variant["walk"])),
            ("idle", int(variant["pause"])),
            ("fall", int(variant["fall"])),
            ("on_floor", int(variant["down"])),
        )
    )


def _shift(landmarks: np.ndarray, offset_x: float) -> np.ndarray:
    shifted = landmarks.copy()
    shifted[:, 0] = np.clip(shifted[:, 0] + offset_x, 0.0, 1.0)
    return shifted


def render_variant(variant: dict[str, object], output: Path, fps: int) -> tuple[Path, SceneScript]:
    """Render one variant clip, translated horizontally so groups differ visually."""
    script = _script_for(variant)
    import cv2

    from ai.pose.synthetic import _draw_watermark, _open_writer

    output.parent.mkdir(parents=True, exist_ok=True)
    writer, target = _open_writer(output, fps, (640, 480))
    offset_x = float(variant["offset_x"])
    try:
        from ai.pose.synthetic import landmarks_for_frame

        for index in range(script.total_frames):
            image = render_frame(_shift(landmarks_for_frame(script, index), offset_x))
            _draw_watermark(image, index, script.total_frames)
            writer.write(image)
    finally:
        writer.release()
    del cv2
    return target, script


def extract_landmarks(
    video_path: Path, *, fps: float, stride: int = 1
) -> tuple[np.ndarray, int, int]:
    """Run the real MediaPipe estimator over a clip. Returns ``(landmarks, detected, total)``."""
    from ai.pose.pose_estimator import PoseEstimator
    from ai.sources.video_source import VideoFileSource

    settings = get_settings()
    estimator = PoseEstimator(settings)
    frames: list[np.ndarray] = []
    total = 0
    try:
        source = VideoFileSource(video_path, settings)
        while True:
            source_frame = source.read()
            if source_frame is None:
                break
            if total % stride == 0:
                pose = estimator.process(
                    source_frame.frame,
                    timestamp=source_frame.timestamp,
                    frame_index=source_frame.index,
                )
                if pose is not None:
                    frames.append(pose.landmarks)
            total += 1
        source.close()
    finally:
        estimator.close()
    if not frames:
        return np.zeros((0, 33, 4), dtype=np.float32), 0, total
    return np.stack(frames, axis=0).astype(np.float32), len(frames), total


def check_detection(video_path: Path, sample_every: int = 10) -> float:
    """Re-measure the pose detection rate of a generated clip."""
    from ai.pose.pose_estimator import PoseEstimator
    from ai.sources.video_source import VideoFileSource

    settings = get_settings()
    estimator = PoseEstimator(settings)
    detected = 0
    total = 0
    try:
        source = VideoFileSource(video_path, settings)
        while True:
            source_frame = source.read()
            if source_frame is None:
                break
            if source_frame.index % sample_every == 0:
                total += 1
                if estimator.process(
                    source_frame.frame,
                    timestamp=source_frame.timestamp,
                    frame_index=source_frame.index,
                ) is not None:
                    detected += 1
        source.close()
    finally:
        estimator.close()
    rate = detected / total if total else 0.0
    logger.info(
        "Detection check on %s: %d/%d frames (%.1f%%)", video_path.name, detected, total,
        100.0 * rate,
    )
    return rate


def save_sample(
    landmarks: np.ndarray,
    *,
    label: int,
    group_id: str,
    output_root: Path,
    source: str,
    frame_labels: np.ndarray | None = None,
) -> Path | None:
    """Write one landmark array plus its sidecar under ``synthetic/<label>/``."""
    if landmarks.shape[0] < 2:
        logger.warning("Skipping %s: only %d pose frames", group_id, landmarks.shape[0])
        return None
    folder = "fall" if label == 1 else "non_fall"
    directory = output_root / "synthetic" / folder
    directory.mkdir(parents=True, exist_ok=True)
    npy_path = directory / f"{group_id}.npy"
    np.save(npy_path, landmarks)
    sidecar: dict[str, object] = {
        "label": label,
        "group_id": group_id,
        "source": source,
        "frames": int(landmarks.shape[0]),
        "synthetic": True,
        "note": SYNTHETIC_BANNER,
    }
    if frame_labels is not None:
        # Per-frame annotation, so the pre-fall seconds of a fall clip are labelled
        # NORMAL. Labelling the whole clip as a fall would teach the model to fire on
        # ordinary standing, which is the classic false-positive source.
        sidecar["frame_labels"] = [int(value) for value in frame_labels]
        sidecar["positive_frames"] = int(np.sum(frame_labels))
    npy_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return npy_path


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1,
                        help="keep every Nth frame when extracting landmarks")
    parser.add_argument("--clips-only", action="store_true",
                        help="only render the demo clip, do not build landmark datasets")
    parser.add_argument("--check-detection", action="store_true",
                        help="re-measure the MediaPipe detection rate after rendering")
    parser.add_argument("--clear", action="store_true",
                        help="delete previously generated synthetic landmark data first")
    parser.add_argument("--min-detection-rate", type=float, default=0.80,
                        help="fail if --check-detection drops below this rate")
    args = parser.parse_args()

    configure_logging()
    print("=" * 78)
    print("  SYNTHETIC DATA GENERATION -- " + SYNTHETIC_BANNER)
    print("=" * 78)

    samples_dir = Path(settings.samples_dir)
    processed = Path(settings.processed_dir)
    clips_dir = samples_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    if args.clear:
        import shutil

        target = processed / "synthetic"
        if target.exists():
            shutil.rmtree(target)
            logger.info("Removed previously generated data at %s", target)

    # 1. The single demo clip the dashboard and scripts/demo.py use.
    demo_clip = generate_sample_video(clips_dir / "synthetic_room.avi", fps=args.fps)
    print(f"demo clip            -> {demo_clip}")

    if args.clips_only:
        return 0

    # 2. Landmark datasets, one group per variant.
    written: list[tuple[Path, int]] = []
    for variant in (*VARIANTS, *ACTIVITY_VARIANTS):
        name = str(variant["name"])
        label = 0 if name.startswith("normal") else 1
        clip_path = clips_dir / f"{name}.avi"
        rendered, script = render_variant(variant, clip_path, args.fps)
        landmarks, detected, total = extract_landmarks(
            rendered, fps=float(args.fps), stride=args.stride
        )
        # Per-frame labels. A pose can be missed on a frame, so the annotation is
        # resampled onto the frames that were actually extracted: frame j of the
        # landmarks array corresponds to roughly the j'th detected frame of the clip.
        fall_start, fall_end = script.fall_frame_range()
        frame_labels = _frame_labels_for(landmarks.shape[0], total, fall_start, fall_end)
        saved = save_sample(
            landmarks,
            label=label,
            group_id=f"synthetic_{name}",
            output_root=processed,
            source=rendered.name,
            frame_labels=frame_labels if label == 1 else np.zeros(landmarks.shape[0], dtype=np.int64),
        )
        rate = detected / total if total else 0.0
        status = f"{saved.relative_to(_ROOT)}" if saved else "skipped"
        positive = int(frame_labels.sum()) if label == 1 else 0
        print(
            f"  {name:14s} label={label} frames={landmarks.shape[0]:3d}/{total:3d} "
            f"({rate:5.1%}) fall_frames={positive:3d} -> {status}"
        )
        if saved:
            written.append((saved, label))

    if not written:
        logger.error("No synthetic samples were written; check the MediaPipe asset")
        return 1

    print("-" * 78)
    print(f"wrote {len(written)} landmark group(s) under {processed / 'synthetic'}")
    print("Reminder: metrics computed on this data are NOT a benchmark.")

    if args.check_detection:
        rate = check_detection(demo_clip)
        if rate < args.min_detection_rate:
            logger.error(
                "Detection rate %.1f%% is below the %.1f%% floor. The renderer or the "
                "MediaPipe asset has changed; the demo would be unreliable.",
                100.0 * rate, 100.0 * args.min_detection_rate,
            )
            return 2
    return 0


def _frame_labels_for(
    detected_frames: int, total_frames: int, fall_start: int, fall_end: int
) -> np.ndarray:
    """Resample a clip-level fall span onto the frames that were actually extracted.

    Landmarks only exist for frames MediaPipe detected, so a clip-level annotation cannot
    be indexed directly. Scaling by the detection rate preserves the temporal position of
    the fall closely enough for window labelling.
    """
    labels = np.zeros(max(detected_frames, 0), dtype=np.int64)
    if detected_frames == 0 or total_frames == 0 or fall_end <= fall_start:
        return labels
    ratio = detected_frames / total_frames
    start_index = min(detected_frames, max(0, int(round(fall_start * ratio))))
    end_index = min(detected_frames, max(start_index, int(round(fall_end * ratio))))
    labels[start_index:end_index] = 1
    return labels


if __name__ == "__main__":
    raise SystemExit(main())
