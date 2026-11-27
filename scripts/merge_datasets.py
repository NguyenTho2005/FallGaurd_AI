#!/usr/bin/env python
"""Merge available fall-detection datasets into ``data/processed/merged/``.

Sources inspected under ``data/raw/`` and ``data/processed/``:
1. **UR Fall Detection Dataset (URFD)**:
   Reuses existing extracted ``cam0`` samples in ``data/processed/urfd/`` and extracts
   remaining ``cam0`` MP4 videos from ``data/raw/urfd/`` using per-frame annotations
   from ``urfall-cam0-falls.csv`` and ``urfall-cam0-adls.csv``.
2. **Le2i / IMVIA Fall Dataset**:
   Processes annotated ``.avi`` videos from ``data/raw/imvia_fall/`` (Coffee_room_01,
   Coffee_room_02, Home_01, Home_02), parsing fall start/end frame intervals from the
   corresponding ``.txt`` annotation files. Videos without fall-interval headers or
   annotation files (Lecture_room, Office) are recorded as skipped rather than guessed.
3. **Multiple Cameras Fall Dataset (MCFD)**:
   Checks ``data/raw/multiple_cameras_fall/``; if extracted scenarios exist, extracts
   their videos, otherwise records the incomplete/missing download status in the
   merge manifest without stalling or fabricating entries.

Outputs
-------
``data/processed/merged/fall/*.npy + *.json``
``data/processed/merged/non_fall/*.npy + *.json``
``data/processed/merged/merge_manifest.json``
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ai.config import get_settings  # noqa: E402
from ai.logging_config import configure_logging, get_logger  # noqa: E402
from ai.pose.landmarks import POSE_LANDMARK_COUNT, POSE_LANDMARK_WIDTH  # noqa: E402

logger = get_logger(__name__)

# Silence verbose TF/MediaPipe C++ logs inside worker processes.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

_WORKER_ESTIMATOR: Any = None


@dataclass(slots=True)
class MergeTask:
    """One video extraction or reuse task."""

    dataset: str
    group_id: str
    source_rel: str
    video_path: Path | None
    label: int
    frame_labels: list[int] | None = None
    existing_npy: Path | None = None
    existing_json: Path | None = None
    fps_fallback: float = 25.0


def _init_worker() -> None:
    """Lazily initialize one MediaPipe PoseEstimator per worker process."""
    global _WORKER_ESTIMATOR
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "2")
    from ai.pose.pose_estimator import PoseEstimator

    _WORKER_ESTIMATOR = PoseEstimator(get_settings())


def _open_video_capture(video_path: Path):
    """Open a video file, preferring CAP_MSMF for uncompressed DIB AVIs on Windows."""
    import cv2

    backends = (
        (cv2.CAP_MSMF, cv2.CAP_ANY, cv2.CAP_DSHOW)
        if video_path.suffix.lower() == ".avi"
        else (cv2.CAP_ANY, cv2.CAP_MSMF, cv2.CAP_DSHOW)
    )
    for backend in backends:
        cap = (
            cv2.VideoCapture(str(video_path))
            if backend == cv2.CAP_ANY
            else cv2.VideoCapture(str(video_path), backend)
        )
        if cap.isOpened():
            try:
                ok, frame = cap.read()
                if ok and frame is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    return cap, None
            except Exception:
                pass
        cap.release()

    return None, f"Could not decode frames from {video_path}"


def _align_frame_labels(raw_labels: list[int] | None, length: int, default_label: int) -> list[int]:
    """Align per-frame annotations to exact extracted frame count ``length``."""
    if raw_labels is None:
        return [int(default_label)] * length
    if len(raw_labels) == length:
        return [int(v) for v in raw_labels]
    if not raw_labels:
        return [int(default_label)] * length
    if len(raw_labels) > length:
        return [int(v) for v in raw_labels[:length]]
    pad_val = int(raw_labels[-1])
    return [int(v) for v in raw_labels] + [pad_val] * (length - len(raw_labels))


def _process_task(
    task: MergeTask,
    output_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Execute one merge task and return a manifest record."""
    folder = "fall" if task.label == 1 else "non_fall"
    out_dir = output_root / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", task.group_id).strip("_")
    out_npy = out_dir / f"{safe_name}.npy"
    out_json = out_dir / f"{safe_name}.json"

    if out_npy.is_file() and out_json.is_file() and not force:
        meta = json.loads(out_json.read_text(encoding="utf-8"))
        frames_n = int(meta.get("frames", 0))
        expected_fl = _align_frame_labels(task.frame_labels, frames_n, task.label)
        if task.label == 0 and sum(meta.get("frame_labels", [])) != 0:
            meta["frame_labels"] = [0] * frames_n
            meta["positive_frames"] = 0
            out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        elif task.frame_labels is not None and meta.get("frame_labels") != expected_fl:
            meta["frame_labels"] = expected_fl
            meta["positive_frames"] = int(sum(expected_fl))
            out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {
            "status": "reused_merged",
            "dataset": task.dataset,
            "group_id": task.group_id,
            "label": task.label,
            "frames": frames_n,
            "detected_frames": int(meta.get("detected_frames", frames_n)),
            "positive_frames": int(meta.get("positive_frames", 0)),
            "npy": out_npy.relative_to(output_root).as_posix(),
            "source": task.source_rel,
        }

    # Reuse existing pre-extracted landmarks (e.g. data/processed/urfd) without touching originals.
    if task.existing_npy is not None and task.existing_npy.is_file():
        arr = np.load(task.existing_npy).astype(np.float32)
        existing_meta: dict[str, Any] = {}
        if task.existing_json is not None and task.existing_json.is_file():
            existing_meta = json.loads(task.existing_json.read_text(encoding="utf-8"))
        raw_fl = existing_meta.get("frame_labels")
        fl = (
            [0] * int(arr.shape[0])
            if task.label == 0
            else _align_frame_labels(
                [int(x) for x in raw_fl] if isinstance(raw_fl, list) else task.frame_labels,
                int(arr.shape[0]),
                task.label,
            )
        )
        detected = int(np.sum(~np.all(arr == 0.0, axis=(1, 2))))
        np.save(out_npy, arr)
        sidecar = {
            "label": int(task.label),
            "group_id": task.group_id,
            "dataset": task.dataset,
            "source": str(existing_meta.get("source") or task.source_rel),
            "frames": int(arr.shape[0]),
            "detected_frames": detected,
            "positive_frames": int(sum(fl)),
            "frame_labels": fl,
        }
        out_json.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        return {
            "status": "copied_existing",
            "dataset": task.dataset,
            "group_id": task.group_id,
            "label": task.label,
            "frames": int(arr.shape[0]),
            "detected_frames": detected,
            "positive_frames": int(sum(fl)),
            "npy": out_npy.relative_to(output_root).as_posix(),
            "source": task.source_rel,
        }

    if task.video_path is None or not task.video_path.is_file():
        return {
            "status": "error",
            "dataset": task.dataset,
            "group_id": task.group_id,
            "reason": f"Missing video: {task.video_path}",
        }

    global _WORKER_ESTIMATOR
    if _WORKER_ESTIMATOR is None:
        _init_worker()

    import cv2

    cap, err = _open_video_capture(task.video_path)
    if cap is None:
        return {
            "status": "error",
            "dataset": task.dataset,
            "group_id": task.group_id,
            "reason": err,
        }

    reported_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = reported_fps if 1.0 <= reported_fps <= 240.0 else task.fps_fallback

    frames_arr: list[np.ndarray] = []
    detected = 0
    idx = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            pose = _WORKER_ESTIMATOR.process(
                bgr,
                timestamp=idx / fps,
                frame_index=idx,
            )
            if pose is not None:
                frames_arr.append(pose.landmarks.astype(np.float32))
                detected += 1
            else:
                frames_arr.append(
                    np.zeros((POSE_LANDMARK_COUNT, POSE_LANDMARK_WIDTH), dtype=np.float32)
                )
            idx += 1
    finally:
        cap.release()

    if len(frames_arr) < 2 or detected < 2:
        return {
            "status": "skipped_low_poses",
            "dataset": task.dataset,
            "group_id": task.group_id,
            "frames": len(frames_arr),
            "detected_frames": detected,
            "source": task.source_rel,
        }

    arr = np.stack(frames_arr, axis=0).astype(np.float32)
    fl = _align_frame_labels(task.frame_labels, int(arr.shape[0]), task.label)
    np.save(out_npy, arr)
    sidecar = {
        "label": int(task.label),
        "group_id": task.group_id,
        "dataset": task.dataset,
        "source": task.source_rel,
        "frames": int(arr.shape[0]),
        "detected_frames": detected,
        "positive_frames": int(sum(fl)),
        "frame_labels": fl,
    }
    out_json.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return {
        "status": "extracted",
        "dataset": task.dataset,
        "group_id": task.group_id,
        "label": task.label,
        "frames": int(arr.shape[0]),
        "detected_frames": detected,
        "positive_frames": int(sum(fl)),
        "npy": out_npy.relative_to(output_root).as_posix(),
        "source": task.source_rel,
    }


def _load_urfd_csv_labels(features_dir: Path) -> dict[str, list[int]]:
    """Parse ``urfall-cam0-falls.csv`` into binary 0/1 fall-sequence frame labels.

    Note: ``urfall-cam0-adls.csv`` records lying posture (e.g. lying on a bed/sofa)
    as ``1`` even though ADLs are non-fall sequences. Therefore only fall sequences
    use CSV per-frame posture labels; all ADL frames are strictly labelled ``0``.
    """
    by_seq: dict[str, list[tuple[int, int]]] = {}
    csv_path = features_dir / "urfall-cam0-falls.csv"
    if csv_path.is_file():
        for line in csv_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            seq_name = parts[0].strip()
            try:
                frame_num = int(parts[1])
                raw_label = int(parts[2])
            except ValueError:
                continue
            # URFD convention: -1 = not lying, 0 = falling, 1 = lying -> binary 1 when >= 0.
            binary = 1 if raw_label >= 0 else 0
            by_seq.setdefault(seq_name, []).append((frame_num, binary))
    return {
        seq: [lbl for _, lbl in sorted(items, key=lambda pair: pair[0])]
        for seq, items in by_seq.items()
    }


def discover_urfd_tasks(raw_urfd: Path, processed_urfd: Path) -> tuple[list[MergeTask], dict[str, Any]]:
    """Build tasks for all 70 URFD cam0 sequences (reusing processed_urfd when present)."""
    tasks: list[MergeTask] = []
    csv_labels = _load_urfd_csv_labels(raw_urfd / "features")
    reused_count = 0
    extract_count = 0

    for seq_type, label in (("fall", 1), ("adl", 0)):
        raw_dir = raw_urfd / seq_type
        proc_dir = processed_urfd / ("fall" if label == 1 else "non_fall")
        if not raw_dir.is_dir():
            continue
        for mp4 in sorted(raw_dir.glob("*-cam0.mp4")):
            seq_id = mp4.stem.replace("-cam0", "")
            existing_npy = proc_dir / f"{seq_id}.npy"
            existing_json = proc_dir / f"{seq_id}.json"
            if existing_npy.is_file() and existing_json.is_file():
                reused_count += 1
            else:
                extract_count += 1
            tasks.append(
                MergeTask(
                    dataset="urfd",
                    group_id=f"urfd_{seq_id}",
                    source_rel=f"urfd/{seq_type}/{mp4.name}",
                    video_path=mp4,
                    label=label,
                    frame_labels=csv_labels.get(seq_id),
                    existing_npy=existing_npy if existing_npy.is_file() else None,
                    existing_json=existing_json if existing_json.is_file() else None,
                    fps_fallback=30.0,
                )
            )

    info = {
        "status": "ready" if tasks else "missing",
        "total_sequences": len(tasks),
        "reused_from_processed_urfd": reused_count,
        "to_extract_from_mp4": extract_count,
        "note": "Uses cam0 (30 falls + 40 ADLs) aligned with urfall-cam0-falls.csv; all ADL frames are strictly 0 (non-fall) to avoid confusing intentional lying with falls.",
    }
    return tasks, info


def _parse_imvia_annotation(txt_path: Path) -> tuple[int, int, int] | None:
    """Parse Le2i annotation file header ``(label, start_frame, end_frame)``.

    Returns ``None`` if the file lacks the 2-line integer fall-interval header.
    """
    lines = [
        line.strip()
        for line in txt_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    if len(lines) < 2:
        return None
    try:
        start_frame = int(lines[0])
        end_frame = int(lines[1])
    except ValueError:
        return None

    if start_frame <= 0 and end_frame <= 0:
        return (0, 0, 0)
    if end_frame < start_frame:
        return (1, start_frame, start_frame)
    return (1, start_frame, end_frame)


def discover_imvia_tasks(raw_imvia: Path) -> tuple[list[MergeTask], dict[str, Any]]:
    """Build tasks for annotated Le2i / IMVIA Fall Dataset videos."""
    tasks: list[MergeTask] = []
    skipped_unannotated: list[str] = []
    skipped_missing_header: list[str] = []
    fall_count = 0
    non_fall_count = 0

    if not raw_imvia.is_dir():
        return tasks, {"status": "missing"}

    import cv2

    # Scenes with Annotation_files / Annotations_files
    annotated_scenes = (
        ("Coffee_room_01", raw_imvia / "Coffee_room_01" / "Coffee_room_01"),
        ("Coffee_room_02", raw_imvia / "Coffee_room_02" / "Coffee_room_02"),
        ("Home_01", raw_imvia / "Home_01" / "Home_01"),
        ("Home_02", raw_imvia / "Home_02" / "Home_02"),
    )
    for scene_name, scene_dir in annotated_scenes:
        if not scene_dir.is_dir():
            continue
        ann_dir = scene_dir / (
            "Annotations_files" if (scene_dir / "Annotations_files").is_dir() else "Annotation_files"
        )
        vid_dir = scene_dir / "Videos"
        for vid in sorted(vid_dir.glob("*.avi")):
            ann = ann_dir / f"{vid.stem}.txt"
            if not ann.is_file():
                skipped_unannotated.append(f"{scene_name}/{vid.name}")
                continue
            parsed = _parse_imvia_annotation(ann)
            if parsed is None:
                skipped_missing_header.append(f"{scene_name}/{vid.name}")
                continue
            label, start_f, end_f = parsed
            # Determine frame count so we can build 1-indexed frame_labels cleanly.
            cap, _ = _open_video_capture(vid)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if cap is not None else 0
            if cap is not None:
                cap.release()
            if label == 1:
                fall_count += 1
                frame_labels = (
                    [1 if start_f <= (i + 1) <= end_f else 0 for i in range(n_frames)]
                    if n_frames > 0
                    else None
                )
            else:
                non_fall_count += 1
                frame_labels = [0] * n_frames if n_frames > 0 else None

            slug = re.sub(r"[^A-Za-z0-9]+", "_", vid.stem).strip("_")
            tasks.append(
                MergeTask(
                    dataset="imvia_fall",
                    group_id=f"imvia_{scene_name}_{slug}",
                    source_rel=f"imvia_fall/{scene_name}/Videos/{vid.name}",
                    video_path=vid,
                    label=label,
                    frame_labels=frame_labels,
                    fps_fallback=25.0,
                )
            )

    # Unannotated scenes (Lecture_room, Office)
    for extra_scene, subdir in (
        ("Lecture_room", raw_imvia / "Lecture_room" / "Lecture room"),
        ("Office", raw_imvia / "Office" / "Office"),
    ):
        if subdir.is_dir():
            for vid in sorted(subdir.glob("*.avi")):
                skipped_unannotated.append(f"{extra_scene}/{vid.name}")

    info = {
        "status": "ready" if tasks else "missing",
        "annotated_videos": len(tasks),
        "fall_videos": fall_count,
        "non_fall_videos": non_fall_count,
        "skipped_no_annotation_count": len(skipped_unannotated),
        "skipped_missing_header_count": len(skipped_missing_header),
        "skipped_missing_header_files": skipped_missing_header,
        "note": "Lecture_room (27) and Office (33) contain no .txt ground-truth files, and 3 .txt files lack fall-interval headers; skipped rather than guessing labels.",
    }
    return tasks, info


def inspect_mcfd(raw_mcfd: Path) -> dict[str, Any]:
    """Inspect Multiple Cameras Fall Dataset availability."""
    if not raw_mcfd.is_dir():
        return {"status": "missing"}
    part = raw_mcfd / "dataset.zip.part"
    scenarios = [
        d for d in raw_mcfd.iterdir()
        if d.is_dir() and d.name not in {".gitkeep"}
    ]
    if scenarios:
        return {
            "status": "extracted",
            "scenario_dirs": len(scenarios),
        }
    if part.is_file():
        size_mib = round(part.stat().st_size / (1024 * 1024), 2)
        return {
            "status": "incomplete_download",
            "partial_file": part.name,
            "downloaded_mib": size_mib,
            "expected_mib": 3607.58,
            "note": "Download was interrupted (492 MiB / 3.52 GiB); excluded from merge until download completes.",
        }
    return {"status": "not_downloaded"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="output merged directory")
    parser.add_argument("--workers", type=int, default=4, help="parallel worker processes")
    parser.add_argument("--force", action="store_true", help="re-extract existing merged files")
    args = parser.parse_args(argv)

    configure_logging()
    settings = get_settings()
    data_dir = Path(settings.data_dir)
    raw_dir = data_dir / "raw"
    output_root = Path(args.output) if args.output else data_dir / "processed" / "merged"
    output_root.mkdir(parents=True, exist_ok=True)

    urfd_tasks, urfd_info = discover_urfd_tasks(raw_dir / "urfd", data_dir / "processed" / "urfd")
    imvia_tasks, imvia_info = discover_imvia_tasks(raw_dir / "imvia_fall")
    mcfd_info = inspect_mcfd(raw_dir / "multiple_cameras_fall")

    all_tasks = [*urfd_tasks, *imvia_tasks]
    logger.info(
        "Merge plan: URFD=%d, IMVIA=%d, MCFD=%s -> total tasks=%d",
        len(urfd_tasks),
        len(imvia_tasks),
        mcfd_info["status"],
        len(all_tasks),
    )

    records: list[dict[str, Any]] = []
    # Run already-merged or pre-extracted copy tasks in-process (no MediaPipe startup needed).
    inproc_tasks: list[MergeTask] = []
    extract_tasks: list[MergeTask] = []
    for t in all_tasks:
        folder = "fall" if t.label == 1 else "non_fall"
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", t.group_id).strip("_")
        already_merged = (
            (output_root / folder / f"{safe_name}.npy").is_file()
            and (output_root / folder / f"{safe_name}.json").is_file()
            and not args.force
        )
        if already_merged or t.existing_npy is not None:
            inproc_tasks.append(t)
        else:
            extract_tasks.append(t)

    for task in inproc_tasks:
        records.append(_process_task(task, output_root, force=args.force))

    if extract_tasks:
        workers = max(1, min(args.workers, os.cpu_count() or 4))
        logger.info("Extracting poses for %d videos using %d worker(s)...", len(extract_tasks), workers)
        if workers == 1:
            _init_worker()
            for idx, task in enumerate(extract_tasks, start=1):
                rec = _process_task(task, output_root, force=args.force)
                records.append(rec)
                if idx % 10 == 0 or idx == len(extract_tasks):
                    logger.info("Progress: %d/%d videos processed", idx, len(extract_tasks))
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
                futures = {
                    pool.submit(_process_task, task, output_root, args.force): task
                    for task in extract_tasks
                }
                done = 0
                for fut in as_completed(futures):
                    done += 1
                    rec = fut.result()
                    records.append(rec)
                    if done % 15 == 0 or done == len(extract_tasks):
                        logger.info("Progress: %d/%d videos processed", done, len(extract_tasks))

    usable = [r for r in records if r.get("status") in {"extracted", "copied_existing", "reused_merged"}]
    skipped = [r for r in records if r.get("status") not in {"extracted", "copied_existing", "reused_merged"}]

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_root),
        "sources": {
            "urfd": urfd_info,
            "imvia_fall": imvia_info,
            "multiple_cameras_fall": mcfd_info,
        },
        "totals": {
            "recordings_usable": len(usable),
            "recordings_fall": sum(1 for r in usable if r.get("label") == 1),
            "recordings_non_fall": sum(1 for r in usable if r.get("label") == 0),
            "recordings_skipped_or_error": len(skipped),
            "total_frames": sum(int(r.get("frames", 0)) for r in usable),
            "total_detected_poses": sum(int(r.get("detected_frames", 0)) for r in usable),
            "total_positive_frames": sum(int(r.get("positive_frames", 0)) for r in usable),
        },
        "by_dataset": {
            ds: {
                "usable": sum(1 for r in usable if r.get("dataset") == ds),
                "fall": sum(1 for r in usable if r.get("dataset") == ds and r.get("label") == 1),
                "non_fall": sum(1 for r in usable if r.get("dataset") == ds and r.get("label") == 0),
                "frames": sum(int(r.get("frames", 0)) for r in usable if r.get("dataset") == ds),
                "detected_poses": sum(int(r.get("detected_frames", 0)) for r in usable if r.get("dataset") == ds),
            }
            for ds in ("urfd", "imvia_fall")
        },
        "skipped_records": skipped,
        "records": sorted(usable, key=lambda r: str(r.get("group_id"))),
    }
    manifest_path = output_root / "merge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Merge complete: %s", json.dumps(manifest["totals"]))
    print(json.dumps(manifest["totals"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
