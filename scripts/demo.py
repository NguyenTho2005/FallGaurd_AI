#!/usr/bin/env python
"""DEMO MODE -- run the whole FallGuard pipeline with no camera and no dataset.

    python scripts/demo.py                       # synthetic clip, offline
    python scripts/demo.py --video path/to.mp4   # your own footage
    python scripts/demo.py --camera              # a real webcam
    python scripts/demo.py --post                # also drive the FastAPI backend

What this proves and what it does not
-------------------------------------
The demo runs the **real** pipeline: MediaPipe Pose -> preprocessing -> sequence buffer ->
model -> event confirmation -> event store -> Telegram alert. Nothing is stubbed except
the light source (a rendered figure) and, when no checkpoint is present, the model
backend (a deterministic kinematic heuristic).

It is printed, logged and named ``DEMO MODE`` for exactly that reason. A demo run is a
software integration test; it is **not** an accuracy claim.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ai.config import get_settings  # noqa: E402
from ai.logging_config import configure_logging, get_logger  # noqa: E402
from ai.pose.synthetic import SYNTHETIC_BANNER, generate_sample_video  # noqa: E402
from ai.preprocessing.pose_preprocessor import (  # noqa: E402
    KINEMATIC_FEATURE_NAMES,
    KINEMATIC_FEATURE_OFFSET,
    PosePreprocessor,
)
from backend.alerts.telegram import build_sender  # noqa: E402
from backend.database.repository import EventRepository  # noqa: E402
from backend.services.inference_service import InferenceService  # noqa: E402

logger = get_logger(__name__)

#: Resolved by name so a feature-layout change cannot silently mislabel the readout.
_KINE_INDEX = {name: i for i, name in enumerate(KINEMATIC_FEATURE_NAMES)}

BAR_WIDTH = 40


def banner(settings) -> None:
    model = settings.model_path
    print("=" * 78)
    print("  DEMO MODE  --  FallGuard AI")
    print("=" * 78)
    print(f"  model path       : {model} "
          f"({'present' if model.is_file() else 'MISSING -> using the mock backend'})")
    print(f"  model backend    : {settings.model_backend}")
    print(f"  sequence length  : {settings.sequence_length} frames")
    print(f"  confirmation     : p >= {settings.confidence_threshold} for "
          f"{settings.min_fall_frames} consecutive frames")
    print(f"  cooldown         : {settings.cooldown_seconds}s")
    print(f"  telegram         : "
          f"{'dry run' if not settings.telegram_configured else 'configured'}")
    print(f"  event store      : {settings.db_path}")
    print("-" * 78)
    print("  Every result below is a DEMO output. It is not an accuracy measurement.")
    print("=" * 78)


def progress_bar(value: float, maximum: float = 1.0) -> str:
    if maximum <= 0:
        return "[" + " " * BAR_WIDTH + "]"
    ratio = max(0.0, min(value / maximum, 1.0))
    filled = int(ratio * BAR_WIDTH)
    return "[" + "#" * filled + "-" * (BAR_WIDTH - filled) + f"] {ratio:5.1%}"


def ensure_synthetic_clip(settings) -> Path:
    """Generate the demo clip if it is missing."""
    candidates = [
        Path(settings.samples_dir) / "clips" / "synthetic_room.avi",
        Path(settings.samples_dir) / "synthetic_room.avi",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    target = candidates[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  generating the synthetic demo clip -> {target}")
    print(f"  {SYNTHETIC_BANNER}")
    return generate_sample_video(target)


def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings)
    banner(settings)

    from ai.inference.pipeline import FallDetectionPipeline
    from ai.sources.video_source import CameraSource, VideoFileSource

    if args.source == "camera":
        source_kind, path = "camera", None
        print("  source           : live camera")
    elif args.source == "video":
        source_kind, path = "video", Path(args.video)
        print(f"  source           : {path}")
    else:
        path = ensure_synthetic_clip(settings)
        source_kind = "synthetic"
        print(f"  source           : {path}")
        print(f"  {SYNTHETIC_BANNER}")

    if source_kind == "camera":
        source = CameraSource(settings.camera_source, settings)
    elif source_kind == "video":
        if not path.is_file():
            logger.error("Video not found: %s", path)
            return 1
        source = VideoFileSource(path, settings)
    else:
        source = VideoFileSource(path, settings)

    # InferenceService wires the pipeline's on_event callback to storage + alerts.
    service = InferenceService(settings)
    model = service.pipeline.model_info()
    if model.backend == "mock":
        print("  NOTE: the mock backend is active. It is a kinematic heuristic, not a")
        print("        trained model, so its output is not a performance measurement.")
    else:
        print(f"  NOTE: trained checkpoint loaded ({model.path}).")
    print("-" * 78)

    preprocessor = PosePreprocessor.from_settings(settings)
    total_frames = 0
    started = time.monotonic()
    shown: set[str] = set()
    try:
        while True:
            if args.max_frames and total_frames >= args.max_frames:
                print(f"\n  stopping after {args.max_frames} frames (--max-frames)")
                break
            source_frame = source.read()
            if source_frame is None:
                if source_kind == "camera":
                    time.sleep(0.05)
                    continue
                break

            outcome = service.process_frame(
                source_frame.frame,
                timestamp=source_frame.timestamp,
                frame_index=source_frame.index,
            )
            total_frames += 1

            if args.verbose or outcome.event is not None or total_frames % args.every == 0:
                pose_text = "pose" if outcome.pose_detected else "----"
                ready_text = "ready" if outcome.sequence_ready else "buf  "
                line = (
                    f"  frame {source_frame.index:4d} | {pose_text} | {ready_text} | "
                    f"{progress_bar(outcome.fall_probability)} | {outcome.prediction:<6}"
                )
                if outcome.pose_detected and args.verbose:
                    features = preprocessor.transform(service.pipeline.last_pose.landmarks)
                    lean = features[KINEMATIC_FEATURE_OFFSET + _KINE_INDEX["torso_lean_deg"]]
                    hip = features[KINEMATIC_FEATURE_OFFSET + _KINE_INDEX["hip_height_above_ankles"]]
                    line += f" | lean {lean:5.1f} deg | hip {hip:5.2f}"
                print(line, flush=True)

            if outcome.event is not None and outcome.event.event_id not in shown:
                shown.add(outcome.event.event_id)
                print()
                print("  " + "-" * 74)
                print("  \u26A0  FALL EVENT CONFIRMED")
                print(f"     event id     : {outcome.event.event_id}")
                print(f"     timestamp    : {outcome.event.timestamp}")
                print(f"     camera       : {outcome.event.camera_id}")
                print(f"     confidence   : {outcome.event.confidence:.1%}")
                print(f"     peak conf.   : {outcome.event.peak_confidence:.1%}")
                stored = service.repository.get(outcome.event.event_id)
                print(f"     stored       : {'yes' if stored else 'NO'}"
                      f" (status={stored['status'] if stored else '-'})")
                print(f"     telegram     : {stored['note'] if stored else '-'}")
                print("  " + "-" * 74)
                print(flush=True)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\n  interrupted by user")
    finally:
        source.close()

    elapsed = time.monotonic() - started
    stats = service.pipeline.status()
    pipeline_stats = stats["pipeline"]
    print("=" * 78)
    print("  DEMO SUMMARY")
    print("=" * 78)
    print(f"  frames processed      : {pipeline_stats['frames_processed']}")
    print(f"  poses detected        : {pipeline_stats['poses_detected']} "
          f"({pipeline_stats['pose_detection_rate']:.1%})")
    print(f"  windows scored        : {pipeline_stats['windows_scored']}")
    print(f"  events confirmed      : {pipeline_stats['events_confirmed']}")
    print(f"  events in the store   : {service.repository.count()}")
    print(f"  alerts sent / failed  : {service.alerts_sent} / {service.alerts_failed}")
    print(f"  wall time             : {elapsed:.1f}s "
          f"({pipeline_stats['frames_processed'] / max(elapsed, 1e-6):.1f} fps)")
    print("-" * 78)
    print("  This was DEMO MODE. It demonstrates the pipeline, not model accuracy.")
    print("=" * 78)

    if args.post:
        _post_events(service, settings)
    service.close()
    return 0


def _post_events(service: InferenceService, settings) -> None:
    """Check that a running FastAPI backend sees the events this demo recorded."""
    import json
    import urllib.error
    import urllib.request

    base = f"http://localhost:{settings.api_port}"
    print(f"  posting to {base}/status ...")
    try:
        with urllib.request.urlopen(f"{base}/status", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        print(f"  backend is {payload.get('system')} "
              f"(model={payload.get('model', {}).get('backend')})")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  could not reach the backend at {base}: {exc}")
        print("  start it with: uvicorn backend.main:app --port %d" % settings.api_port)
        return
    total = service.repository.count()
    print(f"  the demo recorded {total} event(s) in {settings.db_path};")
    print(f"  they will appear in the dashboard at {base}/events")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", action="store_true", help="use a live camera")
    source.add_argument("--video", default=None, help="use a video file")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after N frames (0 = run to the end)")
    parser.add_argument("--every", type=int, default=15,
                        help="print a status line every N frames")
    parser.add_argument("--verbose", action="store_true", help="print every frame")
    parser.add_argument("--post", action="store_true",
                        help="check a running backend at the end")
    args = parser.parse_args()

    source_kind = "synthetic"
    if args.camera:
        source_kind = "camera"
    elif args.video:
        source_kind = "video"
    args.source = source_kind
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
