#!/usr/bin/env python
"""Download the MediaPipe pose landmarker asset required by :mod:`ai.pose.pose_estimator`.

MediaPipe 1.x does not bundle the model, so it must be fetched once:

    python scripts/download_mediapipe_model.py
    python scripts/download_mediapipe_model.py --variant full --force
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ai.config import get_settings  # noqa: E402
from ai.logging_config import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)

BASE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker"
VARIANTS = {
    "lite": "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}
#: Approximate download sizes, used only for the progress log.
SIZES_MB = {"lite": 5.5, "full": 9.3, "heavy": 29.9}


def download(variant: str = "lite", *, destination: Path | None = None,
             force: bool = False) -> Path:
    """Fetch the pose landmarker asset and return the local path."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
    settings = get_settings()
    target = Path(destination) if destination else Path(settings.pose_model_path)
    url = f"{BASE_URL}/{VARIANTS[variant]}"

    if target.is_file() and not force:
        size_mb = target.stat().st_size / (1024 * 1024)
        logger.info("Pose asset already present at %s (%.1f MB); use --force to refetch",
                    target, size_mb)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading the MediaPipe %s pose landmarker (~%.1f MB)...",
                variant, SIZES_MB[variant])
    logger.info("Source: %s", url)
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            if response.status != 200:  # pragma: no cover - urlopen raises instead
                raise RuntimeError(f"unexpected HTTP status {response.status}")
            total = int(response.headers.get("Content-Length") or 0)
            written = 0
            with temporary.open("wb") as handle:
                while True:
                    chunk = response.read(1 << 16)
                    if not chunk:
                        break
                    handle.write(chunk)
                    written += len(chunk)
                    if total:
                        logger.debug("  %d/%d bytes (%.0f%%)", written, total,
                                     100.0 * written / total)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the pose asset from {url}: {exc}. "
            f"Download it manually to {target}."
        ) from exc

    if written < 1_000_000:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded only {written} bytes; that cannot be a valid pose asset. "
            f"Check the network and retry, or fetch it manually to {target}."
        )

    temporary.replace(target)
    logger.info("Saved pose asset to %s (%.1f MB)", target, target.stat().st_size / (1024 * 1024))
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="lite",
                        help="model size/speed trade-off (default: lite)")
    parser.add_argument("--destination", default=None, help="override the output path")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    configure_logging()
    try:
        path = download(args.variant, destination=args.destination, force=args.force)
    except (RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    print(f"MediaPipe pose asset ready: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
