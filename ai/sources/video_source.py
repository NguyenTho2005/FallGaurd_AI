"""Frame sources.

A frame source yields ``(frame, timestamp, index)`` BGR triples and nothing else. This
one-method interface is what lets the same pipeline run against a webcam, a video file or
the synthetic renderer, and it is why the demo exercises the real code path rather than
a special-cased shortcut.

Every source handles its own failure mode: a camera that is unplugged, a file that
disappears, or an encoder error must surface as a clean end-of-stream plus a logged
warning, never as a crash in the inference loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

from ai.config import Settings, get_settings
from ai.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SourceFrame:
    """One frame delivered by a source."""

    frame: np.ndarray
    timestamp: float
    index: int
    fps: float

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.frame.shape)


class FrameSource(Protocol):
    """Protocol implemented by every frame source."""

    name: str

    def read(self) -> SourceFrame | None:
        """Return the next frame, or ``None`` when the stream has ended."""
        ...

    def close(self) -> None:
        """Release any underlying resource."""
        ...

    def __enter__(self) -> "FrameSource": ...

    def __exit__(self, *exc_info: object) -> None: ...


class VideoFileSource:
    """Read frames from a video file (``.mp4``, ``.avi``, ...).

    The file is opened lazily on the first :meth:`read`, so constructing the source
    never raises for a missing path; the failure is reported by the first read and by
    :attr:`open_error`.
    """

    def __init__(self, path: str | Path, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self.path = Path(path)
        self.name = f"file:{self.path.name}"
        self._capture = None
        self._index = 0
        self._start = time.monotonic()
        self._fps = 30.0
        self.open_error: str | None = None

    def _ensure_open(self) -> bool:
        if self._capture is not None:
            return True
        if self.open_error:
            return False
        if not self.path.is_file():
            self.open_error = f"Video file not found: {self.path}"
            logger.error(self.open_error)
            return False
        import cv2

        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            self.open_error = f"OpenCV could not open '{self.path}'"
            logger.error(self.open_error)
            capture.release()
            return False
        reported = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not 1.0 <= reported <= 240.0:
            logger.warning(
                "Video reports an implausible FPS (%.2f); assuming %.1f", reported,
                self._settings.target_fps,
            )
            reported = float(self._settings.target_fps)
        self._fps = reported
        self._capture = capture
        self._start = time.monotonic()
        logger.info("Opened video source %s (fps=%.1f)", self.path.name, self._fps)
        return True

    def read(self) -> SourceFrame | None:
        if not self._ensure_open():
            return None
        assert self._capture is not None
        try:
            ok, frame = self._capture.read()
        except Exception:
            logger.error("Error reading frame from %s", self.path, exc_info=True)
            self.close()
            return None
        if not ok or frame is None:
            logger.info("Reached end of %s after %d frames", self.path.name, self._index)
            return None
        timestamp = self._index / self._fps
        source_frame = SourceFrame(frame=frame, timestamp=timestamp, index=self._index, fps=self._fps)
        self._index += 1
        return source_frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.debug("Closed video source %s", self.path.name)

    def __enter__(self) -> "VideoFileSource":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class CameraSource:
    """Read frames from a webcam index or a network stream URL.

    ``source`` may be a device index (``0``, ``1``, ...), an RTSP/HTTP URL, or a path to
    a video file. Reconnection is attempted at most every
    :attr:`reconnect_interval` seconds, so a disconnected camera degrades to a logged
    warning plus ``None`` frames rather than a tight error loop.
    """

    def __init__(
        self,
        source: str | int | None = None,
        settings: Settings | None = None,
        *,
        reconnect_interval: float = 5.0,
    ) -> None:
        self._settings = settings or get_settings()
        self.source = str(source if source is not None else self._settings.camera_source)
        self.name = f"camera:{self.source}"
        self.reconnect_interval = float(reconnect_interval)
        self._capture = None
        self._index = 0
        self._fps = float(self._settings.target_fps)
        self._next_reconnect_attempt = 0.0
        self.is_connected = False
        self.consecutive_failures = 0

    def _parse_source(self) -> int | str:
        text = self.source.strip()
        if text.lstrip("-").isdigit():
            return int(text)
        return text

    def _ensure_open(self) -> bool:
        if self._capture is not None and self._capture.isOpened():
            return True
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        now = time.monotonic()
        if now < self._next_reconnect_attempt:
            return False
        self._next_reconnect_attempt = now + self.reconnect_interval

        import cv2

        target = self._parse_source()
        logger.info("Opening camera source %r", self.source)
        try:
            capture = cv2.VideoCapture(target)
        except Exception:
            logger.error("Could not construct a capture for %r", self.source, exc_info=True)
            return False
        if not capture.isOpened():
            logger.warning("Camera source %r is not available", self.source)
            capture.release()
            self.is_connected = False
            return False
        try:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._settings.frame_width))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._settings.frame_height))
        except Exception:  # pragma: no cover - not all backends support these
            logger.debug("Capture backend ignored the requested frame size", exc_info=True)
        reported = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._fps = reported if 1.0 <= reported <= 240.0 else float(self._settings.target_fps)
        self._capture = capture
        self._index = 0
        self.is_connected = True
        self.consecutive_failures = 0
        logger.info("Camera connected: %s (fps=%.1f)", self.source, self._fps)
        return True

    def read(self) -> SourceFrame | None:
        if not self._ensure_open():
            return None
        assert self._capture is not None
        try:
            ok, frame = self._capture.read()
        except Exception:
            logger.error("Error reading from camera %r", self.source, exc_info=True)
            self.is_connected = False
            self.consecutive_failures += 1
            return None
        if not ok or frame is None or frame.size == 0:
            self.consecutive_failures += 1
            self.is_connected = False
            if self.consecutive_failures % 30 == 1:
                logger.warning(
                    "Camera %r returned no frame (%d consecutive failures); will retry",
                    self.source,
                    self.consecutive_failures,
                )
            # Force a reopen on the next call.
            try:
                self._capture.release()
            except Exception:  # pragma: no cover - defensive
                logger.debug("Failed to release camera capture", exc_info=True)
            self._capture = None
            return None
        timestamp = self._index / self._fps
        source_frame = SourceFrame(frame=frame, timestamp=timestamp, index=self._index, fps=self._fps)
        self._index += 1
        return source_frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self.is_connected = False
        logger.info("Camera %r closed", self.source)

    def __enter__(self) -> "CameraSource":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class SyntheticSource:
    """Yield frames produced by the offline sample generator.

    Used by the demo and the smoke tests. It goes through the same
    :class:`~ai.sources.video_source.CameraSource`-shaped interface as real hardware,
    so MediaPipe, preprocessing, the model and the event detector are all exercised for
    real -- only the light source is simulated.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        settings: Settings | None = None,
        *,
        fps: float = 30.0,
    ) -> None:
        self._settings = settings or get_settings()
        self.path = Path(path) if path else self._settings.samples_dir / "synthetic_room.mp4"
        self.name = f"synthetic:{self.path.name}"
        self._delegate = VideoFileSource(self.path, self._settings)
        self._fps = float(fps)

    def read(self) -> SourceFrame | None:
        frame = self._delegate.read()
        if frame is None:
            return None
        return SourceFrame(
            frame=frame.frame, timestamp=frame.timestamp, index=frame.index, fps=self._fps
        )

    @property
    def open_error(self) -> str | None:
        return self._delegate.open_error

    def close(self) -> None:
        self._delegate.close()

    def __enter__(self) -> "SyntheticSource":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def build_source(
    kind: str,
    *,
    settings: Settings | None = None,
    path: str | Path | None = None,
    camera_source: str | int | None = None,
) -> FrameSource:
    """Factory for the ``--source`` CLI option.

    Parameters
    ----------
    kind:
        ``camera`` | ``video`` | ``synthetic``.
    path:
        Video file for ``video``; ignored for the other kinds.
    """
    settings = settings or get_settings()
    normalized = kind.strip().lower()
    if normalized == "camera":
        return CameraSource(camera_source, settings)
    if normalized == "video":
        if path is None:
            raise ValueError("source 'video' requires a path")
        return VideoFileSource(path, settings)
    if normalized == "synthetic":
        return SyntheticSource(path, settings)
    raise ValueError(
        f"unknown source kind {kind!r}; expected 'camera', 'video' or 'synthetic'"
    )


def iter_frames(source: FrameSource, limit: int | None = None) -> Iterator[SourceFrame]:
    """Iterate a source, stopping at ``limit`` frames and always closing the source."""
    try:
        count = 0
        while limit is None or count < limit:
            source_frame = source.read()
            if source_frame is None:
                return
            yield source_frame
            count += 1
    finally:
        source.close()
