"""Fall event confirmation.

A single ``FALL`` prediction must never page a human. This module turns a stream of
per-window fall probabilities into confirmed :class:`FallEvent` objects using three
configurable guards, all sourced from :class:`~ai.config.Settings`:

``CONFIDENCE_THRESHOLD``
    Minimum probability for a window to count as evidence at all.
``MIN_FALL_FRAMES``
    Number of *consecutive* evidence windows required before an event is emitted. This
    is the temporal filter: it rejects an isolated spike caused by a bad landmark, a
    passer-by or a detection glitch.
``COOLDOWN_SECONDS``
    Lockout after a confirmed event, so one physical fall produces one alert rather
    than one per window for the next 30 seconds.

The state machine is deliberately explicit and clock-injectable, which is what makes it
unit-testable without sleeping.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from ai.config import Settings, get_settings
from ai.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_CLOCK: Callable[[], float] = time.monotonic


class EventStatus(str, Enum):
    """Lifecycle of a stored event."""

    DETECTED = "detected"
    ALERTED = "alerted"
    ACKNOWLEDGED = "acknowledged"
    FALSE_POSITIVE = "false_positive"

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.value


@dataclass(frozen=True, slots=True)
class FallEvent:
    """A confirmed fall event.

    Only :class:`~ai.inference.fall_detector.FallDetector` constructs these; the
    confirmation policy lives in exactly one place.
    """

    event_id: str
    timestamp: str
    camera_id: str
    prediction: str
    confidence: float
    status: EventStatus = EventStatus.DETECTED
    peak_confidence: float = 0.0
    window_frames: int = 0
    source: str = ""
    notified: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "camera_id": self.camera_id,
            "prediction": self.prediction,
            "confidence": round(float(self.confidence), 6),
            "status": self.status.value,
            "peak_confidence": round(float(self.peak_confidence), 6),
            "window_frames": self.window_frames,
            "source": self.source,
            "notified": self.notified,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Outcome of feeding one window probability into the detector."""

    probability: float
    above_threshold: bool
    consecutive_frames: int
    event: FallEvent | None = None
    in_cooldown: bool = False
    sequence_ready: bool = True

    @property
    def confirmed(self) -> bool:
        return self.event is not None

    @property
    def label(self) -> str:
        return "FALL" if self.above_threshold else "NORMAL"


@dataclass(slots=True)
class DetectorStats:
    """Counters exposed through ``/status``."""

    windows_evaluated: int = 0
    above_threshold: int = 0
    events_confirmed: int = 0
    windows_suppressed_by_cooldown: int = 0
    windows_skipped_no_signal: int = 0
    history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "windows_evaluated": self.windows_evaluated,
            "above_threshold": self.above_threshold,
            "events_confirmed": self.events_confirmed,
            "windows_suppressed_by_cooldown": self.windows_suppressed_by_cooldown,
            "windows_skipped_no_signal": self.windows_skipped_no_signal,
        }


class FallDetector:
    """Confirm fall events from a stream of window probabilities.

    Parameters
    ----------
    settings:
        Source of ``confidence_threshold``, ``min_fall_frames`` and
        ``cooldown_seconds``.
    clock:
        Monotonic time source, injected by tests. Defaults to :func:`time.monotonic`.
    timestamp_fn:
        Wall-clock formatter for event timestamps (ISO-8601, UTC).
    camera_id:
        Camera recorded on each event; the pipeline keeps this in sync with config.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        clock: Callable[[], float] = DEFAULT_CLOCK,
        timestamp_fn: Callable[[], str] | None = None,
        camera_id: str | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._clock = clock
        self._timestamp_fn = timestamp_fn or _utc_now_iso
        self.camera_id = camera_id or self._settings.camera_id
        self.threshold = float(self._settings.confidence_threshold)
        self.min_fall_frames = int(self._settings.min_fall_frames)
        self.cooldown_seconds = float(self._settings.cooldown_seconds)
        self._consecutive = 0
        self._cooldown_until = 0.0
        self._peak = 0.0
        self.stats = DetectorStats()

    # ------------------------------------------------------------------ #
    def update(self, probability: float, *, has_signal: bool = True) -> DetectionResult:
        """Feed one window probability and return the detection outcome.

        Parameters
        ----------
        probability:
            Fall probability in ``[0, 1]`` for the most recent window.
        has_signal:
            ``False`` when the window could not be scored (not enough frames, or too
            many pose dropouts). Such a window is counted and skipped: absence of
            evidence must not be treated as evidence.
        """
        probability = float(probability)
        if not 0.0 <= probability <= 1.0:
            logger.warning("Clamping out-of-range fall probability %.3f into [0, 1]", probability)
            probability = min(max(probability, 0.0), 1.0)

        if not has_signal:
            self._reset_run()
            self.stats.windows_evaluated += 1
            self.stats.windows_skipped_no_signal += 1
            return DetectionResult(
                probability=probability,
                above_threshold=False,
                consecutive_frames=0,
                sequence_ready=False,
            )

        self.stats.windows_evaluated += 1
        above = probability >= self.threshold
        if above:
            self.stats.above_threshold += 1
            self._peak = max(self._peak, probability)
        else:
            self._peak = 0.0

        in_cooldown = self._clock() < self._cooldown_until
        if in_cooldown:
            # Inside the lockout nothing can fire, and the run is cleared so the first
            # window after the cooldown starts a fresh run.
            self.stats.windows_suppressed_by_cooldown += 1
            self._reset_run()
            return DetectionResult(
                probability=probability,
                above_threshold=above,
                consecutive_frames=0,
                in_cooldown=True,
            )

        if above:
            self._consecutive += 1
        else:
            self._consecutive = 0
            self._peak = 0.0

        event: FallEvent | None = None
        if self._consecutive >= self.min_fall_frames:
            event = FallEvent(
                event_id=uuid.uuid4().hex,
                timestamp=self._timestamp_fn(),
                camera_id=self.camera_id,
                prediction="FALL",
                confidence=probability,
                peak_confidence=max(self._peak, probability),
                window_frames=self._consecutive,
            )
            self.stats.events_confirmed += 1
            self._consecutive = 0
            self._peak = 0.0
            self._cooldown_until = self._clock() + self.cooldown_seconds
            logger.warning(
                "FALL EVENT CONFIRMED: event_id=%s camera=%s confidence=%.3f peak=%.3f "
                "frames=%d cooldown=%.1fs",
                event.event_id,
                event.camera_id,
                event.confidence,
                event.peak_confidence,
                event.window_frames,
                self.cooldown_seconds,
            )
        return DetectionResult(
            probability=probability,
            above_threshold=above,
            consecutive_frames=self._consecutive,
            event=event,
        )

    def _reset_run(self) -> None:
        self._consecutive = 0
        self._peak = 0.0

    # ------------------------------------------------------------------ #
    @property
    def in_cooldown(self) -> bool:
        return self._clock() < self._cooldown_until

    @property
    def consecutive_frames(self) -> int:
        return self._consecutive

    def cooldown_remaining(self) -> float:
        """Seconds until the next event may be confirmed (0.0 when not cooling down)."""
        return max(0.0, self._cooldown_until - self._clock())

    def reset(self) -> None:
        """Clear all state, including the cooldown. Used between test cases."""
        self._consecutive = 0
        self._peak = 0.0
        self._cooldown_until = 0.0
        self.stats = DetectorStats()


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with second precision."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
