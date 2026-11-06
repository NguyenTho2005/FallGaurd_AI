"""Telegram alert message formatting.

Kept separate from the transport so the exact text a human will receive can be unit
tested without a network, a token or a running bot. :func:`format_fall_alert` is pure and
deterministic.
"""

from __future__ import annotations

from typing import Any, Mapping

from ai.inference.fall_detector import EventStatus, FallEvent

ALERT_TITLE = "\U0001F6A8 FALL DETECTED"
PROTOTYPE_NOTICE = (
    "FallGuard AI is a prototype that detects possible fall events from a camera. "
    "It is not a medical device and does not provide a diagnosis. Please verify "
    "manually."
)


def _status_text(status: Any) -> str:
    if isinstance(status, EventStatus):
        return status.value
    return str(status)


def format_fall_alert(
    event: FallEvent,
    *,
    camera_id: str | None = None,
    include_notice: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Render a fall event as a Telegram message.

    The layout follows the specification::

        FALL DETECTED

        Time:
        Camera:
        Confidence:
        Status:
    """
    lines = [
        ALERT_TITLE,
        "",
        f"Time: {event.timestamp}",
        f"Camera: {camera_id or event.camera_id}",
        f"Prediction: {event.prediction}",
        f"Confidence: {event.confidence:.1%}",
        f"Peak confidence: {max(event.peak_confidence, event.confidence):.1%}",
        f"Status: {_status_text(event.status)}",
        f"Event ID: {event.event_id}",
    ]
    for key, value in (extra or {}).items():
        lines.append(f"{key}: {value}")
    if include_notice:
        lines += ["", PROTOTYPE_NOTICE]
    return "\n".join(lines)


def format_status_message(
    *,
    system: str,
    camera: str,
    model: str,
    camera_id: str,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Render a system status line (used by the CLI and the dashboard)."""
    lines = [
        "\U0001F4E1 FallGuard AI status",
        "",
        f"System: {system}",
        f"Camera: {camera} ({camera_id})",
        f"Model: {model}",
    ]
    for key, value in (extra or {}).items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def truncate(text: str, limit: int = 4096) -> str:
    """Clamp a message to Telegram's 4096-character limit."""
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    if len(text) <= limit:
        return text
    suffix = "\n... (truncated)"
    return text[: max(0, limit - len(suffix))] + suffix
