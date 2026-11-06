"""Structured logging for FallGuard AI.

Provides a single ``get_logger(name)`` entry point, a console handler, an optional
rotating file handler and a redaction filter that prevents credentials
(``TELEGRAM_BOT_TOKEN`` and friends) from ever reaching a log record -- including
when they are embedded in a third-party exception message.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import re as _re
import sys
from pathlib import Path
from typing import Any, Iterable

from ai.config import Settings, get_settings

LOGGER_ROOT = "fallguard"

#: Environment variables whose values must never be written to a log.
SECRET_ENV_VARS: tuple[str, ...] = (
    "TELEGRAM_BOT_TOKEN",
    "FALLGUARD_TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "FALLGUARD_TELEGRAM_CHAT_ID",
)

_TOKEN_PATTERNS = (
    # Telegram bot tokens look like 123456789:AAH... The lookbehind rejects only a
    # preceding *digit*, deliberately not a letter and not ``\b``: in a request URL the
    # token directly follows "bot", and "t1234567890" has no word boundary there, so a
    # ``\b``-anchored pattern would leave the token in the log.
    _re.compile(r"(?<!\d)\d{8,12}:[A-Za-z0-9_-]{30,}"),
    # Any ``<name>_TOKEN`` / ``<name>_SECRET`` / ``<name>_PASSWORD`` assignment.
    _re.compile(
        r"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY|CHAT_ID))\b\s*[:=]\s*\S+"
    ),
)


def mask_secrets(text: str) -> str:
    """Replace anything that looks like a credential with ``***REDACTED***``."""
    if not text:
        return text
    masked = text
    for env_var in SECRET_ENV_VARS:
        value = os.environ.get(env_var)
        if value and len(value) >= 6:
            masked = masked.replace(value, "***REDACTED***")
    for pattern in _TOKEN_PATTERNS:
        masked = pattern.sub("***REDACTED***", masked)
    return masked


class SecretRedactionFilter(logging.Filter):
    """Scrub credentials from the message and every argument before formatting."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = mask_secrets(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    key: mask_secrets(value) if isinstance(value, str) else value
                    for key, value in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    mask_secrets(value) if isinstance(value, str) else value
                    for value in record.args
                )
        return True


class ConsoleFormatter(logging.Formatter):
    """Human readable single-line formatter."""

    default_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    def __init__(self, fmt: str | None = None) -> None:
        super().__init__(fmt or self.default_format, datefmt=self.date_format)


class JsonFormatter(logging.Formatter):
    """Machine readable formatter, one JSON object per record."""

    _RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
        "message",
        "asctime",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = mask_secrets(self.formatException(record.exc_info))
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = mask_secrets(value) if isinstance(value, str) else value
        return json.dumps(payload, default=str, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger (``fallguard.<name>``)."""
    if name.startswith(LOGGER_ROOT):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_ROOT}.{name}")


def configure_logging(
    settings: Settings | None = None,
    *,
    force: bool = False,
) -> logging.Logger:
    """Install the FallGuard logging configuration exactly once.

    Returns the root FallGuard logger.
    """
    settings = settings or get_settings()
    root = logging.getLogger(LOGGER_ROOT)
    if root.handlers and not force:
        root.setLevel(settings.log_level)
        return root

    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

    root.setLevel(settings.log_level)
    root.propagate = False

    formatter: logging.Formatter = (
        JsonFormatter() if settings.log_json else ConsoleFormatter()
    )
    redaction = SecretRedactionFilter()

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(redaction)
    root.addHandler(stream_handler)

    if settings.log_file:
        log_path = Path(settings.log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redaction)
        root.addHandler(file_handler)

    # Third-party noise suppression; MediaPipe and TensorFlow are very chatty on
    # stderr and would otherwise bury our own INFO records.
    for noisy in ("mediapipe", "absl", "tensorflow", "matplotlib", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    root.debug("Logging configured (level=%s, json=%s)", settings.log_level, settings.log_json)
    return root


def silence_external_loggers(names: Iterable[str] = ("mediapipe", "absl", "tensorflow")) -> None:
    """Raise the level of noisy third-party loggers (used by tests)."""
    for name in names:
        logging.getLogger(name).setLevel(logging.ERROR)
