"""Telegram Bot transport.

Design constraints
------------------
* **No credential ever appears in a log.** The URL is assembled in memory and every log
  line names the *host* only; the logging layer additionally redacts anything that looks
  like a bot token.
* **Dry run is the default.** With no token configured the alert is rendered, logged and
  returned with ``sent=False``, so the pipeline is fully exercisable offline.
* **Failure is not failure of the event.** The event is already persisted by the time an
  alert is attempted; a Telegram outage must not lose it or crash the pipeline. Retries
  use exponential backoff and every failure is logged with its cause.

The HTTP call goes through an injectable :class:`httpx.Client`, so tests exercise the
real code path against a mock transport with no network access.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

from ai.config import Settings, get_settings
from ai.inference.fall_detector import EventStatus, FallEvent
from ai.logging_config import get_logger
from backend.alerts.formatter import format_fall_alert, truncate

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AlertResult:
    """Outcome of one alert attempt."""

    sent: bool
    dry_run: bool
    message: str
    status_code: int | None = None
    error: str | None = None
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sent": self.sent,
            "dry_run": self.dry_run,
            "status_code": self.status_code,
            "error": self.error,
            "attempts": self.attempts,
        }


class TelegramAlertSender:
    """Send fall alerts to a Telegram chat.

    Parameters
    ----------
    settings:
        Supplies the token, chat id, base URL, timeout and retry count. The token is read
        from ``FALLGUARD_TELEGRAM_BOT_TOKEN`` (or ``TELEGRAM_BOT_TOKEN``), never from code.
    client:
        Optional pre-built :class:`httpx.Client`. Injected by tests with a mock transport.
    sleep:
        Injected sleep function, so retry backoff is instant under test.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._owns_client = client is None
        self._sleep = sleep

    # ------------------------------------------------------------------ #
    @property
    def is_configured(self) -> bool:
        return self._settings.telegram_configured

    @property
    def is_dry_run(self) -> bool:
        """``True`` when the alert is rendered but not transmitted."""
        if not self._settings.telegram_enabled:
            return True
        if self._settings.telegram_dry_run:
            return True
        return not self.is_configured

    def status(self) -> dict[str, Any]:
        """Non-secret alert configuration, for ``GET /status``."""
        reason = "ok"
        if not self._settings.telegram_enabled:
            reason = "disabled by configuration"
        elif not self.is_configured:
            reason = "no token/chat id configured"
        elif self._settings.telegram_dry_run:
            reason = "dry_run enabled by configuration"
        return {
            "enabled": bool(self._settings.telegram_enabled),
            "configured": self.is_configured,
            "dry_run": self.is_dry_run,
            "reason": reason,
            "api_base": self._settings.telegram_api_base,
        }

    # ------------------------------------------------------------------ #
    def send_fall_alert(
        self,
        event: FallEvent,
        *,
        camera_id: str | None = None,
        extra: Mapping[str, Any] | None = None,
        include_notice: bool = True,
        parse_mode: str | None = None,
    ) -> AlertResult:
        """Send (or render) an alert for a confirmed fall event."""
        if not self._settings.telegram_enabled:
            message = format_fall_alert(event, camera_id=camera_id, extra=extra)
            logger.info("Telegram alerts are disabled; alert not sent:\n%s", message)
            return AlertResult(sent=False, dry_run=True, message=message, error="disabled")

        message = truncate(format_fall_alert(event, camera_id=camera_id, extra=extra,
                                             include_notice=include_notice))

        if not self.is_configured:
            logger.info(
                "TELEGRAM DRY RUN -- no bot token/chat id configured, alert not sent:\n%s",
                message,
            )
            return AlertResult(
                sent=False, dry_run=True, message=message, error="not configured"
            )
        if self._settings.telegram_dry_run:
            logger.info("TELEGRAM DRY RUN -- alert not sent:\n%s", message)
            return AlertResult(
                sent=False, dry_run=True, message=message, error="dry_run"
            )
        return self._send(message, parse_mode=parse_mode)

    def _send(self, message: str, *, parse_mode: str | None = None) -> AlertResult:
        payload: dict[str, Any] = {
            "chat_id": self._settings.telegram_chat_id,
            "text": message,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        # The token is part of the URL; log the host only, never the full URL.
        host = self._settings.telegram_api_base
        attempts = 0
        last_error = "unknown"
        status_code: int | None = None

        for attempt in range(1, self._settings.telegram_max_retries + 1):
            attempts = attempt
            try:
                status_code, body = self._post(payload, host)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Telegram alert attempt %d/%d failed (%s)", attempt,
                    self._settings.telegram_max_retries, last_error,
                )
            else:
                if status_code == 200 and body.get("ok"):
                    logger.info(
                        "Telegram alert sent to chat %s (event_id=%s)",
                        _mask_chat_id(self._settings.telegram_chat_id), _short(event_id_from(message)),
                    )
                    return AlertResult(
                        sent=True, dry_run=False, message=message,
                        status_code=status_code, attempts=attempt,
                    )
                last_error = f"HTTP {status_code}: {body.get('description', 'unknown error')}"
                logger.warning(
                    "Telegram rejected the alert (HTTP %s): %s", status_code,
                    body.get("description", "no description"),
                )
                if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                    # Client errors will not improve on retry.
                    break
            if attempt < self._settings.telegram_max_retries:
                self._sleep(min(2.0 ** (attempt - 1), 8.0))

        logger.error("Telegram alert failed after %d attempt(s): %s", attempts, last_error)
        return AlertResult(
            sent=False, dry_run=False, message=message, status_code=status_code,
            error=last_error, attempts=attempts,
        )

    def _post(self, payload: dict[str, Any], host: str) -> tuple[int, dict[str, Any]]:
        import httpx

        url = f"{host}/bot{self._settings.telegram_bot_token}/sendMessage"
        client = self._client
        if client is None:
            with httpx.Client(timeout=self._settings.telegram_timeout_seconds) as owned:
                return self._do_post(owned, url, payload)
        return self._do_post(client, url, payload)

    @staticmethod
    def _do_post(client: Any, url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        response = client.post(url, json=payload)
        status_code = int(response.status_code)
        try:
            body = response.json()
        except Exception:
            body = {"ok": False, "description": "response was not JSON"}
        if not isinstance(body, dict):
            body = {"ok": False, "description": "unexpected response body"}
        return status_code, body

    def close(self) -> None:
        if self._owns_client and self._client is not None:  # pragma: no cover - defensive
            try:
                self._client.close()
            except Exception:
                logger.warning("Error closing the Telegram HTTP client", exc_info=True)
        self._client = None

    def __enter__(self) -> "TelegramAlertSender":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _mask_chat_id(chat_id: str | None) -> str:
    """Show only the tail of a chat id -- enough to debug, not enough to be a secret."""
    if not chat_id:
        return "unknown"
    tail = chat_id[-4:]
    return f"***{tail}" if len(chat_id) > 4 else tail


def _short(value: str | None) -> str:
    return (value or "")[:8]


def event_id_from(message: str) -> str | None:  # pragma: no cover - log helper
    for line in message.splitlines():
        if line.startswith("Event ID:"):
            return line.split(":", 1)[1].strip()
    return None


def build_sender(settings: Settings | None = None) -> TelegramAlertSender:
    """Factory used by the application wiring."""
    return TelegramAlertSender(settings or get_settings())
