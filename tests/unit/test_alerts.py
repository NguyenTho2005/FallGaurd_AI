"""Unit tests for Telegram alert formatting, transport, dry-run behaviour and redaction."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from ai.config import Settings
from ai.inference.fall_detector import EventStatus
from ai.logging_config import SecretRedactionFilter, mask_secrets
from backend.alerts.formatter import (
    PROTOTYPE_NOTICE,
    format_fall_alert,
    format_status_message,
    truncate,
)
from backend.alerts.telegram import AlertResult, TelegramAlertSender

FAKE_TOKEN = "1234567890:AAFakeTokenValueForUnitTests_0123456789"
FAKE_CHAT_ID = "-1009876543210"


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeClient:
    """Minimal stand-in for ``httpx.Client`` that records requests."""

    def __init__(self, responses: list[FakeResponse | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.urls: list[str] = []

    def post(self, url: str, json: dict[str, Any]) -> FakeResponse:
        self.urls.append(url)
        self.requests.append(json)
        if not self.responses:
            raise AssertionError("FakeClient ran out of scripted responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:  # pragma: no cover - interface parity
        pass


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def test_alert_contains_every_required_field(make_event):
    message = format_fall_alert(make_event("evt_1", confidence=0.947))
    assert "FALL DETECTED" in message
    assert "Time:" in message
    assert "Camera: camera_01" in message
    assert "Confidence: 94.7%" in message
    assert "Status: detected" in message


def test_alert_includes_the_non_medical_disclaimer(make_event):
    message = format_fall_alert(make_event())
    assert PROTOTYPE_NOTICE in message
    assert "not a medical device" in message
    assert "does not provide a diagnosis" in message


def test_alert_disclaimer_can_be_disabled(make_event):
    assert PROTOTYPE_NOTICE not in format_fall_alert(make_event(), include_notice=False)


def test_alert_camera_override(make_event):
    assert "Camera: lobby" in format_fall_alert(make_event(), camera_id="lobby")


@pytest.mark.parametrize("status", list(EventStatus))
def test_alert_status_is_rendered_from_the_enum(status: EventStatus):
    from ai.inference.fall_detector import FallEvent

    event = FallEvent(
        event_id="evt_2", timestamp="2026-01-01T00:00:00Z", camera_id="camera_01",
        prediction="FALL", confidence=0.9, status=status,
    )
    assert f"Status: {status.value}" in format_fall_alert(event)


def test_alert_extra_fields(make_event):
    message = format_fall_alert(make_event(), extra={"Source": "camera_01", "Zone": "kitchen"})
    assert "Source: camera_01" in message
    assert "Zone: kitchen" in message


def test_status_message_formatting():
    message = format_status_message(
        system="ONLINE", camera="CONNECTED", model="LOADED", camera_id="camera_01",
        extra={"Events": 3},
    )
    assert "System: ONLINE" in message
    assert "Camera: CONNECTED (camera_01)" in message
    assert "Model: LOADED" in message
    assert "Events: 3" in message


def test_truncate_respects_the_telegram_limit():
    assert truncate("short") == "short"
    long_text = "x" * 5000
    result = truncate(long_text)
    assert len(result) <= 4096
    assert result.endswith("(truncated)")
    with pytest.raises(ValueError):
        truncate("x", limit=0)


# --------------------------------------------------------------------------- #
# dry run / disabled
# --------------------------------------------------------------------------- #
def test_no_token_means_dry_run(settings: Settings, make_event):
    settings.telegram_bot_token = None
    settings.telegram_chat_id = None
    sender = TelegramAlertSender(settings)
    result = sender.send_fall_alert(make_event())
    assert isinstance(result, AlertResult)
    assert result.sent is False
    assert result.dry_run is True
    assert result.error == "not configured"
    assert "FALL DETECTED" in result.message


def test_disabled_alerts_do_not_send(settings: Settings, make_event):
    settings.telegram_enabled = False
    settings.telegram_bot_token = FAKE_TOKEN
    settings.telegram_chat_id = FAKE_CHAT_ID
    result = TelegramAlertSender(settings).send_fall_alert(make_event())
    assert result.sent is False
    assert result.error == "disabled"
    assert result.dry_run is True


def test_explicit_dry_run_flag(settings: Settings, make_event):
    settings.telegram_bot_token = FAKE_TOKEN
    settings.telegram_chat_id = FAKE_CHAT_ID
    settings.telegram_dry_run = True
    result = TelegramAlertSender(settings).send_fall_alert(make_event())
    assert result.sent is False
    assert result.dry_run is True
    assert result.error == "dry_run"


def test_status_reports_why_alerts_are_not_sent(settings: Settings):
    settings.telegram_bot_token = None
    settings.telegram_chat_id = None
    status = TelegramAlertSender(settings).status()
    assert status["configured"] is False
    assert status["dry_run"] is True
    assert "no token" in status["reason"]
    assert "token" not in str(status).lower() or FAKE_TOKEN not in str(status)


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #
def _configured(settings: Settings) -> Settings:
    settings.telegram_bot_token = FAKE_TOKEN
    settings.telegram_chat_id = FAKE_CHAT_ID
    settings.telegram_dry_run = False
    settings.telegram_max_retries = 1
    return settings


def test_successful_send(settings: Settings, make_event):
    client = FakeClient([FakeResponse(200, {"ok": True, "result": {"message_id": 1}})])
    sender = TelegramAlertSender(_configured(settings), client=client, sleep=lambda _: None)
    result = sender.send_fall_alert(make_event())
    assert result.sent is True
    assert result.status_code == 200
    assert result.attempts == 1
    assert client.requests[0]["chat_id"] == FAKE_CHAT_ID
    assert "FALL DETECTED" in client.requests[0]["text"]
    assert FAKE_TOKEN in client.urls[0], "the token must be in the request URL"


def test_send_failure_is_reported_not_raised(settings: Settings, make_event):
    client = FakeClient([FakeResponse(500, {"ok": False, "description": "server error"})])
    sender = TelegramAlertSender(_configured(settings), client=client, sleep=lambda _: None)
    result = sender.send_fall_alert(make_event())
    assert result.sent is False
    assert result.status_code == 500
    assert "server error" in (result.error or "")


def test_network_exception_is_reported_not_raised(settings: Settings, make_event):
    client = FakeClient([ConnectionError("connection refused")])
    sender = TelegramAlertSender(_configured(settings), client=client, sleep=lambda _: None)
    result = sender.send_fall_alert(make_event())
    assert result.sent is False
    assert "ConnectionError" in (result.error or "")


def test_client_errors_are_not_retried(settings: Settings, make_event):
    client = FakeClient([FakeResponse(400, {"ok": False, "description": "chat not found"})])
    sender = TelegramAlertSender(_configured(settings), client=client, sleep=lambda _: None)
    sender._settings.telegram_max_retries = 3
    result = sender.send_fall_alert(make_event())
    assert result.sent is False
    assert result.attempts == 1, "a 400 will not improve on retry"
    assert len(client.requests) == 1


def test_transient_errors_are_retried_with_backoff(settings: Settings, make_event):
    sleeps: list[float] = []
    client = FakeClient(
        [
            FakeResponse(502, {"ok": False, "description": "bad gateway"}),
            FakeResponse(200, {"ok": True}),
        ]
    )
    sender = TelegramAlertSender(
        _configured(settings), client=client, sleep=sleeps.append
    )
    sender._settings.telegram_max_retries = 3
    result = sender.send_fall_alert(make_event())
    assert result.sent is True
    assert result.attempts == 2
    assert sleeps == [1.0], "backoff should start at one second"
    assert len(client.requests) == 2


def test_exhausted_retries_report_the_last_error(settings: Settings, make_event):
    client = FakeClient(
        [
            FakeResponse(500, {"ok": False, "description": "one"}),
            FakeResponse(500, {"ok": False, "description": "two"}),
        ]
    )
    sender = TelegramAlertSender(
        _configured(settings), client=client, sleep=lambda _: None
    )
    sender._settings.telegram_max_retries = 2
    result = sender.send_fall_alert(make_event())
    assert result.sent is False
    assert result.attempts == 2
    assert "two" in (result.error or "")


def test_non_json_response_is_handled(settings: Settings, make_event):
    class BadJson(FakeResponse):
        def json(self) -> dict[str, Any]:
            raise ValueError("not json")

    client = FakeClient([BadJson(200, {})])
    sender = TelegramAlertSender(_configured(settings), client=client, sleep=lambda _: None)
    result = sender.send_fall_alert(make_event())
    assert result.sent is False
    assert "not JSON" in (result.error or "")


# --------------------------------------------------------------------------- #
# secret redaction
# --------------------------------------------------------------------------- #
def test_mask_secrets_removes_a_telegram_token():
    text = f"failed to post to https://api.telegram.org/bot{FAKE_TOKEN}/sendMessage"
    masked = mask_secrets(text)
    assert FAKE_TOKEN not in masked
    assert "REDACTED" in masked


def test_mask_secrets_removes_generic_credential_assignments():
    assert "hunter2" not in mask_secrets("API_KEY=hunter2")
    assert "s3cr3t" not in mask_secrets("db_password: s3cr3t")


def test_mask_secrets_leaves_ordinary_text_alone():
    message = "FALL EVENT CONFIRMED event_id=abc123 confidence=0.947"
    assert mask_secrets(message) == message


def test_mask_secrets_handles_empty_input():
    assert mask_secrets("") == ""


def test_redaction_filter_scrubs_message_and_args():
    # The message embeds the token directly; the token also travels as a %s argument.
    record = logging.LogRecord(
        "test", logging.INFO, __file__, 1,
        f"posting with token {FAKE_TOKEN} to %s", ("chat",), None,
    )
    assert SecretRedactionFilter().filter(record) is True
    assert FAKE_TOKEN not in record.getMessage()
    assert "REDACTED" in record.getMessage()

    arg_record = logging.LogRecord(
        "test", logging.INFO, __file__, 1, "posting to %s", (f"bot{FAKE_TOKEN}",), None
    )
    SecretRedactionFilter().filter(arg_record)
    assert FAKE_TOKEN not in arg_record.getMessage()


@pytest.fixture
def captured_logs():
    """Capture from the fallguard logger.

    It sets ``propagate = False``, so pytest's own ``caplog`` (which hooks the root
    logger) never sees it; a handler has to be attached to it directly.
    """
    records: list[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Collector()
    handler.addFilter(SecretRedactionFilter())
    logger = logging.getLogger("fallguard")
    previous_level, previous_propagate = logger.level, logger.propagate
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


def test_exception_text_with_a_token_is_redacted(captured_logs):
    logging.getLogger("fallguard.test").error("request failed for bot %s", FAKE_TOKEN)
    text = " ".join(record.getMessage() for record in captured_logs)
    assert FAKE_TOKEN not in text
    assert "REDACTED" in text


def test_successful_send_does_not_log_the_token(settings, make_event, captured_logs):
    client = FakeClient([FakeResponse(200, {"ok": True})])
    sender = TelegramAlertSender(_configured(settings), client=client, sleep=lambda _: None)
    sender.send_fall_alert(make_event())
    text = " ".join(record.getMessage() for record in captured_logs)
    assert text, "the sender should log something on success"
    assert FAKE_TOKEN not in text
    assert FAKE_CHAT_ID not in text, "the chat id must be masked in logs"
    assert "***3210" in text
