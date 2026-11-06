"""Integration tests for the service layer: event persistence, alerting and the monitor."""

from __future__ import annotations

import json

import pytest

from ai.config import Settings
from ai.inference.fall_detector import EventStatus, FallEvent
from backend.alerts.telegram import AlertResult
from backend.services.inference_service import InferenceService
from tests.conftest import ON_FLOOR, STANDING, landmark_array

pytestmark = pytest.mark.integration


class RecordingAlerter:
    """Captures alerts instead of sending them."""

    def __init__(self, *, result: AlertResult | None = None, raises: bool = False) -> None:
        self.sent: list[FallEvent] = []
        self._result = result
        self._raises = raises

    def send_fall_alert(self, event: FallEvent, **kwargs: object) -> AlertResult:
        if self._raises:
            raise RuntimeError("telegram is down")
        self.sent.append(event)
        return self._result or AlertResult(sent=True, dry_run=False, message="ok")

    def status(self) -> dict[str, object]:
        return {"enabled": True, "configured": True, "dry_run": False, "reason": "mock",
                "api_base": "https://example.invalid"}

    def close(self) -> None:
        pass


@pytest.fixture
def service(settings: Settings) -> InferenceService:
    instance = InferenceService(settings)
    yield instance
    instance.close()


def test_confirmed_event_is_persisted_then_alerted(service: InferenceService, settings: Settings):
    alerter = RecordingAlerter()
    service.alerter = alerter
    event = FallEvent(
        event_id="evt_1", timestamp="2026-01-01T00:00:00Z", camera_id="camera_01",
        prediction="FALL", confidence=0.91,
    )
    result = service.handle_event(event)
    assert result["persisted"] is True
    assert alerter.sent and alerter.sent[0].event_id == "evt_1"
    stored = service.repository.get("evt_1")
    assert stored is not None
    assert stored["status"] == EventStatus.ALERTED.value
    assert stored["notified"] is True
    assert service.alerts_sent == 1


def test_event_is_persisted_even_when_the_alert_fails(
    service: InferenceService, settings: Settings
):
    """A Telegram outage must never cost us the event."""
    service.alerter = RecordingAlerter(
        result=AlertResult(sent=False, dry_run=False, message="", error="HTTP 500")
    )
    event = FallEvent(
        event_id="evt_2", timestamp="2026-01-01T00:00:00Z", camera_id="camera_01",
        prediction="FALL", confidence=0.88,
    )
    result = service.handle_event(event)
    assert result["persisted"] is True
    stored = service.repository.get("evt_2")
    assert stored is not None, "the event must survive a failed alert"
    assert stored["status"] == EventStatus.DETECTED.value, "it must not claim to be alerted"
    assert stored["note"] == "HTTP 500"
    assert service.alerts_failed == 1


def test_alert_exception_does_not_propagate(service: InferenceService):
    service.alerter = RecordingAlerter(raises=True)
    event = FallEvent(
        event_id="evt_3", timestamp="2026-01-01T00:00:00Z", camera_id="camera_01",
        prediction="FALL", confidence=0.9,
    )
    result = service.handle_event(event)  # must not raise
    assert result["persisted"] is True
    assert service.repository.get("evt_3") is not None
    assert service.alerts_failed == 1


def test_persistence_failure_is_reported_not_raised(service: InferenceService):
    def explode(event: FallEvent) -> str:
        raise RuntimeError("database is locked")

    service.repository.add = explode  # type: ignore[method-assign]
    event = FallEvent(
        event_id="evt_4", timestamp="2026-01-01T00:00:00Z", camera_id="camera_01",
        prediction="FALL", confidence=0.9,
    )
    result = service.handle_event(event)
    assert result["persisted"] is False
    assert result["alert"]["sent"] is False


def test_full_loop_from_landmarks_to_stored_event(service: InferenceService, settings: Settings):
    alerter = RecordingAlerter()
    service.alerter = alerter
    service.pipeline.detector.min_fall_frames = 3
    service.pipeline.detector.cooldown_seconds = 60.0

    confirmed: list[FallEvent] = []
    # Wrap the service's own callback so persistence and alerting still run.
    original = service.pipeline._on_event
    assert original is not None
    service.pipeline._on_event = lambda event: (confirmed.append(event), original(event))

    for index in range(settings.sequence_length + 5):
        service.process_landmarks(landmark_array(ON_FLOOR), frame_index=index)

    assert confirmed, "no fall event was produced"
    assert service.repository.get(confirmed[0].event_id) is not None
    assert alerter.sent


def test_standing_activity_produces_no_events(service: InferenceService, settings: Settings):
    for index in range(settings.sequence_length + 5):
        service.process_landmarks(landmark_array(STANDING), frame_index=index)
    assert service.repository.count() == 0


def test_status_reports_the_whole_stack(service: InferenceService, settings: Settings):
    status = service.status()
    assert status["system"] == "online"
    assert status["model"]["loaded"] is True
    assert status["model"]["sequence_length"] == settings.sequence_length
    assert status["camera"]["id"] == settings.camera_id
    assert status["config"]["store_raw_video"] is False
    assert status["telegram"]["dry_run"] is True
    assert "events" in status and "pipeline" in status and "detector" in status


def test_latest_pose_payload_carries_no_imagery(service: InferenceService, settings: Settings):
    assert service.latest_pose() is None
    service.process_landmarks(landmark_array(STANDING), frame_index=0)
    payload = service.latest_pose()
    assert payload is not None
    assert len(payload["landmarks"]) == 33
    serialised = json.dumps(payload).lower()
    for forbidden in ("image", "jpeg", "png", "frame_bytes", "b64"):
        assert forbidden not in serialised


def test_monitor_is_not_started_when_disabled(service: InferenceService):
    service.start_monitor("camera")
    assert service.monitor is None
    assert service.status()["camera"]["monitor_enabled"] is False


def test_monitor_can_be_started_and_stopped(service: InferenceService, settings: Settings,
                                            tmp_path):
    settings.enable_monitor = True
    service.start_monitor("video", path=str(tmp_path / "absent.avi"))
    monitor = service.monitor
    assert monitor is not None
    assert monitor.is_running
    service.stop_monitor()
    assert not monitor.is_running


def test_close_is_idempotent(settings: Settings):
    service = InferenceService(settings)
    service.close()
    service.close()
