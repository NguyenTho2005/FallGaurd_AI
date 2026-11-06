"""API tests for the FallGuard FastAPI service.

Uses FastAPI's in-process ``TestClient``, so no server, no network and no GPU. The model
backend is the deterministic mock, so no trained checkpoint is required.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ai.config import Settings
from ai.inference.fall_detector import EventStatus
from backend.main import create_app
from tests.conftest import ON_FLOOR, STANDING, landmark_array

pytestmark = pytest.mark.integration


class _PoseUnavailable:
    """Stands in for a missing MediaPipe asset.

    The API must remain usable when the pose estimator cannot start: clients can still
    post landmark arrays. Substituting this also keeps the suite fast, since loading the
    native landmarker takes seconds per app instance.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("MediaPipe is not available in this test")

    def close(self) -> None:  # pragma: no cover - never constructed
        pass


@pytest.fixture
def client(settings: Settings, monkeypatch) -> Iterator[TestClient]:
    import ai.inference.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "PoseEstimator", _PoseUnavailable)
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def landmarks(points: dict[int, tuple[float, float]]) -> list[list[float]]:
    return landmark_array(points).astype(float).tolist()


def fill_buffer(client: TestClient, points, count: int) -> None:
    payload = {"landmarks": landmarks(points)}
    for _ in range(count):
        client.post("/predict", json=payload)


# --------------------------------------------------------------------------- #
# health & index
# --------------------------------------------------------------------------- #
def test_index_lists_the_endpoints(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "FallGuard AI"
    assert "GET /health" in body["endpoints"]
    assert "Not a medical device" in body["disclaimer"]


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["service"] == "FallGuard AI"
    assert body["timestamp"].endswith("Z")


def test_health_does_not_depend_on_the_camera(client: TestClient):
    assert client.get("/health").status_code == 200


def test_openapi_schema_is_served(client: TestClient):
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    paths = schema.json()["paths"]
    for route in ("/health", "/status", "/predict", "/events", "/events/{event_id}"):
        assert route in paths


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
def test_status_shape(client: TestClient):
    response = client.get("/status")
    assert response.status_code == 200
    body = response.json()
    assert body["system"] == "online"
    assert body["model"]["loaded"] is True
    assert body["model"]["sequence_length"] > 0
    assert body["camera"]["id"] == "camera_01"
    assert body["pipeline"]["frames_processed"] == 0
    assert body["detector"]["min_fall_frames"] >= 1
    assert body["events"]["total"] == 0
    assert body["last_prediction"] is None


def test_status_reports_the_privacy_posture(client: TestClient):
    privacy = client.get("/status").json()["privacy"]
    assert privacy["raw_video_stored"] == "never"
    assert "not implemented" in privacy["identity_recognition"]
    assert "not a medical device" in privacy["disclaimer"]


def test_status_flags_the_mock_backend(client: TestClient):
    model = client.get("/status").json()["model"]
    assert model["backend"] == "mock"
    assert "not a trained model" in model["detail"]


def test_status_telegram_is_not_configured_by_default(client: TestClient):
    telegram = client.get("/status").json()["telegram"]
    assert telegram["enabled"] is True
    assert telegram["configured"] is False
    assert telegram["dry_run"] is True


# --------------------------------------------------------------------------- #
# predict
# --------------------------------------------------------------------------- #
def test_predict_without_landmarks_and_without_a_buffer_is_409(client: TestClient):
    response = client.post("/predict", json={})
    assert response.status_code == 409
    assert "buffer is not ready" in response.json()["detail"]


def test_predict_rejects_a_wrong_landmark_count(client: TestClient):
    response = client.post("/predict", json={"landmarks": [[0.1, 0.2, 0.0, 1.0]] * 10})
    assert response.status_code == 422


def test_predict_rejects_a_wrong_landmark_width(client: TestClient):
    response = client.post("/predict", json={"landmarks": [[0.1, 0.2, 1.0]] * 33})
    assert response.status_code == 422


def test_predict_rejects_unknown_fields(client: TestClient):
    response = client.post("/predict", json={"landmarks": None, "surprise": 1})
    assert response.status_code == 422


def test_predict_rejects_an_out_of_range_threshold(client: TestClient):
    response = client.post("/predict", json={"threshold": 1.5})
    assert response.status_code == 422


def test_predict_returns_the_documented_shape(client: TestClient, settings: Settings):
    fill_buffer(client, STANDING, 3)
    response = client.post("/predict", json={"landmarks": landmarks(STANDING)})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"FALL", "NORMAL", "UNKNOWN"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["fall_probability"] <= 1.0
    assert body["timestamp"]
    assert body["camera_id"] == "camera_01"
    assert body["model_backend"] == "mock"
    assert isinstance(body["event_confirmed"], bool)
    assert "message" in body


def test_predict_reports_buffering_until_the_window_is_full(
    client: TestClient, settings: Settings
):
    response = client.post("/predict", json={"landmarks": landmarks(STANDING)})
    body = response.json()
    assert body["sequence_ready"] is False
    assert "buffering" in body["message"]


def test_predict_reports_a_fall_once_the_window_is_full(
    client: TestClient, settings: Settings
):
    fill_buffer(client, ON_FLOOR, settings.sequence_length + 2)
    body = client.post("/predict", json={"landmarks": landmarks(ON_FLOOR)}).json()
    assert body["sequence_ready"] is True
    assert body["pose_detected"] is True
    assert body["fall_probability"] > 0.5
    assert body["prediction"] == "FALL"


def test_predict_confirms_an_event_and_stores_it(
    client: TestClient, settings: Settings, make_pipeline=None
):
    settings.min_fall_frames = 3
    settings.cooldown_seconds = 60.0
    client.app.state.service.pipeline.detector.min_fall_frames = 3
    client.app.state.service.pipeline.detector.cooldown_seconds = 60.0

    # The confirmation fires on one specific request; the cooldown then suppresses the
    # rest, so every response has to be inspected rather than only the last one.
    confirmed: dict = {}
    for _ in range(settings.sequence_length + 8):
        body = client.post("/predict", json={"landmarks": landmarks(ON_FLOOR)}).json()
        if body["event_confirmed"]:
            confirmed = body
            break
    assert confirmed, "no fall event was confirmed"
    assert confirmed["event_id"]

    events = client.get("/events").json()
    assert events["total"] == 1
    assert events["events"][0]["event_id"] == confirmed["event_id"]


def test_predict_without_landmarks_scores_the_buffer(
    client: TestClient, settings: Settings
):
    fill_buffer(client, ON_FLOOR, settings.sequence_length + 2)
    response = client.post("/predict", json={})
    assert response.status_code == 200
    assert response.json()["sequence_ready"] is True


def test_buffer_rescore_does_not_advance_the_detector(
    client: TestClient, settings: Settings
):
    """Re-reading the buffer must not double-count evidence."""
    fill_buffer(client, ON_FLOOR, settings.sequence_length + 2)
    before = client.get("/status").json()["detector"]["windows_evaluated"]
    for _ in range(5):
        assert client.post("/predict", json={}).status_code == 200
    after = client.get("/status").json()["detector"]["windows_evaluated"]
    assert after == before


# --------------------------------------------------------------------------- #
# events
# --------------------------------------------------------------------------- #
def test_events_is_empty_initially(client: TestClient):
    body = client.get("/events").json()
    assert body == {"total": 0, "limit": 50, "offset": 0, "events": []}


def test_events_pagination_validation(client: TestClient):
    assert client.get("/events", params={"limit": 0}).status_code == 422
    assert client.get("/events", params={"limit": 5000}).status_code == 422
    assert client.get("/events", params={"offset": -1}).status_code == 422


def test_unknown_event_is_404(client: TestClient):
    response = client.get("/events/does_not_exist")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_event_lifecycle(client: TestClient, settings: Settings):
    from ai.inference.fall_detector import FallEvent

    service = client.app.state.service
    event = FallEvent(
        event_id="evt_lifecycle",
        timestamp="2026-01-01T00:00:00Z",
        camera_id="camera_01",
        prediction="FALL",
        confidence=0.93,
        peak_confidence=0.97,
        window_frames=8,
    )
    service.repository.add(event)

    fetched = client.get("/events/evt_lifecycle")
    assert fetched.status_code == 200
    assert fetched.json()["confidence"] == pytest.approx(0.93)
    assert fetched.json()["status"] == "detected"

    patched = client.patch("/events/evt_lifecycle", json={"status": "acknowledged"})
    assert patched.status_code == 200
    assert patched.json()["status"] == "acknowledged"

    listed = client.get("/events", params={"status": "acknowledged"}).json()
    assert listed["total"] == 1

    patched = client.patch(
        "/events/evt_lifecycle",
        json={"status": "false_positive", "note": "verified by a human"},
    )
    assert patched.status_code == 200
    assert patched.json()["note"] == "verified by a human"


def test_patch_unknown_event_is_404(client: TestClient):
    assert client.patch("/events/nope", json={"status": "acknowledged"}).status_code == 404


def test_patch_rejects_an_unknown_status(client: TestClient, settings: Settings, make_event):
    service = client.app.state.service
    service.repository.add(make_event("evt_x"))
    response = client.patch("/events/evt_x", json={"status": "deleted"})
    assert response.status_code == 422


def test_events_filter_by_camera(client: TestClient, settings: Settings, make_event):
    service = client.app.state.service
    service.repository.add(make_event("evt_a", camera_id="camera_01"))
    service.repository.add(make_event("evt_b", camera_id="camera_02"))
    filtered = client.get("/events", params={"camera_id": "camera_02"}).json()
    assert filtered["total"] == 1
    assert filtered["events"][0]["event_id"] == "evt_b"


# --------------------------------------------------------------------------- #
# pose
# --------------------------------------------------------------------------- #
def test_pose_latest_is_404_before_any_frame(client: TestClient):
    response = client.get("/pose/latest")
    assert response.status_code == 404
    assert "no pose" in response.json()["detail"]


def test_pose_latest_returns_landmarks_only(client: TestClient):
    fill_buffer(client, STANDING, 2)
    response = client.get("/pose/latest")
    assert response.status_code == 200
    body = response.json()
    assert len(body["landmarks"]) == 33
    assert len(body["landmarks"][0]) == 4
    assert body["connections"]
    assert "does not retain or serve raw video" in body["privacy_note"]


# --------------------------------------------------------------------------- #
# error handling
# --------------------------------------------------------------------------- #
def test_wrong_method_is_405(client: TestClient):
    assert client.get("/predict").status_code == 405


def test_malformed_json_is_rejected(client: TestClient):
    response = client.post(
        "/predict", content=b"{not json}", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_nan_landmarks_are_handled_without_a_500(client: TestClient, settings: Settings):
    """Non-finite coordinates must not become a 500; they are sanitised or rejected."""
    rows = ",".join("[NaN,NaN,NaN,NaN]" for _ in range(33))
    response = client.post(
        "/predict", content=f'{{"landmarks":[{rows}]}}'.encode(),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code in {200, 422}
    assert response.status_code != 500
    if response.status_code == 200:
        assert response.json()["pose_detected"] is False


def test_malformed_landmarks_type_is_rejected(client: TestClient):
    response = client.post("/predict", json={"landmarks": "not a list"})
    assert response.status_code == 422


def test_events_survive_a_restart(settings: Settings, make_event):
    """Durability: a confirmed event must still be there after a process restart."""
    from backend.database.repository import EventRepository

    with TestClient(create_app(settings)) as client:
        client.app.state.service.repository.add(make_event("evt_durable"))
    reopened = EventRepository(settings.db_path)
    try:
        assert reopened.get("evt_durable") is not None
    finally:
        reopened.close()


def test_alert_outcome_is_recorded_on_the_event(client: TestClient, settings: Settings,
                                                make_event):
    """A failed alert must not erase the event, and must not claim to have been sent."""
    service = client.app.state.service
    service.handle_event(make_event("evt_alert"))
    stored = service.repository.get("evt_alert")
    assert stored is not None
    assert stored["status"] == EventStatus.DETECTED.value
    assert stored["notified"] is False
    assert stored["note"] == "not configured"
    assert service.alerts_failed == 1
    assert service.alerts_sent == 0
