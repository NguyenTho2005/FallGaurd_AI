"""Integration tests for the Streamlit dashboard.

Streamlit's ``AppTest`` executes the real script, so these catch the failures a health
endpoint cannot: a renamed variable, a stale key, a bad widget argument, an unhandled
exception on the "backend is down" path.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="Streamlit is required for the dashboard tests")

from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.integration

APP_PATH = str((Path(__file__).resolve().parents[2] / "frontend" / "streamlit_app.py"))


@pytest.fixture
def offline_app() -> AppTest:
    """A dashboard pointed at a port nothing is listening on."""
    app = AppTest.from_file(APP_PATH, default_timeout=120)
    os.environ["FALLGUARD_API_URL"] = "http://127.0.0.1:1"
    app.run()
    return app


class FakeBackend:
    """Serves the subset of the API the dashboard consumes."""

    def __init__(self, *, with_events: bool = True, with_pose: bool = True) -> None:
        self.with_events = with_events
        self.with_pose = with_pose
        self.acknowledged: list[str] = []

    def _pose_payload(self):
        from tests.conftest import STANDING, landmark_array

        return {
            "camera_id": "camera_01",
            "landmarks": landmark_array(STANDING).astype(float).tolist(),
            "connections": [[11, 12], [11, 23], [23, 24]],
            "frame_index": 7,
            "privacy_note": "Skeleton landmarks only.",
        }

    def _event(self, index: int) -> dict:
        return {
            "event_id": f"evt{index:032d}",
            "timestamp": "2026-01-01T00:00:00Z",
            "camera_id": "camera_01",
            "prediction": "FALL",
            "confidence": 0.947,
            "peak_confidence": 0.99,
            "status": "detected",
            "window_frames": 8,
            "source": "camera_01",
            "notified": False,
            "note": "",
            "created_at": "2026-01-01T00:00:00Z",
        }

    def _status(self) -> dict:
        return {
            "system": "online",
            "app": {"name": "FallGuard AI", "version": "0.1.0", "environment": "test"},
            "model": {
                "backend": "mock", "loaded": True, "path": None, "sequence_length": 30,
                "feature_dim": 145, "detail": "deterministic heuristic",
            },
            "camera": {"id": "camera_01", "source": "0", "connected": True,
                       "monitor_enabled": True},
            "config": {"confidence_threshold": 0.8},
            "telegram": {"enabled": True, "configured": False, "dry_run": True,
                         "reason": "not configured", "api_base": "https://example.invalid"},
            "alerts": {"sent": 0, "failed": 1},
            "events": {"total": 1 if self.with_events else 0},
            "pipeline": {
                "frames_processed": 100, "poses_detected": 99, "pose_dropouts": 1,
                "pose_detection_rate": 0.99, "windows_scored": 70, "events_confirmed": 1,
                "uptime_seconds": 12.0, "buffered_frames": 30, "sequence_length": 30,
                "sequence_ready": True, "missing_ratio": 0.0,
                "pose_estimator_available": True,
            },
            "detector": {"windows_evaluated": 70, "confidence_threshold": 0.8,
                         "min_fall_frames": 8, "cooldown_seconds": 30.0},
            "last_prediction": {
                "prediction": "FALL", "confidence": 0.947, "fall_probability": 0.947,
                "pose_detected": True, "sequence_ready": True, "event_confirmed": True,
                "event_id": "evt0" * 16, "timestamp": "2026-01-01T00:00:00Z",
                "frame_index": 99, "camera_id": "camera_01",
            },
            "privacy": {"retention": "skeleton landmarks and event metadata only",
                        "raw_video_stored": "never",
                        "identity_recognition": "not implemented",
                        "disclaimer": "Not a medical device."},
        }

    def handle(self, request) -> tuple[int, object]:
        path = request.path_url.split("?")[0]
        if path == "/status":
            return 200, self._status()
        if path == "/pose/latest":
            return (200, self._pose_payload()) if self.with_pose else (404, {"detail": "none"})
        if path == "/events":
            events = [self._event(0)] if self.with_events else []
            return 200, {"total": len(events), "limit": 50, "offset": 0, "events": events}
        if path.startswith("/events/") and request.method == "PATCH":
            self.acknowledged.append(path.rsplit("/", 1)[-1])
            return 200, self._event(0)
        return 404, {"detail": "not found"}


class _BackendServer(FakeBackend):
    """Serves :class:`FakeBackend` over real HTTP on a loopback port.

    A real server is used rather than a mocked ``requests`` so the dashboard exercises
    its actual transport, including URL construction and error handling.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        backend = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _respond(self, method: str) -> None:
                status, payload = backend.handle(_Request(method, self.path))
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                self._respond("GET")

            def do_PATCH(self) -> None:  # noqa: N802 - http.server API
                self._respond("PATCH")

            def log_message(self, *args: object) -> None:
                """Silence the default stderr access log."""

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


class _Request:
    def __init__(self, method: str, url: str) -> None:
        self.method = method
        self.path_url = url


@pytest.fixture
def fake_backend():
    server = _BackendServer()
    os.environ["FALLGUARD_API_URL"] = server.url
    try:
        yield server
    finally:
        os.environ.pop("FALLGUARD_API_URL", None)
        server.close()


def test_dashboard_renders_against_a_live_backend(fake_backend) -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=120)
    app.run()

    assert not app.exception, [e.value for e in app.exception]

    labels = {metric.label: metric.value for metric in app.metric}
    assert labels["Backend"] == "ONLINE"
    assert labels["System"] == "ONLINE"
    assert labels["Camera"] == "CONNECTED"
    assert labels["Model"] == "LOADED"
    assert labels["Poses"] == "99%"
    assert labels["Confidence"] == "94.7%"
    assert labels["Sequence"] == "READY"
    assert labels["Pose"] == "DETECTED"

    assert [h.value for h in app.subheader] == [
        "System status", "Live prediction", "Event history"
    ]
    # A confirmed event must be unmissable, not just a row in the table.
    assert "FALL DETECTED" in " ".join(e.value for e in app.error)
    assert len(app.dataframe) == 1
    frame = app.dataframe[0].value
    assert len(frame) == 1
    assert list(frame.columns) == [
        "Timestamp", "Prediction", "Confidence", "Peak", "Camera", "Status", "Alerted",
        "Event ID",
    ]
    assert frame.iloc[0]["Confidence"] == "94.7%"


def test_dashboard_warns_about_the_mock_backend(fake_backend) -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=120)
    app.run()
    warnings = " ".join(w.value for w in app.warning)
    assert "Mock model backend" in warnings
    # Streamlit renders markdown emphasis, so the asterisks are stripped from the text.
    assert "not a trained model" in warnings.replace("*", "")


def test_dashboard_always_disclaims_being_a_medical_device(fake_backend) -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=120)
    app.run()
    text = " ".join(w.value for w in app.warning) + " ".join(e.value for e in app.success)
    assert "not a medical device" in text.lower()


def test_dashboard_handles_an_unreachable_backend(offline_app: AppTest) -> None:
    assert not offline_app.exception, [e.value for e in offline_app.exception]
    errors = " ".join(e.value for e in offline_app.error)
    assert "Cannot reach the FallGuard API" in errors
    labels = {metric.label: metric.value for metric in offline_app.metric}
    assert labels["Backend"] == "OFFLINE"


def test_dashboard_sidebar_controls_exist() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=120)
    app.run()
    labels = {item.label for item in app.sidebar.text_input} | {
        item.label for item in app.sidebar.checkbox
    } | {item.label for item in app.sidebar.slider}
    assert "Backend URL" in labels
    assert "Auto refresh" in labels
    assert "Events to show" in labels


def test_empty_event_history_is_reported_clearly() -> None:
    server = _BackendServer(with_events=False, with_pose=False)
    os.environ["FALLGUARD_API_URL"] = server.url
    try:
        app = AppTest.from_file(APP_PATH, default_timeout=120)
        app.run()
    finally:
        os.environ.pop("FALLGUARD_API_URL", None)
        server.close()
    assert not app.exception, [e.value for e in app.exception]
    assert "No fall events recorded" in " ".join(s.value for s in app.success)
    assert not app.dataframe
