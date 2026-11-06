"""Unit tests for the SQLite event store."""

from __future__ import annotations

import json
import threading

import pytest

from ai.inference.fall_detector import EventStatus
from backend.database.repository import EventRepository


def test_add_and_get(repository: EventRepository, make_event):
    event = make_event("evt_001")
    assert repository.add(event) == "evt_001"
    stored = repository.get("evt_001")
    assert stored is not None
    assert stored["camera_id"] == "camera_01"
    assert stored["prediction"] == "FALL"
    assert stored["confidence"] == pytest.approx(0.95)
    assert stored["status"] == "detected"
    assert stored["notified"] is False
    assert stored["created_at"].endswith("Z")


def test_get_unknown_event_returns_none(repository: EventRepository):
    assert repository.get("does_not_exist") is None


def test_list_is_newest_first(repository: EventRepository, make_event):
    for index in range(3):
        repository.add(make_event(f"evt_{index:03d}"))
    events = repository.list()
    assert len(events) == 3
    created = [event["created_at"] for event in events]
    assert created == sorted(created, reverse=True)


def test_pagination(repository: EventRepository, make_event):
    for index in range(10):
        repository.add(make_event(f"evt_{index:03d}"))
    first = repository.list(limit=4, offset=0)
    second = repository.list(limit=4, offset=4)
    assert len(first) == 4 and len(second) == 4
    assert {e["event_id"] for e in first}.isdisjoint({e["event_id"] for e in second})
    assert len(repository.list(limit=100, offset=8)) == 2


def test_filter_by_camera(repository: EventRepository, make_event):
    repository.add(make_event("evt_a", camera_id="camera_01"))
    repository.add(make_event("evt_b", camera_id="camera_02"))
    assert len(repository.list(camera_id="camera_02")) == 1
    assert repository.count(camera_id="camera_02") == 1
    assert repository.count() == 2


def test_filter_by_status(repository: EventRepository, make_event):
    repository.add(make_event("evt_a"))
    repository.add(make_event("evt_b"))
    repository.update_status("evt_b", EventStatus.ACKNOWLEDGED)
    assert len(repository.list(status="acknowledged")) == 1
    assert repository.count(status=EventStatus.DETECTED) == 1


@pytest.mark.parametrize(
    "kwargs", [{"limit": 0}, {"limit": 1001}, {"limit": 10, "offset": -1}]
)
def test_invalid_pagination_is_rejected(repository: EventRepository, kwargs):
    with pytest.raises(ValueError):
        repository.list(**kwargs)


def test_update_status_and_notified_flag(repository: EventRepository, make_event):
    repository.add(make_event("evt_a"))
    assert repository.update_status("evt_a", EventStatus.ALERTED, notified=True) is True
    stored = repository.get("evt_a")
    assert stored["status"] == "alerted"
    assert stored["notified"] is True


def test_update_status_of_unknown_event_returns_false(repository: EventRepository):
    assert repository.update_status("nope", EventStatus.ACKNOWLEDGED) is False


def test_update_note(repository: EventRepository, make_event):
    repository.add(make_event("evt_a"))
    repository.update_status("evt_a", EventStatus.DETECTED, notified=False, note="telegram down")
    assert repository.get("evt_a")["note"] == "telegram down"


def test_delete(repository: EventRepository, make_event):
    repository.add(make_event("evt_a"))
    assert repository.delete("evt_a") is True
    assert repository.get("evt_a") is None
    assert repository.delete("evt_a") is False


def test_adding_the_same_id_replaces_the_row(repository: EventRepository, make_event):
    repository.add(make_event("evt_a", confidence=0.5))
    repository.add(make_event("evt_a", confidence=0.9))
    assert repository.count() == 1
    assert repository.get("evt_a")["confidence"] == pytest.approx(0.9)


def test_summary_groups_by_status_and_camera(repository: EventRepository, make_event):
    repository.add(make_event("evt_a", camera_id="camera_01"))
    repository.add(make_event("evt_b", camera_id="camera_02"))
    repository.add(make_event("evt_c", camera_id="camera_01"))
    repository.update_status("evt_b", EventStatus.FALSE_POSITIVE)
    summary = repository.summary()
    assert summary["total"] == 3
    assert summary["by_camera"] == {"camera_01": 2, "camera_02": 1}
    assert summary["by_status"] == {"detected": 2, "false_positive": 1}
    assert summary["latest_event_at"] is not None


def test_retention_prunes_the_oldest_rows(make_event):
    store = EventRepository(":memory:", max_events=5)
    try:
        for index in range(12):
            store.add(make_event(f"evt_{index:03d}"))
        assert store.count() == 5
        remaining = {event["event_id"] for event in store.list(limit=10)}
        assert "evt_000" not in remaining
        assert "evt_011" in remaining
    finally:
        store.close()


def test_concurrent_writes_are_serialised(tmp_path, make_event):
    """The API thread and the monitor thread may both write; SQLite must not choke."""
    store = EventRepository(tmp_path / "concurrent.db")
    errors: list[BaseException] = []

    def writer(worker: int) -> None:
        try:
            for index in range(20):
                store.add(make_event(f"w{worker}_evt_{index:03d}"))
        except BaseException as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(worker,)) for worker in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    try:
        assert not errors, f"concurrent writes failed: {errors}"
        assert store.count() == 80
    finally:
        store.close()


def test_export_json_contains_metadata_only(repository: EventRepository, make_event, tmp_path):
    repository.add(make_event("evt_a"))
    destination = repository.export_json(tmp_path / "export.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert len(payload["events"]) == 1
    event = payload["events"][0]
    assert set(event) >= {"event_id", "timestamp", "camera_id", "prediction", "confidence",
                          "status"}
    # Nothing resembling imagery may be exported. "window_frames" is a legitimate
    # metadata field, so the forbidden terms are specific enough not to catch it.
    serialised = json.dumps(payload).lower()
    for forbidden in ("image", "video", "jpeg", "png", "landmarks", "raw_frame",
                      "thumbnail", "screenshot", "b64"):
        assert forbidden not in serialised


def test_database_file_is_created_on_disk(settings, make_event):
    path = settings.db_path
    assert not path.exists()
    store = EventRepository(path)
    try:
        store.add(make_event("evt_a"))
        assert path.exists()
    finally:
        store.close()
    # A second handle must see the persisted row.
    reopened = EventRepository(path)
    try:
        assert reopened.count() == 1
    finally:
        reopened.close()


def test_context_manager_closes(repository: EventRepository):
    with repository as store:
        assert store.count() == 0
