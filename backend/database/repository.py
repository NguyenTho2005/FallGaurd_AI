"""SQLite-backed event store.

MVP scope only: one table, the standard library driver, no ORM and no migrations
framework. What matters for the rest of the system is that the store is:

* **durable** -- a confirmed event survives a restart;
* **thread-safe** -- the API thread writes while the monitor thread may also write, so
  every statement runs under one lock on a connection created with
  ``check_same_thread=False``;
* **bounded** -- retention is capped by ``max_events`` so a long-running deployment
  cannot fill the disk.

Privacy: only event metadata is stored. Video frames, images and skeleton coordinates
are never written -- see ``docs/privacy.md``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from ai.inference.fall_detector import EventStatus, FallEvent
from ai.logging_config import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 1

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS fall_events (
    event_id        TEXT PRIMARY KEY,
    timestamp       TEXT    NOT NULL,
    camera_id       TEXT    NOT NULL,
    prediction      TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    peak_confidence REAL    NOT NULL DEFAULT 0.0,
    status          TEXT    NOT NULL DEFAULT 'detected',
    window_frames   INTEGER NOT NULL DEFAULT 0,
    source          TEXT    NOT NULL DEFAULT '',
    notified        INTEGER NOT NULL DEFAULT 0,
    note            TEXT    NOT NULL DEFAULT '',
    created_at      TEXT    NOT NULL
)
"""

_CREATE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON fall_events (timestamp DESC)",
    "CREATE INDEX IF NOT EXISTS idx_events_camera ON fall_events (camera_id)",
    "CREATE INDEX IF NOT EXISTS idx_events_status ON fall_events (status)",
)


class EventRepository:
    """CRUD access to confirmed fall events.

    Parameters
    ----------
    db_path:
        SQLite file. ``":memory:"`` is supported, which is what the tests use.
    max_events:
        Retention cap. The oldest rows are pruned after each insert.
    """

    def __init__(self, db_path: str | Path, *, max_events: int = 5000) -> None:
        self.db_path = str(db_path)
        self.max_events = int(max_events)
        self._lock = threading.RLock()
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.db_path, check_same_thread=False, timeout=10.0
        )
        self._connection.row_factory = sqlite3.Row
        self._initialise()
        logger.info("Event store ready at %s (schema v%d)", self.db_path, SCHEMA_VERSION)

    def _initialise(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute(_CREATE_TABLE)
            for statement in _CREATE_INDEXES:
                self._connection.execute(statement)
            self._connection.commit()

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cursor = self._connection.cursor()
            try:
                yield cursor
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
            finally:
                cursor.close()

    def close(self) -> None:
        with self._lock:
            try:
                self._connection.close()
            except Exception:  # pragma: no cover - defensive
                logger.warning("Error closing the event store", exc_info=True)

    # ------------------------------------------------------------------ #
    # writes
    # ------------------------------------------------------------------ #
    def add(self, event: FallEvent) -> str:
        """Persist a confirmed event and return its id."""
        created_at = _utc_now()
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT OR REPLACE INTO fall_events (
                    event_id, timestamp, camera_id, prediction, confidence,
                    peak_confidence, status, window_frames, source, notified, note, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.timestamp,
                    event.camera_id,
                    event.prediction,
                    float(event.confidence),
                    float(event.peak_confidence),
                    str(event.status.value if isinstance(event.status, EventStatus) else event.status),
                    int(event.window_frames),
                    event.source,
                    1 if event.notified else 0,
                    event.note,
                    created_at,
                ),
            )
        logger.info(
            "Stored event %s (camera=%s, confidence=%.3f, status=%s)",
            event.event_id, event.camera_id, event.confidence, event.status,
        )
        self._prune()
        return event.event_id

    def update_status(
        self, event_id: str, status: EventStatus | str, *, notified: bool | None = None,
        note: str | None = None,
    ) -> bool:
        """Update an event's status. Returns ``False`` when the id is unknown."""
        value = status.value if isinstance(status, EventStatus) else str(status)
        assignments = ["status = ?"]
        parameters: list[Any] = [value]
        if notified is not None:
            assignments.append("notified = ?")
            parameters.append(1 if notified else 0)
        if note is not None:
            assignments.append("note = ?")
            parameters.append(note)
        parameters.append(event_id)
        with self._cursor() as cursor:
            cursor.execute(
                f"UPDATE fall_events SET {', '.join(assignments)} WHERE event_id = ?",
                parameters,
            )
            return cursor.rowcount > 0

    def delete(self, event_id: str) -> bool:
        with self._cursor() as cursor:
            cursor.execute("DELETE FROM fall_events WHERE event_id = ?", (event_id,))
            return cursor.rowcount > 0

    def _prune(self) -> None:
        if self.max_events <= 0:
            return
        with self._cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS n FROM fall_events")
            total = int(cursor.fetchone()["n"])
            if total <= self.max_events:
                return
            excess = total - self.max_events
            cursor.execute(
                """
                DELETE FROM fall_events WHERE event_id IN (
                    SELECT event_id FROM fall_events ORDER BY created_at ASC, rowid ASC LIMIT ?
                )
                """,
                (excess,),
            )
            logger.info("Pruned %d old events (retention cap %d)", excess, self.max_events)

    # ------------------------------------------------------------------ #
    # reads
    # ------------------------------------------------------------------ #
    def get(self, event_id: str) -> dict[str, Any] | None:
        with self._cursor() as cursor:
            cursor.execute("SELECT * FROM fall_events WHERE event_id = ?", (event_id,))
            row = cursor.fetchone()
        return _row_to_dict(row) if row is not None else None

    def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        camera_id: str | None = None,
        status: EventStatus | str | None = None,
    ) -> list[dict[str, Any]]:
        if limit < 1 or limit > 1000:
            raise ValueError(f"limit must be between 1 and 1000, got {limit}")
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        clauses: list[str] = []
        parameters: list[Any] = []
        if camera_id:
            clauses.append("camera_id = ?")
            parameters.append(camera_id)
        if status is not None:
            clauses.append("status = ?")
            parameters.append(status.value if isinstance(status, EventStatus) else str(status))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.extend([int(limit), int(offset)])
        with self._cursor() as cursor:
            cursor.execute(
                f"SELECT * FROM fall_events{where} ORDER BY created_at DESC, rowid DESC "
                "LIMIT ? OFFSET ?",
                parameters,
            )
            return [_row_to_dict(row) for row in cursor.fetchall()]

    def count(self, *, camera_id: str | None = None, status: EventStatus | str | None = None) -> int:
        clauses: list[str] = []
        parameters: list[Any] = []
        if camera_id:
            clauses.append("camera_id = ?")
            parameters.append(camera_id)
        if status is not None:
            clauses.append("status = ?")
            parameters.append(status.value if isinstance(status, EventStatus) else str(status))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) AS n FROM fall_events{where}", parameters)
            return int(cursor.fetchone()["n"])

    def summary(self) -> dict[str, Any]:
        """Counts by status, for the dashboard header."""
        with self._cursor() as cursor:
            cursor.execute("SELECT status, COUNT(*) AS n FROM fall_events GROUP BY status")
            by_status = {row["status"]: int(row["n"]) for row in cursor.fetchall()}
            cursor.execute(
                "SELECT camera_id, COUNT(*) AS n FROM fall_events GROUP BY camera_id"
            )
            by_camera = {row["camera_id"]: int(row["n"]) for row in cursor.fetchall()}
            cursor.execute("SELECT MAX(timestamp) AS latest FROM fall_events")
            latest = cursor.fetchone()["latest"]
        return {
            "total": sum(by_status.values()),
            "by_status": by_status,
            "by_camera": by_camera,
            "latest_event_at": latest,
            "database": self.db_path,
        }

    def export_json(self, path: str | Path) -> Path:
        """Dump every event to a JSON file (metadata only, no imagery)."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "exported_at": _utc_now(),
            "schema_version": SCHEMA_VERSION,
            "events": self.list(limit=1000, offset=0),
        }
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Exported %d events to %s", len(payload["events"]), destination)
        return destination

    def __enter__(self) -> "EventRepository":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "event_id": row["event_id"],
        "timestamp": row["timestamp"],
        "camera_id": row["camera_id"],
        "prediction": row["prediction"],
        "confidence": float(row["confidence"]),
        "peak_confidence": float(row["peak_confidence"]),
        "status": row["status"],
        "window_frames": int(row["window_frames"]),
        "source": row["source"],
        "notified": bool(row["notified"]),
        "note": row["note"],
        "created_at": row["created_at"],
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def rows_to_events(rows: Sequence[dict[str, Any]]) -> list[FallEvent]:  # pragma: no cover
    """Rebuild :class:`FallEvent` objects from repository rows (used by tests)."""
    return [
        FallEvent(
            event_id=row["event_id"],
            timestamp=row["timestamp"],
            camera_id=row["camera_id"],
            prediction=row["prediction"],
            confidence=row["confidence"],
            status=EventStatus(row["status"]),
            peak_confidence=row.get("peak_confidence", 0.0),
            window_frames=row.get("window_frames", 0),
            source=row.get("source", ""),
            notified=row.get("notified", False),
            note=row.get("note", ""),
        )
        for row in rows
    ]
