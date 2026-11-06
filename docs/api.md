# API reference

Base URL: `http://localhost:8000`
Interactive docs: `/docs` (Swagger) and `/redoc`

Start the server:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## Conventions

* All timestamps are ISO-8601 UTC with second precision, e.g. `2026-01-01T12:00:05Z`.
* `prediction` is one of `FALL`, `NORMAL`, `UNKNOWN`.
* `event_confirmed` is the field that matters. `prediction: "FALL"` means the window is
  suspicious; `event_confirmed: true` means the confirmation policy was satisfied, the
  event was persisted, and an alert was attempted.
* Errors use `{"detail": "..."}`.
* **Landmarks are always `[x, y, z, visibility]`, 33 of them, x/y normalised to `[0, 1]`.**

---

## `GET /health`

Liveness probe. Performs no I/O, so it never fails slowly.

```json
{
  "status": "ok",
  "service": "FallGuard AI",
  "version": "0.1.0",
  "environment": "development",
  "timestamp": "2026-01-01T12:00:05Z"
}
```

| Field | Notes |
|---|---|
| `status` | `ok`, or `degraded` when the pose estimator is unavailable |

---

## `GET /status`

Full runtime state. Consumed by the dashboard.

```json
{
  "system": "online",
  "app": { "name": "FallGuard AI", "version": "0.1.0", "environment": "development" },
  "model": {
    "backend": "keras",
    "loaded": true,
    "path": "models/fallguard_lstm.keras",
    "sequence_length": 30,
    "feature_dim": 145,
    "architecture": "lstm",
    "checkpoint_present": true,
    "detail": ""
  },
  "camera": { "id": "camera_01", "source": "0", "connected": false, "monitor_enabled": false },
  "config": {
    "sequence_length": 30,
    "confidence_threshold": 0.8,
    "min_fall_frames": 8,
    "cooldown_seconds": 30.0,
    "max_missing_ratio": 0.5,
    "store_raw_video": false
  },
  "telegram": {
    "enabled": true, "configured": false, "dry_run": true,
    "reason": "no token/chat id configured", "api_base": "https://api.telegram.org"
  },
  "alerts": { "sent": 0, "failed": 1 },
  "events": { "total": 1, "by_status": {"detected": 1}, "by_camera": {"camera_01": 1},
              "latest_event_at": "2026-01-01T12:00:05Z", "database": "data/fallguard.db" },
  "pipeline": {
    "frames_processed": 195, "poses_detected": 194, "pose_dropouts": 1,
    "pose_detection_rate": 0.9949, "windows_scored": 166, "events_confirmed": 1,
    "uptime_seconds": 25.7, "buffered_frames": 30, "sequence_length": 30,
    "sequence_ready": false, "missing_ratio": 0.0, "pose_estimator_available": true
  },
  "detector": {
    "windows_evaluated": 166, "above_threshold": 20, "events_confirmed": 1,
    "windows_suppressed_by_cooldown": 0, "windows_skipped_no_signal": 0,
    "confidence_threshold": 0.8, "min_fall_frames": 8, "cooldown_seconds": 30.0,
    "in_cooldown": false, "cooldown_remaining_seconds": 0.0,
    "consecutive_above_threshold": 0
  },
  "last_prediction": { "prediction": "NORMAL", "confidence": 0.0, "...": "..." },
  "privacy": {
    "retention": "skeleton landmarks and event metadata only",
    "raw_video_stored": "never",
    "identity_recognition": "not implemented",
    "disclaimer": "FallGuard AI is a prototype ... not a medical device and does not diagnose."
  }
}
```

`telegram.reason` distinguishes *"not configured"* from *"dry_run enabled"* from
*"disabled"* without exposing the token.

---

## `POST /predict`

Score a frame, or re-score the live buffer.

### Request

| Field | Type | Required | Notes |
|---|---|---|---|
| `landmarks` | `[[x, y, z, visibility] × 33]` | no | 33 landmarks. Omit to score the live buffer |
| `camera_id` | string ≤ 64 | no | Defaults to the configured camera |
| `timestamp` | string ≤ 64 | no | Defaults to now |
| `threshold` | float 0–1 | no | Overrides the configured threshold for this call |

Unknown fields are rejected.

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"camera_id":"living_room","landmarks":[[0.5,0.13,0.0,1.0],"…32 more…"]}'
```

### Response — `200`

```json
{
  "prediction": "FALL",
  "confidence": 0.9738,
  "fall_probability": 0.9738,
  "timestamp": "2026-01-01T12:00:05Z",
  "camera_id": "camera_01",
  "pose_detected": true,
  "sequence_ready": true,
  "event_confirmed": true,
  "event_id": "5a514798a6874106a0f2c9d584d99b69",
  "model_backend": "keras",
  "message": "Fall event confirmed and stored: 5a514798a6874106a0f2c9d584d99b69"
}
```

| Field | Notes |
|---|---|
| `sequence_ready` | `false` until `SEQUENCE_LENGTH` frames are buffered. `message` says so |
| `pose_detected` | `false` when the frame yielded no usable pose |
| `fall_probability` | raw model output, before thresholding |
| `confidence` | same value; present for the dashboard's convenience |
| `model_backend` | `keras` or `mock` — a `mock` response is **not** a trained prediction |
| `message` | human-readable: buffered, above threshold, or confirmed |

An array containing `NaN` or `inf` is **rejected** and reported as `pose_detected: false`
with a warning. A frame of non-finite values is not a detection; accepting it would report
a pose while producing a gap and corrupt the detection rate.

### Errors

| Status | When | Detail |
|---|---|---|
| `409 Conflict` | No `landmarks` and the live buffer is not ready | *"no landmarks were supplied and the live buffer is not ready (3/30 frames)…"* |
| `422 Unprocessable` | Wrong landmark count or width, or an unknown field | pydantic message |
| `400 Bad Request` | A malformed array slipped past validation | — |

### The two modes differ

| Mode | Advances the confirmation state machine? |
|---|---|
| With `landmarks` | **Yes** — a real frame enters the buffer |
| Without `landmarks` | **No** — a read-only re-score of the buffer |

Re-scoring the buffer without new evidence would otherwise advance the detector twice per
request, letting a client manufacture events by polling.

---

## `GET /events`

Stored events, newest first.

| Query | Default | Notes |
|---|---|---|
| `limit` | 50 | 1–1000 |
| `offset` | 0 | ≥ 0 |
| `camera_id` | — | exact match |
| `status` | — | `detected`, `alerted`, `acknowledged`, `false_positive` |

```bash
curl "http://localhost:8000/events?limit=20&status=detected"
```

```json
{
  "total": 1,
  "limit": 20,
  "offset": 0,
  "events": [
    {
      "event_id": "5a514798a6874106a0f2c9d584d99b69",
      "timestamp": "2026-01-01T12:00:05Z",
      "camera_id": "camera_01",
      "prediction": "FALL",
      "confidence": 0.9738,
      "peak_confidence": 0.9911,
      "status": "detected",
      "window_frames": 8,
      "source": "",
      "notified": false,
      "note": "",
      "created_at": "2026-01-01T12:00:05Z"
    }
  ]
}
```

`total` reflects the filters, not just the page — so a dashboard can page correctly.

`note` carries the alert outcome, e.g. `"not configured"`, `"HTTP 500"`. It is how a
failed notification stays visible without falsely marking the event as alerted.

### Status lifecycle

```text
detected ──alert sent──▶ alerted ──human reviews──▶ acknowledged
   │                                               acknowledged
   └──human reviews──▶ false_positive
```

`detected` means "recorded, not necessarily announced".

---

## `GET /events/{event_id}`

One event, or `404` with `{"detail": "event 'x' not found"}`.

---

## `PATCH /events/{event_id}`

```json
{ "status": "acknowledged", "note": "checked on person, OK" }
```

| Field | Required | Notes |
|---|---|---|
| `status` | yes | `detected`, `alerted`, `acknowledged`, `false_positive` |
| `note` | no | ≤ 500 chars |

Returns the updated event, or `404` for an unknown id, `422` for an unknown status.

---

## `GET /pose/latest`

The latest skeleton — **landmarks only, never pixels**.

```json
{
  "camera_id": "camera_01",
  "landmarks": [[0.5, 0.13, -0.02, 0.99], "…32 more…"],
  "connections": [[11, 12], [11, 13], "…"],
  "frame_index": 142,
  "privacy_note": "Skeleton landmarks only. FallGuard AI does not retain or serve raw video."
}
```

Returns `404` before any frame has been processed. Landmarks with visibility below 0.3
are off-canvas markers, so a client can skip drawing a joint rather than connecting a line
to the origin.

This endpoint exists so a dashboard can show posture without receiving imagery.

---

## `GET /`

Service index: name, version, endpoint list, and the non-medical disclaimer.

---

## Monitoring

```bash
# Liveness
curl -fsS http://localhost:8000/health

# Readiness: is the model loaded and the pose estimator available?
curl -fsS http://localhost:8000/status | jq '.model.loaded, .pipeline.pose_estimator_available'

# Detection health over the last N frames
curl -fsS http://localhost:8000/status | jq '.pipeline.pose_detection_rate'

# Event volume
curl -fsS http://localhost:8000/status | jq '.events.total, .alerts'
```

Worth alerting on:

| Signal | Meaning |
|---|---|
| `model.loaded == false` | fell back to the mock backend — do not trust predictions |
| `pose_estimator_available == false` | the MediaPipe asset is missing |
| `pose_detection_rate` collapses | camera moved, lighting changed, or the subject left |
| `alerts.failed` climbing | Telegram outage; events are still being stored |
| `events_confirmed` flat while `poses_detected` is high | the model may be stuck |

---

## Security

> **The API has no authentication. Do not expose it to an untrusted network.**

Before any deployment beyond localhost:

* Bind to a private interface, or put it behind a reverse proxy that enforces
  authentication and TLS.
* Narrow the CORS policy. It is `allow_origins: ["*"]` because the dashboard is a separate
  process; restrict it to the dashboard's origin in production.
* Never commit `.env`. Tokens are read from the environment and are redacted from logs.

See [`deployment.md`](deployment.md).
