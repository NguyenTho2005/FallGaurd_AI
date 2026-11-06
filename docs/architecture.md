# Architecture

How FallGuard AI is put together, and why.

---

## 1. Layering

```text
frontend/  ──HTTP──▶  backend/  ──▶  ai/
```

The dependency arrow points **one way**. The `ai` package knows nothing about HTTP,
Telegram or SQLite, and contains no web framework. Two consequences:

* `scripts/demo.py` and the FastAPI service drive the **same** `FallDetectionPipeline`
  instance, so a demo run and a production run execute identical inference code.
* The AI layer is testable without a server, and the API is testable without an AI
  change rippling outward.

| Layer | Packages | Responsibility |
|---|---|---|
| Capture | `ai/sources` | yield `(frame, timestamp, index)`; nothing else |
| Perception | `ai/pose` | BGR frame → 33 landmarks, or `None` |
| Representation | `ai/preprocessing` | 33 landmarks → 145-D feature vector, or a gap |
| Memory | `ai/inference` | ring buffer → `(30, 145)` window |
| Prediction | `ai/models` | window → fall probability |
| Decision | `ai/inference/fall_detector` | probability stream → confirmed `FallEvent` |
| Orchestration | `ai/inference/pipeline` | wires the above, owns the counters |
| Persistence | `backend/database` | `FallEvent` → SQLite |
| Notification | `backend/alerts` | `FallEvent` → Telegram (or dry run) |
| Delivery | `backend/api` | JSON over HTTP |
| Presentation | `frontend` | Streamlit dashboard |

---

## 2. Perception: `ai/pose`

`PoseEstimator` is the only module that talks to MediaPipe.

MediaPipe 1.x removed `mp.solutions.pose`, so the Tasks API is used:

```python
options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=...),
    running_mode=vision.RunningMode.VIDEO,   # enables temporal tracking
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
```

`min_tracking_confidence` is resolved by inspecting the constructor signature, because
the keyword was renamed between MediaPipe releases and a hard-coded name breaks on upgrade.

### Failure handling — the contract

`process()` **never raises** for a per-frame problem. It returns `None`, which the pipeline
records as a gap:

| Failure | Behaviour |
|---|---|
| No person in frame | `None` |
| `None` / empty / non-array frame | `None`, warning, `frames_processed` unchanged |
| Wrong channel count | `None`, warning |
| 2-D frame | expanded to 3 channels |
| 4-channel (BGRA) | accepted, alpha dropped |
| `float` frame | clipped and cast to `uint8` |
| NaN / inf landmark | landmark zeroed and its visibility set to 0 |
| Wrong landmark count | frame rejected, warning |
| TFLite internal error | `None`, warning, `detector_errors` incremented |
| Called after `close()` | `None`, error log |

MediaPipe's `VIDEO` mode requires strictly increasing millisecond timestamps, so
`_next_timestamp_ms` enforces monotonicity even when the source clock stalls.

---

## 3. Representation: `ai/preprocessing`

This is where the accuracy of the whole system is won or lost.

### Two views of the same pose

`PoseNormalizer.normalize()` returns both:

| View | Transform | Used for |
|---|---|---|
| `coordinates` | hip-centred → torso-aligned → torso-scaled | the 99-D coordinate block fed to the LSTM |
| `unaligned` | hip-centred → torso-scaled, **not** rotated | the 12 kinematic descriptors |

Both are needed, and the reason is a genuine trap. Forcing the torso vertical makes the
model invariant to camera tilt — but it also forces `torso_lean_deg` to be identically
zero and flips the sign of the hip height, destroying exactly the two cues that separate a
fall from ordinary movement. Computing the descriptors from the unrotated view keeps them
interpretable while the coordinate block stays tilt-invariant. A unit test pins both
behaviours.

### The pipeline

1. **Visibility mask** — landmarks with `visibility < MIN_LANDMARK_VISIBILITY` are invalid.
   A valid shoulder or hip pair repairs an invalid one; the torso scale falls back to the
   bounding-box diagonal of the valid landmarks, so losing a hip costs precision, not the
   frame.
2. **Translate** — subtract the hip midpoint (shoulder midpoint as fallback).
3. **Rotate** — rotate by `−θ` so the shoulder→hip axis points up, cancelling camera tilt.
   The sign matters: with the wrong sign the transform amplifies rotation rather than
   cancelling it. A test asserts invariance.
4. **Scale** — divide by torso length, so an adult and a child, or a subject at 2 m and at
   5 m, produce comparable features.
5. **Re-mask and sanitise** — invalid joints are exactly zero after transformation; the
   array is `nan_to_num`'d and cast to `float32`.

Frames with no usable reference return `None`, which the preprocessor turns into a zero
vector with `present = 0`. It never raises: a bad frame is a gap, not an exception.

### Feature layout (145-D)

| Block | Size | Content |
|---|---|---|
| `xyz` | 99 | 33 normalised landmarks × (x, y, z) |
| `visibility` | 33 | per-landmark confidence, 0 for missing joints |
| `kinematic` | 12 | see below |
| `present` | 1 | 1.0 if a usable pose was extracted, else 0.0 |

The 12 descriptors, all in torso lengths:

`torso_lean_deg`, `hip_height_above_ankles`, `shoulder_height_above_hip`,
`bbox_aspect_ratio`, `bbox_center_y`, `bbox_height`, `elbow_angle_left`,
`elbow_angle_right`, `knee_angle_left`, `knee_angle_right`, `wrist_separation`,
`ankle_separation`.

`torso_lean_deg` uses the **horizontal** component only. Using the full 2-D norm floors
the result at 45° for a perfectly vertical torso.

The layout is versioned (`FEATURE_SPEC_VERSION = "1.0.0"`) and recorded in every
checkpoint's sidecar, so a layout change cannot silently invalidate old weights.

---

## 4. Memory: `ai/inference/sequence_buffer.py`

A fixed-length ring buffer, dependency-free and clock-injectable.

* **Gaps occupy a slot.** A pose-less frame is stored as a zero vector with `present = 0`.
  Dropping it would compress real time and hide dropouts — the model would see a short
  burst of good frames as a long window.
* **`is_ready` requires density.** A full window *and* a missing ratio within
  `max_missing_ratio`. Otherwise the model would be asked to score mostly absence.
* **Eviction keeps the counters honest.** `poses_in_buffer` tracks the buffer, not the
  lifetime total, so `missing_ratio` stays correct after the window slides.

---

## 5. Prediction: `ai/models`

### Architectures

`build_lstm` and `build_bilstm` both consume `(batch, sequence_length, 145)` and emit a
sigmoid probability.

* `BatchNormalization` on the input: the feature blocks have very different scales
  (coordinates in ±2, angles in degrees, ratios in 0–5) and an LSTM is scale-sensitive.
* `return_sequences=False`: only a window summary is needed, and dropping the time axis
  before the dense head keeps the model small.
* Sigmoid single-unit head: the event-confirmation logic needs a *probability*, not a
  label.

The BiLSTM is roughly twice the parameters and cannot be streamed frame-by-frame, so the
LSTM is the serving default.

### The predictor's three jobs

1. **Load** a checkpoint, or substitute `MockFallModel`.
2. **Validate the serving contract** against the sidecar — sequence length, feature width,
   feature-spec version. A mismatch raises `ModelContractError` rather than silently
   scoring garbage.
3. **Infer defensively.** A shape mismatch or an inference exception returns `0.0` after
   logging. A broken model must degrade to "no fall", never take the process down.

### `MockFallModel`

A documented, deterministic kinematic heuristic that exists so the API, the event logic
and the dashboard are runnable before a checkpoint exists, and on machines with no GPU.
It has **zero learned parameters**, says so in `summary()`, and its `/status` entry says
`"deterministic kinematic heuristic, not a trained model"`.

It scores a frame as fall-like only when **both** cues fire — the hip has dropped towards
the ankles *and* the torso has rotated away from vertical. A forward bend has a low hip
but stays vertical; leaning back in a chair tilts but keeps the hip high. A naive "hip
dropped" rule fails on the first; a naive "torso tilted" rule fails on the second.

---

## 6. Decision: `ai/inference/fall_detector.py`

A three-guard state machine over the probability stream.

```text
p >= CONFIDENCE_THRESHOLD ?  ──no──▶  reset run
        │ yes
        ▼
consecutive += 1
        │
consecutive >= MIN_FALL_FRAMES ?
        │ yes                      │ no
        ▼                          ▼
  emit FallEvent            wait for more evidence
  enter cooldown
```

`COOLDOWN_SECONDS` suppresses further events, and any window evaluated *during* cooldown
clears the run, so the first window afterwards starts fresh.

A window that could not be scored (`has_signal=False`) is counted and skipped, and resets
the run. **Absence of evidence is not evidence.**

Confirmed events also reset the run. Otherwise a second fall shortly after the first would
be confirmed on fewer frames than the policy requires.

The clock is injected, so cooldown behaviour is tested without sleeping and without
flakiness.

---

## 7. Orchestration: `ai/inference/pipeline.py`

Owns one instance of every stage and the counters behind `GET /status`.

* A re-entrant lock guards all mutable state, so `POST /predict` can run while the
  optional camera monitor pushes frames.
* `process_frame` → `process_pose` is the single scoring path, so a client-posted landmark
  array and a camera frame are treated identically.
* A non-finite landmark array is **rejected**. A frame of NaNs is not a detection;
  accepting it would report `pose_detected = true` while the preprocessor produced a gap,
  corrupting the detection rate.
* A failing `on_event` callback is logged and swallowed: the event is already confirmed,
  and downstream persistence must not undo that.
* If MediaPipe cannot initialise, the pipeline still constructs. Landmark arrays can be
  posted directly, and `/status` reports the pose estimator as unavailable.
* `reset()` clears the buffer and the detector but keeps the lifetime counters, so
  `/status` keeps a meaningful uptime count.

---

## 8. Service: `backend/services/inference_service.py`

The one place where pipeline, storage and alerting are combined.

```text
event confirmed
      ↓
1. persist          ← first, always
      ↓
2. attempt alert
      ↓
3. record the outcome on the event
```

Persisting before alerting is deliberate: **a Telegram outage can delay a notification but
must never lose an event.** When the alert fails, the event stays `detected` — claiming
`alerted` would be a lie — and the reason lands in its `note`.

`CameraMonitor` is an optional daemon thread, off by default. It reconnects at most every
`reconnect_interval` seconds, so a disconnected camera degrades to a logged warning rather
than a tight error loop.

---

## 9. Storage: `backend/database/repository.py`

One table, stdlib `sqlite3`, no ORM.

```sql
CREATE TABLE fall_events (
    event_id        TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    camera_id       TEXT NOT NULL,
    prediction      TEXT NOT NULL,
    confidence      REAL NOT NULL,
    peak_confidence REAL NOT NULL DEFAULT 0.0,
    status          TEXT NOT NULL DEFAULT 'detected',
    window_frames   INTEGER NOT NULL DEFAULT 0,
    source          TEXT NOT NULL DEFAULT '',
    notified        INTEGER NOT NULL DEFAULT 0,
    note            TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);
```

* **Thread-safe** — `check_same_thread=False` plus one lock around every statement.
  A test hammers the store from four threads.
* **Bounded** — `max_events` prunes oldest-first, so a long deployment cannot fill the disk.
* **Queryable** — index on `timestamp`, `camera_id`, `status`.
* **Metadata only** — no imagery column exists, and `export_json` is asserted to contain
  none.

---

## 10. API: `backend/api`

A single `InferenceService` lives on `app.state.service`, created and released in the
lifespan handler so resources are released deterministically.

`POST /predict` has two modes:

* **With `landmarks`** — push one frame through the pipeline. This advances the state
  machine.
* **Without `landmarks`** — re-score the live buffer. This is deliberately a **read**:
  feeding the probability back into the confirmation logic would advance it twice per
  request. A test pins that.

`/health` does no I/O, so it can never fail slowly. `GET /status` publishes the whole
runtime state *including the privacy posture*.

---

## 11. Error handling summary

| Condition | Where handled | Result |
|---|---|---|
| Camera disconnected | `CameraSource` | warning, `None` frames, reconnect with backoff |
| Video file missing | `VideoFileSource` | `open_error` set, `None` frames |
| No pose detected | `PoseEstimator` | `None` → gap in the buffer |
| Bad / NaN frame | `PoseEstimator` | warning, `None` |
| Model asset missing | `FallPredictor` | raises; `create_predictor` falls back to mock |
| Contract mismatch | `FallPredictor` | `ModelContractError`, refuses to serve |
| Inference exception | `FallPredictor` | error log, probability `0.0` |
| Empty sequence | `SequenceBuffer` | `window() is None`, detector skips the window |
| Malformed request | FastAPI + pydantic | 422 |
| Buffer not ready | route | 409 with actionable text |
| Unknown event id | route | 404 |
| Unhandled route error | app handler | 500 with a traceback in the log, no internals leaked |
| Telegram unreachable | `TelegramAlertSender` | retries with backoff, event already persisted |
| Persistence failure | `InferenceService` | logged, reported, never crashes the loop |

There is no bare `except: pass` anywhere. Every handler either returns a defined value or
logs with `exc_info=True`.

---

## 12. Logging

One entry point, `get_logger(name)`, namespaced under `fallguard`.

* Console handler on **stderr**, so stdout stays clean for CLI output.
* Optional rotating file handler (5 MB × 3).
* JSON formatter available via `FALLGUARD_LOG_JSON=true`.
* **`SecretRedactionFilter` on every handler** — masks Telegram-shaped tokens and any
  `*_TOKEN` / `*_SECRET` / `*_PASSWORD` / `*_API_KEY` / `*_CHAT_ID` assignment, in the
  message, in the args and in exception text. The token pattern uses a narrow negative
  lookbehind rather than `\b`, because in a request URL the token directly follows `bot`
  and `t1234567890` has no word boundary there.
* MediaPipe, TensorFlow, matplotlib, urllib3 and httpx are pinned to `ERROR` so their
  native chatter cannot bury the application's own records.

The key events, all logged: application startup, model loaded, camera connected, pose
unavailable, fall confirmed, alert sent, alert failed, API errors.

---

## 13. Privacy architecture

The dataflow is the guarantee:

```text
Camera → Frame → Pose landmarks → Skeleton features → Model
                └─ in memory only ─┘
```

After landmark extraction the frame is never needed and is never retained. There is no
code path that writes a frame to disk. `FALLGUARD_STORE_RAW_VIDEO` defaults to `false` and
is a documented, auditable flag.

`/pose/latest` returns coordinates rather than pixels, so a client can display posture
without ever receiving imagery. The dashboard draws a skeleton by default.

No face or identity recognition is implemented. See [`privacy.md`](privacy.md).
