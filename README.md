# FallGuard AI

**Prototype fall-event detection from camera pose data.**

FallGuard AI watches a camera, extracts a 33-point human skeleton with MediaPipe Pose,
runs a short window of that skeleton through an LSTM, and — only after the evidence holds
up over time — raises a fall event, stores it, and sends a Telegram alert to a
Streamlit dashboard.

> ### ⚠ This is a prototype, not a medical device
>
> FallGuard AI flags **possible** fall events so a human can check on a person. It does
> **not** diagnose injuries, does not assess a person's condition, and must not replace
> emergency services or human supervision. A notification is a prompt to look, not a
> clinical assessment. Never rely on it as the sole means of detecting a fall.

---

## A note on `PRD.md`

`PRD.md` in this repository is **empty (0 bytes)**. The requirements this implementation was
built against are the detailed specification supplied with the task, and every decision
derived from it is recorded as an explicit assumption (A1–A14) in
[`IMPLEMENTATION_PLAN.md §8`](IMPLEMENTATION_PLAN.md#8-assumptions). If a real PRD is
supplied later, that section is the place to reconcile the two.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Requirements](#3-requirements)
4. [Installation](#4-installation)
5. [Configuration](#5-configuration)
6. [Dataset setup](#6-dataset-setup)
7. [Training](#7-training)
8. [Evaluation](#8-evaluation)
9. [Running the backend](#9-running-the-backend)
10. [Running the frontend](#10-running-the-frontend)
11. [Telegram setup](#11-telegram-setup)
12. [Testing](#12-testing)
13. [Privacy](#13-privacy)
14. [Limitations](#14-limitations)
15. [Project layout](#15-project-layout)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. Project overview

### What it does

```text
Camera / Video
      ↓
OpenCV                       frame capture
      ↓
MediaPipe Pose               33 landmarks [x, y, z, visibility]
      ↓
Preprocessing                hip-centred, torso-aligned, torso-scaled → 145-D feature vector
      ↓
Temporal Sequence            ring buffer of 30 frames
      ↓
LSTM / BiLSTM                fall probability ∈ [0, 1]
      ↓
Event Confirmation          threshold + consecutive frames + cooldown
      ↓
FastAPI                      /health /status /predict /events /pose/latest
      ↓
Streamlit Dashboard          status, live prediction, event history, skeleton
      ↓
Telegram Alert               only for a *confirmed* event
```

### Design decisions that matter

**A prediction is not an alert.** A single `FALL` output means nothing. An event requires
`p >= CONFIDENCE_THRESHOLD` on `MIN_FALL_FRAMES` **consecutive** windows, then a
`COOLDOWN_SECONDS` lockout. This is what stops a bad landmark, a passer-by or a detector
glitch from paging a human at 3 a.m.

**Pose dropouts are preserved, not hidden.** A frame with no person is stored as a
zero-vector with a `present = 0` flag rather than being skipped. Skipping it would
compress real time and quietly hide the fact that the camera lost the subject.

**Training and inference share one preprocessor.** `ai/training/dataset.py` imports the
same `PosePreprocessor` class that `ai/inference/pipeline.py` uses. The classic
train/serve skew is prevented structurally, not by discipline.

**Splits are over recordings, not windows.** Windows from one video are near-duplicates.
Splitting them at random inflates every metric. `ai/training/dataset.py` assigns whole
recordings to splits and a unit test asserts no group ever appears twice.

**The evaluator refuses to invent numbers.** If the test split is empty, or contains one
class, `scripts/evaluate.py` exits non-zero with an explanation instead of printing a
score. Synthetic results are prefixed `SYNTHETIC_` and carry a note saying so.

### Current state

See [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md) for a precise breakdown of
what is complete, partial, and blocked, plus the **only** measured metric numbers.

---

## 2. Architecture

```text
┌──────────────────────────── frame sources ────────────────────────────┐
│  CameraSource (webcam / RTSP)   VideoFileSource   SyntheticSource     │
└──────────────────────────────┬────────────────────────────────────────┘
                               │ BGR ndarray
                ┌──────────────▼──────────────┐
                │ ai/pose/pose_estimator.py  │   MediaPipe PoseLandmarker (VIDEO)
                │   33 landmarks             │   returns None when no person
                └──────────────┬──────────────┘
                               │ PoseFrame | None
                ┌──────────────▼──────────────┐
                │ ai/preprocessing/          │   visibility mask → hip centre
                │   pose_preprocessor.py     │   → torso rotation → torso scale
                │                            │   → 145-D feature vector
                └──────────────┬──────────────┘
                               │ (145,) + presence flag
                ┌──────────────▼──────────────┐
                │ ai/inference/               │   ring buffer, gaps preserved
                │   sequence_buffer.py        │   → (30, 145) window
                └──────────────┬──────────────┘
       ┌──────────────────────▼──────────────────────┐
       │ ai/models/predictor.py                      │
       │   Keras LSTM/BiLSTM  or  MockFallModel      │
       └──────────────────────┬──────────────────────┘
                              │ p(fall)
                ┌─────────────▼─────────────┐
                │ ai/inference/             │   threshold → consecutive
                │   fall_detector.py        │   → cooldown → FallEvent
                └─────┬───────────────┬──────┘
                      │               │
          ┌───────────▼──────┐  ┌─────▼──────────────────┐
          │ backend/database/│  │ backend/alerts/       │
          │  SQLite store    │  │  Telegram + dry run   │
          │  (persist first) │  └─────┬──────────────────┘
          └───────────┬──────┘        │
                      └───────┬───────┘
                              │
                ┌─────────────▼──────────────┐
                │ backend/api  FastAPI       │
                │  /health /status /predict  │
                │  /events /pose/latest      │
                └─────────────┬──────────────┘
                              │ JSON
                ┌─────────────▼──────────────┐
                │ frontend/streamlit_app.py  │
                │  skeleton view, no footage │
                └────────────────────────────┘
```

Package dependencies point one way only: `backend → ai`. The `ai` package has no
knowledge of HTTP, Telegram or SQLite, which is what lets the CLI demo and the API run
identical inference code.

Detailed design notes: [`docs/architecture.md`](docs/architecture.md).

---

## 3. Requirements

| | Version | Why |
|---|---|---|
| Python | ≥ 3.11 (tested on 3.12) | `X \| None` unions, `dataclass(slots=True)` |
| numpy | ≥ 1.26 | feature tensors |
| opencv-contrib-python-headless | ≥ 4.9 | frame capture. The *headless* build avoids an X11 dependency on servers |
| mediapipe | ≥ 1.0 | pose landmarks. **1.x exposes only the Tasks API** — see below |
| tensorflow-cpu | ≥ 2.16 | Keras 3. CPU-only keeps the test suite GPU-free |
| fastapi + uvicorn | ≥ 0.110 | API |
| streamlit | ≥ 1.32 | dashboard. `st.fragment` requires ≥ 1.37 |
| scikit-learn / pandas | | metrics and reports |
| pytest | ≥ 8 | tests |

**No GPU is required for anything**, including training and testing. `tensorflow-cpu` is
used deliberately.

### A note on mediapipe 1.x

MediaPipe 1.0 **removed** the legacy `mp.solutions.pose` namespace. The pose estimator
therefore uses the Tasks API and downloads a model asset:

```bash
python scripts/download_mediapipe_model.py
```

This stores `pose_landmarker_lite.task` (5.5 MB) in `ai/pose/mediapipe_models/`. The
asset is git-ignored. If it is missing, the system still runs: `POST /predict` accepts
landmark arrays directly, and the API reports the pose estimator as unavailable.

---

## 4. Installation

```bash
git clone <your-repo-url> fallguard-ai
cd fallguard-ai

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

# One-time: fetch the MediaPipe pose asset
python scripts/download_mediapipe_model.py

cp .env.example .env               # optional; defaults work out of the box
```

### Quick sanity check — no camera, no dataset, no checkpoint

```bash
python scripts/generate_sample_data.py   # render a synthetic clip + landmark datasets
python scripts/demo.py                   # run the full pipeline offline
```

`scripts/demo.py` prints a `DEMO MODE` banner and a frame-by-frame trace ending in a
confirmed fall event, stored in SQLite, with a Telegram alert rendered in dry run.

---

## 5. Configuration

Every tunable lives in `ai/config.py` and is overridable with a `FALLGUARD_` environment
variable or an entry in `.env`. No module reads an environment variable directly for a
tunable — that is what keeps training and inference from drifting apart.

| Variable | Default | Meaning |
|---|---|---|
| `FALLGUARD_SEQUENCE_LENGTH` | `30` | Frames per model window. ~1 s at 30 FPS |
| `FALLGUARD_CONFIDENCE_THRESHOLD` | `0.80` | Minimum probability for a window to count as evidence |
| `FALLGUARD_MIN_FALL_FRAMES` | `8` | Consecutive evidence windows before an event fires |
| `FALLGUARD_COOLDOWN_SECONDS` | `30` | Alert lockout after a confirmed event |
| `FALLGUARD_MAX_MISSING_RATIO` | `0.5` | Largest share of pose-less frames still scorable |
| `FALLGUARD_MODEL_PATH` | `models/fallguard_lstm.keras` | Checkpoint location |
| `FALLGUARD_MODEL_BACKEND` | `keras` | `keras` or `mock` (deterministic heuristic) |
| `FALLGUARD_MIN_LANDMARK_VISIBILITY` | `0.5` | Below this, a landmark is treated as missing |
| `FALLGUARD_ROTATION_NORMALIZATION` | `true` | Align the torso to vertical |
| `FALLGUARD_CAMERA_ID` | `camera_01` | Camera label recorded on events |
| `FALLGUARD_CAMERA_SOURCE` | `0` | Webcam index, RTSP URL, or a video path |
| `FALLGUARD_DB_PATH` | `data/fallguard.db` | SQLite event store |
| `FALLGUARD_ENABLE_MONITOR` | `false` | Run the camera thread inside the API process |
| `FALLGUARD_STORE_RAW_VIDEO` | `false` | Kept false always — see [Privacy](#13-privacy) |
| `FALLGUARD_TELEGRAM_BOT_TOKEN` | *(empty)* | Bot token. Empty ⇒ dry run |
| `FALLGUARD_TELEGRAM_CHAT_ID` | *(empty)* | Destination chat |
| `FALLGUARD_TELEGRAM_DRY_RUN` | `true` | Render and log alerts without sending |
| `FALLGUARD_LOG_LEVEL` | `INFO` | `DEBUG` … `ERROR` |
| `FALLGUARD_LOG_JSON` | `false` | JSON log records instead of console text |

Full list with comments: [`.env.example`](.env.example). Source of truth:
[`ai/config.py`](ai/config.py).

### Tuning guidance

* **`SEQUENCE_LENGTH`** — 30 frames (~1 s) covers a descent plus ground contact. Longer
  adds latency without adding information about a single fall; shorter starts missing the
  fall. **Changing it invalidates existing checkpoints** — the predictor refuses to serve
  a mismatched model, and tells you to retrain.
* **`CONFIDENCE_THRESHOLD`** — higher means fewer false alarms and more missed falls. For
  a prototype that alerts a human, bias towards false negatives. Tune on a **real**
  validation split; the synthetic smoke test is not a basis for choosing a threshold.
* **`MIN_FALL_FRAMES`** — the main false-positive control. 8 frames ≈ 0.27 s of sustained
  evidence.
* **`COOLDOWN_SECONDS`** — one physical fall should produce one alert, not one per window.

---

## 6. Dataset setup

> **The repository ships no dataset.** `scripts/generate_sample_data.py` creates
> *synthetic* data for pipeline testing only. For any real accuracy claim, supply a real
> fall dataset — see [`docs/model.md`](docs/model.md).

### Option A — synthetic (smoke test only)

```bash
python scripts/generate_sample_data.py --clear --check-detection
```

Produces 18 rendered recordings in `data/samples/clips/` and their extracted landmarks in
`data/processed/synthetic/{fall,non_fall}/`. Every file is watermarked `SYNTHETIC` and
carries a sidecar saying so. A model trained on this validates the *software*, nothing
about fall detection on people.

### Option B — real videos

```text
data/raw/
├── fall/       subject01_fall_01.mp4  subject02_fall_01.mp4 ...
└── non_fall/   subject01_walk_01.mp4  subject01_sit_01.mp4  ...
```

`ai.training.dataset.label_for_path` infers the label from the folder name, checking the
negative patterns first (because `non_fall` contains `fall`).

### Option C — pre-extracted landmarks (fastest to train)

```text
data/processed/
├── fall/       subject01_fall_01.npy + subject01_fall_01.json
└── non_fall/   subject01_walk_01.npy + subject01_walk_01.json
```

* `*.npy` — `(T, 33, 4)` float array of MediaPipe landmarks.
* `*.json` sidecar — optional:

```json
{
  "label": 1,
  "group_id": "subject01_fall_01",
  "source": "subject01_fall_01.mp4",
  "frame_labels": [0, 0, 0, 1, 1, 1, 1, 1]
}
```

`frame_labels` is a **per-frame** 0/1 temporal annotation. It matters more than it looks:
a fall recording is not entirely a fall. Without it, the seconds of ordinary standing
before the fall get labelled `1`, and the model learns to fire on people standing still —
the single most common source of false alarms. When present, a window is labelled `1` if
at least `FALL_LABEL_WINDOW_RATIO` (0.5) of its frames are fall-annotated.

`group_id` is what the splitter uses. Set it to the **subject**, not the clip, if a
subject appears in several recordings — otherwise the same person lands in both train and
test and the score is optimistic.

---

## 7. Training

```bash
# Smoke test on synthetic data (no real dataset required)
python scripts/generate_sample_data.py --clear
python scripts/train.py --synthetic --epochs 150

# Train on a real dataset
python scripts/train.py --data data/processed --epochs 60
python scripts/train.py --data data/processed --architecture bilstm --output models/fallguard_bilstm.keras
```

Training writes two files:

| File | Purpose |
|---|---|
| `models/fallguard_lstm.keras` | the weights |
| `models/fallguard_lstm.meta.json` | the **serving contract** — feature spec version, feature dim, sequence length, architecture, class balance, measured metrics, hyperparameters |

The sidecar is not decoration. `FallPredictor` reads it and **refuses to serve** a
checkpoint whose feature layout, sequence length or feature-spec version disagrees with
the running code. Without that check, a stale checkpoint would silently produce
meaningless probabilities.

### Splitting

Splits are assigned over whole **groups**, deterministically, from a hash of
`(seed, group_id)`. Not one recording appears in two splits — this is enforced by
`assert_no_group_leakage` in the trainer and by
`tests/unit/test_dataset.py::test_random_window_level_split_would_leak`, which pins the
failure mode.

The test split's positive rate is data-dependent. With few fall recordings, the positive
class can end up almost absent from a split; the split logger reports the rate per split
so you can see it.

### If the split is degenerate

`train_model` raises `InsufficientDataError` rather than training on nothing. If the
*test* split is empty, training still completes and the message says plainly that no
metrics were produced — it does not report zeros.

---

## 8. Evaluation

```bash
python scripts/evaluate.py                                        # the trained checkpoint
python scripts/evaluate.py --mock                                 # the mock backend
python scripts/evaluate.py --data data/processed --threshold 0.75
```

Writes to `reports/`:

| File | Contents |
|---|---|
| `*_metrics.json` | accuracy, precision, recall, F1, support, confusion matrix, provenance |
| `*_classification_report.txt` | scikit-learn per-class report plus the confusion matrix |
| `*_confusion_matrix.png` | rendered matrix |

Artefacts derived from synthetic data are prefixed **`SYNTHETIC_`** and embed a note
stating that the numbers describe the software pipeline, not fall-detection performance.

The evaluator **exits non-zero** rather than reporting a number when:

* there are no test samples,
* the test split contains only one class (so positive-class precision/recall are
  undefined).

If you are reading a metric in this repository, check `dataset_kind` in the JSON before
quoting it.

---

## 9. Running the backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
# development reload:
uvicorn backend.main:app --reload
```

Interactive API docs: <http://localhost:8000/docs>

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness. Does no I/O, so it never fails slowly |
| `GET` | `/status` | Model, camera, config, pipeline counters, alert state, privacy posture |
| `POST` | `/predict` | Score a frame (send `landmarks`) or the live buffer (send `{}`) |
| `GET` | `/events` | Stored events, newest first. Filter by `camera_id`, `status` |
| `GET` | `/events/{event_id}` | One event |
| `PATCH` | `/events/{event_id}` | Acknowledge, or mark as a false positive |
| `GET` | `/pose/latest` | Latest 33 landmarks — **no pixels** |

### Example

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
        "camera_id": "living_room",
        "landmarks": [[0.5,0.13,0.0,1.0], "…32 more…"]
      }'
```

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
  "event_id": "5a51...",
  "model_backend": "keras",
  "message": "Fall event confirmed and stored: 5a51..."
}
```

`event_confirmed` is the field that matters. `prediction: "FALL"` on its own means the
window is suspicious; `event_confirmed: true` means the confirmation policy was satisfied,
the event was written to SQLite, and an alert was attempted.

### Running the camera inside the API

```bash
FALLGUARD_ENABLE_MONITOR=true FALLGUARD_CAMERA_SOURCE=0 uvicorn backend.main:app
```

Off by default, which keeps the API deterministic and testable. A camera failure
degrades to a logged warning and `null` frames, never to a crash.

---

## 10. Running the frontend

```bash
streamlit run frontend/streamlit_app.py
```

| Panel | Contents |
|---|---|
| **System status** | `System / Camera / Model` and the pose-detection rate, exactly as specified |
| **Live prediction** | `Activity: NORMAL \| FALL`, confidence bar, sequence state, pose state, and a **skeleton** |
| **Event history** | Timestamp, prediction, confidence, camera, status, alerted, event id. Acknowledge inline |
| **Privacy** | What is stored, what is never stored, and the non-medical disclaimer |

The live panels refresh via `st.fragment(run_every=…)`, which re-runs the panels on a
timer **without blocking the script thread**. The skeleton is drawn with matplotlib from
landmark coordinates; the dashboard never displays camera footage.

Point it at a different backend with the sidebar field or `FALLGUARD_API_URL`.

---

## 11. Telegram setup

1. **Create a bot** — message [@BotFather](https://t.me/BotFather) → `/newbot` → copy the
   token.
2. **Get your chat id** — message [@userinfobot](https://t.me/userinfobot), or add the bot
   to the target group and read the id from `getUpdates`.
3. **Configure** — in `.env`:

   ```dotenv
   FALLGUARD_TELEGRAM_BOT_TOKEN=123456789:AA...
   FALLGUARD_TELEGRAM_CHAT_ID=-1001234567890
   FALLGUARD_TELEGRAM_DRY_RUN=false
   ```

4. **Restart the backend.**

### Dry run

With no token, or with `FALLGUARD_TELEGRAM_DRY_RUN=true` (the default), the alert is
rendered and logged but **not sent**:

```text
🚨 FALL DETECTED

Time: 2026-01-01T12:00:05Z
Camera: camera_01
Prediction: FALL
Confidence: 97.4%
Peak confidence: 99.1%
Status: detected
Event ID: 5a514798a6874106a0f2c9d584d99b69

FallGuard AI is a prototype that detects possible fall events from a camera. It is not
a medical device and does not provide a diagnosis. Please verify manually.
```

`GET /status` reports `telegram.dry_run` and a `reason` so you can tell "not configured"
from "configured but disabled" without exposing the token.

### Security

* Tokens are read from the environment. **Never hard-code one; never commit `.env`.**
* Every log record passes through a redaction filter that masks Telegram-shaped tokens
  and any `*_TOKEN` / `*_SECRET` / `*_PASSWORD` assignment — including inside third-party
  exception text. A test asserts a token sent in a log request never appears in the
  output.
* Chat ids are masked to the last four characters in logs.
* Failed alerts are retried with exponential backoff. 4xx responses are not retried;
  429 and 5xx are.

### Failure behaviour

The event is **persisted before the alert is attempted**. A Telegram outage can delay a
notification but can never lose an event. When an alert fails, the event stays
`detected` (never falsely `alerted`) and the reason is recorded in its `note` field.

---

## 12. Testing

```bash
pytest                 # 356 tests, ~40 s, no GPU, no network
pytest -m slow         # training/end-to-end runs
pytest -m "not slow"   # explicit
pytest tests/unit -q   # unit only
```

| Suite | Location | Covers |
|---|---|---|
| Preprocessing | `tests/unit/test_preprocessing.py` | scale/rotation/translation invariance, missing landmarks, degenerate poses, feature layout, scale/rotation invariance of the kinematic block |
| Sequences | `tests/unit/test_sequence_buffer.py` | window shape and order, eviction, gap preservation, missing-ratio policy |
| Event confirmation | `tests/unit/test_fall_detector.py` | threshold, min-frames, cooldown, single-spike rejection, dropout not firing |
| Models | `tests/unit/test_models.py` | LSTM/BiLSTM build, checkpoint round trip, contract rejection, mock determinism |
| Alerts | `tests/unit/test_alerts.py` | formatting, dry run, retry/backoff, 4xx no-retry, **token redaction** |
| Storage | `tests/unit/test_repository.py` | CRUD, filters, pagination, retention pruning, concurrent writes, durability |
| Config | `tests/unit/test_config.py` | env overrides, validation, prefixed-variable isolation |
| Dataset | `tests/unit/test_dataset.py` | windowing, temporal labels, **group splits, leakage** |
| Evaluation | `tests/unit/test_evaluation.py` | metric maths, **refusal to report on empty input**, SYNTHETIC marking |
| Pose & sources | `tests/unit/test_pose_and_sources.py` | asset errors, NaN/empty frames, detector exceptions, camera/video failure paths |
| Frontend & logging | `tests/unit/test_frontend_and_logging.py` | skeleton rendering, formatters, redaction filter |
| Pipeline | `tests/integration/test_pipeline.py` | frame → event, dropouts, callback faults, NaN landmarks |
| Service | `tests/integration/test_service.py` | persist-then-alert, alert-failure isolation, monitor lifecycle |
| API | `tests/integration/test_api.py` | all endpoints, validation, 404/409/422, durability across restart |
| Dashboard | `tests/integration/test_dashboard.py` | real HTTP server, every panel, offline path |
| Training | `tests/integration/test_training_smoke.py` | train → checkpoint → sidecar → reload → serve (`-m slow`) |

The suite never requires a trained checkpoint: it uses `FALLGUARD_MODEL_BACKEND=mock` and
builds tiny Keras models in-process. See [`docs/testing.md`](docs/testing.md).

---

## 13. Privacy

> FallGuard AI is a prototype that supports the detection of **possible** fall events.
> It is **not** a medical device and does not provide a diagnosis.

### The dataflow is privacy-preserving by construction

```text
Camera → Frame → Pose landmarks → Skeleton features → Model
                └─ in memory only ─┘
```

Once landmarks are extracted, the frame is no longer needed and is never retained.

| Data | Stored? | Where |
|---|---|---|
| Raw video frames | **Never** | discarded after landmark extraction |
| Images / thumbnails | **Never** | not produced by any code path |
| Skeleton landmarks | Transiently | in memory, for the live skeleton view |
| Event metadata | **Yes** | SQLite: `event_id`, `timestamp`, `camera_id`, `prediction`, `confidence`, `status` |

### What is not implemented, on purpose

* No face recognition.
* No identity recognition.
* No biometric templates of any kind.
* No raw-video retention or upload.

### Enforcement

* `FALLGUARD_STORE_RAW_VIDEO` defaults to `false` and no code path writes frames.
* `.gitignore` excludes `data/raw/*`, `*.mp4`, `*.avi` and the MediaPipe asset.
* `EventRepository.export_json` is metadata-only, and a unit test asserts no imagery key
  can appear in an export.
* The dashboard draws a **skeleton** by default.
* `GET /pose/latest` returns coordinates, not pixels — so a client can show posture
  without receiving imagery.
* `GET /status` publishes the privacy posture, so an operator can see it at a glance.

Operational guidance (signage, retention, camera placement, consent, DPIA):
[`docs/privacy.md`](docs/privacy.md).

---

## 14. Limitations

Read this before drawing any conclusion from a number this repository produces.

### The model

* **No real dataset is included, and no benchmark has been run.** The only metrics present
  were measured on *synthetic rendered data* — a fictional stick figure. They validate
  the software pipeline. They say **nothing** about fall-detection performance on people.
* A 6-epoch or 150-epoch run on 313 windows is a smoke test. Real evaluation needs a
  real dataset, multiple subjects, and a subject-disjoint test split.
* Single class only: `FALL` / `NORMAL`. No severity, no recovery detection.
* The BiLSTM cannot be streamed frame-by-frame, so the default LSTM is the serving path.

### The system

* **Not a medical device.** No injury assessment, no clinical decision support.
* Alerts depend on a Telegram bot being reachable. There is no SMS, e-mail or local siren
  fallback.
* A fixed background assumption. Normalisation cancels translation, rotation and scale,
  but a person walking in front of a busy background, in loose clothing, or heavily
  occluded will degrade detection.
* Partial occlusion is handled (invalid landmarks are zeroed and flagged) but not
  reconstructed.
* No multi-camera fusion, no per-room zones, no escalation policy.
* The camera monitor is single-threaded; at high resolution on CPU, throughput is limited
  by MediaPipe.
* English-language Telegram formatting only.

### Engineering

* The event store is a single SQLite file. Fine for one camera; not a distributed store.
* No authentication on the API. **Do not expose it to an untrusted network.**
* `PATCH /events/{id}` has no audit trail beyond the status change.
* Retention is a row count (`max_events`), not a time window.

---

## 15. Project layout

```text
fallguard-ai/
├── ai/                     # everything model-side; no web framework
│   ├── config.py           #   single source of truth for all tunables
│   ├── logging_config.py   #   structured logging + secret redaction
│   ├── pose/               #   MediaPipe wrapper, landmark constants, synthetic renderer
│   ├── preprocessing/      #   normalisation + feature engineering
│   ├── models/             #   LSTM/BiLSTM, predictor, mock backend
│   ├── training/           #   dataset, group split, trainer
│   ├── inference/          #   sequence buffer, event confirmation, pipeline
│   ├── evaluation/         #   metrics + report writer
│   └── sources/            #   camera / video / synthetic frame sources
├── backend/
│   ├── main.py             #   FastAPI app factory
│   ├── api/                #   routes, schemas
│   ├── services/           #   inference service, camera monitor
│   ├── alerts/             #   Telegram client + message formatter
│   └── database/           #   SQLite event repository
├── frontend/
│   ├── streamlit_app.py    #   dashboard
│   └── skeleton.py         #   matplotlib skeleton renderer
├── data/{raw,processed,samples}/
├── models/                 # checkpoints (git-ignored) + README
├── reports/                # evaluation output (git-ignored)
├── tests/
│   ├── unit/               # 286 tests
│   └── integration/        # 70 tests + 8 slow
├── scripts/                # demo, train, evaluate, sample data, model download
├── docs/                   # architecture, model, api, testing, deployment, privacy
├── IMPLEMENTATION_PLAN.md
└── IMPLEMENTATION_STATUS.md
```

---

## 16. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `MediaPipe pose asset not found` | asset not downloaded | `python scripts/download_mediapipe_model.py` |
| `MediaPipe is not installed` | deps missing | `pip install -r requirements.txt` |
| `Model checkpoint not found` | never trained | `python scripts/generate_sample_data.py && python scripts/train.py --synthetic --epochs 150` |
| `Checkpoint feature spec vX does not match vY` | code changed under an old checkpoint | retrain; the message names the mismatch |
| `FALLGUARD_SEQUENCE_LENGTH=30 but the checkpoint expects 45` | window length changed | retrain, or set the variable to match |
| `Cannot reach the FallGuard API` | backend not running | `uvicorn backend.main:app --port 8000` |
| `POST /predict` returns **409** | no landmarks sent and the buffer is not full | send a `landmarks` array, or enable the monitor |
| `pose_detected: false` on every frame | camera disconnected, or no person visible | check `FALLGUARD_CAMERA_SOURCE`; `GET /status` shows the pose-estimator state |
| Alerts never arrive | dry run, or no token | `GET /status` → `telegram.reason` |
| `sklearn` import error | sklearn missing | `pip install -r requirements.txt` |
| Slow tests | MediaPipe loading per test | the suite substitutes a stub pose estimator; see `tests/conftest.py` |
| `OSError: externally-managed-environment` | PEP 668 system Python | use a virtualenv (see [Installation](#4-installation)) |

---

## Documentation map

| Document | Contents |
|---|---|
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | Plan, architecture decisions, assumptions |
| [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md) | What is done / partial / blocked, measured results, run commands |
| [`docs/architecture.md`](docs/architecture.md) | Module-by-module design and dataflow |
| [`docs/model.md`](docs/model.md) | Features, architectures, training on a real dataset |
| [`docs/api.md`](docs/api.md) | Endpoint reference with request/response examples |
| [`docs/testing.md`](docs/testing.md) | Test strategy and how to run each suite |
| [`docs/deployment.md`](docs/deployment.md) | Deployment, operations, monitoring, security |
| [`docs/privacy.md`](docs/privacy.md) | Privacy design and operational guidance |

---

**FallGuard AI is a prototype for detecting possible fall events. It is not a medical
device, does not diagnose, and must not replace human supervision or emergency services.**
