# Privacy

> FallGuard AI is a prototype that supports the detection of **possible** fall events. It
> is **not** a medical device, does not provide a diagnosis, and does not assess a person's
> condition.

---

## 1. Data flow

```text
Camera
  ↓
Frame  ── in memory only, never written to disk ──┐
  ↓                                                │
MediaPipe Pose  →  33 landmarks                   │
  ↓                                                │
Preprocessing → skeleton feature vector            │
  ↓                                                │
LSTM → fall probability                           │
  ↓                                                │
Event confirmation → FallEvent                    │
  ↓                                                │
Event metadata → SQLite                           │
  ↓                                                │
Telegram alert / dashboard skeleton ◄─────────────┘
```

The privacy property is structural: **after landmark extraction, the frame is never needed
and never retained.** The only renderer that writes video to disk is the *synthetic* sample
generator, which draws a fictional figure.

---

## 2. What is stored

| Data | Stored? | Where | Retention |
|---|---|---|---|
| Raw video frames | **Never** | — | discarded after landmark extraction |
| Images / thumbnails | **Never** | — | no code path produces them |
| Skeleton landmarks | Transiently | process memory | until the next frame; the last one is kept only for the live view |
| Event metadata | **Yes** | SQLite `data/fallguard.db` | capped at `max_events` (default 5000), oldest pruned |

### Event record fields

```text
event_id, timestamp, camera_id, prediction, confidence, peak_confidence,
status, window_frames, source, notified, note, created_at
```

All of these are derived metadata. None identify a person, and none contain imagery.

---

## 3. What is deliberately not implemented

* **No face recognition.**
* **No identity recognition.**
* **No biometric templates of any kind** — no embeddings, no face crops, no gait signatures.
* **No raw-video retention or upload.**
* No third-party analytics or telemetry. The application makes no outbound request except
  the Telegram `sendMessage` call you configure.

---

## 4. How the guarantees are enforced

| Guarantee | Mechanism | Verified by |
|---|---|---|
| Frames are never persisted | no code path writes a frame; `FALLGUARD_STORE_RAW_VIDEO` defaults to `false` | manual review of `ai/sources`, `ai/pose` |
| Storage is metadata-only | `fall_events` has no imagery column | `test_repository.py` |
| Exports are metadata-only | `export_json` emits event rows only | `test_export_json_contains_metadata_only` |
| Datasets are not committed | `.gitignore` excludes `data/raw/*`, `*.mp4`, `*.avi`, `*.npy` | — |
| Checkpoints are not committed | `.gitignore` excludes `models/*.keras` and sidecars | — |
| Clients cannot obtain imagery | `/pose/latest` returns coordinates, not pixels | `test_pose_latest_returns_landmarks_only` |
| The UI defaults to a skeleton | `render_skeleton_figure` draws landmarks only | `test_dashboard.py` |
| The posture is auditable | `GET /status` publishes the privacy posture | `test_status_reports_the_privacy_posture` |
| Credentials never leak | redaction filter on every log handler | `test_alerts.py` |

---

## 5. The dashboard's privacy default

The live panel draws a **skeleton**, not a camera feed:

```text
Camera → Frame → Pose landmarks → Skeleton features → Model
                └─ in memory only ─┘
```

Rendering the skeleton rather than the video is deliberate. A skeleton conveys the
posture that matters for fall detection — hip height, torso angle, limb position — and
discloses nothing about the person's appearance or surroundings.

Note that this is a *display* choice as well as a storage one: the backend has no endpoint
that returns image data, so there is nothing for a client to display even if it wanted to.

---

## 6. Operational guidance

Software design cannot make a camera deployment acceptable. Before installing FallGuard AI
in a real space:

### Lawful basis and consent

* Establish a lawful basis for processing in your jurisdiction (GDPR Art. 6, CCPA, and
  equivalents). Legitimate interest is commonly cited for eldercare and workplace safety;
  consent is the alternative.
* **Inform** everyone in the monitored area. A sign at the camera is the minimum; a
  privacy notice covering the purpose, retention and recipients is better.
* Offer a **non-camera alternative** (a wearable call button) where one exists. A system
  that only works via video should not be the only option.

### Placement

* Avoid private spaces: bedrooms, bathrooms, changing areas.
* Avoid audio capture. FallGuard AI processes **video only**; do not add a microphone.
* Do not place a camera where a subject has no reasonable expectation of privacy.
* Angle the camera at the floor area that matters, not at rest positions — this limits
  what is captured and improves detection.

### Retention and access

* Keep the SQLite file on an encrypted volume.
* Restrict filesystem access; the event store contains timestamps and room labels that can
  reveal occupancy patterns.
* Set a retention policy. `max_events` bounds the row count; a time-based purge job should
  be added for a real deployment.
* Define who may read events, and log every acknowledgement (`PATCH /events/{id}` currently
  changes the status but has no audit trail — see `IMPLEMENTATION_STATUS.md`).

### Network

* The API has **no authentication**. Do not expose it beyond a trusted network. See
  [`deployment.md`](deployment.md).
* Telegram sends event metadata to a third party. If that crosses a legal boundary for
  your deployment, set `FALLGUARD_TELEGRAM_ENABLED=false`; events are still stored and the
  dashboard still works.
* On a shared network, use TLS.

### Incident response

* If the database is exposed, it contains no imagery and no identity — but it does contain
  timestamps and camera labels, which can be sensitive. Treat it accordingly.
* If `.env` is exposed, **rotate the Telegram bot token** via @BotFather. Tokens are
  redacted from logs, but a leaked `.env` is a leaked credential.

---

## 7. What this system does not do

* It does not identify people.
* It does not store footage.
* It does not transmit video.
* It does not diagnose, triage, or assess injury.
* It does not replace a personal alarm, a call button, or human supervision.

A FallGuard AI alert is a **prompt to check on a person**. The safety of the person
concerned still depends entirely on a human responding to it. Build that response process
before you trust the alert.
