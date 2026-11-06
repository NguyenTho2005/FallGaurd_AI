# FallGuard AI — Implementation Plan

> **Status of source requirements:** `PRD.md` in this repository is **empty (0 bytes)**.
> The detailed specification supplied with the implementation request is therefore used as the
> authoritative requirements document. All derived decisions are recorded as **assumptions**
> in section [8. Assumptions](#8-assumptions) and repeated in `docs/` and `IMPLEMENTATION_STATUS.md`.

---

## 1. Current state (repository analysis)

| Item | Finding |
|---|---|
| Repository | Not a git repository. Single file present: `PRD.md`, size **0 bytes**. |
| Source code | None. |
| Tests | None. |
| Configuration | None. |
| Documentation | Only the empty `PRD.md`. |
| Reusable assets | None. |

**Conclusion:** this is a green-field build. Nothing can be preserved or refactored, so the full
architecture from the specification is created from scratch. No existing behaviour is at risk.

### Local environment (measured, not assumed)

| Component | Version used | Notes |
|---|---|---|
| Python | 3.12.3 | venv created at `.venv` (PEP 668 externally-managed system env). |
| numpy | 2.5.3 | |
| opencv-contrib-python-headless | 5.0.0 | headless build → no GUI/X11 dependency on servers. |
| mediapipe | 1.0.1 | **Only the new `Tasks` API is exposed** (`mp.solutions.pose` was removed in 1.x). Pose code is written against `mediapipe.tasks.python.vision.PoseLandmarker`. |
| tensorflow-cpu | 2.21.0 (Keras 3.15) | CPU-only wheel, so tests never require a GPU. |
| fastapi / uvicorn | 0.142.2 / 0.54.0 | |
| streamlit | 1.64.0 | |
| scikit-learn / pandas | 1.9.1 / 3.0.6 | evaluation + tabular export. |
| pytest | 9.1.1 | |

MediaPipe 1.x does **not** bundle the pose model, so `pose_landmarker_lite.task` (5.5 MB) is
downloaded by `scripts/download_mediapipe_model.py` and stored at
`ai/pose/mediapipe_models/`. This is a code change versus the legacy `mp.solutions.pose` design and
is a direct consequence of the pinned dependency version.

---

## 2. Target architecture

```text
       ┌───────────────────────────── frame sources ─────────────────────────────┐
       │  CameraSource (webcam/RTSP)   VideoFileSource (*.mp4)   SyntheticSource  │
       └───────────────────────────────────┬──────────────────────────────────────┘
                                           │ BGR ndarray
                    ┌──────────────────────▼──────────────────────┐
                    │ ai/pose/pose_estimator.py                    │
                    │   MediaPipe PoseLandmarker (VIDEO mode)      │
                    │   33 landmarks × [x, y, z, visibility]       │
                    └──────────────────────┬──────────────────────┘
                                           │ PoseFrame | None (no pose)
                    ┌──────────────────────▼──────────────────────┐
                    │ ai/preprocessing/pose_preprocessor.py       │
                    │   visibility mask → hip-centre → torso      │
                    │   rotation → scale → feature vector (145-D) │
                    └──────────────────────┬──────────────────────┘
                                           │ (145,) float32 + present flag
                    ┌──────────────────────▼──────────────────────┐
                    │ ai/inference/sequence_buffer.py             │
                    │   ring buffer → window (SEQUENCE_LENGTH,145)│
                    └──────────────────────┬──────────────────────┘
                                           │
        ┌──────────────────────────────────▼─────────────────────────────────┐
        │ ai/models/predictor.py → LSTM / BiLSTM  (or MockFallModel)         │
        └──────────────────────────────────┬─────────────────────────────────┘
                                           │ p(fall)
                    ┌──────────────────────▼──────────────────────┐
                    │ ai/inference/fall_detector.py               │
                    │   confidence threshold → MIN_FALL_FRAMES    │
                    │   consecutive → cooldown → FallEvent        │
                    └───────────────┬──────────────────┬───────────┘
                                    │                  │
                    ┌───────────────▼──────────┐  ┌────▼──────────────────┐
                    │ backend/database/        │  │ backend/alerts/       │
                    │ EventRepository (SQLite) │  │ TelegramAlert +      │
                    │ event_id, timestamp,     │  │ formatter + dry-run  │
                    │ camera_id, prediction,   │  └────┬──────────────────┘
                    │ confidence, status       │       │
                    └───────────────┬──────────┘       │
                                    └────────┬─────────┘
                                             │
                    ┌────────────────────────▼────────────────────────┐
                    │ backend/api  FastAPI                            │
                    │  /health /status /predict /events /events/{id} │
                    │  /pose/latest  (skeleton only, no raw video)    │
                    └────────────────────────┬────────────────────────┘
                                             │ HTTP (JSON)
                    ┌────────────────────────▼────────────────────────┐
                    │ frontend/streamlit_app.py                      │
                    │  system status · live prediction · event       │
                    │  history · skeleton overlay                    │
                    └────────────────────────────────────────────────┘
```

### Privacy by construction

The dataflow is `frame → pose landmarks → skeleton features → model`. Raw frames are used **only**
in memory, are never persisted, and never leave the process. The only renderer that writes video to
disk is the *synthetic* sample generator, which draws a fictional figure. `/pose/latest` returns
landmark coordinates, never pixels. No face or identity recognition is implemented anywhere.

---

## 3. Repository layout

```text
fallguard-ai/
├── ai/
│   ├── config.py                  # single source of truth for all tunables
│   ├── logging_config.py          # structured logging, secret redaction
│   ├── pose/                      # MediaPipe wrapper, landmark constants, synthetic renderer
│   ├── preprocessing/             # normalisation + feature engineering
│   ├── models/                    # LSTM/BiLSTM builders, predictor, mock model
│   ├── training/                  # dataset assembly, group-safe split, trainer, CLI
│   ├── inference/                 # sequence buffer, event confirmation, pipeline
│   ├── evaluation/                # metrics + report writer
│   └── sources/                   # camera / video-file / synthetic frame sources
├── backend/
│   ├── main.py                    # FastAPI app factory
│   ├── api/                       # routes, schemas, dependencies
│   ├── services/                  # inference service, optional monitor thread
│   ├── alerts/                    # Telegram client + message formatter
│   ├── database/                  # SQLite event repository
│   └── logging_config.py          # re-export for convenience
├── frontend/streamlit_app.py
├── data/{raw,processed,samples}/
├── models/                        # checkpoints (.gitignored) + README
├── reports/                       # evaluation output
├── tests/{unit,integration,model}/
├── scripts/                       # demo, sample generation, model download, shell helpers
├── notebooks/
├── docs/{architecture,model,api,testing,deployment,privacy}.md
├── requirements.txt  .env.example  .gitignore  README.md  PRD.md
├── IMPLEMENTATION_PLAN.md  IMPLEMENTATION_STATUS.md
```

---

## 4. Tasks, dependencies and implementation order

Phases are strictly sequential; each phase ends with its own verification step.

| # | Phase | Tasks | Depends on | Exit check |
|---|---|---|---|---|
| 1 | Requirements & architecture | Read spec, repo audit, this plan, scaffold | — | tree + `requirements.txt` import cleanly |
| 2 | Pose & preprocessing | `landmarks.py`, `pose_estimator.py`, `synthetic.py`, `pose_preprocessor.py` | 1 | 33 landmarks extracted; normalisation unit-checked |
| 3 | Sequence & model | `sequence_buffer.py`, `lstm.py`, `predictor.py`, `mock_model.py` | 2 | window shape `(L,145)`; model builds + saves + reloads |
| 4 | Inference & evaluation | `fall_detector.py`, `pipeline.py`, `evaluate.py` | 3 | event confirmation unit-tested; report files written |
| 5 | Backend | `EventRepository`, `InferenceService`, schemas, routes, `main.py` | 4 | all 6 endpoints answer via `TestClient` |
| 6 | Alerts & frontend | `telegram.py`, `formatter.py`, `streamlit_app.py` | 5 | dry-run alert renders; dashboard imports & serves |
| 7 | Training & demo data | `dataset.py`, `trainer.py`, `generate_sample_data.py`, `demo.py` | 2–4 | training runs end-to-end on synthetic data |
| 8 | Tests & QA | unit, integration, API test modules; run `pytest` | all | `pytest` green |
| 9 | Documentation | `README.md`, `docs/*`, `IMPLEMENTATION_STATUS.md` | all | acceptance checklist satisfied |

**Dependency graph:** `config` → `preprocessing` → `models` → `inference` → `services` → `api`/`alerts` → `frontend`.
`training` and `evaluation` sit on `preprocessing` + `models`. Nothing imports `backend` from `ai`
(the dependency arrow points one way only: `backend → ai`).

---

## 5. Key design decisions

1. **One config object, imported everywhere.** `ai.config.get_settings()` reads environment
   variables prefixed `FALLGUARD_` plus an optional `.env`. No module reads `os.environ` directly for
   a tunable, so training and inference cannot drift apart.
2. **One preprocessor, used by training and inference.** `PosePreprocessor.transform()` is the only
   place landmark→feature conversion happens. `ai/training/dataset.py` imports that exact class. This
   structurally prevents the classic train/serve skew.
3. **Group-aware splitting.** Splits are computed over *videos* (or subjects) via a deterministic
   hash of the group id, never per window, so windows from one video cannot appear in both train and
   test. This is asserted by a unit test.
4. **Model backend is pluggable.** `FALLGUARD_MODEL_BACKEND=keras|mock`. Tests and the demo can run
   with no checkpoint present; production uses Keras. The mock model is a documented, deterministic
   kinematic heuristic — never presented as an AI result.
5. **Model metadata is stored beside the weights.** A JSON sidecar records the feature spec version
   and sequence length, and the predictor refuses to run when the checkpoint disagrees with the
   current config (fail fast instead of silently scoring garbage).
6. **Event confirmation is a state machine**, not a threshold: consecutive-frame counting plus a
   cooldown, both configurable, so one noisy frame cannot page a human.
7. **Alerts are decoupled from persistence.** Persist first, then alert; a Telegram outage can never
   lose an event, and alert failures are logged and retried with backoff.
8. **Logging with redaction.** A formatter masks any value that looks like a bot token, so
   `TELEGRAM_BOT_TOKEN` cannot leak through an exception traceback.

---

## 6. Testing strategy

| Suite | Location | Scope | Notes |
|---|---|---|---|
| Unit — preprocessing | `tests/unit/test_preprocessing.py` | hip-centring, scale invariance, rotation invariance, missing-landmark handling, frame-length invariance | pure numpy, no TF |
| Unit — sequences | `tests/unit/test_sequence_buffer.py` | window shape, ordering, not-ready state, missing-frame policy, overflow eviction | |
| Unit — event confirmation | `tests/unit/test_fall_detector.py` | threshold, min-frames, cooldown, single noisy frame rejected, pose-dropout does not fire | fake clock injected |
| Unit — model | `tests/unit/test_models.py` | LSTM/BiLSTM build, save/reload, input-dim validation, mock model determinism | real TF on CPU, small shapes |
| Unit — alerts | `tests/unit/test_alerts.py` | message formatting, dry-run, failure logging, token never rendered | `httpx` transport mocked |
| Unit — storage | `tests/unit/test_repository.py` | insert, list, filter, pagination, concurrent access | tmp SQLite |
| Unit — config | `tests/unit/test_config.py` | env override, validation errors, feature-dim consistency | |
| Unit — data hygiene | `tests/unit/test_dataset_split.py` | no group appears in two splits | |
| Integration | `tests/integration/test_pipeline.py` | frame → pose → features → sequence → model → event, with a stub pose estimator | |
| Integration | `tests/integration/test_api_events.py` | confirmed event is persisted and listed through the API | |
| API | `tests/integration/test_api.py` | `/health`, `/status`, `/predict` (valid, invalid, empty sequence), `/events`, `/events/{id}`, `/pose/latest` | FastAPI `TestClient` |

* **No GPU dependency.** `tensorflow-cpu` is pinned; the model tests use 4-unit sequences.
* **No network dependency.** Telegram tests inject a mock transport; `TestClient` is in-process.
* **No checkpoint dependency.** API/integration tests run with `FALLGUARD_MODEL_BACKEND=mock`.
* **Markers.** `slow` (training/eval smoke tests) and `model` are deselected by default via
  `pytest.ini` addopts, and can be run with `pytest -m "not slow"` / `pytest -m slow`.

---

## 7. Expected deliverables

```text
ai/            7 sub-packages, import-clean, fully exercised
backend/       FastAPI app + 6 endpoints + SQLite store + Telegram alerts
frontend/      Streamlit dashboard (status, live prediction, history, skeleton)
scripts/       demo.py, generate_sample_data.py, download_mediapipe_model.py, train.py
tests/         ~90+ unit/integration/API test cases, all green
models/        README explaining how to obtain/produce a checkpoint (weights gitignored)
reports/       produced by scripts/evaluate.py, generated at run time
docs/          architecture, model, api, testing, deployment, privacy
requirements.txt  .env.example  .gitignore  README.md
IMPLEMENTATION_PLAN.md  IMPLEMENTATION_STATUS.md
```

---

## 8. Assumptions

Recorded here because the PRD is empty. Each is conservative and reversible via environment
variables.

| # | Assumption | Rationale | Override |
|---|---|---|---|
| A1 | The task specification is the requirements source. | `PRD.md` is 0 bytes; refusing to start would deliver nothing. | — |
| A2 | `SEQUENCE_LENGTH = 30` frames at ~30 FPS ≈ 1 s of motion, which covers the descent + ground-contact of a fall. | 1 s is the shortest window that still contains a fall; a longer window adds latency. | `FALLGUARD_SEQUENCE_LENGTH` |
| A3 | 33 landmarks → 145-D feature vector (99 normalised xyz + 33 visibility + 12 kinematic descriptors + 1 presence flag). | Keeps every landmark while adding the discriminative joint angles and hip-drop cues. | implemented in `ai/preprocessing/features.py` |
| A4 | Normalisation = visibility mask → hip-centre translation → torso-axis rotation → torso-length scaling. | Translation/rotation/scale invariance is what makes the model robust to camera position, tilt and subject size. | implemented in `ai/preprocessing/pose_preprocessor.py` |
| A5 | Frames with no detectable pose are **kept in the window and zero-filled** with `present=0`, rather than being dropped. | Dropping them would compress real time and hide dropouts; the presence flag lets the model and the detector see the gap. | `FALLGUARD_MAX_MISSING_RATIO` |
| A6 | `CONFIDENCE_THRESHOLD = 0.80`, `MIN_FALL_FRAMES = 8`, `COOLDOWN_SECONDS = 30`. | ~0.27 s of sustained evidence, then a 30 s lockout, keeps alert volume low. Deliberately conservative for a safety-adjacent tool. | env vars |
| A7 | Two classes only: `FALL` and `NORMAL`. | MVP scope; matches the binary confirmation logic. | — |
| A8 | Binary cross-entropy + sigmoid head (probabilistic, thresholdable). | A probability is required by the confidence-threshold confirmation logic. | — |
| A9 | MediaPipe Tasks API + downloaded `.task` asset. | Forced by `mediapipe==1.0.1`, which removed `mp.solutions`. | `scripts/download_mediapipe_model.py` |
| A10 | SQLite at `data/fallguard.db` via the stdlib `sqlite3` module. | Specification allows "SQLite or an equally lightweight option"; zero extra dependency. | `FALLGUARD_DB_PATH` |
| A11 | No face/identity recognition, no raw-video retention, no medical claims. | Hard privacy requirement in the specification. | non-negotiable |
| A12 | Accuracy/precision/recall figures are **not** reported unless produced from a real dataset. | Explicitly forbidden to fabricate. A synthetic smoke test is labelled as such everywhere. | see `IMPLEMENTATION_STATUS.md` |
| A13 | The optional background monitor thread that drives the camera is **off by default**; `/predict` operates on the live buffer or on client-supplied landmarks. | Keeps the API deterministic and testable; a camera thread is a deployment concern, not an MVP one. | `FALLGUARD_ENABLE_MONITOR=true` |
| A14 | Synthetic demo data is rendered from a **fictional** stick-figure animator and is labelled `SYNTHETIC` in the code, in the report filenames and in the console banner. | A self-contained demo is required, and it must not be mistaken for a benchmark. | `scripts/generate_sample_data.py` |
