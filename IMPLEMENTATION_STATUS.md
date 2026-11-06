# Implementation Status

**Project:** FallGuard AI — prototype fall-event detection from camera pose data
**Date:** 2026-09-30
**Verdict:** MVP complete and verified end to end. **No real-dataset benchmark exists.**

---

## 0. Source requirements

`PRD.md` in this repository is **empty (0 bytes)**. The detailed specification supplied
with the implementation request was therefore used as the requirements source, and all
derived decisions are recorded as assumptions in
[`IMPLEMENTATION_PLAN.md §8](IMPLEMENTATION_PLAN.md#8-assumptions) (A1–A14).

The repository was otherwise bare: no source, no tests, no configuration. This is a
green-field build.

---

## 1. Completed

### Perception

| Item | Notes |
|---|---|
| MediaPipe Pose integration | Tasks API (`PoseLandmarker`), since mediapipe 1.x removed `mp.solutions` |
| 33 landmarks `[x, y, z, visibility]` | Verified on real frames, not just declared |
| Pose asset download script | `scripts/download_mediapipe_model.py`, idempotent, with integrity floor |
| Graceful per-frame failure | `None` frame → warning; never a crash (10 failure modes tested) |
| NaN/inf handling | Neutralised per landmark in the estimator; arrays rejected at the pipeline |
| Monotonic timestamps | Enforced for MediaPipe `VIDEO` mode, so a stalled camera clock cannot error |
| Deterministic resource release | Context manager + idempotent `close()` |

### Preprocessing

| Item | Notes |
|---|---|
| Visibility masking | Configurable threshold; invalid landmarks zeroed after transformation |
| Hip-centred translation | Falls back to shoulder midpoint |
| Torso-axis rotation | Correct sign — verified invariant to camera tilt |
| Torso-length scaling | Falls back to the bounding-box diagonal when hips are lost |
| Two coordinate views | Rotated (LSTM input) and unrotated (kinematic descriptors) — see [why](#why-two-coordinate-views) |
| 145-D feature vector | Versioned layout (`FEATURE_SPEC_VERSION = "1.0.0"`) |
| 12 kinematic descriptors | Hip height, torso lean, joint angles, bounding box, separations |
| Train/serve parity | One `PosePreprocessor` class, asserted by test |
| Graceful degradation | A bad frame becomes a zero vector with `present = 0`, never an exception |

### Sequence & model

| Item | Notes |
|---|---|
| Ring buffer | Configurable `SEQUENCE_LENGTH`; gaps preserved, not closed |
| Density gate | `is_ready` requires a full window **and** missing ratio within limit |
| Correct eviction counters | `poses_in_buffer` tracks the buffer, not the lifetime total |
| LSTM baseline | 56,453 parameters, 30×145 → 1 |
| BiLSTM variant | ~2× parameters; documented as non-streamable, so LSTM is the serving default |
| Checkpoint save/reload | Round trip verified to 1e-5 |
| Metadata sidecar | Serving contract: feature spec, dims, sequence length, metrics, hyperparameters |
| **Contract validation** | A mismatched checkpoint is **refused**, not silently served |
| Inference hardening | Shape mismatch or exception → `0.0` + error log, never a crash |
| Mock backend | Deterministic heuristic, 0 learned parameters, self-described as not a model |

### Training & evaluation

| Item | Notes |
|---|---|
| `.npy` landmark loader | With JSON sidecars for label, group and temporal annotation |
| Video → landmark extraction | On-the-fly MediaPipe decoding |
| Sliding-window builder | Configurable length and stride |
| **Temporal labels** | A window is a fall only if ≥50 % of its frames are fall-annotated |
| **Group-aware splitting** | Whole recordings to splits, deterministic, seeded |
| **Leakage assertion** | Enforced in the trainer; the random-split failure mode is itself a test |
| Class weighting | Inverse frequency, so imbalance does not collapse to one class |
| Callbacks | Early stopping + `ReduceLROnPlateau` (the latter only with a validation split) |
| Determinism | numpy, Python and TensorFlow seeded; op determinism enabled |
| Degenerate-data handling | `InsufficientDataError` with actionable guidance; never a silent bad run |
| Metrics | Accuracy, precision, recall, F1, confusion matrix, per-class report |
| **Refusal to fabricate** | Empty or single-class test split ⇒ exit non-zero, no number printed |
| Provenance | `SYNTHETIC_` prefix and embedded note, derived from the **data**, not the path |

### Event handling

| Item | Notes |
|---|---|
| Three-guard state machine | Threshold → consecutive frames → cooldown |
| Single-spike rejection | One `p = 0.99` never fires (tested) |
| Run reset on interruption | Evidence does not accumulate across a sub-threshold window |
| Run reset after an event | Prevents a second fall being confirmed on fewer frames |
| Missing signal ≠ evidence | An unscorable window resets the run and is counted |
| Injected clock | Cooldown tested deterministically, no sleeping |
| Configurable | `CONFIDENCE_THRESHOLD`, `MIN_FALL_FRAMES`, `COOLDOWN_SECONDS` |

### Backend

| Item | Notes |
|---|---|
| `GET /health` | No I/O, so it never fails slowly |
| `GET /status` | Model, camera, config, counters, alert state, **privacy posture** |
| `POST /predict` | Landmark mode (advances state) and buffer mode (read-only) |
| `GET /events` | Filters, pagination, `total` reflects the filters |
| `GET /events/{id}` | 404 for unknown ids |
| `PATCH /events/{id}` | Acknowledge / false-positive |
| `GET /pose/latest` | Landmarks only, never pixels |
| Buffer read-only re-score | Polling cannot manufacture events (tested) |
| Lifespan management | Deterministic creation and release |
| Global error handler | Traceback logged, no internals leaked |

### Storage

| Item | Notes |
|---|---|
| SQLite repository | Stdlib only, one table, indexed |
| Thread-safe | One lock; verified with 4 concurrent writers × 20 events |
| Bounded | `max_events` prunes oldest-first |
| Persist-then-alert | Ordering is deliberate and tested |
| Alert-failure honesty | A failed alert leaves the event `detected`, with the reason in `note` |
| Durability | Verified across a process restart |
| Metadata-only export | A test asserts no imagery key can appear |

### Alerts & dashboard

| Item | Notes |
|---|---|
| Telegram integration | Token from the environment only; never in source |
| Message format | Time / Camera / Confidence / Status, plus event id and the non-medical notice |
| Dry run by default | No token ⇒ rendered and logged, not sent; `reason` surfaced in `/status` |
| Retry policy | Exponential backoff; 4xx not retried, 429 and 5xx are |
| Credential redaction | Every handler filters tokens, in messages, args and exception text |
| Chat-id masking | Last four characters only |
| Dashboard | Status, live prediction, event history, privacy — all panels verified |
| **Skeleton by default** | matplotlib renderer; the UI never shows footage |
| Non-blocking refresh | `st.fragment(run_every=…)` instead of `sleep()` + `rerun()` |
| Acknowledgement | Inline from the history table |
| Offline path | Renders a clear error instead of crashing |

### Tooling, QA & documentation

| Item | Notes |
|---|---|
| `scripts/demo.py` | Offline demo, `DEMO MODE` banner, generates its own clip if needed |
| `scripts/generate_sample_data.py` | 18 synthetic recordings, watermarked, with a detection-rate check |
| `scripts/train.py` | `--synthetic` and `--data`, both architectures |
| `scripts/evaluate.py` | Reports to `reports/`, refuses to fabricate |
| Logging | Structured, stderr, optional file + JSON, redaction filter |
| Tests | **356 passing** (+8 slow), no GPU, no network, no checkpoint needed |
| Docs | `README.md` + 6 documents in `docs/`, `models/README.md` |
| `.gitignore` | Excludes `.env`, checkpoints, datasets, video, reports |
| `.env.example` | Documented, no credentials |

---

## 2. Partial

| Item | What works | What is missing |
|---|---|---|
| **Model accuracy** | Pipeline trains, checkpoints, reloads, scores | Trained only on synthetic data. No real-dataset model exists, so **no benchmark** |
| **Camera monitor** | Runs, reconnects with backoff, degrades on failure | Off by default. Not load-tested with real hardware; no multi-camera support |
| **BiLSTM** | Builds, trains, checkpoints | Not the serving path — it cannot stream frame-by-frame, so it adds window latency |
| **Video → landmarks** | `extract_samples_from_videos` works | Not wired to a CLI. A user must write a wrapper or export `.npy` first |
| **Event retention** | Row-count cap with oldest-first pruning | No time-based purge; no `updated_at` column |
| **Event audit** | `PATCH` changes status | No audit trail of who acknowledged what, and when |
| **Auth** | — | None. Documented as a blocker for untrusted networks |
| **CORS** | Works for the separate dashboard process | `allow_origins: ["*"]`; must be narrowed in production |
| **Structured logs** | Formatter available, redaction enforced | Log-to-metrics pipeline not built |

---

## 3. Blocked

| Item | Why | Unblock |
|---|---|---|
| **Real benchmark** | No real fall dataset is present or licensed | Supply a labelled fall dataset with subject ids and per-frame temporal annotations. See [`docs/model.md`](docs/model.md) §7 |
| **Real-dataset checkpoint** | Depends on the above | Same |
| **Threshold tuning** | Requires a labelled **real** validation set and a false-alarm/false-negative cost ratio | Same |
| **Telegram delivery test** | No bot token or chat id available | Create a bot via @BotFather, set `FALLGUARD_TELEGRAM_BOT_TOKEN` / `..._CHAT_ID`, `DRY_RUN=false` |
| **Real-camera validation** | No camera in this environment | Attach a camera, run `scripts/demo.py --camera`, then `--max-frames 600` and inspect the drop rate |
| **Load testing** | No production-like target defined | Define fps, camera count and latency budget, then benchmark |

---

## 4. Tests

```text
356 passed, 8 deselected in 61.72s     # pytest
  8 passed, 356 deselected in 15.64s   # pytest -m slow
```

| Suite | Count | Focus |
|---|---|---|
| `test_preprocessing.py` | 32 | invariance, missing landmarks, degenerate poses, feature layout, scale/rotation invariance of the kinematic block |
| `test_sequence_buffer.py` | 19 | window shape/order, eviction, gap preservation, density policy |
| `test_fall_detector.py` | 18 | threshold, min-frames, cooldown, spike rejection, dropout, run resets |
| `test_models.py` | 27 | LSTM/BiLSTM build, checkpoint round trip, contract rejection, mock determinism |
| `test_alerts.py` | 29 | formatting, dry run, retry/backoff, 4xx no-retry, **token redaction** |
| `test_repository.py` | 20 | CRUD, filters, pagination, pruning, 4-thread concurrency, durability |
| `test_config.py` | 33 | env override, validation, prefixed-variable isolation, paths |
| `test_dataset.py` | 34 | windowing, temporal labels, **group splits, leakage** |
| `test_evaluation.py` | 20 | metric maths, **refusal on empty input**, SYNTHETIC marking |
| `test_pose_and_sources.py` | 34 | asset errors, invalid frames, detector exceptions, camera/video failures, **real MediaPipe when available** |
| `test_frontend_and_logging.py` | 20 | skeleton rendering, formatters, redaction filter |
| `test_pipeline.py` | 19 | frame → event, dropouts, callback faults, NaN landmarks |
| `test_service.py` | 11 | persist-then-alert, failure isolation, monitor lifecycle |
| `test_api.py` | 34 | every endpoint, validation, 404/409/422, restart durability |
| `test_dashboard.py` | 6 | real loopback HTTP, every panel, offline path |
| `test_training_smoke.py` | 8 | train → checkpoint → sidecar → reload → serve |

### Quality gate — all verified

| Check | Result |
|---|---|
| `python -m compileall` | clean |
| Import every module (42) | 0 failures |
| `--help` on all 5 scripts | exit 0 |
| `pytest` | 356 passed |
| `pytest -m slow` | 8 passed |
| `uvicorn backend.main:app` | starts; `/health` 200, `/status` correct, `/docs` 200, `/events` 200, `/pose/latest` 404 before any frame |
| `streamlit run frontend/streamlit_app.py` | starts; health 200; no errors in the log |
| `AppTest` against a **live** server | 0 exceptions; all 8 metrics, 3 subheaders, event table with 1 row and correct columns |
| `AppTest` with the backend **down** | 0 exceptions; clear error, `Backend: OFFLINE` |
| `scripts/demo.py` | 195 frames, 99.5 % pose detection, 1 confirmed event, stored, alert rendered in dry run |
| `scripts/generate_sample_data.py` | 18 recordings, 48–100 % detection per clip |
| `scripts/train.py --synthetic` | checkpoint + sidecar written |
| `scripts/evaluate.py` | 3 report artefacts written |

---

## 5. Metrics

> **Every number below was measured on SYNTHETIC rendered data — a fictional stick figure
> produced by `ai/pose/synthetic.py`.** They validate the software pipeline. They say
> **nothing** about fall-detection performance on people. **No real benchmark exists.**

### Trained LSTM, held-out synthetic test split

Produced by `python scripts/train.py --synthetic --epochs 150` then
`python scripts/evaluate.py`. Artefacts: `reports/SYNTHETIC_evaluation_*`.

| Metric | Value |
|---|---|
| Accuracy | 0.9459 |
| Precision | 0.7143 |
| Recall | 0.7143 |
| F1 | 0.7143 |
| Samples | 74 windows |
| Positive rate | 0.0946 |
| Confusion matrix | `[[65, 2], [2, 5]]` (rows true, columns predicted) |
| Decision threshold | 0.80 |
| Split | train 179 / val 60 / test 74, 18 groups, group-disjoint |

**Interpretation, stated honestly:** precision 0.71 means roughly 2 of every 7 alerts would
be false. At 74 windows with 7 positives, the confidence interval on that F1 is very wide.
The dominant failure mode is **false negatives** — 2 of 7 falls missed — which is the
dangerous direction for a safety-adjacent tool.

### Demo run — `scripts/demo.py`

| Measurement | Value |
|---|---|
| Frames processed | 195 |
| Poses detected | 194 (**99.5 %**) |
| Windows scored | 166 |
| Events confirmed | 1 |
| Events stored | 1 |
| Alerts sent / failed | 0 / 1 (dry run — Telegram not configured) |
| Wall time | 25.7 s (7.6 fps, CPU) |
| Peak confidence on the event | 99.9 % |

### Synthetic generator — per-clip MediaPipe detection

18 clips, 195–260 frames each. Best 100 %, worst 48.1 % (`fall_late`, the slowest descent).
The two low clips are genuinely hard for the detector and their low rate is preserved
honestly rather than filtered out.

### Mock backend — kinematic heuristic

| Measurement | Value |
|---|---|
| Standing pose | 0.000 |
| Bending over | 0.000 |
| Lying on the floor | 1.000 |
| Empty window | 0.000 |
| Confusion on the synthetic test split | `[[65, 2], [5, 2]]` — accuracy 0.9054, F1 0.3636 |

A heuristic, not a model. It exists so the software is exercisable without a checkpoint,
and it is labelled as such everywhere it appears.

### Two earlier results, recorded for honesty

| Run | Result | Lesson |
|---|---|---|
| 4 fall recordings, 6 epochs | F1 **0.0000** | Group count, not window count, is the binding constraint on a smoke test |
| 4 fall recordings, 150 epochs | F1 **0.0000** | More epochs cannot fix too few positive windows in the training split |

Both led to generating 18 recordings instead of 7.

---

## 6. Bugs found and fixed during implementation

Recorded because each was a genuine defect caught by a test, not a typo.

| Defect | Impact | Fix |
|---|---|---|
| Rotation normalisation used the **wrong sign** | Amplified camera tilt instead of cancelling it — the model would have been *less* robust to exactly the variation normalisation exists for | Rotate by `−θ`; invariance test added |
| Losing both hips **rejected the whole frame** | A partial detection caused a hard gap | Fall back to the bounding-box diagonal as the scale reference |
| `torso_lean_deg` used the full 2-D norm | Floored at 45° for a perfectly vertical torso, destroying the cue | Use the horizontal component only |
| Kinematic features were computed on the **rotated** pose | `torso_lean_deg` identically zero, hip height sign flipped — the mock backend scored a lying pose at 0.0 | Split into `coordinates` (rotated) and `unaligned` (not rotated) |
| Token redaction regex used `\b` | A token following `bot` in a URL has no word boundary, so **the token was logged in clear text** | Narrow negative lookbehind instead of `\b` |
| `MockFallModel.predict` handled only one window | `scripts/evaluate.py --mock` crashed on a batch | Made it fully Keras-compatible (`(N, 1)`) |
| Pipeline accepted an all-`NaN` landmark array | Reported `pose_detected: true` while producing a gap, corrupting the detection rate | Reject non-finite arrays with a warning |
| `ReduceLROnPlateau` installed without a validation split | Keras warned every epoch and never fired | Install only when a validation split exists |
| Evaluation provenance read from the **path** | `data/processed` contains both real and synthetic data, so synthetic results were labelled "real" | Derive from the data (group ids) instead |
| Dashboard auto-refresh used `sleep()` + `rerun()` | Blocked the script thread every cycle; the app never settled | `st.fragment(run_every=…)` |
| Synthetic demo video used lossy `mp4v` | Compression artefacts cost **55 %** of MediaPipe's detections (99.5 % → 43 %) | Lossless `FFV1` in `.avi` |
| `SYNTHETIC` watermark drawn as a full-width top banner | The horizontal edge suppressed detection — 99.5 % → 53 % | Outlined corner caption instead |
| Fall clips labelled `1` end to end | The pre-fall seconds taught the model to fire on ordinary standing | Per-frame temporal labels; windows labelled by fall-frame share |

The last two are worth emphasising: the demo was made to work by *measuring* detection rate
under each rendering choice, not by guessing. `scripts/generate_sample_data.py
--check-detection` now fails below 80 % so a regression cannot slip back in.

---

## 7. How to run

### Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_mediapipe_model.py
cp .env.example .env
```

### Demo (no camera, no dataset, no checkpoint)

```bash
python scripts/generate_sample_data.py
python scripts/demo.py
```

### Full pipeline

```bash
python scripts/train.py --synthetic --epochs 150   # or --data data/processed
python scripts/evaluate.py
uvicorn backend.main:app --host 0.0.0.0 --port 8000
streamlit run frontend/streamlit_app.py
```

### Tests

```bash
pytest                 # 356 passed
pytest -m slow         # 8 passed
```

---

## 8. Known limitations

### The model

* **No real benchmark.** The only metrics are synthetic. Do not quote them as performance.
* Precision 0.71 on the smoke test means false alarms are likely, and 2 of 7 falls were
  missed — the dangerous direction.
* Trained on 313 windows from 18 rendered recordings. A toy dataset.
* Binary `FALL` / `NORMAL`. No severity, no recovery, no near-miss class.
* The BiLSTM cannot stream, so it adds window latency and is not the serving default.
* Normalisation assumes a visible torso. Busy backgrounds, loose clothing and heavy
  occlusion degrade detection. Partial occlusion is flagged but not reconstructed.

### The system

* **Not a medical device.** No injury assessment, no triage, no clinical decision support.
* Alerts depend solely on Telegram. No SMS, e-mail or local siren fallback.
* Single camera, single process. `--workers > 1` is wrong: each worker would open the
  camera and hold a partial window.
* **No API authentication.** Do not expose it to an untrusted network.
* CORS is `allow_origins: ["*"]`; narrow it in production.
* Event store is a single SQLite file — fine for one camera, not a distributed store.
* Retention is a row count, not a time window. No audit trail on acknowledgement.
* CPU throughput is roughly 8–30 fps at 640×480, bounded by MediaPipe.
* English-language alert formatting only.

---

## 9. Next steps

Ordered by value. The first item is the one that matters.

### 1. Get a real dataset and a real benchmark — **blocks every accuracy claim**

* Select a fall dataset with per-frame temporal annotations and subject identifiers. Check
  the licence and follow its terms.
* Split **by subject**, not by clip. Record the subject as `group_id`.
* Train, then `scripts/evaluate.py` on an untouched test split.
* Report accuracy, precision, recall, F1, the confusion matrix, the number of subjects,
  **a per-subject breakdown**, and the decision threshold. Aggregate F1 over a handful of
  subjects can hide total failure on two of them.
* Then tune `CONFIDENCE_THRESHOLD` against a false-alarm / false-negative cost ratio.

### 2. Raise the detection ceiling

* Most of the smoke test's misses come from *recall*, and the 48 % detection on slow
  descents suggests the fall frames themselves are the bottleneck. Train on real
  recordings with varied fall styles — forward, backward, sideways, with objects.
* Add temporal augmentation and `frame_labels`-consistent labelling for every real clip.
* Consider a second model for a different view (e.g. a top-down floor camera), which sees
  hip height far more directly than a frontal one.

### 3. Close the production gaps

* Add authentication to the API, or document the reverse-proxy requirement as enforced.
* Narrow CORS to the dashboard's origin.
* Add a time-based retention purge and an acknowledgement audit trail.
* Add a second alerting channel (SMS, e-mail, a local siren). A single Telegram dependency
  is a single point of failure for the notification path.

### 4. Make the responses real

* Rehearse the human response: who is alerted, who checks, within what time.
* Add a confirmation step (a caregiver acknowledges in the dashboard) so a missed alert is
  visible.
* Consider a complementary non-camera device. A system that only works via video should not
  be the sole option.

### 5. Engineering follow-ups

* Wire video → landmark extraction into a CLI so users need not write a wrapper.
* Add a `updated_at` column and an audit table.
* Add load testing once a throughput target is defined.
* Make the monitor thread's queue bounded, dropping **oldest** frames — for fall detection
  latency matters more than completeness.

---

## 10. Acceptance criteria

| Criterion | Status |
|---|---|
| PRD read and understood | ✅ `PRD.md` empty; the task specification used, assumptions A1–A14 recorded |
| `IMPLEMENTATION_PLAN.md` exists | ✅ |
| MediaPipe Pose works | ✅ real landmarker, verified on real frames |
| 33 landmarks extracted | ✅ asserted in tests |
| Preprocessing works | ✅ 32 tests, invariances verified |
| Sequence pipeline works | ✅ gaps preserved, density gated |
| LSTM/BiLSTM interface works | ✅ both build, train, save, reload |
| Model inference works with a checkpoint | ✅ round trip verified; contract validation active |
| Evaluation pipeline works | ✅ 3 artefacts; refuses to fabricate |
| FastAPI runs | ✅ verified with a live server |
| Event manager works | ✅ confirmed → persisted → alerted |
| Telegram works or has a mock fallback | ✅ dry-run fallback verified; live send untested (no token) |
| Streamlit dashboard runs | ✅ verified against a live backend and offline |
| Event history works | ✅ list, filter, detail, acknowledge, paginate |
| Privacy requirements met | ✅ no raw video stored, no identity recognition, skeleton-first UI |
| Unit tests exist | ✅ 286 unit tests |
| Integration tests exist | ✅ 70 integration/API/dashboard tests, plus 8 slow |
| `pytest` succeeds | ✅ 356 passed, 8 slow passed |
| README complete | ✅ 16 sections |
| `.env.example` exists | ✅ |
| No hard-coded credentials | ✅ asserted by tests; `.env` git-ignored |
| No fabricated AI results | ✅ synthetic results labelled everywhere; empty splits exit non-zero |
