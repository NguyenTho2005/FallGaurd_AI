# Testing

```bash
pytest                 # full suite
pytest -m slow         # training / end-to-end runs
pytest -m "not slow"   # default: everything except slow
pytest tests/unit -q   # unit only
pytest tests/integration -q
```

`slow` is deselected by default via `pytest.ini`, so `pytest` alone stays fast.

---

## Principles

1. **No GPU.** `tensorflow-cpu` is pinned. Model tests build tiny networks in-process
   (6 frames × 8 features) rather than loading a checkpoint.
2. **No network.** Telegram tests inject a fake HTTP client; the API tests use
   `TestClient`; the dashboard tests spin up a real loopback `http.server`.
3. **No trained checkpoint.** The suite runs with `FALLGUARD_MODEL_BACKEND=mock`, so it is
   meaningful on a fresh clone.
4. **No MediaPipe requirement for the default run.** Tests that need the native landmarker
   skip themselves when the asset is absent. Download it to enable them:
   ```bash
   python scripts/download_mediapipe_model.py
   ```
5. **No sleeping.** `FallDetector` and `SequenceBuffer` take an injected clock, so cooldown
   behaviour is verified deterministically.
6. **Isolation.** Every test gets a `Settings` object rooted in `tmp_path`, so nothing
   touches the real `data/` directory.

---

## Layout

```text
tests/
├── conftest.py                      # settings, clock, pose fixtures, sample poses
├── unit/
│   ├── test_preprocessing.py        # normalisation, features, layout
│   ├── test_sequence_buffer.py      # window, gaps, eviction, density
│   ├── test_fall_detector.py        # threshold, min-frames, cooldown
│   ├── test_models.py               # LSTM/BiLSTM, checkpoints, contract, mock
│   ├── test_alerts.py               # formatting, transport, redaction
│   ├── test_repository.py           # SQLite CRUD, threads, durability
│   ├── test_config.py               # env override, validation
│   ├── test_dataset.py              # windowing, temporal labels, leakage
│   ├── test_evaluation.py           # metric maths, refusal to invent numbers
│   ├── test_pose_and_sources.py     # MediaPipe wrapper, frame sources
│   └── test_frontend_and_logging.py # skeleton rendering, formatters, redaction
└── integration/
    ├── test_pipeline.py             # frame → event, dropouts, callback faults
    ├── test_service.py              # persist-then-alert, monitor lifecycle
    ├── test_api.py                  # every endpoint
    ├── test_dashboard.py            # real HTTP, every panel, offline path
    └── test_training_smoke.py       # train → checkpoint → reload (marked slow)
```

---

## What the important tests actually assert

### Data leakage (`test_dataset.py`)

```python
def test_no_group_appears_in_two_splits(...)
def test_leakage_assertion_catches_a_violation(...)
def test_random_window_level_split_would_leak(...)
```

The third one documents the failure mode itself: it generates a *random* window-level split
and asserts that `assert_no_group_leakage` rejects it. A guard with no test is a comment.

### False positives (`test_fall_detector.py`)

```python
def test_a_single_spike_does_not_confirm(...)
def test_interrupted_evidence_resets_the_run(...)
def test_cooldown_suppresses_immediate_repeat_alerts(...)
def test_windows_without_signal_do_not_accumulate_evidence(...)
def test_run_resets_after_a_confirmed_event(...)
```

A single `p = 0.99` must never produce an event. Evidence accumulated before an event must
not carry into the next one.

### Train/serve skew (`test_preprocessing.py`)

```python
def test_training_and_inference_share_one_preprocessor(...)
```

Asserts the two paths produce byte-identical feature vectors from the same class — and that
`ai.training.dataset` imports `PosePreprocessor` rather than reimplementing it.

### No fabricated metrics (`test_evaluation.py`)

```python
def test_empty_input_raises_rather_than_reporting_zero(...)
def test_synthetic_reports_are_marked_and_warned_about(...)
def test_synthetic_report_files_are_prefixed(...)
```

A zero metric is indistinguishable from a terrible model, so the evaluator refuses.

### Alert failure isolation (`test_service.py`, `test_alerts.py`)

```python
def test_event_is_persisted_even_when_the_alert_fails(...)
def test_alert_exception_does_not_propagate(...)
def test_client_errors_are_not_retried(...)
def test_transient_errors_are_retried_with_backoff(...)
```

A Telegram outage must never cost an event, and the event must not claim to have been
alerted.

### Credential handling (`test_alerts.py`)

```python
def test_mask_secrets_removes_a_telegram_token(...)
def test_redaction_filter_scrubs_exception_text(...)
def test_successful_send_does_not_log_the_token(...)
```

The token is posted in a real request; the test asserts it appears nowhere in the captured
log, and that the chat id is masked to its last four characters.

### Robustness (`test_pose_and_sources.py`, `test_pipeline.py`)

```python
@pytest.mark.parametrize("frame", [None, np.zeros((0,0,3)), "not an array", 42])
def test_invalid_frames_return_none_without_raising(...)
def test_non_finite_landmarks_are_treated_as_no_pose(...)
def test_detector_exception_is_contained(...)
def test_camera_source_handles_a_missing_device(...)
```

No bare `except: pass` anywhere — every handler returns a defined value or logs with
`exc_info=True`.

### Buffer semantics (`test_pipeline.py`)

```python
def test_pose_dropouts_do_not_produce_events(...)
def test_dropouts_are_kept_in_the_window(...)
def test_interleaved_dropouts_do_not_reset_a_real_fall(...)
```

Gaps occupy a slot, and a brief dropout does not cancel an ongoing fall.

---

## Fixtures

| Fixture | Purpose |
|---|---|
| `settings` | `Settings` rooted in `tmp_path`; function-scoped |
| `fake_clock` | manually advanced monotonic clock |
| `preprocessor` | `PosePreprocessor` from the test settings |
| `standing_landmarks` / `on_floor_landmarks` / `bending_landmarks` | 33×4 reference poses |
| `standing_window` / `on_floor_window` | `(30, 145)` model inputs |
| `repository` | in-memory `EventRepository` |
| `make_event` | `FallEvent` factory |
| `StubPoseEstimator` | replays a fixed pose list; no MediaPipe needed |
| `make_pipeline` | pipeline wired to a stub estimator, closed on teardown |

`transform_points()` applies scale/translate/rotate to the **complete** 33-landmark set,
filling unspecified joints first — otherwise an invariance test measures the fixture
rather than the normaliser. That bug happened during development and the helper's docstring
now records it.

---

## Coverage

| Area | Approach |
|---|---|
| Preprocessing | Direct assertions on invariance and degradation |
| Sequence buffer | Property-style: order, eviction, density, gap handling |
| Event confirmation | State-machine transitions with an injected clock |
| Models | Tiny real Keras nets; checkpoint round trip; contract rejection |
| Alerts | Fake transport with scripted responses and errors |
| Storage | Real SQLite, including a 4-thread concurrency test |
| Config | Env-var override with `monkeypatch` |
| Dataset | Split invariants, leakage, label inference |
| Evaluation | Known confusion matrices; refusal paths |
| Pose / sources | Real MediaPipe when available, otherwise a stub landmarker |
| Pipeline / service / API | Stub pose estimator, in-process client |
| Dashboard | Real loopback HTTP server, `AppTest` executes the script |

---

## Writing new tests

* Prefer asserting **behaviour** to asserting implementation details. Assert that a single
  spike does not fire, not that a counter equals a particular value.
* Inject the clock rather than sleeping. A cooldown test that sleeps 30 s is a test nobody
  runs.
* When a test fails, ask whether the code is wrong or the test is wrong, and prefer fixing
  the test when the code is right. Several tests in this suite were wrong about the
  normaliser's behaviour and the fix was to correct the *fixture*, with a comment
  recording what the correct behaviour is.
* If a metric or a result could be mistaken for a benchmark, label it in the test.
