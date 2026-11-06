# The model

Features, architectures, and how to train on a real dataset.

> **No benchmark exists in this repository.** The only metrics present were measured on
> *synthetic rendered data* — a fictional stick figure produced by
> `ai/pose/synthetic.py`. They validate the software pipeline and say nothing about
> fall-detection performance on people. See `IMPLEMENTATION_STATUS.md` for the exact
> numbers and their provenance.

---

## 1. Pipeline in one line

```text
33 landmarks → normalisation → 145-D feature vector → 30-frame window → sigmoid
```

---

## 2. Features (`ai/preprocessing/pose_preprocessor.py`)

### Normalisation

Applied identically at training and inference time, by the same class:

1. **Visibility mask** — landmarks below `MIN_LANDMARK_VISIBILITY` are invalid. A valid
   shoulder or hip pair repairs an invalid one; the scale falls back to the bounding-box
   diagonal of the valid landmarks.
2. **Hip-centred translation** — cancels where the subject is in frame.
3. **Torso-axis rotation** — cancels camera tilt and transient lean.
4. **Torso-length scaling** — cancels subject size and camera distance.

### Two coordinate views

| View | Transform | Purpose |
|---|---|---|
| `coordinates` | hip-centred → rotated → scaled | the 99-D coordinate block |
| `unaligned` | hip-centred → scaled, **not** rotated | the 12 kinematic descriptors |

This split is essential. Forcing the torso vertical would make `torso_lean_deg`
identically zero and flip the sign of the hip height — destroying the two strongest fall
cues while nominally adding tilt invariance to the *descriptors*. The coordinate block
keeps the invariance; the descriptors keep the meaning.

### Layout — `FEATURE_DIM = 145`

| Block | Size | Content |
|---|---|---|
| `xyz` | 99 | 33 landmarks × (x, y, z), hip-centred, torso-aligned, torso-scaled |
| `visibility` | 33 | per-landmark confidence; 0 for missing joints |
| `kinematic` | 12 | descriptors below |
| `present` | 1 | 1.0 if a usable pose was extracted, else 0.0 |

All descriptor values are in **torso lengths**, so they are independent of subject size
and camera distance.

| Descriptor | Why it is there |
|---|---|
| `torso_lean_deg` | 0° upright, ~90° on the floor. Note: uses the *horizontal* component only — the full 2-D norm would floor a vertical torso at 45° |
| `hip_height_above_ankles` | ~1.5 standing, ~0 on the floor. The single strongest cue |
| `shoulder_height_above_hip` | separates an upright torso from a collapsed one |
| `bbox_aspect_ratio` | a lying body is wide and flat, a standing one tall and narrow |
| `bbox_center_y`, `bbox_height` | vertical position and extent of the body in frame |
| `elbow_angle_left/right` | arms flailing is the mechanics of a fall |
| `knee_angle_left/right` | legs folding or buckling |
| `wrist_separation`, `ankle_separation` | a spread-eagle fall differs from a compact one |

The two primary cues must **both** fire. A forward bend gives a low hip with a vertical
torso; leaning back in a chair gives a tilted torso with a high hip. A single-cue rule
fails on one of them, which is exactly what `test_bending_is_not_mistaken_for_lying_down`
pins down.

### Versioning

`FEATURE_SPEC_VERSION = "1.0.0"` is written into every checkpoint's sidecar. The predictor
rejects a checkpoint whose spec version differs from the running code, so a feature-layout
change can never silently invalidate old weights.

---

## 3. Architectures (`ai/models/lstm.py`)

Both consume `(batch, sequence_length, 145)` and emit a sigmoid probability.

### `build_lstm` — the baseline and the serving default

```text
Input (30, 145)
  └─ BatchNormalization          feature blocks have very different scales
  └─ LSTM(64)                    return_sequences=False
  └─ Dropout(0.3)
  └─ Dense(32, relu)
  └─ Dropout(0.15)
  └─ Dense(1, sigmoid)           → P(fall)
```

56,453 parameters. Causal, cheap, and streamable frame-by-frame.

### `build_bilstm` — the optional variant

Identical, with `Bidirectional(LSTM(64))` in the middle. Roughly twice the parameters.

It reads the window in both directions, which helps because the *aftermath* of a fall — a
person lying still — is as informative as the descent. The cost: it cannot be streamed,
so the window is only available once complete. That is why the LSTM is the default.

### Why these choices

* **`BatchNormalization` on the input** — coordinates span ±2, angles span 0–180, ratios
  0–5. An LSTM is sensitive to input scale; without this, the kinematic block dominates the
  gradient simply by being larger.
* **`return_sequences=False`** — the classifier needs a window summary, not a per-frame
  output. Dropping the time axis keeps the model small.
* **Sigmoid** — event confirmation needs a probability to threshold, not a hard label.

---

## 4. Sequence length

`SEQUENCE_LENGTH = 30` frames ≈ 1 s at 30 FPS.

One second is the shortest window that still contains a fall: a controlled descent takes
roughly 0.6–0.9 s, and ground contact arrives just after. A shorter window can miss the
fall; a longer one adds latency without adding information about a *single* fall.

Shorter windows give lower detection latency; longer windows give fewer false alarms on
brief transients (a dropped object, a person sitting down abruptly). Tune on a real
validation split — not on synthetic data.

**Changing it invalidates existing checkpoints.** The predictor detects the mismatch and
tells you to retrain.

---

## 5. Training

### On synthetic data (smoke test)

```bash
python scripts/generate_sample_data.py --clear
python scripts/train.py --synthetic --epochs 150
```

Validates the pipeline end to end. The console and the sidecar both state that the metrics
are not a benchmark.

**Group count is the binding constraint, not window count.** Splits are over recordings,
so with 4 fall clips the training split receives almost no positive windows and the model
learns nothing. The generator produces 18 recordings for this reason. A 4-epoch run gave
F1 = 0.00; 150 epochs on 18 groups gave F1 = 0.71 — still on synthetic data, still not a
benchmark, but it shows the pipeline works.

### On a real dataset

```bash
python scripts/train.py --data data/processed --epochs 60 --architecture lstm
python scripts/train.py --data data/processed --architecture bilstm \
                        --output models/fallguard_bilstm.keras
```

`--data` expects:

```text
data/processed/
├── fall/       subjectNN_fall_01.npy + .json
└── non_fall/   subjectNN_walk_01.npy + .json
```

### Hyperparameters

| Setting | Default | Notes |
|---|---|---|
| `EPOCHS` | 20 | 60+ on a real dataset; the smoke test uses 150 |
| `BATCH_SIZE` | 32 | capped at the training-window count |
| `LEARNING_RATE` | 1e-3 | halved by `ReduceLROnPlateau` on `val_loss` |
| `DROPOUT_RATE` | 0.3 | 0.15 on the second dense block |
| `EARLY_STOPPING_PATIENCE` | 5 | restores the best weights |
| `TEST_SPLIT` / `VALIDATION_SPLIT` | 0.2 / 0.2 | must sum to < 1.0 |
| `RANDOM_SEED` | 20240917 | seeds numpy, Python and TensorFlow; enables op determinism |

Class weights are inverse-frequency, so an imbalanced dataset does not collapse to
predicting `NORMAL` for everything.

### Splits and leakage

Windows are slices of a longer recording, so two windows from the same video are nearly
identical. Splitting windows at random puts near-duplicates on both sides of the
train/test boundary and inflates every metric.

`group_split` therefore assigns **whole recordings** to splits, deterministically, from a
hash of `(seed, group_id)`. `assert_no_group_leakage` enforces it in the trainer, and
`tests/unit/test_dataset.py::test_random_window_level_split_would_leak` demonstrates that
a window-level split is caught.

Set `group_id` to the **subject** when one person appears in several recordings.
Otherwise the same person lands in both train and test and the score is optimistic.

### Temporal labels

Without `frame_labels`, a fall recording is labelled `1` end to end — including the
seconds of ordinary standing before the fall. The model then learns to fire on people
standing still, the most common false-positive source.

With a per-frame annotation, a window is labelled `1` if at least
`FALL_LABEL_WINDOW_RATIO` (0.5) of its frames are fall-annotated. The generator emits
these automatically.

### Degenerate data

| Situation | Behaviour |
|---|---|
| Empty training set | `InsufficientDataError` with instructions |
| Fewer than 2 training windows | `InsufficientDataError` naming the group count |
| Training split has one class | warning that metrics will be meaningless |
| Test split empty | training completes, `evaluation` is `None`, a message explains why |
| Test split has one class | `scripts/evaluate.py` exits non-zero; no metric is reported |

---

## 6. Evaluation

```bash
python scripts/evaluate.py
python scripts/evaluate.py --mock --data data/processed
python scripts/evaluate.py --data data/processed --threshold 0.75
```

Outputs land in `reports/`:

| File | Contents |
|---|---|
| `*_metrics.json` | accuracy, precision, recall, F1, support, confusion matrix, `dataset_kind`, notes |
| `*_classification_report.txt` | scikit-learn per-class report, confusion matrix, notes |
| `*_confusion_matrix.png` | rendered matrix |

The threshold is `FALLGUARD_CONFIDENCE_THRESHOLD` unless `--threshold` overrides it, so
evaluation measures the *deployed* decision rule rather than an arbitrary 0.5 cut.

### Refusing to invent numbers

`compute_metrics` raises `NoSamplesError` on an empty input rather than returning zeros,
because a zero metric is indistinguishable from a genuinely terrible model.
`scripts/evaluate.py` exits non-zero when the test split is empty or single-class, and says
why.

Results derived from synthetic data are prefixed **`SYNTHETIC_`** and carry a note:

> Metrics computed on SYNTHETIC rendered data. They describe the software pipeline only
> and are NOT a benchmark of fall-detection performance.

Provenance is derived from the **data** (group ids), not the path — the default directory
`data/processed` legitimately holds both real and synthetic subdirectories, so a path check
would mislabel synthetic results as real benchmarks.

### Reading the confusion matrix

Rows are true, columns are predicted, order `(non_fall, fall)`.

```text
[[ TN,  FP ],
 [ FN,  TP ]]
```

For fall detection, **FN (a missed fall) is the dangerous cell.** Optimising F1 alone can
buy recall at the cost of alert fatigue, which causes humans to ignore alerts — so a real
deployment should pick a threshold on a labelled validation set against a *cost* ratio,
and measure the operating point, not just the F1.

---

## 7. Getting a real benchmark

1. **Collect a labelled dataset.** A fall dataset plus ordinary daily activity, ideally
   with per-frame temporal annotations and subject identifiers.
2. **Check the licence** of anything you use, and follow its terms for redistribution.
3. **Export landmarks** with `ai.training.dataset.extract_samples_from_videos`, or with
   `scripts/generate_sample_data.py`'s extraction loop adapted to your files.
4. **Split by subject**, not by clip, and record the subject as `group_id`.
5. **Train**, then `scripts/evaluate.py` on the untouched test split.
6. **Report** accuracy, precision, recall, F1, the confusion matrix, the number of
   subjects, the per-subject breakdown, and the decision threshold.
7. **Test on people who were not in training.** A model that has never seen a wheelchair,
   a walking aid or a very large or very small subject is not ready.

The honest version of step 6 includes a per-subject breakdown. Aggregate F1 across a
handful of subjects can hide that the model fails completely on two of them.

---

## 8. Limitations

* Binary `FALL` / `NORMAL` only. No severity, no recovery, no near-miss class.
* The BiLSTM cannot stream, so it adds window latency.
* Normalisation assumes a visible torso. A person behind a counter, in loose clothing, or
  heavily occluded degrades detection — partial occlusion is handled (invalid landmarks are
  zeroed and flagged) but not reconstructed.
* No background modelling. A busy background raises false positives.
* A single camera, no multi-view fusion.
* Training is CPU-feasible but slow for large datasets; a GPU helps but is not required.
