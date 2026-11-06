# Model checkpoints

Checkpoints (`*.keras`) and their metadata sidecars (`*.meta.json`) are **not committed** —
they are large, regenerable, and specific to a training run.

## Producing a checkpoint

```bash
# Smoke test on synthetic data — validates the pipeline, proves nothing about accuracy
python scripts/generate_sample_data.py --clear
python scripts/train.py --synthetic --epochs 150

# Real dataset
python scripts/train.py --data data/processed --epochs 60
```

Two files are written:

| File | Purpose |
|---|---|
| `fallguard_lstm.keras` | the Keras weights |
| `fallguard_lstm.meta.json` | the **serving contract**: feature-spec version, feature dim, sequence length, architecture, parameter count, dataset summary, split sizes, hyperparameters, measured test metrics, and any warnings |

`FallPredictor` reads the sidecar and **refuses to serve** a checkpoint whose feature layout,
sequence length or feature-spec version disagrees with the running code. **Keep the two
files together.** A checkpoint without its sidecar still loads, but the contract check then
falls back to the architecture's own input shape — which is why a warning is logged.

## Distributing a checkpoint

The file is small enough to move directly:

```bash
# On the training machine
tar czf fallguard_lstm-ckpt.tar.gz \
    models/fallguard_lstm.keras models/fallguard_lstm.meta.json

# On the target
tar xzf fallguard_lstm-ckpt.tar.gz -C /opt/fallguard-ai/
```

Then point the service at it:

```dotenv
FALLGUARD_MODEL_PATH=/opt/fallguard-ai/models/fallguard_lstm.keras
FALLGUARD_MODEL_METADATA_PATH=/opt/fallguard-ai/models/fallguard_lstm.meta.json
```

## Compatibility

A checkpoint is valid only for the exact feature layout and sequence length it was trained
with. If the predictor rejects one, the message names the mismatch:

| Message | Fix |
|---|---|
| `Checkpoint feature spec vX does not match vY` | The feature layout changed. Retrain on the new code |
| `FALLGUARD_SEQUENCE_LENGTH=30 but the checkpoint expects 45` | Align the variable with the checkpoint, or retrain |
| `Checkpoint expects N input features but this build produces M` | Retrain |

## Without a checkpoint

The backend does not fail. It falls back to `MockFallModel` — a deterministic kinematic
heuristic with **zero learned parameters** — so the API, event logic and dashboard remain
fully exercisable. `GET /status` then reports:

```json
"model": {
  "backend": "mock",
  "loaded": true,
  "detail": "deterministic kinematic heuristic, not a trained model"
}
```

The dashboard displays a warning banner, and `scripts/demo.py` prints one too.

**A mock-backend prediction is never a trained prediction.** Do not quote it as an accuracy
result.

## Verifying a checkpoint before deploying

```bash
# It loads and its contract matches
FALLGUARD_MODEL_BACKEND=keras python -c "
from ai.config import reload_settings
from ai.models.predictor import FallPredictor
p = FallPredictor(reload_settings())
print(p.info().to_dict())
"

# It has a measured, non-synthetic evaluation
python -c "
import json; m = json.load(open('models/fallguard_lstm.meta.json'))
print('synthetic:', m['dataset']['synthetic'])
print('evaluation:', m['evaluation'] and m['evaluation']['metrics'])
"
```

`"synthetic": true` means the checkpoint was trained on rendered data and **no benchmark
exists** for it.
