# Deployment

FallGuard AI is a **prototype**. The guidance below covers running it reliably, not
certifying it.

---

## 1. Local run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_mediapipe_model.py
cp .env.example .env

# Terminal 1
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Terminal 2
streamlit run frontend/streamlit_app.py
```

With a camera attached to the backend:

```bash
FALLGUARD_ENABLE_MONITOR=true FALLGUARD_CAMERA_SOURCE=0 uvicorn backend.main:app
```

---

## 2. Systemd

`/etc/systemd/system/fallguard.service`:

```ini
[Unit]
Description=FallGuard AI backend
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=fallguard
Group=fallguard
WorkingDirectory=/opt/fallguard-ai
EnvironmentFile=/opt/fallguard-ai/.env
Environment=FALLGUARD_ENVIRONMENT=production
ExecStart=/opt/fallguard-ai/.venv/bin/uvicorn backend.main:app \
          --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=5

# The process needs the camera device, and nothing else.
SupplementaryGroups=video
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/fallguard-ai/data /opt/fallguard-ai/reports

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now fallguard
sudo journalctl -u fallguard -f
```

> **`--workers 1` is deliberate.** The pipeline holds a sequence buffer and detector state
> in memory, and a background camera thread runs per worker. Multiple workers would each
> open the camera and each hold a partial window, so detections would be split and
> incoherent. To scale out, run one process per camera with a distinct
> `FALLGUARD_CAMERA_ID` and `FALLGUARD_DB_PATH` behind a reverse proxy.

---

## 3. systemd for the dashboard

`/etc/systemd/system/fallguard-dashboard.service`:

```ini
[Unit]
Description=FallGuard AI dashboard
After=fallguard.service
Requires=fallguard.service

[Service]
Type=simple
User=fallguard
Group=fallguard
WorkingDirectory=/opt/fallguard-ai
EnvironmentFile=/opt/fallguard-ai/.env
Environment=FALLGUARD_API_URL=http://127.0.0.1:8000
ExecStart=/opt/fallguard-ai/.venv/bin/streamlit run frontend/streamlit_app.py \
          --server.address 127.0.0.1 --server.port 8501 \
          --server.headless true --browser.gatherUsageStats false
Restart=always
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

---

## 4. Reverse proxy

The backend **has no authentication**. Put it behind a proxy that does.

```nginx
server {
    listen 443 ssl http2;
    server_name fallguard.internal;

    ssl_certificate     /etc/ssl/certs/fallguard.crt;
    ssl_certificate_key /etc/ssl/private/fallguard.key;

    # Video-adjacent infrastructure: a long timeout keeps SSE/streaming usable.
    client_max_body_size 1m;
    proxy_read_timeout 120s;

    # Authenticate at the edge (choose one):
    #   auth_basic "FallGuard";
    #   allow 10.0.0.0/8;  deny all;
    # Or put it behind an SSO-aware proxy.

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
    }
}

server {
    listen 8501;
    server_name localhost;
    location / {
        proxy_pass       http://127.0.0.1:8501;
        proxy_set_header Host $host;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
    }
}
```

Also narrow the CORS policy in `backend/main.py`: `allow_origins=["*"]` is a development
convenience because the dashboard is a separate process. In production, set it to the
dashboard's origin.

---

## 5. GPU (optional)

Nothing requires a GPU. If you have one, install CUDA-enabled TensorFlow in place of the
CPU wheel:

```bash
pip uninstall -y tensorflow-cpu
pip install "tensorflow[and-cuda]"
```

Verify:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

> The test suite is written to stay GPU-free either way. If you install the GPU build,
> consider pinning `TF_ENABLE_ONEDNN_OPTS=0` and running the model tests on CPU so results
> do not depend on hardware.

---

## 6. Operational checklist

Before you trust the system with a person's safety:

### Correctness

- [ ] `GET /status` shows `model.loaded: true` and `model.backend: "keras"`. If it says
      `mock`, the mock heuristic is running and **no prediction is a trained one**.
- [ ] `model.path` points at a checkpoint you trained on a **real** dataset.
- [ ] `reports/` contains a report whose `dataset_kind` is **not** synthetic. If it is, you
      have no benchmark.
- [ ] The false-alarm rate is acceptable **for your environment**, measured over hours of
      real footage — not over a smoke test.
- [ ] A missed fall is understood and accepted, or mitigated with a second system.
- [ ] Someone has watched the skeleton output and confirmed it reflects reality.

### Operations

- [ ] Someone is **available to respond** to an alert. The system is a prompt, not a
      response.
- [ ] The response process is written down and rehearsed.
- [ ] Monitoring alerts on: `model.loaded == false`, `pose_estimator_available == false`,
      a collapse in `pose_detection_rate`, rising `alerts.failed`, and the process being
      down.
- [ ] `/health` is on a probe.
- [ ] Log rotation is configured (`FALLGUARD_LOG_FILE`, 5 MB × 3).
- [ ] The SQLite file is on a backed-up, encrypted volume.
- [ ] `max_events` suits your retention requirement.

### Privacy

- [ ] Lawful basis established and documented.
- [ ] Everyone in the monitored area knows, with signage.
- [ ] No cameras in private spaces.
- [ ] No audio capture.
- [ ] Event-metadata access is restricted.
- [ ] The Telegram bot token is stored with `chmod 600` and can be rotated.
- [ ] A data-subject access and deletion process exists for the event store.

---

## 7. Monitoring

```bash
# Liveness
curl -fsS http://localhost:8000/health

# Readiness
curl -fsS http://localhost:8000/status \
  | jq '.model.loaded, .pipeline.pose_estimator_available'

# Detection health
curl -fsS http://localhost:8000/status | jq '.pipeline.pose_detection_rate'

# Alerting channel health
curl -fsS http://localhost:8000/status | jq '.alerts, .telegram.reason'
```

Suggested alerts:

| Condition | Severity | Meaning |
|---|---|---|
| `/health` unreachable | critical | backend down |
| `model.loaded == false` | critical | fell back to the mock backend |
| `pose_estimator_available == false` | critical | MediaPipe asset missing |
| `alerts.failed` increasing | high | Telegram outage; events still stored |
| `pose_detection_rate` drops sharply | high | camera moved, lighting changed, or subject absent |
| `events_confirmed == 0` for hours with poses detected | medium | the model may be stuck or the threshold too high |

---

## 8. Upgrading

1. `pytest` before and after.
2. Read the changelog for a changed `FEATURE_SPEC_VERSION`. If it moved, **every existing
   checkpoint is invalid** and must be retrained — the predictor will refuse to serve them
   and will say so.
3. Re-run `scripts/evaluate.py` on a real test split and compare against the previous
   report. A regression is a reason not to deploy.
4. Deploy the backend first, confirm `/status`, then the dashboard.

---

## 9. Hardware notes

| Component | Guidance |
|---|---|
| Camera | 640×480 @ 30 fps is the tested configuration. Higher resolution costs CPU in MediaPipe without improving 33-landmark accuracy much |
| CPU | MediaPipe dominates. Throughput is roughly 8–30 fps single-stream on a modern desktop CPU, depending on resolution |
| RAM | ~1.5 GB with TensorFlow loaded |
| Disk | The event store is small. Checkpoints are a few MB |
| GPU | Optional. Helpful for training and for higher-resolution inference, not required |
| Camera placement | Elevated, angled down at the floor area. Frontal eye-level captures the torso but often loses the ankles, which are the strongest cue |

---

## 10. Scaling

The MVP is single-camera by design. Beyond that:

* **Multiple cameras** — one process per camera, distinct `FALLGUARD_CAMERA_ID` and
  `FALLGUARD_DB_PATH`, behind a reverse proxy that aggregates `/events`. Do **not** use
  `--workers > 1` for this.
* **Shared storage** — replace `EventRepository` with a networked database. The interface is
  small; the SQLite specifics (WAL, rowid ordering, `INSERT OR REPLACE`) would need
  revisiting.
* **Queueing** — if inference falls behind, decouple capture from inference with a bounded
  queue. Drop the **oldest** frames, never the newest: for fall detection, latency matters
  more than completeness.
* **Model serving** — TensorFlow Serving or Triton, behind the same `FallPredictor`
  contract. The checkpoint sidecar carries everything a serving container needs.

---

## 11. Rollback

* **Configuration** — revert `.env`. Safe and immediate.
* **Code** — deploy the previous version; the SQLite schema is additive and forward
  compatible.
* **Model** — keep the previous checkpoint and its sidecar. `MODEL_PATH` makes rollback a
  config change, provided the sequence length and feature-spec version match.
