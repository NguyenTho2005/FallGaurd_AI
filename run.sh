#!/usr/bin/env bash
# FallGuard AI -- convenience launcher.
#
# Solves the one thing that trips people up: this project needs the virtualenv in
# .venv, and `streamlit`/`uvicorn` are NOT installed system-wide. Forgetting to
# activate gives you "streamlit: command not found".
#
#   ./run.sh demo        offline demo, no camera needed
#   ./run.sh test        run the test suite
#   ./run.sh backend     start the FastAPI server only
#   ./run.sh dashboard   start the Streamlit dashboard only
#   ./run.sh both        start backend + dashboard together (Ctrl-C stops both)
#   ./run.sh setup       install dependencies + fetch the MediaPipe pose asset
#   ./run.sh train       train on the synthetic data
#   ./run.sh evaluate    evaluate the checkpoint, write reports/
#
# Options are passed through, e.g.:
#   ./run.sh demo --camera --max-frames 600

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
    echo "ERROR: virtualenv not found at $VENV" >&2
    echo "       Run: ./run.sh setup" >&2
    exit 1
fi

# Activate so that bare `streamlit` / `uvicorn` resolve to this venv. This is what
# makes `./run.sh dashboard` behave identically to an activated shell.
# shellcheck disable=SC1091
source "$VENV/bin/activate"

banner() {
    echo
    echo "=============================================================="
    echo "  FallGuard AI -- $1"
    echo "=============================================================="
}

wait_for_backend() {
    local port="${1:-8000}"
    echo "waiting for the backend on port $port ..."
    for _ in $(seq 1 60); do
        if curl -fsS -m 2 "http://localhost:$port/health" >/dev/null 2>&1; then
            echo "backend is up:  http://localhost:$port"
            echo "API docs:       http://localhost:$port/docs"
            return 0
        fi
        sleep 2
    done
    echo "WARNING: the backend did not become healthy in time." >&2
    echo "         Check its output above for an error." >&2
}

cmd="${1:-demo}"
shift || true

case "$cmd" in
    setup)
        banner "SETUP"
        "$PY" -m pip install --upgrade pip
        "$PY" -m pip install -r requirements.txt
        "$PY" scripts/download_mediapipe_model.py
        [[ -f .env ]] || { cp .env.example .env; echo "created .env from .env.example"; }
        echo
        echo "Setup complete. Next:  ./run.sh demo"
        ;;

    demo)
        banner "DEMO MODE"
        echo "No camera, no real dataset. Every result is a DEMO output, not an"
        echo "accuracy measurement. See IMPLEMENTATION_STATUS.md."
        echo
        "$PY" scripts/demo.py "$@"
        ;;

    test)
        banner "TESTS"
        "$PY" -m pytest "$@"
        ;;

    backend)
        banner "BACKEND"
        echo "API docs: http://localhost:8000/docs"
        exec uvicorn backend.main:app --host 0.0.0.0 --port "${FALLGUARD_API_PORT:-8000}"
        ;;

    dashboard)
        banner "DASHBOARD"
        echo "Dashboard: http://localhost:8501"
        echo "Make sure the backend is running, otherwise the page shows OFFLINE."
        exec streamlit run frontend/streamlit_app.py \
            --server.address 0.0.0.0 --server.port 8501 \
            --server.headless true --browser.gatherUsageStats false
        ;;

    both)
        banner "BACKEND + DASHBOARD"
        echo "Dashboard: http://localhost:8501"
        echo "API docs:  http://localhost:8000/docs"
        echo "Press Ctrl-C to stop both."
        echo
        PORT="${FALLGUARD_API_PORT:-8000}"
        uvicorn backend.main:app --host 0.0.0.0 --port "$PORT" &
        BACKEND_PID=$!
        # Ctrl-C must take down the dashboard too, otherwise it lingers holding port 8501.
        trap 'kill "$BACKEND_PID" 2>/dev/null || true' EXIT INT TERM
        wait_for_backend "$PORT"
        echo
        streamlit run frontend/streamlit_app.py \
            --server.address 0.0.0.0 --server.port 8501 \
            --server.headless true --browser.gatherUsageStats false
        ;;

    train)
        banner "TRAIN (synthetic)"
        "$PY" scripts/train.py --synthetic "$@"
        ;;

    evaluate)
        banner "EVALUATE"
        "$PY" scripts/evaluate.py "$@"
        ;;

    sample-data)
        banner "GENERATE SYNTHETIC SAMPLE DATA"
        "$PY" scripts/generate_sample_data.py "$@"
        ;;

    *)
        sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 1
        ;;
esac
