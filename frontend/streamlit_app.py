"""FallGuard AI -- Streamlit dashboard.

    streamlit run frontend/streamlit_app.py

Panels
------
1. **System status** -- system / camera / model, matching the specification.
2. **Live prediction** -- activity, confidence and a skeleton visualisation.
3. **Event history** -- timestamp, prediction, confidence, camera, status.
4. **Privacy** -- what is stored, and what is deliberately not.

Privacy: the live panel draws a **skeleton**, never camera footage. Raw frames are never
retained by the backend, so there is nothing to display even if this page wanted to.
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

# Allow `streamlit run frontend/streamlit_app.py` from the repository root.
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ai.config import get_settings  # noqa: E402
from ai.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)

st.set_page_config(
    page_title="FallGuard AI",
    page_icon="\U0001F6A8",
    layout="wide",
    initial_sidebar_state="expanded",
)

settings = get_settings()
DEFAULT_API_URL = os.environ.get("FALLGUARD_API_URL", f"http://localhost:{settings.api_port}")


# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=2.0, show_spinner=False)
def fetch_status(api_url: str) -> dict[str, Any] | None:
    """GET /status. Returns ``None`` when the backend is unreachable."""
    try:
        response = requests.get(f"{api_url.rstrip('/')}/status", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        logger.debug("Could not reach %s/status: %s", api_url, exc)
        return None


@st.cache_data(ttl=2.0, show_spinner=False)
def fetch_events(api_url: str, limit: int = 50) -> list[dict[str, Any]]:
    """GET /events. An unreachable backend yields an empty list, not an exception."""
    try:
        response = requests.get(
            f"{api_url.rstrip('/')}/events", params={"limit": limit}, timeout=5
        )
        response.raise_for_status()
        return response.json().get("events", [])
    except requests.RequestException as exc:
        logger.debug("Could not reach %s/events: %s", api_url, exc)
        return []


@st.cache_data(ttl=2.0, show_spinner=False)
def fetch_pose(api_url: str) -> dict[str, Any] | None:
    """GET /pose/latest -- landmarks only, never video."""
    try:
        response = requests.get(f"{api_url.rstrip('/')}/pose/latest", timeout=5)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        logger.debug("Could not reach %s/pose/latest: %s", api_url, exc)
        return None


def clear_caches() -> None:
    for func in (fetch_status, fetch_events, fetch_pose):
        func.clear()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("\U0001F6A8 FallGuard AI")
    st.caption("Fall-event detection from camera pose data")

    api_url = st.text_input("Backend URL", value=DEFAULT_API_URL, help="Base URL of the FastAPI server")
    auto_refresh = st.checkbox("Auto refresh", value=True)
    refresh_interval = st.slider("Refresh interval (s)", 1, 30, 3)
    limit = st.slider("Events to show", 5, 200, 50, step=5)
    st.divider()
    if st.button("Refresh now", use_container_width=True):
        clear_caches()
        st.rerun()

    st.divider()
    st.markdown(
        """
        **Privacy**
        - Skeleton data only. Raw video is never stored.
        - No face or identity recognition.
        - Events are metadata: time, camera, confidence, status.
        """
    )
    st.caption(
        "Prototype fall-event detection. Not a medical device and not a diagnosis. "
        "Always verify an alert manually."
    )


# --------------------------------------------------------------------------- #
# Live panels
# --------------------------------------------------------------------------- #
def render_live_panels(api_url: str, limit: int) -> None:
    """Render the status, live-prediction and event-history panels.

    Kept as a plain function so it can be called directly (no auto-refresh) or from
    inside a ``st.fragment`` that re-runs on a timer. It is never decorated itself, so
    the fragment is always in control of the refresh cadence.
    """
    status = fetch_status(api_url)
    header, subheader = st.columns([3, 2])
    with header:
        st.title("FallGuard AI")
    with subheader:
        st.metric("Backend", "ONLINE" if status else "OFFLINE")
        st.caption(api_url)

    if not status:
        st.error(
            f"Cannot reach the FallGuard API at `{api_url}`. "
            "Start it with `uvicorn backend.main:app --port 8000` and reload this page."
        )
        st.stop()

    # -- system status --------------------------------------------------- #
    st.subheader("System status")
    cols = st.columns(4)
    with cols[0]:
        st.metric("System", str(status.get("system", "unknown")).upper())
        st.caption(f"v{status.get('app', {}).get('version', '?')}")
    with cols[1]:
        camera = status.get("camera", {})
        st.metric("Camera", "CONNECTED" if camera.get("connected") else "DISCONNECTED")
        st.caption(f"{camera.get('id', '?')} · `{camera.get('source', '?')}`")
    with cols[2]:
        model = status.get("model", {})
        st.metric("Model", "LOADED" if model.get("loaded") else "NOT LOADED")
        st.caption(f"{model.get('backend', '?')} · seq {model.get('sequence_length', '?')}")
    with cols[3]:
        pipeline = status.get("pipeline", {})
        st.metric("Poses", f"{float(pipeline.get('pose_detection_rate', 0.0)):.0%}")
        st.caption(
            f"{pipeline.get('poses_detected', 0)}/{pipeline.get('frames_processed', 0)} frames"
        )

    if model.get("backend") == "mock":
        st.warning(
            "**Mock model backend active.** This is a deterministic kinematic heuristic for "
            "pipeline testing, *not* a trained model. No accuracy claim applies."
        )

    with st.expander("Runtime detail", expanded=False):
        left, right = st.columns(2)
        with left:
            st.json({"config": status.get("config", {}), "detector": status.get("detector", {})})
        with right:
            st.json(
                {
                    "pipeline": pipeline,
                    "telegram": status.get("telegram", {}),
                    "alerts": status.get("alerts", {}),
                    "events": status.get("events", {}),
                }
            )

    # -- live prediction -------------------------------------------------- #
    st.subheader("Live prediction")
    last = status.get("last_prediction")
    left, right = st.columns([3, 2])

    with left:
        if not last:
            st.info(
                "No prediction yet. Enable the camera monitor "
                "(`FALLGUARD_ENABLE_MONITOR=true`) or post landmarks to `/predict`."
            )
        elif last.get("event_confirmed"):
            st.error("### \u26A0 FALL DETECTED")
            st.write(
                f"Event `{str(last.get('event_id', ''))[:12]}` confirmed at "
                f"{last.get('timestamp')} on camera `{last.get('camera_id')}`."
            )
        else:
            st.info(f"### Activity: {last.get('prediction', 'UNKNOWN')}")

        if last:
            confidence = float(last.get("confidence", 0.0))
            st.progress(min(max(confidence, 0.0), 1.0))
            metric_cols = st.columns(3)
            metric_cols[0].metric("Confidence", f"{confidence:.1%}")
            metric_cols[1].metric("Sequence", "READY" if last.get("sequence_ready") else "BUFFERING")
            metric_cols[2].metric("Pose", "DETECTED" if last.get("pose_detected") else "NO POSE")

    with right:
        st.markdown("**Pose skeleton (privacy-preserving view)**")
        pose = fetch_pose(api_url)
        if pose and pose.get("landmarks"):
            from frontend.skeleton import render_skeleton_figure, skeleton_summary

            state = str((last or {}).get("prediction", "normal")).lower()
            figure = render_skeleton_figure(
                pose["landmarks"], state=state, title=f"Camera {pose.get('camera_id', '?')}"
            )
            st.pyplot(figure, width="stretch")
            summary = skeleton_summary(pose["landmarks"])
            st.caption(
                f"{summary['detected_joints']}/{summary['landmarks']} joints · "
                f"mean visibility {summary['mean_visibility']:.2f} · no video displayed"
            )
        else:
            st.caption("No pose available. The camera monitor may be disabled.")

    # -- event history ---------------------------------------------------- #
    st.subheader("Event history")
    events = fetch_events(api_url, limit=limit)
    if not events:
        st.success("No fall events recorded. \u2714")
        return

    st.dataframe(
        [
            {
                "Timestamp": event["timestamp"],
                "Prediction": event["prediction"],
                "Confidence": f"{float(event['confidence']):.1%}",
                "Peak": f"{float(event['peak_confidence']):.1%}",
                "Camera": event["camera_id"],
                "Status": event["status"],
                "Alerted": "yes" if event["notified"] else "no",
                "Event ID": event["event_id"][:12],
            }
            for event in events
        ],
        width="stretch",
        hide_index=True,
    )
    total = status.get("events", {}).get("total", len(events))
    st.caption(f"{len(events)} shown of {total} stored event(s).")

    with st.expander("Acknowledge the most recent event", expanded=False):
        newest = events[0]
        st.write(f"Event `{newest['event_id']}` · {newest['timestamp']} · {newest['status']}")
        if st.button("Mark as acknowledged", key="ack_event"):
            try:
                response = requests.patch(
                    f"{api_url.rstrip('/')}/events/{newest['event_id']}",
                    json={"status": "acknowledged"},
                    timeout=5,
                )
                response.raise_for_status()
                clear_caches()
                st.success("Event acknowledged.")
            except requests.RequestException as exc:
                st.error(f"Could not update the event: {exc}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
@st.fragment(run_every=refresh_interval)
def _live_panels() -> None:
    render_live_panels(api_url, limit)


if auto_refresh:
    # A fragment re-runs on a timer without blocking the script thread, which is why
    # this is not implemented as sleep() + st.rerun().
    _live_panels()
else:
    render_live_panels(api_url, limit)


# --------------------------------------------------------------------------- #
# Privacy footer
#
# Rendered outside the fragment on purpose: it is static policy text plus one line from
# the API, so it has no reason to re-render on the refresh timer.
# --------------------------------------------------------------------------- #
def render_privacy_footer(api_url: str) -> None:
    st.divider()
    privacy = (fetch_status(api_url) or {}).get("privacy", {})
    with st.expander("Privacy & limitations", expanded=False):
        st.markdown(
            f"""
**What is stored**

| Item | Stored? |
|---|---|
| Skeleton landmarks | In memory only, for live display |
| Event metadata (time, camera, confidence, status) | Yes, in `{settings.db_path}` |
| Raw video / images | **Never** |
| Face or identity recognition | **Not implemented** |

{privacy.get('retention', '')}
"""
        )
        st.warning(
            "FallGuard AI is a prototype that detects **possible** fall events. It is not a "
            "medical device, it does not diagnose injuries, and it must not replace emergency "
            "services or human supervision. A notification is a prompt to check on a person, "
            "not a clinical assessment."
        )


render_privacy_footer(api_url)
