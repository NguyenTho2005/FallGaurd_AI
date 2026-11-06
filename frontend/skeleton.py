"""Skeleton rendering for the FallGuard dashboard.

The dashboard's default visualisation is a **skeleton**, not footage. This module converts
33 normalised landmarks into the normalised coordinates Streamlit's drawing primitives
expect, so the UI never has to touch pixels.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ai.pose.landmarks import POSE_CONNECTIONS, POSE_LANDMARK_COUNT

#: Landmarks drawn as dots: the head plus the joints that carry fall information.
KEY_LANDMARKS: tuple[int, ...] = (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)

COLORS = {
    "background": "#0f1117",
    "skeleton": "#34d399",
    "joint": "#a7f3d0",
    "head": "#60a5fa",
    "fall": "#f87171",
    "normal": "#34d399",
    "unknown": "#94a3b8",
}


def landmarks_to_canvas(
    landmarks: Sequence[Sequence[float]] | np.ndarray,
    *,
    width: int = 640,
    height: int = 480,
    min_visibility: float = 0.3,
) -> tuple[list[list[float]], list[tuple[float, float, float, float]]]:
    """Return ``(points, segments)`` in canvas pixel coordinates.

    Parameters
    ----------
    landmarks:
        A ``(33, 4)`` array of ``[x, y, z, visibility]`` with ``x``/``y`` normalised to
        ``[0, 1]``.
    """
    array = np.asarray(landmarks, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != POSE_LANDMARK_COUNT:
        raise ValueError(
            f"expected a ({POSE_LANDMARK_COUNT}, 4) landmark array, got {array.shape}"
        )
    points: list[list[float]] = []
    for index in range(POSE_LANDMARK_COUNT):
        x, y, visibility = float(array[index, 0]), float(array[index, 1]), float(array[index, 3])
        if visibility < min_visibility or not (np.isfinite(x) and np.isfinite(y)):
            # Out-of-view joints are pushed off-canvas so no line is drawn to them.
            points.append([-1.0, -1.0])
            continue
        points.append([x * width, y * height])

    segments: list[tuple[float, float, float, float]] = []
    for start, end in POSE_CONNECTIONS:
        p1, p2 = points[start], points[end]
        if p1[0] < 0 or p2[0] < 0:
            continue
        segments.append((p1[0], p1[1], p2[0], p2[1]))
    return points, segments


def render_skeleton_figure(
    landmarks: Sequence[Sequence[float]] | np.ndarray,
    *,
    width: int = 640,
    height: int = 480,
    state: str = "normal",
    min_visibility: float = 0.3,
    title: str = "",
):
    """Return a matplotlib figure of a pose skeleton.

    Deliberately implemented with matplotlib rather than a canvas widget: it keeps the
    dependency list unchanged, it renders reliably on a headless server, and it draws
    *only* skeleton coordinates -- so the default visualisation never contains pixels
    from the camera.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    points, segments = landmarks_to_canvas(
        landmarks, width=width, height=height, min_visibility=min_visibility
    )
    color = COLORS.get(state, COLORS["unknown"])
    figure, axes = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)
    figure.patch.set_facecolor(COLORS["background"])
    axes.set_facecolor(COLORS["background"])
    axes.set_xlim(0, width)
    axes.set_ylim(height, 0)  # image coordinates: y grows downwards
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)

    for x1, y1, x2, y2 in segments:
        axes.add_line(
            Line2D([x1, x2], [y1, y2], color=color, linewidth=2.5, solid_capstyle="round")
        )
    for index in KEY_LANDMARKS:
        if points[index][0] < 0:
            continue
        radius = 12 if index == 0 else 7
        axes.add_patch(
            plt.Circle(
                (points[index][0], points[index][1]),
                radius,
                facecolor=COLORS["head"] if index == 0 else COLORS["joint"],
                edgecolor=color,
                linewidth=1.5,
            )
        )
    if title:
        axes.set_title(title, color="#e2e8f0", fontsize=11)
    figure.tight_layout(pad=0.4)
    return figure


def skeleton_summary(landmarks: Sequence[Sequence[float]] | np.ndarray) -> dict[str, Any]:
    """Small human-readable description of a pose, for the dashboard side panel."""
    array = np.asarray(landmarks, dtype=np.float32)
    visibility = array[:, 3]
    return {
        "landmarks": int(array.shape[0]),
        "detected_joints": int((visibility >= 0.3).sum()),
        "mean_visibility": float(visibility.mean()),
    }
