"""Matplotlib rendering of the hand poses."""

from __future__ import annotations

import numpy as np

from .hand import FINGERS, Pose, pose_for, pose_geometry

# Drawn with line width rather than fill so the finger articulation stays
# visible; a filled silhouette hides exactly the joint angles that carry the
# difference between one grasp and another.
_PALM_FACE = "#cbd5e1"
_PALM_EDGE = "#475569"
_FINGER = "#334155"


def draw_hand(ax, pose: Pose, color: str | None = None, alpha: float = 1.0,
              linewidth: float = 7.0, label: str | None = None) -> None:
    """Draw `pose` onto a matplotlib axis."""
    from matplotlib.patches import Polygon

    geo = pose_geometry(pose)
    ax.add_patch(Polygon(geo["palm"], closed=True, facecolor=color or _PALM_FACE,
                         edgecolor=_PALM_EDGE, linewidth=1.6, alpha=alpha, zorder=1))

    # Supination and pronation are mirror images of each other in a flat drawing,
    # so tilt alone cannot tell them apart. What actually distinguishes them is
    # which surface faces the viewer: palm creases when supinated, knuckles when
    # pronated. Without this the two classes are indistinguishable on screen even
    # when the model has them exactly right.
    if abs(pose.wrist_rot) > 0.25 and color is None:
        palm = geo["palm"]
        cx, cy = palm[:, 0].mean(), palm[:, 1].mean()
        w = (palm[:, 0].max() - palm[:, 0].min()) or 1e-6
        h = (palm[:, 1].max() - palm[:, 1].min()) or 1e-6
        if pose.wrist_rot > 0:                      # palm towards the viewer
            for dy, half in ((0.22, 0.26), (0.02, 0.30), (-0.20, 0.22)):
                ax.plot([cx - half * w, cx + half * w * 0.7],
                        [cy + dy * h, cy + (dy - 0.06) * h],
                        "-", color=_PALM_EDGE, alpha=0.45 * alpha, linewidth=1.3, zorder=2)
        else:                                       # back of the hand
            for dx in (-0.26, -0.09, 0.09, 0.26):
                ax.plot([cx + dx * w], [cy + 0.30 * h], ".", color=_PALM_EDGE,
                        alpha=0.5 * alpha, markersize=4.5, zorder=2)
    for name in FINGERS:
        pts = geo[name]
        ax.plot(pts[:, 0], pts[:, 1], "-", color=color or _FINGER, alpha=alpha,
                linewidth=linewidth if name != "thumb" else linewidth * 1.15,
                solid_capstyle="round", zorder=2)
        ax.plot(pts[-1, 0], pts[-1, 1], ".", color=color or _FINGER, alpha=alpha,
                markersize=linewidth * 0.8, zorder=3)

    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.15, 1.75)
    ax.set_aspect("equal")
    ax.axis("off")
    if label:
        ax.set_title(label, fontsize=9)


def pose_sheet(path: str = "results/hand_poses.png", classes: list[str] | None = None):
    """Render every pose in one figure, to eyeball that they read as distinct.

    Two grasps that look the same here will look the same in the demonstration,
    where the viewer has no labels to fall back on.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..data.movements import DEFAULT_SUBSET

    names = classes or [DEFAULT_SUBSET[k] for k in sorted(DEFAULT_SUBSET)]
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.9 * rows))
    for ax, name in zip(np.ravel(axes), names):
        draw_hand(ax, pose_for(name), label=name.replace("_", " "))
    for ax in np.ravel(axes)[len(names):]:
        ax.axis("off")
    fig.suptitle("Hand poses by movement class", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
