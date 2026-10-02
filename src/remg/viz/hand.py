"""A 2D articulated hand, posed by movement class.

The demonstration replays recorded EMG and animates this hand from what the
model predicts, which is the only part of the project a non-specialist can read
directly. No prosthetic hardware is involved.

The hand is drawn rather than illustrated: each finger is a three-segment chain
whose joints bend with a flexion parameter, so a pose is a small vector rather
than a picture. That keeps the poses honest -- a grasp that needs the thumb
opposed cannot be faked by swapping in a nicer drawing -- and it means an
intermediate pose can be interpolated when the prediction changes.

Pose parameters, all 0..1 unless noted:

    flexion     per finger (thumb, index, middle, ring, little); 0 extended,
                1 fully curled
    spread      how far the fingers fan apart
    thumb_opp   thumb opposition: 0 alongside the index, 1 across the palm
    wrist_rot   -1 full pronation, +1 full supination (drawn as foreshortening)
    wrist_flex  -1 full extension, +1 full flexion
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "little")

# Finger length relative to palm height, and where each meets the palm.
_FINGER_LENGTH = {"thumb": 0.62, "index": 1.00, "middle": 1.08, "ring": 0.98, "little": 0.78}
_FINGER_BASE_X = {"thumb": -0.52, "index": -0.27, "middle": -0.02, "ring": 0.22, "little": 0.44}
_FINGER_SPREAD = {"thumb": -52.0, "index": -12.0, "middle": -1.0, "ring": 9.0, "little": 20.0}

# Proportions of a finger taken by its proximal, middle and distal segments.
_SEGMENTS = (0.45, 0.32, 0.23)

PALM_W, PALM_H = 1.15, 1.0


@dataclass(frozen=True)
class Pose:
    """One hand configuration."""

    flexion: tuple[float, float, float, float, float] = (0.2, 0.2, 0.2, 0.2, 0.2)
    spread: float = 0.35
    thumb_opp: float = 0.2
    wrist_rot: float = 0.0
    wrist_flex: float = 0.0

    def lerp(self, other: "Pose", t: float) -> "Pose":
        """Blend towards `other`, so a changed prediction moves rather than jumps."""
        t = float(np.clip(t, 0.0, 1.0))
        mix = lambda a, b: a + (b - a) * t          # noqa: E731
        return replace(
            self,
            flexion=tuple(mix(a, b) for a, b in zip(self.flexion, other.flexion)),
            spread=mix(self.spread, other.spread),
            thumb_opp=mix(self.thumb_opp, other.thumb_opp),
            wrist_rot=mix(self.wrist_rot, other.wrist_rot),
            wrist_flex=mix(self.wrist_flex, other.wrist_flex),
        )


# The twelve classes in DEFAULT_SUBSET. Names match remg.data.movements so a
# prediction maps straight to a pose; POSES is checked against that subset in
# the tests, which is what stops the two drifting apart.
POSES: dict[str, Pose] = {
    "rest":                 Pose(flexion=(0.28, 0.24, 0.26, 0.28, 0.30), spread=0.30,
                                 thumb_opp=0.25),
    "open_hand":            Pose(flexion=(0.05, 0.03, 0.03, 0.03, 0.04), spread=1.00,
                                 thumb_opp=0.05),
    "close_hand":           Pose(flexion=(0.85, 0.97, 0.97, 0.97, 0.97), spread=0.05,
                                 thumb_opp=0.55),
    "point_index":          Pose(flexion=(0.70, 0.02, 0.95, 0.95, 0.95), spread=0.15,
                                 thumb_opp=0.45),
    "wrist_supination":     Pose(flexion=(0.20, 0.15, 0.15, 0.15, 0.18), spread=0.40,
                                 thumb_opp=0.20, wrist_rot=+0.95),
    "wrist_pronation":      Pose(flexion=(0.20, 0.15, 0.15, 0.15, 0.18), spread=0.40,
                                 thumb_opp=0.20, wrist_rot=-0.95),
    "wrist_flexion":        Pose(flexion=(0.22, 0.20, 0.20, 0.20, 0.22), spread=0.35,
                                 thumb_opp=0.20, wrist_flex=+0.90),
    "wrist_extension":      Pose(flexion=(0.22, 0.20, 0.20, 0.20, 0.22), spread=0.35,
                                 thumb_opp=0.20, wrist_flex=-0.90),
    # Grasps: aperture is what separates them, so the flexion values carry the
    # object size rather than a label.
    "large_diameter_grasp": Pose(flexion=(0.42, 0.52, 0.54, 0.56, 0.58), spread=0.30,
                                 thumb_opp=0.85),
    "medium_wrap":          Pose(flexion=(0.55, 0.70, 0.72, 0.74, 0.76), spread=0.18,
                                 thumb_opp=0.88),
    "tripod_grasp":         Pose(flexion=(0.62, 0.68, 0.66, 0.30, 0.24), spread=0.22,
                                 thumb_opp=0.92),
    "lateral_grasp":        Pose(flexion=(0.58, 0.88, 0.90, 0.90, 0.90), spread=0.05,
                                 thumb_opp=0.35),   # key grip: thumb across the index
}


def _rot(points: np.ndarray, degrees: float, about: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    m = np.array([[c, -s], [s, c]])
    p = np.asarray(points, dtype=float) - np.array(about)
    return p @ m.T + np.array(about)


def finger_path(name: str, pose: Pose) -> np.ndarray:
    """Joint positions of one finger, from knuckle to fingertip.

    The hand is drawn from the back, so a finger curling towards the palm bends
    out of the picture plane. What the viewer sees is the finger getting shorter,
    not swinging sideways -- so flexion is applied as foreshortening (each
    segment projected through its accumulated joint angle) with only a slight
    inward lean for shape. Sweeping the fingers sideways instead, which is the
    obvious thing to write, makes every grasp look like the same hand waving.
    """
    idx = FINGERS.index(name)
    flex = float(np.clip(pose.flexion[idx], 0.0, 1.0))
    length = _FINGER_LENGTH[name]

    if name == "thumb":
        return _thumb_path(pose, flex, length)

    spread_deg = _FINGER_SPREAD[name] * (0.35 + 1.25 * pose.spread)
    base = np.array([_FINGER_BASE_X[name] * PALM_W / 1.15, PALM_H / 2.0])

    # Accumulated joint angle away from the plane, capped at a right angle so a
    # fingertip never projects back behind its own knuckle.
    joint_angles = np.array([45.0, 70.0, 90.0]) * flex

    points = [base]
    direction = spread_deg
    for seg, theta in zip(_SEGMENTS, joint_angles):
        direction += 9.0 * flex * np.sign(spread_deg or 1.0) * -1.0   # slight inward lean
        step = length * seg * PALM_H * np.cos(np.radians(theta))
        a = np.radians(direction)
        points.append(points[-1] + np.array([np.sin(a), np.cos(a)]) * step)
    return np.array(points)


def _thumb_path(pose: Pose, flex: float, length: float) -> np.ndarray:
    """The thumb, which opposes across the palm rather than curling into it.

    Opposition is the parameter that separates a grasp from a closed fist, so it
    moves the thumb across the palm and shortens it, rather than folding it the
    way the fingers fold.
    """
    opp = float(np.clip(pose.thumb_opp, 0.0, 1.0))
    base = np.array([-PALM_W * 0.44, -PALM_H * 0.12])

    # Out from the palm at rest, swinging across it as opposition increases.
    direction = -68.0 + 96.0 * opp
    joint_angles = np.array([30.0, 55.0, 70.0]) * flex

    points = [base]
    for seg, theta in zip(_SEGMENTS, joint_angles):
        direction += 16.0 * flex
        step = length * seg * PALM_H * np.cos(np.radians(theta))
        a = np.radians(direction)
        points.append(points[-1] + np.array([np.sin(a), np.cos(a)]) * step)
    return np.array(points)


def palm_path(pose: Pose) -> np.ndarray:
    """Outline of the palm, rounded so it reads as a hand rather than a box."""
    w, h = PALM_W / 2.0, PALM_H / 2.0
    # Wrist, up the little-finger side, across the knuckles, down the thumb side.
    pts = [
        (-w * 0.62, -h * 1.05), (w * 0.60, -h * 1.05),
        (w * 0.92, -h * 0.45), (w * 1.00, h * 0.30), (w * 0.94, h * 0.92),
        (w * 0.40, h * 1.02), (-w * 0.30, h * 1.00), (-w * 0.86, h * 0.80),
        (-w * 1.02, h * 0.10), (-w * 0.88, -h * 0.55),
    ]
    return _smooth_closed(np.array(pts), subdivisions=3)


def _smooth_closed(points: np.ndarray, subdivisions: int = 3) -> np.ndarray:
    """Chaikin corner-cutting: rounds a polygon without needing splines."""
    p = np.asarray(points, dtype=float)
    for _ in range(subdivisions):
        nxt = np.roll(p, -1, axis=0)
        p = np.reshape(np.stack([0.75 * p + 0.25 * nxt, 0.25 * p + 0.75 * nxt], axis=1),
                       (-1, 2))
    return p


def pose_geometry(pose: Pose) -> dict[str, np.ndarray]:
    """Palm outline and finger paths for `pose`, after the wrist is applied.

    Pronation and supination are a rotation about the forearm axis, which in a
    flat drawing shows up as the hand narrowing -- so it is drawn as horizontal
    foreshortening plus a slight tilt, not as an in-plane spin that would read
    as radial deviation instead.
    """
    parts = {"palm": palm_path(pose)}
    for name in FINGERS:
        parts[name] = finger_path(name, pose)

    squash = 1.0 - 0.55 * abs(pose.wrist_rot)
    tilt = 16.0 * pose.wrist_rot
    wrist = (0.0, -PALM_H / 2.0)

    out = {}
    for key, pts in parts.items():
        p = np.asarray(pts, dtype=float).copy()
        p[:, 0] *= squash
        p = _rot(p, tilt, about=wrist)
        p = _rot(p, -38.0 * pose.wrist_flex, about=wrist)
        out[key] = p
    return out


def pose_for(class_name: str) -> Pose:
    """Pose for a predicted class, falling back to rest for anything unknown."""
    return POSES.get(class_name, POSES["rest"])
