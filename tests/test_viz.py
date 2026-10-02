"""Tests for the hand rendering.

The demonstration is the part of this project a non-specialist reads directly,
and it can be wrong in a way no accuracy number catches: if two movements are
drawn the same, a correct prediction is still unreadable on screen. These check
the poses are defined for the right classes and that they actually look
different from one another.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from remg.data.movements import DEFAULT_SUBSET
from remg.viz.hand import FINGERS, POSES, Pose, pose_for, pose_geometry


def pose_signature(pose: Pose) -> np.ndarray:
    """What the viewer can actually see: fingertips and the palm's placement."""
    geo = pose_geometry(pose)
    tips = np.array([geo[name][-1] for name in FINGERS]).ravel()
    palm = geo["palm"]
    extent = [np.ptp(palm[:, 0]), np.ptp(palm[:, 1])]
    return np.concatenate([tips, palm.mean(axis=0), extent])


def test_poses_cover_exactly_the_movement_subset():
    """A movement the model can predict but the hand cannot show is a dead end."""
    assert set(POSES) == set(DEFAULT_SUBSET.values())


@pytest.mark.parametrize("a,b", list(itertools.combinations(sorted(POSES), 2)))
def test_every_pair_of_poses_is_visually_distinct(a, b):
    """Two movements drawn alike make a correct prediction unreadable."""
    d = float(np.linalg.norm(pose_signature(POSES[a]) - pose_signature(POSES[b])))
    assert d > 0.12, f"{a} and {b} are drawn too similarly (signature distance {d:.3f})"


def test_supination_and_pronation_are_not_mere_mirrors():
    """They are opposite rotations, so tilt alone cannot separate them.

    The renderer distinguishes them by which surface faces the viewer; this
    guards the underlying parameter that cue keys on.
    """
    sup, pro = POSES["wrist_supination"], POSES["wrist_pronation"]
    assert np.sign(sup.wrist_rot) == -np.sign(pro.wrist_rot)
    assert abs(sup.wrist_rot) > 0.25 and abs(pro.wrist_rot) > 0.25


def test_flexion_shortens_the_visible_finger():
    """Drawn from the back, curling reads as foreshortening, not a sideways sweep.

    A sideways sweep is the natural thing to implement and makes every grasp
    look like the same hand waving.
    """
    from remg.viz.hand import finger_path

    extended = Pose(flexion=(0.0,) * 5)
    curled = Pose(flexion=(1.0,) * 5)
    for name in FINGERS:
        span = lambda p: float(np.linalg.norm(finger_path(name, p)[-1]      # noqa: E731
                                              - finger_path(name, p)[0]))
        assert span(curled) < span(extended) * 0.8, f"{name} does not foreshorten"


def test_a_fist_is_more_closed_than_an_open_hand():
    """The most basic thing a viewer reads off the drawing."""
    from remg.viz.hand import finger_path

    reach = lambda pose: sum(  # noqa: E731
        float(np.linalg.norm(finger_path(n, pose)[-1] - finger_path(n, pose)[0]))
        for n in FINGERS
    )
    assert reach(POSES["close_hand"]) < reach(POSES["open_hand"]) * 0.6


def test_grasps_differ_in_aperture():
    """Grasp classes are separated by how far the hand opens, not by a label."""
    from remg.viz.hand import finger_path

    def aperture(pose):
        tips = np.array([finger_path(n, pose)[-1] for n in FINGERS])
        return float(np.linalg.norm(tips[0] - tips[1:].mean(axis=0)))   # thumb to fingers

    wide = aperture(POSES["large_diameter_grasp"])
    tight = aperture(POSES["medium_wrap"])
    assert wide != pytest.approx(tight, abs=0.03), "the two wraps have the same aperture"


def test_lerp_moves_between_poses():
    a, b = POSES["rest"], POSES["close_hand"]
    assert a.lerp(b, 0.0) == a
    assert a.lerp(b, 1.0).flexion == pytest.approx(b.flexion)
    mid = a.lerp(b, 0.5)
    for x, y, z in zip(a.flexion, mid.flexion, b.flexion):
        assert min(x, z) <= y <= max(x, z)


def test_lerp_clamps_out_of_range_t():
    a, b = POSES["rest"], POSES["open_hand"]
    assert a.lerp(b, -5.0).flexion == pytest.approx(a.flexion)
    assert a.lerp(b, 5.0).flexion == pytest.approx(b.flexion)


def test_unknown_class_falls_back_to_rest():
    """A prediction the renderer does not know must not crash the demonstration."""
    assert pose_for("no_such_movement") == POSES["rest"]
    assert pose_for("tripod_grasp") == POSES["tripod_grasp"]


def test_geometry_is_finite_for_every_pose():
    for name, pose in POSES.items():
        for part, pts in pose_geometry(pose).items():
            assert np.isfinite(pts).all(), f"{name}/{part} produced non-finite points"
