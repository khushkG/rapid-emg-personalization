"""Guards on the temporal decision rules.

Debounce exists to stop a single confident mistake from moving the hand. The ways
it can be wrong all make it look better than it is: fire early by accumulating a
run across a gap in time, or across a seam between recordings, or by counting
windows that disagree about *which* movement.
"""

from __future__ import annotations

import numpy as np
import pytest

from remg.evaluate.temporal import debounce, majority_vote

STRIDE = 100


def call(pred, n, start=None, group=None):
    m = len(pred)
    return debounce(
        np.array(pred), n=n,
        start=np.arange(m) * STRIDE if start is None else np.array(start),
        group=np.zeros(m, dtype=int) if group is None else np.array(group),
        stride_samples=STRIDE,
    )


def test_n_one_is_a_no_op():
    p = [0, 3, 0, 5, 5]
    assert np.array_equal(call(p, 1), np.array(p))


def test_a_lone_spike_is_suppressed():
    assert np.array_equal(call([0, 4, 0], 2), np.array([0, 0, 0]))


def test_a_sustained_movement_fires_after_n_windows():
    """The first n-1 windows are the latency cost; after that it emits."""
    assert np.array_equal(call([3, 3, 3, 3], 3), np.array([0, 0, 3, 3]))


def test_rest_is_never_delayed():
    """Returning to rest must be immediate -- a hand that keeps moving is the danger."""
    assert np.array_equal(call([3, 3, 3, 0], 2), np.array([0, 3, 3, 0]))


def test_a_changed_movement_restarts_the_count():
    """Two different movements are not two windows of agreement."""
    assert np.array_equal(call([3, 4, 4], 2), np.array([0, 0, 4]))


def test_a_time_gap_restarts_the_count():
    """Dropped boundary windows mean consecutive rows can be far apart in time."""
    out = call([4, 4], 2, start=[0, 100_000])
    assert np.array_equal(out, np.array([0, 0])), "a gap must not let the run accumulate"


def test_a_group_boundary_restarts_the_count():
    out = call([4, 4], 2, start=[0, 100], group=[1, 2])
    assert np.array_equal(out, np.array([0, 0]))


def test_debounce_never_invents_a_movement():
    rng = np.random.default_rng(0)
    p = rng.integers(0, 6, size=400)
    for n in (2, 3, 5):
        out = call(p, n)
        moved = out != 0
        assert np.array_equal(out[moved], p[moved]), "emitted a label it was not given"
        assert (out != 0).sum() <= (p != 0).sum(), "debounce must only ever suppress"


def test_debounce_is_order_independent():
    """Rows may arrive in any order; the rule is defined by start and group."""
    p = np.array([0, 3, 3, 3, 0])
    st = np.arange(5) * STRIDE
    gp = np.zeros(5, dtype=int)
    straight = debounce(p, start=st, group=gp, n=2, stride_samples=STRIDE)
    perm = np.array([3, 0, 4, 1, 2])
    shuffled = debounce(p[perm], start=st[perm], group=gp[perm], n=2,
                        stride_samples=STRIDE)
    assert np.array_equal(straight[perm], shuffled)


def test_rejects_bad_n():
    with pytest.raises(ValueError):
        call([0, 1], 0)


def test_majority_vote_span_one_is_a_no_op():
    p = [0, 3, 0]
    assert np.array_equal(majority_vote(np.array(p), group=np.zeros(3, int), span=1),
                          np.array(p))


def test_majority_vote_removes_an_isolated_flip():
    p = np.array([3, 3, 0, 3, 3])
    out = majority_vote(p, group=np.zeros(5, dtype=int), span=3)
    assert out[2] == 3


def test_majority_vote_does_not_cross_groups():
    p = np.array([3, 3, 4, 4])
    out = majority_vote(p, group=np.array([1, 1, 2, 2]), span=3)
    assert np.array_equal(out, p)
