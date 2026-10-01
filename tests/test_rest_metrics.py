"""Guards on the rest-behaviour metrics.

The event rate is the one number here that cannot be derived from the existing
scores, and it is also the one that is easy to compute wrongly in the direction
that flatters the model: merge two distinct false activations into one and the
device looks twice as calm as it is. These tests pin the merge rules.
"""

from __future__ import annotations

import numpy as np
import pytest

from remg.evaluate.metrics import rest_metrics

FS = 1000
STRIDE = 100  # samples, so one window per 100 ms


def call(true, pred, start=None, group=None):
    n = len(true)
    return rest_metrics(
        np.array(true), np.array(pred),
        start=np.arange(n) * STRIDE if start is None else np.array(start),
        group=np.zeros(n, dtype=int) if group is None else np.array(group),
        stride_samples=STRIDE, fs=FS,
    )


def test_perfect_rest_has_no_false_activations():
    r = call([0, 0, 0, 0], [0, 0, 0, 0])
    assert r.rest_recall == 1.0
    assert r.false_activation_rate == 0.0
    assert r.false_activations_per_min == 0.0
    assert r.mean_false_burst_ms == 0.0


def test_false_activation_rate_is_one_minus_rest_recall():
    r = call([0, 0, 0, 0], [0, 3, 3, 0])
    assert r.rest_recall == pytest.approx(0.5)
    assert r.false_activation_rate == pytest.approx(1.0 - r.rest_recall)


def test_movement_windows_are_ignored():
    """Only true-rest windows can produce a false activation."""
    r = call([0, 5, 5, 0], [0, 2, 2, 0])
    assert r.rest_windows == 2
    assert r.false_activation_rate == 0.0


def test_one_sustained_error_is_one_event_not_many():
    r = call([0] * 11, [0] + [4] * 10)
    assert r.false_activations_per_min == pytest.approx(60.0 / 1.1, rel=1e-6)
    assert r.mean_false_burst_ms == pytest.approx(1000.0)
    assert r.longest_false_burst_ms == pytest.approx(1000.0)


def test_scattered_errors_are_separate_events():
    """Same per-window rate as above, five times the events -- a different device."""
    sustained = call([0] * 11, [0] + [4] * 10)
    scattered = call([0] * 20, [4, 0] * 10)
    assert scattered.false_activation_rate == pytest.approx(0.5)
    assert scattered.false_activations_per_min > sustained.false_activations_per_min
    assert scattered.mean_false_burst_ms == pytest.approx(100.0)


def test_a_different_predicted_movement_does_not_split_a_burst():
    """The user feels one unwanted motion, not two, if the label flickers."""
    r = call([0] * 5, [0, 3, 7, 3, 0])
    assert r.mean_false_burst_ms == pytest.approx(300.0)


def test_a_gap_in_time_splits_a_burst():
    """Consecutive rows are not always adjacent -- boundary windows get dropped."""
    starts = [0, 100, 100_000, 100_100]
    r = call([0, 0, 0, 0], [4, 4, 4, 4], start=starts)
    assert r.false_activation_rate == 1.0
    assert r.mean_false_burst_ms == pytest.approx(200.0), "the time gap must end the burst"


def test_a_group_boundary_splits_a_burst():
    """Two recordings must not merge into one event across the seam."""
    r = call([0, 0, 0, 0], [4, 4, 4, 4],
             start=[0, 100, 200, 300], group=[1, 1, 2, 2])
    assert r.mean_false_burst_ms == pytest.approx(200.0)


def test_rest_seconds_uses_stride_not_window_length():
    """Overlapping windows must not double-count wall-clock time."""
    r = call([0] * 10, [0] * 10)
    assert r.rest_seconds == pytest.approx(1.0)


def test_no_rest_windows_is_nan_not_zero():
    """A subject with no rest cannot score a perfect false-activation rate."""
    r = call([5, 5], [5, 5])
    assert np.isnan(r.rest_recall)
    assert np.isnan(r.false_activation_rate)
    assert r.rest_windows == 0


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        rest_metrics(np.zeros(3), np.zeros(2), start=np.zeros(3), group=np.zeros(3),
                     stride_samples=STRIDE, fs=FS)
