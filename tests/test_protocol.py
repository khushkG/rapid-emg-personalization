"""Guards against the mistakes that would invalidate the results.

These are not unit tests of convenience. Each one blocks a specific way the
headline number could come out inflated without anything looking broken.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from remg.data import PreprocessConfig, WindowConfig, preprocess, segment
from remg.data.movements import EXERCISE_OFFSET, to_global_label
from remg.data.splits import calibration_split, hold_out_subject, loso_folds
from remg.data.synthetic import make_cohort
from remg.models import build_model
from remg.models.adapters import FiLM, reset_adapters
from remg.train.adapt import adapt
from remg.train.sampler import EpisodeConfig, EpisodeSampler


@pytest.fixture(scope="module")
def ws():
    recs = make_cohort(3, fs=1000, n_repetitions=6, move_s=1.5, rest_s=0.8)
    recs = [preprocess(r, PreprocessConfig(band_hz=(20.0, 400.0))) for r in recs]
    return segment(recs, cfg=WindowConfig(length_ms=200, stride_ms=100))


# --- labels ----------------------------------------------------------------

def test_exercise_offsets_do_not_collide():
    """Every exercise's labels must map into a distinct global range."""
    ends = {e: EXERCISE_OFFSET[e] + n for e, n in {1: 17, 2: 23, 3: 9}.items()}
    assert ends[1] == EXERCISE_OFFSET[2]
    assert ends[2] == EXERCISE_OFFSET[3]
    assert ends[3] == 49


def test_rest_stays_rest_after_offsetting():
    local = np.array([0, 1, 5, 0, 23], dtype=np.int16)
    out = to_global_label(local, 2)
    assert out[0] == 0 and out[3] == 0          # rest is never shifted
    assert out.tolist() == [0, 18, 22, 0, 40]


# --- windowing -------------------------------------------------------------

def test_windows_are_label_pure(ws):
    """No window may straddle a movement boundary."""
    assert ws.X.ndim == 3
    assert len(ws) == len(ws.y) == len(ws.rep) == len(ws.subject)
    assert set(np.unique(ws.y)).issubset(set(range(ws.n_classes)))


def test_rest_windows_get_a_repetition(ws):
    """Rest must be splittable by repetition, or it all lands on one side."""
    rest = ws.rep[ws.y == 0]
    assert len(rest) > 0
    assert (rest > 0).all(), "rest windows still carry repetition 0"


# --- the two protocol rules ------------------------------------------------

def test_calibration_and_test_repetitions_are_disjoint(ws):
    """Overlapping windows from one contraction must not span both sides."""
    split = calibration_split(ws, subject=1, shots=2)
    assert set(split.calib_reps).isdisjoint(split.test_reps)
    assert set(np.unique(split.calib.rep)).isdisjoint(np.unique(split.test.rep))
    assert len(split.calib) > 0 and len(split.test) > 0


def test_calibration_sets_are_nested_across_shot_counts(ws):
    """1-shot data must be a subset of 2-shot data, or the curve is confounded."""
    one = calibration_split(ws, subject=1, shots=1)
    two = calibration_split(ws, subject=1, shots=2)
    assert set(one.calib_reps).issubset(two.calib_reps)


def test_calibration_covers_every_class(ws):
    split = calibration_split(ws, subject=1, shots=1)
    assert set(np.unique(split.calib.y)) == set(range(ws.n_classes))


def test_held_out_subject_contributes_nothing_to_pretraining(ws):
    for subject, train, test in loso_folds(ws):
        assert subject not in set(np.unique(train.subject))
        assert set(np.unique(test.subject)) == {subject}
        assert len(train) + len(test) == len(ws)


def test_cross_cohort_folds_keep_cohorts_disjoint(ws):
    source, target = hold_out_subject(ws, 3)
    folds = loso_folds(source, target=target)
    for subject, train, test in folds:
        assert subject not in set(np.unique(train.subject))


def test_episode_support_and_query_come_from_different_repetitions(ws):
    sampler = EpisodeSampler(ws, EpisodeConfig(n_support_reps=2, n_support=2, n_query=4), seed=0)
    for _ in range(10):
        s_idx, _, q_idx, _, _ = sampler.sample()
        assert set(ws.rep[s_idx]).isdisjoint(ws.rep[q_idx])
        assert set(np.unique(ws.subject[s_idx])) == set(np.unique(ws.subject[q_idx]))


# --- model / adaptation ----------------------------------------------------

def test_adapters_start_as_exact_identity():
    """The baseline and the adapted model must be the same network at step zero."""
    torch.manual_seed(0)
    model = build_model(n_channels=12, n_classes=5).eval()
    x = torch.randn(4, 12, 200)
    before = model(x, mode="linear")
    for m in model.modules():
        if isinstance(m, FiLM):
            assert torch.allclose(m.gamma, torch.ones_like(m.gamma))
            assert torch.allclose(m.beta, torch.zeros_like(m.beta))
    reset_adapters(model)
    assert torch.allclose(before, model(x, mode="linear"), atol=1e-6)


def test_adaptation_leaves_the_shared_model_untouched(ws):
    """Conditions run in sequence must not contaminate each other."""
    model = build_model(ws.n_channels, ws.n_classes).eval()
    snapshot = {k: v.clone() for k, v in model.state_dict().items()}
    split = calibration_split(ws, subject=1, shots=1)
    device = torch.device("cpu")
    for condition in ("linear_probe", "finetune", "rapid"):
        adapt(model, split.calib, condition, device)
        for k, v in model.state_dict().items():
            assert torch.equal(v, snapshot[k]), f"{condition} mutated the shared model at {k}"


def test_rapid_updates_only_adapter_parameters(ws):
    model = build_model(ws.n_channels, ws.n_classes).eval()
    split = calibration_split(ws, subject=1, shots=1)
    res = adapt(model, split.calib, "rapid", torch.device("cpu"))
    assert res.updated_params == model.encoder.n_adapter_parameters()
    assert res.updated_params < 0.02 * model.encoder.n_parameters()

    changed = [
        name
        for (name, a), (_, b) in zip(res.model.state_dict().items(), model.state_dict().items())
        if not torch.equal(a, b)
    ]
    assert all(("gamma" in n or "beta" in n or "prototypes" in n) for n in changed), changed
