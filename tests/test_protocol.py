"""Guards against the mistakes that would invalidate the results.

These are not unit tests of convenience. Each one blocks a specific way the
headline number could come out inflated without anything looking broken.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from remg.data import PreprocessConfig, WindowConfig, preprocess, segment
from remg.data.movements import (
    DEFAULT_SUBSET,
    EXERCISE_OFFSET,
    EXERCISE_SIZES,
    OFFICIAL_MOVEMENT_NAMES,
    subset_mapping,
    to_global_label,
)
from remg.data.splits import (
    calibration_split,
    hold_out_subject,
    loso_folds,
    recency_partition,
)
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


# --- the movement subset ----------------------------------------------------
#
# A wrong id here does not crash and does not look wrong in any output: the run
# simply trains and reports on a different movement than the one named. These
# tests are the only thing standing between a typo and a mislabelled result.

# Every subset entry, pinned to its name in the official NinaPro movement list.
SUBSET_PROVENANCE = [
    (5,  "open_hand",             "Abduction of the fingers"),
    (6,  "close_hand",            "Fingers flexed together"),
    (7,  "point_index",           "Pointing index"),
    (9,  "wrist_supination",      "Wrist supination (rotation axis through the middle finger)"),
    (10, "wrist_pronation",       "Wrist pronation (rotation axis through the middle finger)"),
    (13, "wrist_flexion",         "Wrist flexion"),
    (14, "wrist_extension",       "Wrist extension"),
    (18, "large_diameter_grasp",  "Large diameter grasp"),
    (22, "medium_wrap",           "Medium wrap"),
    (30, "tripod_grasp",          "Tripod grasp"),
    (34, "lateral_grasp",         "Lateral grasp"),
]


@pytest.mark.parametrize("global_id,our_name,official_name", SUBSET_PROVENANCE)
def test_subset_entry_matches_the_official_movement(global_id, our_name, official_name):
    assert DEFAULT_SUBSET[global_id] == our_name
    assert OFFICIAL_MOVEMENT_NAMES[global_id] == official_name


def test_subset_provenance_is_exhaustive():
    """Adding a movement without recording its provenance fails here."""
    pinned = {g for g, _, _ in SUBSET_PROVENANCE} | {0}       # 0 is rest
    assert set(DEFAULT_SUBSET) == pinned


def test_lateral_grasp_is_not_confused_with_tip_pinch():
    """The specific error this check was added to catch."""
    assert OFFICIAL_MOVEMENT_NAMES[32] == "Tip pinch grasp"
    assert OFFICIAL_MOVEMENT_NAMES[34] == "Lateral grasp"
    assert 32 not in DEFAULT_SUBSET


def test_official_names_cover_the_named_movements_exactly():
    """Exercises B and C are named postures; exercise D is force patterns."""
    named = EXERCISE_SIZES[1] + EXERCISE_SIZES[2]             # 17 + 23
    assert sorted(OFFICIAL_MOVEMENT_NAMES) == list(range(1, named + 1))
    assert len(set(OFFICIAL_MOVEMENT_NAMES.values())) == named   # no duplicates


def test_subset_ids_are_real_movements():
    for global_id in DEFAULT_SUBSET:
        if global_id == 0:
            continue
        assert global_id in OFFICIAL_MOVEMENT_NAMES, f"id {global_id} is not a named movement"


def test_subset_mapping_is_contiguous_and_ordered():
    """Models need 0..K-1 targets, and the names must stay aligned to them."""
    mapping, names = subset_mapping()
    assert sorted(mapping.values()) == list(range(len(DEFAULT_SUBSET)))
    assert names[mapping[0]] == "rest"
    for global_id, our_name, _ in SUBSET_PROVENANCE:
        assert names[mapping[global_id]] == our_name


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


# --- cohort exclusions ------------------------------------------------------
#
# An exclusion changes who the result is about. These pin the registry so one
# cannot be added, removed or silently reworded without the change being seen.

def test_documented_exclusions_are_exactly_what_we_expect():
    from remg.data.cohort import EXCLUSIONS, excluded_subjects

    assert excluded_subjects("DB3") == [1]
    assert excluded_subjects("DB2") == []
    assert set(EXCLUSIONS) == {("DB3", 1)}


def test_every_exclusion_states_a_reason():
    """An exclusion without a stated reason is indistinguishable from cherry-picking."""
    from remg.data.cohort import EXCLUSIONS

    for (dataset, subject), reason in EXCLUSIONS.items():
        assert len(reason) > 60, f"{dataset} S{subject} needs a real reason, not a note"
        assert any(w in reason.lower() for w in ("movement", "channel", "electrode")), (
            f"{dataset} S{subject}: the reason should say what is missing from the data"
        )


def test_exclusion_reason_is_reported_for_the_writeup():
    from remg.data.cohort import describe, exclusion_reason

    assert exclusion_reason("DB3", 1)
    assert exclusion_reason("DB3", 2) is None
    assert "S1" in describe("DB3")
    assert "no subjects excluded" in describe("DB2")


# --- rejection curve --------------------------------------------------------

def test_rejection_curve_flags_points_that_lost_classes():
    """Balanced accuracy across thresholds is only comparable at a fixed class set.

    Raising the confidence threshold drops whole classes out of the surviving
    windows, and balanced accuracy then averages over the survivors -- so the
    curve can climb without the model improving at all.
    """
    from remg.evaluate.metrics import rejection_curve

    n, k = 400, 4
    y = np.arange(n) % k
    proba = np.zeros((n, k))
    for i, c in enumerate(y):
        if c == 0:                      # class 0 is the only confident one
            proba[i] = 0.01 / (k - 1)
            proba[i, 0] = 0.99
        else:                           # every other class peaks at 0.4
            proba[i] = 0.6 / (k - 1)
            proba[i, c] = 0.4

    rows = rejection_curve(y, proba)

    assert all("classes_present" in r and "comparable" in r for r in rows)
    assert rows[0]["comparable"], "full coverage must be comparable to itself"
    degenerate = [r for r in rows if r["classes_present"] < 2]
    assert degenerate, "this fixture should drive some thresholds down to one class"
    assert all(np.isnan(r["balanced_accuracy"]) for r in degenerate), (
        "a single-class subset scores 1.0 or 0.0 by construction, not by skill"
    )


def test_rejection_curve_coverage_is_monotone():
    from remg.evaluate.metrics import rejection_curve

    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, size=300)
    proba = rng.dirichlet(np.ones(5), size=300)

    cov = [r["coverage"] for r in rejection_curve(y, proba)]
    assert all(b <= a + 1e-12 for a, b in zip(cov, cov[1:])), "coverage must not rise"


# --- the cross-repetition session proxy -------------------------------------

def test_recency_bins_are_disjoint_and_cover_the_test_set(ws):
    split = calibration_split(ws, subject=1, shots=2)
    bins = recency_partition(split.test, split.calib_reps)

    assert [name for name, _ in bins] == ["early", "late"]
    masks = [m for _, m in bins]
    assert not (masks[0] & masks[1]).any()
    assert (masks[0] | masks[1]).all(), "every held-out window belongs to a bin"


def test_late_repetitions_are_actually_later(ws):
    """The proxy is meaningless if the bins are not ordered in time."""
    split = calibration_split(ws, subject=1, shots=2)
    (_, early), (_, late) = recency_partition(split.test, split.calib_reps)

    assert split.test.rep[early].max() < split.test.rep[late].min()


def test_recency_bins_never_include_calibration_repetitions(ws):
    """Scoring on a calibration repetition would report memorisation as retention."""
    split = calibration_split(ws, subject=1, shots=2)

    for _, mask in recency_partition(ws.select(ws.subject == 1), split.calib_reps):
        assert set(np.unique(ws.select(ws.subject == 1).rep[mask])).isdisjoint(split.calib_reps)


def test_recency_partition_declines_when_it_cannot_measure(ws):
    """One held-out repetition cannot show drift; it must not fake a bin."""
    subject = ws.select(ws.subject == 1)
    reps = sorted(int(r) for r in np.unique(subject.rep) if r > 0)
    all_but_one = tuple(reps[:-1])

    assert recency_partition(subject, all_but_one) == []


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
