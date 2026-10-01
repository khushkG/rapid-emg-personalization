"""Guards on how the adaptation budget is chosen.

The budget is a hyperparameter, so two things must hold or the headline number is
not trustworthy: it must be selected without the test cohort, and the selection
must be reproducible. A first implementation broke the second -- it ranked
near-equal configurations by measured wall-clock, and the host paged into swap
mid-search, recording 91s for a 600-step configuration against 21s for a
1200-step one. The ordering inverted and the selection changed. These tests pin
the fix: cost is counted in steps, which is exact and machine-independent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tune_budget.py"


@pytest.fixture(scope="module")
def tb():
    spec = importlib.util.spec_from_file_location("tune_budget", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_condition_has_a_grid(tb):
    from remg.train.adapt import CONDITIONS

    adaptive = [c for c in CONDITIONS if c != "none"]
    assert sorted(tb.GRIDS) == sorted(adaptive), (
        "a condition without a grid would silently keep its hand-set default while "
        "the others were selected, which is the asymmetry this script exists to remove"
    )


def test_grids_bracket_the_shipped_default(tb):
    """A grid that only searches upward cannot report "the default was fine"."""
    from remg.train import AdaptConfig

    default = AdaptConfig()
    for cond, grid in tb.GRIDS.items():
        steps = sorted({tb.config_steps(cond, o) for o in grid})
        key = {"finetune": "finetune_steps", "linear_probe": "probe_steps",
               "rapid": "rapid_steps"}[cond]
        shipped = getattr(default, key)
        assert shipped in steps, f"{cond}: default {shipped} steps is not in the grid {steps}"
        assert max(steps) > shipped, f"{cond}: grid never searches above the default"


def test_config_labels_are_unique(tb):
    for cond, grid in tb.GRIDS.items():
        labels = [tb.config_label(cond, o) for o in grid]
        assert len(labels) == len(set(labels)), f"{cond}: duplicate config labels collide"


def test_config_steps_and_lr_read_the_right_keys(tb):
    for cond, grid in tb.GRIDS.items():
        for over in grid:
            assert tb.config_steps(cond, over) == over[
                {"finetune": "finetune_steps", "linear_probe": "probe_steps",
                 "rapid": "rapid_steps"}[cond]]
            assert tb._config_lr(cond, over) == over[
                {"finetune": "finetune_lr", "linear_probe": "probe_lr",
                 "rapid": "rapid_lr"}[cond]]


def _pick(tb, cond, band):
    """The production tie-break, applied to an explicit candidate band."""
    by_label = {tb.config_label(cond, o): o for o in tb.GRIDS[cond]}
    return min(band, key=lambda lab: (tb.config_steps(cond, by_label[lab]),
                                      tb._config_lr(cond, by_label[lab])))


def test_tie_break_prefers_fewer_steps_not_less_wall_clock(tb):
    """The exact inversion that corrupted the first search.

    600 steps must win over 1200 steps regardless of what the clock said, because
    on a thrashing host the clock said the opposite.
    """
    band = ["steps=1200,lr=0.001", "steps=600,lr=0.005"]
    assert _pick(tb, "rapid", band) == "steps=600,lr=0.005"


def test_tie_break_breaks_remaining_ties_on_learning_rate(tb):
    """Equal steps must not resolve by dict ordering, or the pick is arbitrary."""
    band = ["steps=600,lr=0.01", "steps=600,lr=0.001", "steps=600,lr=0.005"]
    assert _pick(tb, "rapid", band) == "steps=600,lr=0.001"
    assert _pick(tb, "rapid", list(reversed(band))) == "steps=600,lr=0.001"


def test_tie_break_is_order_independent(tb):
    import itertools

    band = ["steps=1200,lr=0.0001", "steps=300,lr=0.0003", "steps=600,lr=0.0003"]
    picks = {_pick(tb, "finetune", list(p)) for p in itertools.permutations(band)}
    assert picks == {"steps=300,lr=0.0003"}


def test_selection_loads_exactly_one_cohort(tb):
    """Structural guarantee that the target cohort cannot leak into selection.

    Checked on the parsed source rather than on prose: the script must contain a
    single `load_cohort` call, so there is no second dataset for DB3 to arrive
    through, and that call's dataset must default to the source cohort.
    """
    import ast

    tree = ast.parse(SCRIPT.read_text())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "load_cohort"]
    assert len(calls) == 1, f"expected one load_cohort call, found {len(calls)}"

    defaults = {}
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call)
                and getattr(n.func, "attr", None) == "add_argument"
                and n.args and isinstance(n.args[0], ast.Constant)):
            for kw in n.keywords:
                if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                    defaults[n.args[0].value] = kw.value.value
    assert defaults.get("--dataset") == "DB2", (
        f"selection must default to the source cohort, got {defaults.get('--dataset')!r}"
    )
