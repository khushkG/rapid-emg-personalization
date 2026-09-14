"""Evaluation protocol: who is held out, and what counts as calibration data.

Two rules are enforced here because breaking either one inflates results in ways
that are invisible in the final numbers:

1. Leave-one-subject-out. A test subject contributes nothing to pretraining --
   not one window. Cross-subject generalization is the whole question.

2. Calibration is sampled by *repetition*, never by random windows. Consecutive
   windows inside one repetition overlap and share the same muscle contraction,
   so a random split puts near-duplicates on both sides and can lift accuracy by
   tens of points without the model having learned anything transferable. "Three
   calibration examples per movement" means three repetitions, which is also what
   a user would actually be asked to perform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .windows import WindowSet


@dataclass
class CalibrationSplit:
    """One target subject, partitioned into calibration and evaluation windows."""

    calib: WindowSet
    test: WindowSet
    subject: int
    calib_reps: tuple[int, ...]
    test_reps: tuple[int, ...]

    @property
    def shots(self) -> int:
        return len(self.calib_reps)

    def describe(self) -> str:
        return (
            f"S{self.subject}: calib reps {self.calib_reps} ({len(self.calib)} windows) | "
            f"test reps {self.test_reps} ({len(self.test)} windows)"
        )


def subjects(ws: WindowSet) -> list[int]:
    return sorted(int(s) for s in np.unique(ws.subject))


def sessions(ws: WindowSet) -> list[int]:
    return sorted(int(s) for s in np.unique(ws.session))


def hold_out_subject(ws: WindowSet, subject: int) -> tuple[WindowSet, WindowSet]:
    """Split into (everyone else, this subject)."""
    is_target = ws.subject == subject
    if not is_target.any():
        raise ValueError(f"subject {subject} not present in this window set")
    return ws.select(~is_target), ws.select(is_target)


def calibration_split(
    ws: WindowSet,
    subject: int,
    *,
    shots: int,
    session: int | None = None,
    test_session: int | None = None,
    rep_order: tuple[int, ...] | None = None,
) -> CalibrationSplit:
    """Carve `shots` repetitions out of a target subject as calibration data.

    `rep_order` fixes which repetitions are used for which shot count, so the
    1-shot calibration set is a subset of the 2-shot set. Without that, the
    shots-vs-accuracy curve mixes "more data" with "luckier repetitions".

    Passing `test_session` different from `session` gives the cross-session
    experiment: calibrate on day 1, evaluate on day 5.
    """
    target = ws.select(ws.subject == subject)
    if len(target) == 0:
        raise ValueError(f"subject {subject} not present in this window set")

    calib_pool = target if session is None else target.select(target.session == session)
    if len(calib_pool) == 0:
        raise ValueError(f"subject {subject} has no data in session {session}")

    reps = sorted(int(r) for r in np.unique(calib_pool.rep) if r > 0)
    order = list(rep_order) if rep_order else reps
    unknown = [r for r in order if r not in reps]
    if unknown:
        raise ValueError(f"rep_order lists repetitions {unknown} absent for subject {subject}")
    if shots >= len(order):
        raise ValueError(
            f"shots={shots} leaves no held-out repetitions (subject {subject} has {len(order)})"
        )

    calib_reps = tuple(order[:shots])
    calib = calib_pool.select(np.isin(calib_pool.rep, calib_reps))

    test_pool = target if test_session is None else target.select(target.session == test_session)
    if test_session is None or test_session == session:
        # Same session: evaluate on the repetitions calibration did not touch.
        test_reps = tuple(r for r in reps if r not in calib_reps)
        test = test_pool.select(np.isin(test_pool.rep, test_reps))
    else:
        # Different session: every repetition of the later session is unseen.
        test_reps = tuple(sorted(int(r) for r in np.unique(test_pool.rep) if r > 0))
        test = test_pool

    if len(test) == 0:
        raise ValueError(f"empty evaluation set for subject {subject}")

    missing = set(range(ws.n_classes)) - set(int(c) for c in np.unique(calib.y))
    if missing:
        names = [ws.class_names[i] for i in sorted(missing)]
        raise ValueError(
            f"subject {subject} calibration reps {calib_reps} are missing classes {names} -- "
            "the few-shot conditions assume every movement is demonstrated"
        )

    return CalibrationSplit(
        calib=calib, test=test, subject=subject, calib_reps=calib_reps, test_reps=test_reps
    )


def loso_folds(
    ws: WindowSet,
    target: WindowSet | None = None,
) -> list[tuple[int, WindowSet, WindowSet]]:
    """Yield (held-out subject, pretraining windows, that subject's windows).

    With `target` given, pretraining draws from `ws` (e.g. the intact DB2 cohort)
    while the held-out subjects come from `target` (the DB3 amputees) -- the two
    cohorts are disjoint, so no subject is ever in both.
    """
    if target is None:
        return [(s, *hold_out_subject(ws, s)) for s in subjects(ws)]
    return [(s, ws, target.select(target.subject == s)) for s in subjects(target)]
