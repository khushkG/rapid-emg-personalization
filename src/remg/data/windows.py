"""Sliding-window segmentation: continuous recordings -> labelled examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .movements import subset_mapping
from .records import SubjectRecord


@dataclass
class WindowConfig:
    length_ms: float = 200.0     # controller latency budget; 200 ms is the usual ceiling
    stride_ms: float = 100.0
    purity: float = 0.9          # fraction of samples that must share the window's label
    keep_rest: bool = True


@dataclass
class WindowSet:
    """A bag of windows with everything needed to split them honestly.

    X:       (N, C, L) float32
    y:       (N,) int64, contiguous class index
    rep:     (N,) int16, repetition the window came from. Rest is given the
             repetition of the movement it adjoins (see `with_rest_repetitions`)
             so repetition-based splits keep rest on both sides.
    subject: (N,) int32
    session: (N,) int16
    start:   (N,) int64, index of the window's first sample in its recording.
             Kept because several questions are about *time*, not just counts:
             how many separate false activations a controller would emit, how
             long each one lasts, and whether two windows are actually adjacent.
             Boundary windows are dropped during segmentation, so consecutive
             rows are not necessarily consecutive in time -- `start` is what makes
             that detectable rather than assumed.
    """

    X: np.ndarray
    y: np.ndarray
    rep: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    start: np.ndarray
    class_names: list[str]
    fs: int

    def __len__(self) -> int:
        return self.X.shape[0]

    @property
    def n_channels(self) -> int:
        return self.X.shape[1]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def select(self, mask: np.ndarray) -> "WindowSet":
        """Index into the set with a boolean mask or an integer index array."""
        return WindowSet(
            X=self.X[mask],
            y=self.y[mask],
            rep=self.rep[mask],
            subject=self.subject[mask],
            session=self.session[mask],
            start=self.start[mask],
            class_names=self.class_names,
            fs=self.fs,
        )

    def class_counts(self) -> dict[str, int]:
        counts = np.bincount(self.y, minlength=self.n_classes)
        return dict(zip(self.class_names, (int(c) for c in counts)))


def segment(
    records: list[SubjectRecord],
    *,
    cfg: WindowConfig | None = None,
    subset: dict[int, str] | None = None,
) -> WindowSet:
    """Cut every record into fixed-length windows and drop impure ones.

    A window is kept only if at least `cfg.purity` of its samples carry the same
    movement label *and* come from the same repetition. Windows straddling a
    rest/movement boundary are therefore discarded rather than assigned to
    whichever class happens to dominate -- otherwise the model gets trained on
    transitions it will be scored on as if they were steady-state movements.
    """
    cfg = cfg or WindowConfig()
    id_to_idx, class_names = subset_mapping(subset)

    Xs, ys, reps, subs, sess, starts_out = [], [], [], [], [], []
    for rec in records:
        # Rest carries repetition 0 in NinaPro; give it one so repetition-based
        # calibration/test splits can keep the rest class balanced on both sides.
        rec = rec.with_rest_repetitions()
        L = int(round(cfg.length_ms * rec.fs / 1000.0))
        S = int(round(cfg.stride_ms * rec.fs / 1000.0))
        if L <= 0 or S <= 0:
            raise ValueError("window length and stride must be positive")
        n = rec.emg.shape[0]
        if n < L:
            continue

        starts = np.arange(0, n - L + 1, S)
        # (n_windows, L) views of the label/repetition streams.
        lab_win = np.lib.stride_tricks.sliding_window_view(rec.label, L)[::S]
        rep_win = np.lib.stride_tricks.sliding_window_view(rec.repetition, L)[::S]

        centre = lab_win[:, L // 2]
        pure = (lab_win == centre[:, None]).mean(axis=1) >= cfg.purity
        rep_centre = rep_win[:, L // 2]
        pure &= (rep_win == rep_centre[:, None]).mean(axis=1) >= cfg.purity

        in_subset = np.isin(centre, list(id_to_idx))
        keep = pure & in_subset
        if not cfg.keep_rest:
            keep &= centre != 0
        if not keep.any():
            continue

        sel = starts[keep]
        # (n_kept, L, C) -> (n_kept, C, L); copy so we own the memory, not a view
        # into the full recording (which would pin the whole array alive).
        win = np.stack([rec.emg[s : s + L] for s in sel], axis=0).transpose(0, 2, 1)
        Xs.append(np.ascontiguousarray(win, dtype=np.float32))
        ys.append(np.array([id_to_idx[int(c)] for c in centre[keep]], dtype=np.int64))
        reps.append(rep_centre[keep].astype(np.int16))
        subs.append(np.full(len(sel), rec.subject, dtype=np.int32))
        sess.append(np.full(len(sel), rec.session, dtype=np.int16))
        starts_out.append(sel.astype(np.int64))

    if not Xs:
        raise ValueError("no windows survived segmentation -- check labels, subset and purity")

    return WindowSet(
        X=np.concatenate(Xs),
        y=np.concatenate(ys),
        rep=np.concatenate(reps),
        subject=np.concatenate(subs),
        session=np.concatenate(sess),
        start=np.concatenate(starts_out),
        class_names=class_names,
        fs=records[0].fs,
    )
