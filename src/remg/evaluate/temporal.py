"""Temporal decision rules: what a controller actually emits, moment to moment.

A classifier produces one label per window. A prosthesis produces *motion*, and
the difference is latency and commitment. Both rules here operate on prediction
sequences only -- never on labels -- so either can run on a live device.

`debounce` is the conservative one and the reason this module exists. Requiring a
movement to be predicted N windows in a row before acting means a single confident
mistake cannot move the hand; it costs N-1 strides of latency on every genuine
movement onset, which is the honest price.

`majority_vote` is the softer, non-causal cousin used in the literature benchmark:
it takes the modal prediction over a window centred on the current sample, which
cannot run live but is what published numbers usually assume.
"""

from __future__ import annotations

import numpy as np


def debounce(
    pred: np.ndarray,
    *,
    start: np.ndarray,
    group: np.ndarray,
    n: int,
    stride_samples: int,
    rest_index: int = 0,
) -> np.ndarray:
    """Emit a movement only after it has been predicted `n` windows running.

    Strictly causal: window i is decided from windows i-n+1..i and nothing later,
    so the rule is implementable on a device. Anything short of n consecutive
    agreeing windows emits rest, which is the safe default for a hand.

    `n=1` is a no-op and returns the input unchanged.

    Adjacency is tested on `start`, not row order: segmentation drops windows that
    straddle a rest/movement boundary, so consecutive rows can be seconds apart.
    Treating them as adjacent would let a run accumulate across a hole and fire
    early -- the failure that silently weakens the rule.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    pred = np.asarray(pred)
    if n == 1:
        return pred.copy()

    start = np.asarray(start)
    group = np.asarray(group)
    if not (len(pred) == len(start) == len(group)):
        raise ValueError("pred, start and group must be the same length")

    out = np.full(len(pred), rest_index, dtype=pred.dtype)
    order = np.lexsort((start, group))
    g, s, p = group[order], start[order], pred[order]

    run = 0
    for i in range(len(p)):
        contiguous = (
            i > 0
            and g[i] == g[i - 1]
            and int(s[i] - s[i - 1]) <= stride_samples
            and p[i] == p[i - 1]
        )
        run = run + 1 if contiguous else 1
        if p[i] != rest_index and run >= n:
            out[order[i]] = p[i]
        # else: leave it at rest -- not yet committed, or genuinely rest
    return out


def majority_vote(
    pred: np.ndarray,
    *,
    group: np.ndarray,
    span: int,
) -> np.ndarray:
    """Modal prediction over `span` consecutive windows within one group.

    Non-causal (the window is centred), so this is an offline smoother. Kept here
    so the benchmark's smoothing and the few-shot pipeline's smoothing are one
    implementation rather than two that agree by coincidence.
    """
    if span <= 1:
        return np.asarray(pred).copy()
    pred = np.asarray(pred)
    out = pred.copy()
    half = span // 2
    for gval in np.unique(group):
        idx = np.flatnonzero(group == gval)
        seq = pred[idx]
        for j in range(len(seq)):
            lo, hi = max(0, j - half), min(len(seq), j + half + 1)
            vals, counts = np.unique(seq[lo:hi], return_counts=True)
            out[idx[j]] = vals[counts.argmax()]
    return out
