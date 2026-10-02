"""Evaluation-time sensor degradation.

Separate from `remg.train.augment` on purpose. The training augmentations teach
the model that electrodes can fail; these functions test it under failures at
severities and counts it was not trained on, including the worst case rather
than only the average. A controller that survives a random electrode dropping
out but fails whenever *one particular* electrode drops out has not been shown
to be robust -- the per-channel sweep is what exposes that.

Degradation is applied after normalization, because in deployment the electrode
fails during use, long after any calibration statistics were computed.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ..data.windows import WindowSet


def drop_channels(ws: WindowSet, channels: list[int]) -> WindowSet:
    """Zero the named electrodes -- a lead that has lost skin contact."""
    return _with_X(ws, _zeroed(ws.X, channels))


def corrupt_channels(
    ws: WindowSet,
    channels: list[int],
    *,
    kind: str = "noise",
    severity: float = 3.0,
    seed: int = 0,
) -> WindowSet:
    """Replace the named electrodes with a characteristic failure signal.

    kind="noise"      broadband junk: a floating or disconnected lead
    kind="saturate"   clipped at a large constant: an amplifier railed
    kind="offset"     a large DC shift: motion artefact or baseline wander
    """
    rng = np.random.default_rng(seed)
    X = ws.X.copy()
    scale = np.abs(ws.X).mean() + 1e-8
    for c in channels:
        if kind == "noise":
            X[:, c, :] = rng.normal(0.0, severity * scale, size=X[:, c, :].shape)
        elif kind == "saturate":
            X[:, c, :] = np.sign(rng.normal(size=(X.shape[0], 1))) * severity * scale
        elif kind == "offset":
            X[:, c, :] = X[:, c, :] + severity * scale
        else:
            raise ValueError(f"unknown corruption kind {kind!r}")
    return _with_X(ws, X)


def _with_X(ws: WindowSet, X: np.ndarray) -> WindowSet:
    return WindowSet(
        X=np.ascontiguousarray(X, dtype=np.float32),
        y=ws.y,
        rep=ws.rep,
        subject=ws.subject,
        session=ws.session,
        start=ws.start,
        class_names=ws.class_names,
        fs=ws.fs,
    )


def channel_failure_sweep(
    ws: WindowSet,
    n_failed: int,
    *,
    kind: str = "drop",
    severity: float = 3.0,
    max_combinations: int | None = 20,
    seed: int = 0,
) -> list[tuple[tuple[int, ...], WindowSet]]:
    """Enumerate which electrodes fail, yielding one degraded set per combination.

    With 12 electrodes, every single failure (12 cases) and every double failure
    (66 cases) is cheap to enumerate exhaustively; beyond that `max_combinations`
    samples uniformly so the sweep stays tractable. Report the mean *and the
    minimum* over these -- the minimum is the worst-case guarantee.
    """
    rng = np.random.default_rng(seed)
    all_combos = list(combinations(range(ws.n_channels), n_failed))
    if max_combinations is not None and len(all_combos) > max_combinations:
        pick = rng.choice(len(all_combos), size=max_combinations, replace=False)
        all_combos = [all_combos[i] for i in sorted(pick)]

    out = []
    for combo in all_combos:
        chans = list(combo)
        degraded = (
            _with_X(ws, _zeroed(ws.X, chans))
            if kind == "drop"
            else corrupt_channels(ws, chans, kind=kind, severity=severity, seed=seed)
        )
        out.append((combo, degraded))
    return out


def _zeroed(X: np.ndarray, channels: list[int]) -> np.ndarray:
    Y = X.copy()
    Y[:, channels, :] = 0.0
    return Y
