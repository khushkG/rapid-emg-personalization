"""Per-channel normalization -- and why the mode matters for the headline claim.

Normalizing a new subject's data with *that subject's own* statistics is already
a form of personalization. If the no-personalization baseline is denied it while
the proposed method gets it for free, the comparison is rigged and the reported
gain is partly just rescaling.

So the mode is explicit and the same object is handed to every condition:

  "source"  -- statistics from the pretraining subjects only. Honest definition
               of a truly un-personalized model, and the harshest baseline.
  "calib"   -- statistics from the target subject's calibration examples only.
               The default: every condition gets the identical few calibration
               windows, so the comparison isolates what the *adaptation method*
               adds on top of rescaling.
  "oracle"  -- statistics from the target subject's full recording, test windows
               included. Leaks test data; kept only as a diagnostic ceiling and
               never reported as a result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .windows import WindowSet

MODES = ("source", "calib", "oracle")


@dataclass
class ChannelStats:
    """Per-channel location/scale, shape (C, 1) for broadcasting over time."""

    mean: np.ndarray
    scale: np.ndarray
    mode: str

    def apply(self, ws: WindowSet) -> WindowSet:
        X = (ws.X - self.mean[None]) / self.scale[None]
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


def normalize_per_subject(ws: WindowSet, *, robust: bool = True) -> WindowSet:
    """Normalize each subject-session by its own statistics.

    Used for the *pretraining* cohort, where every window is training data, so a
    subject using its own statistics is not leakage. It removes the between-
    subject amplitude differences that would otherwise dominate the gradient and
    let the encoder waste capacity on who is wearing the sleeve.
    """
    X = ws.X.copy()
    for s in np.unique(ws.subject):
        for sess in np.unique(ws.session[ws.subject == s]):
            m = (ws.subject == s) & (ws.session == sess)
            stats = fit(ws.select(m), "oracle", robust=robust)
            X[m] = (ws.X[m] - stats.mean[None]) / stats.scale[None]
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


def fit(ws: WindowSet, mode: str, *, robust: bool = True, eps: float = 1e-6) -> ChannelStats:
    """Fit per-channel statistics on whatever window set is passed in.

    `robust` uses median / interquartile-based scale instead of mean / standard
    deviation. EMG is heavy-tailed and a single motion artefact can otherwise
    dominate a channel's scale -- which is exactly the kind of thing that
    differs between a clean lab session and a worn-sensor session.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    # (N, C, L) -> (C, N*L)
    flat = ws.X.transpose(1, 0, 2).reshape(ws.n_channels, -1)
    if robust:
        centre = np.median(flat, axis=1)
        q75, q25 = np.percentile(flat, [75, 25], axis=1)
        scale = (q75 - q25) / 1.349  # IQR -> std-equivalent for a normal
    else:
        centre = flat.mean(axis=1)
        scale = flat.std(axis=1)
    scale = np.maximum(scale, eps)
    return ChannelStats(
        mean=centre.astype(np.float32)[:, None],
        scale=scale.astype(np.float32)[:, None],
        mode=mode,
    )
