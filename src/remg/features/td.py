"""Hudgins-style time-domain features: MAV, RMS, waveform length, ZC, SSC.

The reference feature set for surface EMG classification since Hudgins et al.
(1993), and still the baseline any new method has to beat. Five numbers per
channel, computed on the same windows the network sees, so a comparison between
them differs only in the model.
"""

from __future__ import annotations

import numpy as np

TD_FEATURE_NAMES = ("MAV", "RMS", "WL", "ZC", "SSC")


def fit_zc_threshold(X_train: np.ndarray, frac: float = 0.01) -> np.ndarray:
    """Per-channel deadzone for the ZC and SSC counts, from training windows only.

    Must be fitted on training data: a threshold derived from the test windows
    would be a small but real leak, and these counts are sensitive to it.
    """
    return frac * np.sqrt((X_train ** 2).mean(axis=(0, 2)))


def td_features(X: np.ndarray, zc_threshold: np.ndarray) -> np.ndarray:
    """The five features, per channel, concatenated to (N, 5*C).

    X is (N, C, L). `zc_threshold` is (C,), a per-channel deadzone -- without one,
    amplifier noise around zero inflates the zero-crossing and slope-sign-change
    counts into nonsense.
    """
    mav = np.abs(X).mean(axis=2)
    rms = np.sqrt((X ** 2).mean(axis=2))
    wl = np.abs(np.diff(X, axis=2)).sum(axis=2)

    thr = zc_threshold[None, :, None]
    a, b = X[:, :, :-1], X[:, :, 1:]
    zc = (((a * b) < 0) & (np.abs(a - b) >= thr)).sum(axis=2)

    d = np.diff(X, axis=2)
    d1, d2 = d[:, :, :-1], d[:, :, 1:]
    ssc = (((d1 * d2) < 0) & ((np.abs(d1) >= thr) | (np.abs(d2) >= thr))).sum(axis=2)

    return np.concatenate([mav, rms, wl, zc, ssc], axis=1).astype(np.float64)
