"""Signal conditioning applied to a continuous recording before windowing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from .records import SubjectRecord


@dataclass
class PreprocessConfig:
    notch_hz: float | None = 50.0   # mains interference; None to skip
    notch_q: float = 30.0
    band_hz: tuple[float, float] | None = (20.0, 450.0)
    rectify_lowpass_hz: float | None = None  # set e.g. 5.0 to feed envelopes
    target_fs: int | None = None    # decimate to this rate after filtering


def _sos_bandpass(low: float, high: float, fs: int):
    nyq = fs / 2.0
    high = min(high, nyq * 0.99)
    return sps.butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")


def preprocess(rec: SubjectRecord, cfg: PreprocessConfig | None = None) -> SubjectRecord:
    """Filter (and optionally rectify/decimate) one recording.

    Filtering is zero-phase (`sosfiltfilt`) so movement onsets do not drift
    relative to their labels -- a causal filter would smear the boundary between
    rest and movement and quietly corrupt the window labels.
    """
    cfg = cfg or PreprocessConfig()
    x = rec.emg.astype(np.float64, copy=True)
    fs = rec.fs

    if cfg.notch_hz:
        # Notch the mains frequency and its harmonics up to Nyquist.
        f = cfg.notch_hz
        while f < fs / 2.0:
            b, a = sps.iirnotch(f, cfg.notch_q, fs)
            x = sps.filtfilt(b, a, x, axis=0)
            f += cfg.notch_hz

    if cfg.band_hz:
        x = sps.sosfiltfilt(_sos_bandpass(*cfg.band_hz, fs), x, axis=0)

    if cfg.rectify_lowpass_hz:
        x = np.abs(x)
        sos = sps.butter(4, cfg.rectify_lowpass_hz / (fs / 2.0), btype="lowpass", output="sos")
        x = sps.sosfiltfilt(sos, x, axis=0)

    label, repetition = rec.label, rec.repetition
    if cfg.target_fs and cfg.target_fs < fs:
        if fs % cfg.target_fs:
            raise ValueError(f"target_fs {cfg.target_fs} must divide source fs {fs}")
        step = fs // cfg.target_fs
        # Signal is already band-limited by the filters above, so plain striding
        # is safe; labels are stepped identically to stay aligned sample-for-sample.
        x = x[::step]
        label = label[::step]
        repetition = repetition[::step]
        fs = cfg.target_fs

    return SubjectRecord(
        emg=np.ascontiguousarray(x, dtype=np.float32),
        label=np.ascontiguousarray(label),
        repetition=np.ascontiguousarray(repetition),
        subject=rec.subject,
        dataset=rec.dataset,
        fs=fs,
        session=rec.session,
        meta={**rec.meta, "preprocessed": True},
    )
