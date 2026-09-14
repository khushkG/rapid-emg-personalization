"""The one in-memory format every data source produces.

Real NinaPro recordings and the synthetic generator both return `SubjectRecord`,
so the rest of the pipeline never has to know which it is looking at.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SubjectRecord:
    """One subject's full continuous recording, all exercises concatenated.

    emg:        (T, C) float32, raw signal in whatever units the source uses
    label:      (T,)   int16, global movement id (0 = rest)
    repetition: (T,)   int16, repetition index (0 = between repetitions)
    subject:    subject id within its dataset
    dataset:    "DB2" | "DB3" | "synthetic" | ...
    fs:         sampling rate in Hz
    session:    recording day/session, used by the cross-session experiment
    """

    emg: np.ndarray
    label: np.ndarray
    repetition: np.ndarray
    subject: int
    dataset: str
    fs: int
    session: int = 0
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.emg.ndim != 2:
            raise ValueError(f"emg must be (T, C), got shape {self.emg.shape}")
        t = self.emg.shape[0]
        for name in ("label", "repetition"):
            arr = getattr(self, name)
            if arr.shape != (t,):
                raise ValueError(f"{name} must be ({t},) to match emg, got {arr.shape}")

    @property
    def n_channels(self) -> int:
        return self.emg.shape[1]

    @property
    def duration_s(self) -> float:
        return self.emg.shape[0] / self.fs

    @property
    def uid(self) -> str:
        return f"{self.dataset}-S{self.subject}-sess{self.session}"

    def with_rest_repetitions(self) -> "SubjectRecord":
        """Attach every rest period to the repetition block it belongs to.

        NinaPro marks rest with repetition 0, which would leave every rest window
        unassignable when calibration/test sets are split by repetition -- all the
        rest data would land in one side or be dropped. Each rest stretch is
        assigned to the repetition that *follows* it (the movement it precedes),
        falling back to the preceding one for the trailing rest.
        """
        rep = self.repetition.copy()
        idx = np.arange(len(rep))
        nz = rep != 0
        if not nz.any():
            return self
        pos = np.where(nz, idx, -1)
        # Nearest non-zero to the right, then to the left for what remains.
        nxt = np.minimum.accumulate(np.where(nz, idx, len(rep))[::-1])[::-1]
        prv = np.maximum.accumulate(pos)
        src = np.where(nxt < len(rep), nxt, prv)
        filled = np.where(nz, rep, rep[np.clip(src, 0, len(rep) - 1)])
        return SubjectRecord(
            emg=self.emg,
            label=self.label,
            repetition=filled.astype(np.int16),
            subject=self.subject,
            dataset=self.dataset,
            fs=self.fs,
            session=self.session,
            meta=self.meta,
        )

    def summary(self) -> str:
        movements = sorted(int(v) for v in np.unique(self.label) if v > 0)
        reps = sorted(int(v) for v in np.unique(self.repetition) if v > 0)
        return (
            f"{self.uid}: {self.duration_s:7.1f}s  {self.n_channels}ch @ {self.fs}Hz  "
            f"{len(movements)} movements  {len(reps)} repetitions"
        )
