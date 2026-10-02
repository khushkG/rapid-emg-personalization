"""A hybrid controller: classic features decide *whether* to move, the network decides *what*.

The measurement that motivates this is in the benchmark. On identical windows and
an identical protocol, time-domain features with a random forest reach 0.982 rest
recall and 1.8 false activations per minute, while the CNN -- better than all of
them on balanced accuracy -- manages 0.505 and 28.8. The classic model is far
better at exactly one thing, and it happens to be the thing a prosthesis cannot get
wrong.

So use each for what it is good at:

  stage 1   random forest on MAV/RMS/WL/ZC/SSC, rest vs movement, trained on the
            natural class balance. Amplitude features are close to a physical
            measurement of whether the muscle is active, which is why a shallow
            model is hard to beat here.
  stage 2   the fine-tuned CNN's movement argmax, with rest removed from the
            running -- the part where learned representations genuinely win.

This is the same `td_rf` as the benchmark, from the same `remg.features` code, so
the gate and the baseline cannot drift into being different methods.

The forest's settings are fixed a priori (300 trees, the benchmark's value). Its
decision threshold is an operating point and is selected on held-out *source*
subjects, never on the target cohort.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..data.windows import WindowSet
from ..features import fit_zc_threshold, td_features


@dataclass
class ClassicGateConfig:
    n_estimators: int = 300        # the benchmark's value, fixed before any sweep
    threshold: float = 0.5
    random_state: int = 0
    class_balanced: bool = False   # keep the natural rest prior -- the whole point
    rest_index: int = 0


@dataclass
class ClassicModel:
    """A fitted td_rf, binary or multiclass, with the deadzone it was fitted with."""

    pipe: object | None
    zc_threshold: np.ndarray | None
    seconds: float
    degenerate: bool
    classes: int

    def describe(self) -> str:
        if self.degenerate:
            return "td_rf not fitted (calibration had fewer than two classes)"
        return f"td_rf over {self.classes} classes"


def fit_td_rf(
    calib: WindowSet,
    *,
    n_estimators: int = 300,
    random_state: int = 0,
    class_balanced: bool = False,
    binary: bool = False,
    rest_index: int = 0,
) -> ClassicModel:
    """Fit a random forest on time-domain features of one subject's windows.

    `binary=True` collapses the targets to rest vs movement, which is stage 1 of the
    hybrid controller. `binary=False` fits the full movement set, which is the
    classic baseline in its own right -- the same model the literature benchmark
    runs, so the two are directly comparable rather than merely similar.

    The deadzone and the feature scaler are both fitted here, on calibration data
    only: a statistic taken from the evaluation windows is a leak however small.
    """
    t0 = time.time()
    y = (calib.y != rest_index).astype(np.int64) if binary else calib.y
    if len(np.unique(y)) < 2:
        return ClassicModel(None, None, time.time() - t0, True, int(len(np.unique(y))))
    zc = fit_zc_threshold(calib.X)
    pipe = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced" if class_balanced else None,
        ),
    )
    pipe.fit(td_features(calib.X, zc), y)
    return ClassicModel(pipe, zc, time.time() - t0, False, int(len(np.unique(y))))


def predict_td_rf(ws: WindowSet, model: ClassicModel) -> np.ndarray:
    """Hard labels in the original label space."""
    if model.degenerate or model.pipe is None:
        raise ValueError("cannot predict with a degenerate td_rf")
    return model.pipe.predict(td_features(ws.X, model.zc_threshold))


@dataclass
class ClassicGateResult:
    pipe: object | None
    zc_threshold: np.ndarray | None
    threshold: float
    seconds: float
    degenerate: bool

    def describe(self) -> str:
        if self.degenerate:
            return "classic gate not fitted (calibration had only one of rest/movement)"
        return f"classic gate (td_rf), threshold {self.threshold:.2f}"


def fit_classic_gate(
    calib: WindowSet,
    cfg: ClassicGateConfig | None = None,
) -> ClassicGateResult:
    """Train stage 1 on one subject's calibration windows.

    The zero-crossing deadzone and the feature scaler are both fitted here, on
    calibration data only, for the same reason the network's normalizer is: a
    statistic taken from the evaluation windows is a leak however small.
    """
    cfg = cfg or ClassicGateConfig()
    m = fit_td_rf(calib, n_estimators=cfg.n_estimators, random_state=cfg.random_state,
                  class_balanced=cfg.class_balanced, binary=True,
                  rest_index=cfg.rest_index)
    return ClassicGateResult(m.pipe, m.zc_threshold, cfg.threshold, m.seconds, m.degenerate)


def classic_gate_scores(ws: WindowSet, result: ClassicGateResult) -> np.ndarray:
    """Stage-1 P(movement) per window, for sweeping thresholds without refitting."""
    if result.degenerate or result.pipe is None:
        return np.full(len(ws), np.nan, dtype=np.float64)
    return result.pipe.predict_proba(td_features(ws.X, result.zc_threshold))[:, 1]
