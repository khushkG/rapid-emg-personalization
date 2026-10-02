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
    t0 = time.time()
    is_move = (calib.y != cfg.rest_index).astype(np.int64)
    if len(np.unique(is_move)) < 2:
        return ClassicGateResult(None, None, cfg.threshold, time.time() - t0, True)

    zc = fit_zc_threshold(calib.X)
    feats = td_features(calib.X, zc)
    pipe = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            random_state=cfg.random_state,
            class_weight="balanced" if cfg.class_balanced else None,
        ),
    )
    pipe.fit(feats, is_move)
    return ClassicGateResult(pipe, zc, cfg.threshold, time.time() - t0, False)


def classic_gate_scores(ws: WindowSet, result: ClassicGateResult) -> np.ndarray:
    """Stage-1 P(movement) per window, for sweeping thresholds without refitting."""
    if result.degenerate or result.pipe is None:
        return np.full(len(ws), np.nan, dtype=np.float64)
    return result.pipe.predict_proba(td_features(ws.X, result.zc_threshold))[:, 1]
