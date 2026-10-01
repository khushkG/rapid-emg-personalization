"""A rest gate: decide whether to move at all, then decide what movement.

Why this and not more capacity. Adaptation trains class-balanced on purpose --
rest is ~79% of calibration windows, and without balancing cross-entropy collapses
onto predicting rest and every baseline looks crippled. That fix works, and it has
a cost nobody priced: throwing away the rest prior means the model no longer knows
that holding still is the common case. Measured on DB3, `finetune` keeps only 0.44
rest recall, so more than half of every rest period commands a movement. In a
prosthesis that is not a scoring artifact, it is the hand opening while the user
is carrying a glass.

The decomposition puts the prior back exactly where it belongs:

  stage 1   rest or movement, trained on the *natural* class balance, so it learns
            that rest is common
  stage 2   which movement, the existing class-balanced head with rest removed
            from the argmax, so it stays unbiased across the eleven grasps

Neither stage is asked to do the other's job, which is what the single head was
doing badly. Stage 1 is a logistic regression on the already-adapted embeddings:
no gradient steps, no new architecture, and nothing added to the parameter count
the headline comparison reports.

The threshold is a real operating-point choice, not a free win -- raising it buys
rest recall and spends movement recall. It is selected on held-out *source*
subjects (`scripts/tune_gate.py`), never on the amputee cohort.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from ..data.windows import WindowSet
from ..models.heads import EMGClassifier


@dataclass
class GateConfig:
    """Stage-1 settings.

    `class_balanced=False` is the point of the whole design, not an oversight: the
    gate is the one component that is *supposed* to know rest is common.
    """

    threshold: float = 0.5
    C: float = 1.0
    class_balanced: bool = False
    max_iter: int = 2000
    rest_index: int = 0


@dataclass
class TwoStageResult:
    gate: LogisticRegression | None
    threshold: float
    seconds: float
    gate_params: int
    degenerate: bool          # calibration held only one of {rest, movement}

    def describe(self) -> str:
        if self.degenerate:
            return "gate not fitted (calibration had only one of rest/movement)"
        return f"gate on {self.gate_params} params, threshold {self.threshold:.2f}"


@torch.no_grad()
def embed(model: EMGClassifier, ws: WindowSet, device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(ws), batch_size):
        x = torch.from_numpy(ws.X[start : start + batch_size]).to(device)
        out.append(model.embed(x).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)


def fit_gate(
    model: EMGClassifier,
    calib: WindowSet,
    device,
    cfg: GateConfig | None = None,
) -> TwoStageResult:
    """Fit stage 1 on the calibration windows of one subject."""
    cfg = cfg or GateConfig()
    t0 = time.time()
    is_move = (calib.y != cfg.rest_index).astype(np.int64)
    # A subject whose calibration is all rest or all movement cannot train a gate.
    # Returning a degenerate result lets the caller fall back to the single-stage
    # prediction rather than silently fitting a constant classifier.
    if len(np.unique(is_move)) < 2:
        return TwoStageResult(None, cfg.threshold, time.time() - t0, 0, True)

    z = embed(model, calib, device)
    gate = LogisticRegression(
        C=cfg.C,
        max_iter=cfg.max_iter,
        class_weight="balanced" if cfg.class_balanced else None,
    )
    gate.fit(z, is_move)
    return TwoStageResult(
        gate, cfg.threshold, time.time() - t0,
        int(gate.coef_.size + gate.intercept_.size), False,
    )


def predict_two_stage(
    model: EMGClassifier,
    ws: WindowSet,
    device,
    result: TwoStageResult,
    *,
    mode: str = "linear",
    threshold: float | None = None,
    rest_index: int = 0,
    proba: np.ndarray | None = None,
) -> np.ndarray:
    """Gate, then classify the movement.

    `proba` lets a caller pass stage-2 probabilities it has already computed, so a
    threshold sweep costs one forward pass rather than one per threshold.
    """
    from .adapt import predict_proba

    if proba is None:
        proba = predict_proba(model, ws, device, mode=mode)
    movement_pred = _argmax_excluding_rest(proba, rest_index)
    if result.degenerate or result.gate is None:
        return proba.argmax(axis=1)

    tau = result.threshold if threshold is None else threshold
    p_move = result.gate.predict_proba(embed(model, ws, device))[:, 1]
    return np.where(p_move >= tau, movement_pred, rest_index)


def gate_scores(model: EMGClassifier, ws: WindowSet, device,
                result: TwoStageResult) -> np.ndarray:
    """Stage-1 P(movement) for every window, for sweeping thresholds cheaply."""
    if result.degenerate or result.gate is None:
        return np.full(len(ws), np.nan, dtype=np.float64)
    return result.gate.predict_proba(embed(model, ws, device))[:, 1]


def _argmax_excluding_rest(proba: np.ndarray, rest_index: int) -> np.ndarray:
    """Best movement, with rest removed from the running rather than down-weighted.

    Masking to -inf rather than deleting the column keeps the returned indices in
    the original label space, so callers never have to remap.
    """
    masked = proba.copy()
    masked[:, rest_index] = -np.inf
    return masked.argmax(axis=1)


def apply_threshold(p_move: np.ndarray, movement_pred: np.ndarray, tau: float,
                    rest_index: int = 0) -> np.ndarray:
    """Vectorised decision for one threshold, given cached stage-1/stage-2 outputs."""
    return np.where(p_move >= tau, movement_pred, rest_index)
