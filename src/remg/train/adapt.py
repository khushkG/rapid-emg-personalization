"""The four personalization conditions the main experiment compares.

All four start from the identical pretrained checkpoint and see the identical
calibration windows. They differ only in what calibration is allowed to change:

  none          nothing. The general model, applied to a new person as-is.
  linear_probe  the final linear layer only. The cheap, strong classical baseline
                that new methods often forget to beat.
  finetune      every weight in the network. Ordinary transfer learning.
  rapid         the FiLM adapters (a few hundred numbers) plus prototypes
                recomputed from the calibration embeddings. The proposed method.

Hyperparameters for `finetune` and `rapid` -- step counts, learning rates, the
drift penalty -- must be chosen using held-out *source* subjects, never the test
subject. Tuning them per test subject would mean the test subject's data chose
the hyperparameters, and the reported few-shot accuracy would be optimistic.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..data.windows import WindowSet
from ..models.adapters import adapter_drift, reset_adapters
from ..models.heads import EMGClassifier, PrototypeHead

CONDITIONS = ("none", "linear_probe", "finetune", "rapid")


@dataclass
class AdaptConfig:
    finetune_steps: int = 100
    finetune_lr: float = 1e-4        # small: the backbone is already good
    probe_steps: int = 200
    probe_lr: float = 1e-3
    rapid_steps: int = 50
    rapid_lr: float = 5e-3           # larger: only a few hundred parameters
    drift_weight: float = 0.05       # penalty on moving away from the general model
    batch_size: int = 128
    # Balance the calibration classes for the gradient-trained conditions. Off
    # makes the comparison against `rapid` unfair rather than conservative --
    # see _batches.
    class_balanced: bool = True
    seed: int = 0


@dataclass
class AdaptResult:
    """Everything needed to score a condition and report what it cost."""

    model: EMGClassifier
    mode: str                 # which head to predict with: "linear" or "proto"
    condition: str
    seconds: float
    updated_params: int
    calib_windows: int
    calib_reps: tuple[int, ...]

    def predict(self, ws: WindowSet, device, batch_size: int = 512) -> np.ndarray:
        return predict(self.model, ws, device, mode=self.mode, batch_size=batch_size)

    def predict_proba(self, ws: WindowSet, device, batch_size: int = 512) -> np.ndarray:
        return predict_proba(self.model, ws, device, mode=self.mode, batch_size=batch_size)


@torch.no_grad()
def predict_proba(model, ws: WindowSet, device, mode: str = "linear", batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(ws), batch_size):
        x = torch.from_numpy(ws.X[start : start + batch_size]).to(device)
        out.append(F.softmax(model(x, mode=mode), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, ws.n_classes), dtype=np.float32)


def predict(model, ws: WindowSet, device, mode: str = "linear", batch_size: int = 512) -> np.ndarray:
    return predict_proba(model, ws, device, mode=mode, batch_size=batch_size).argmax(axis=1)


def _batches(n: int, batch_size: int, steps: int, rng: np.random.Generator,
             weights: np.ndarray | None = None):
    """Calibration batches, optionally sampled to balance the classes.

    Rest is around 79% of real windows and up to 70x any single movement, so a
    uniformly-sampled batch is mostly rest and cross-entropy on it converges to
    predicting rest. Pretraining already samples class-balanced; not doing the
    same here handicapped `linear_probe` and `finetune` against `rapid`, which
    classifies by per-class prototypes and is prior-free by construction. The
    resulting gap looked like the adapters working and was partly just the
    baselines inheriting a prior the proposed method never sees.
    """
    replace_ = n < batch_size
    for _ in range(steps):
        if weights is None:
            yield rng.choice(n, size=min(batch_size, n), replace=replace_)
        else:
            # Balanced sampling oversamples the rare classes, so it must draw
            # with replacement.
            yield rng.choice(n, size=min(batch_size, n), replace=True, p=weights)


def _class_balanced_weights(calib: WindowSet) -> np.ndarray:
    """Per-window sampling probabilities that give every present class equal mass."""
    counts = np.bincount(calib.y, minlength=calib.n_classes).astype(np.float64)
    per_class = np.zeros_like(counts)
    nonzero = counts > 0
    per_class[nonzero] = 1.0 / counts[nonzero]
    w = per_class[calib.y]
    return w / w.sum()


def _to_device(ws: WindowSet, idx, device):
    x = torch.from_numpy(ws.X[idx]).to(device)
    y = torch.from_numpy(ws.y[idx]).to(device)
    return x, y


def adapt(
    base_model: EMGClassifier,
    calib: WindowSet,
    condition: str,
    device,
    cfg: AdaptConfig | None = None,
    calib_reps: tuple[int, ...] = (),
) -> AdaptResult:
    """Personalize a copy of `base_model` to one subject's calibration windows."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    cfg = cfg or AdaptConfig()
    rng = np.random.default_rng(cfg.seed)

    # Deep copy so the shared pretrained model is never mutated -- otherwise the
    # conditions would contaminate each other in the order they happen to run.
    model = copy.deepcopy(base_model).to(device)
    t0 = time.time()

    if condition == "none":
        model.eval()
        return AdaptResult(model, "linear", condition, time.time() - t0, 0, len(calib), calib_reps)

    if condition == "linear_probe":
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.linear_head.parameters():
            p.requires_grad_(True)
        params = list(model.linear_head.parameters())
        opt = torch.optim.AdamW(params, lr=cfg.probe_lr)
        # Backbone stays in eval mode so its BatchNorm running statistics are not
        # rewritten by a handful of calibration windows.
        model.eval()
        weights = _class_balanced_weights(calib) if cfg.class_balanced else None
        for idx in _batches(len(calib), cfg.batch_size, cfg.probe_steps, rng, weights):
            x, y = _to_device(calib, idx, device)
            loss = F.cross_entropy(model(x, mode="linear"), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        return AdaptResult(
            model, "linear", condition, time.time() - t0,
            sum(p.numel() for p in params), len(calib), calib_reps,
        )

    if condition == "finetune":
        for p in model.parameters():
            p.requires_grad_(True)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.finetune_lr)
        model.train()
        weights = _class_balanced_weights(calib) if cfg.class_balanced else None
        for idx in _batches(len(calib), cfg.batch_size, cfg.finetune_steps, rng, weights):
            x, y = _to_device(calib, idx, device)
            loss = F.cross_entropy(model(x, mode="linear"), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        return AdaptResult(
            model, "linear", condition, time.time() - t0,
            sum(p.numel() for p in model.parameters()), len(calib), calib_reps,
        )

    # --- rapid personalization -------------------------------------------------
    reset_adapters(model)          # start from the un-personalized general model
    for p in model.parameters():
        p.requires_grad_(False)
    adapter_params = list(model.encoder.adapter_parameters())
    for p in adapter_params:
        p.requires_grad_(True)

    x_all = torch.from_numpy(calib.X).to(device)
    y_all = torch.from_numpy(calib.y).to(device)
    model.eval()                    # frozen BatchNorm statistics throughout

    if cfg.rapid_steps > 0:
        opt = torch.optim.AdamW(adapter_params, lr=cfg.rapid_lr)
        for idx in _batches(len(calib), cfg.batch_size, cfg.rapid_steps, rng):
            sel = torch.from_numpy(idx).to(device)
            # Split the batch into support and query halves so the adapters are
            # trained on the same leave-some-out objective used in pretraining,
            # rather than on reconstructing labels they already have.
            half = max(len(sel) // 2, 1)
            sup, qry = sel[:half], sel[half:]
            if len(qry) == 0:
                continue
            zs = model.embed(x_all[sup])
            zq = model.embed(x_all[qry])
            protos = PrototypeHead.compute_prototypes(zs, y_all[sup], model.n_classes)
            loss = F.cross_entropy(model.proto_head(zq, protos), y_all[qry])
            loss = loss + cfg.drift_weight * adapter_drift(model)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    # Final prototypes use every calibration window, through the adapted encoder.
    model.fit_prototypes(x_all, y_all)
    model.eval()
    return AdaptResult(
        model, "proto", condition, time.time() - t0,
        sum(p.numel() for p in adapter_params), len(calib), calib_reps,
    )
