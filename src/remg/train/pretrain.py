"""Pretraining the shared backbone on the source cohort.

Two losses are optimised together on one network:

  classification  standard cross-entropy through the linear head, over batches
                  mixed across source subjects. This is the model the
                  no-personalization baseline uses and the one ordinary
                  fine-tuning starts from.

  episodic        cross-entropy through the prototype head, where the prototypes
                  come from a few support repetitions of a single subject. This
                  is what makes the encoder personalizable from a few examples.

Training both on one backbone is what keeps the headline comparison honest: all
three conditions are then the same weights, differing only in what calibration
is allowed to change.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from ..data.windows import WindowSet
from ..models import build_model
from ..models.heads import PrototypeHead
from ..utils import pick_device, seed_everything
from .augment import AugmentConfig, apply as augment
from .sampler import EpisodeConfig, EpisodeSampler, to_tensors


@dataclass
class PretrainConfig:
    steps: int = 3000
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    episodic_weight: float = 1.0      # 0.0 gives the plain-supervised ablation
    class_balanced: bool = True       # rest outnumbers every movement ~6:1
    warmup_steps: int = 100
    grad_clip: float = 5.0
    log_every: int = 100
    seed: int = 0
    device: str | None = None
    model: dict = field(default_factory=dict)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)


def _sampling_weights(ws: WindowSet) -> np.ndarray:
    counts = np.bincount(ws.y, minlength=ws.n_classes).astype(np.float64)
    w = np.zeros(len(ws))
    nonzero = counts > 0
    per_class = np.zeros_like(counts)
    per_class[nonzero] = 1.0 / counts[nonzero]
    w = per_class[ws.y]
    return w / w.sum()


def pretrain(
    train: WindowSet,
    val: WindowSet | None = None,
    cfg: PretrainConfig | None = None,
    verbose: bool = True,
):
    """Train the shared backbone. Returns (model, history)."""
    cfg = cfg or PretrainConfig()
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)

    model = build_model(train.n_channels, train.n_classes, **cfg.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.steps, pct_start=cfg.warmup_steps / max(cfg.steps, 1)
    )

    rng = np.random.default_rng(cfg.seed)
    weights = _sampling_weights(train) if cfg.class_balanced else None
    sampler = (
        EpisodeSampler(train, cfg.episode, seed=cfg.seed) if cfg.episodic_weight > 0 else None
    )

    history: list[dict] = []
    t0 = time.time()
    model.train()

    for step in range(1, cfg.steps + 1):
        idx = rng.choice(len(train), size=cfg.batch_size, replace=False, p=weights)
        x, y = to_tensors(train, idx, train.y[idx], device)
        x = augment(x, cfg.augment)
        loss_ce = F.cross_entropy(model(x, mode="linear"), y, label_smoothing=0.05)

        loss_proto = torch.zeros((), device=device)
        if sampler is not None:
            s_idx, s_y, q_idx, q_y, _ = sampler.sample()
            xs, ys = to_tensors(train, s_idx, s_y, device)
            xq, yq = to_tensors(train, q_idx, q_y, device)
            # Support windows are augmented: real calibration data is collected
            # under whatever sensor conditions the day happens to bring, so the
            # model should not learn to trust pristine support examples.
            zs = model.embed(augment(xs, cfg.augment))
            zq = model.embed(augment(xq, cfg.augment))
            protos = PrototypeHead.compute_prototypes(zs, ys, train.n_classes)
            loss_proto = F.cross_entropy(model.proto_head(zq, protos), yq)

        loss = loss_ce + cfg.episodic_weight * loss_proto
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        if step % cfg.log_every == 0 or step == cfg.steps:
            row = {
                "step": step,
                "loss": float(loss.item()),
                "loss_ce": float(loss_ce.item()),
                "loss_proto": float(loss_proto.item()),
                "lr": sched.get_last_lr()[0],
                "elapsed_s": time.time() - t0,
            }
            if val is not None:
                row["val_acc"] = evaluate_linear(model, val, device)
                model.train()
            history.append(row)
            if verbose:
                extra = f" val_acc {row['val_acc']:.3f}" if "val_acc" in row else ""
                print(
                    f"  step {step:5d}/{cfg.steps}  loss {row['loss']:.4f} "
                    f"(ce {row['loss_ce']:.4f} proto {row['loss_proto']:.4f}){extra} "
                    f" {row['elapsed_s']:.0f}s",
                    flush=True,
                )

    model.eval()
    return model, history


@torch.no_grad()
def evaluate_linear(model, ws: WindowSet, device, batch_size: int = 512) -> float:
    """Plain accuracy through the linear head -- a training monitor, not a result.

    Reported metrics come from `remg.evaluate.metrics`, which is balanced; this
    is only here to watch the loss curve do something sensible.
    """
    model.eval()
    correct = 0
    for start in range(0, len(ws), batch_size):
        sl = slice(start, start + batch_size)
        x = torch.from_numpy(ws.X[sl]).to(device)
        pred = model(x, mode="linear").argmax(dim=1).cpu().numpy()
        correct += int((pred == ws.y[sl]).sum())
    return correct / max(len(ws), 1)
