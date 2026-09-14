"""Lightweight adaptation layers.

A FiLM layer applies a per-feature scale and shift:  y = gamma * x + beta.

It is initialised to the exact identity (gamma=1, beta=0), so inserting adapters
never changes the pretrained model's predictions before personalization begins.
That matters for the experiment: the no-personalization baseline and the adapted
model are then literally the same network, differing only in the few hundred
numbers that calibration fits, and any accuracy difference is attributable to
adaptation rather than to a different architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """Per-channel affine modulation, identity at initialisation.

    Accepts (B, C, L) feature maps or (B, C) vectors.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_features))
        self.beta = nn.Parameter(torch.zeros(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x * self.gamma[None, :, None] + self.beta[None, :, None]
        if x.dim() == 2:
            return x * self.gamma[None, :] + self.beta[None, :]
        raise ValueError(f"FiLM expects (B, C) or (B, C, L), got {tuple(x.shape)}")

    def reset(self) -> None:
        """Return to identity -- called before personalizing a new subject."""
        with torch.no_grad():
            self.gamma.fill_(1.0)
            self.beta.zero_()

    def drift(self) -> torch.Tensor:
        """How far this adapter has moved from identity.

        Used as a regulariser during few-shot adaptation: with only a handful of
        calibration repetitions, an unconstrained adapter will happily overfit
        them, so the amount of personalization is explicitly penalised.
        """
        return ((self.gamma - 1.0) ** 2).sum() + (self.beta**2).sum()


def reset_adapters(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, FiLM):
            module.reset()


def adapter_drift(model: nn.Module) -> torch.Tensor:
    terms = [m.drift() for m in model.modules() if isinstance(m, FiLM)]
    if not terms:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(terms).sum()
