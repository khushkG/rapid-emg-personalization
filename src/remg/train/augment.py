"""Training-time signal augmentation.

Every augmentation here corresponds to a specific thing that goes wrong when an
EMG controller is worn rather than demonstrated:

  channel_dropout    an electrode loses skin contact mid-use
  channel_corrupt    an electrode saturates or picks up an artefact
  electrode_shift    the sleeve is donned rotated relative to yesterday
  gain_jitter        skin impedance changes, so per-channel gain changes
  time_shift         the window boundary lands elsewhere in the contraction
  noise              baseline noise level differs between sessions

Training under these is what the robustness evaluation later measures, so the
evaluation-time corruptions in `remg.evaluate.robustness` are deliberately *not*
the same code path: the model is tested on corruption severities and channel
counts it was not trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AugmentConfig:
    channel_dropout_p: float = 0.3     # probability a batch item gets any dropout
    channel_dropout_max: int = 2       # at most this many electrodes zeroed
    channel_corrupt_p: float = 0.1
    electrode_shift_p: float = 0.2
    electrode_shift_max: int = 1       # circular positions around the forearm
    gain_jitter_sd: float = 0.15       # log-normal, per channel
    time_shift_max: float = 0.1        # fraction of window length
    noise_sd: float = 0.05
    enabled: bool = True


def _bernoulli(p: float, n: int, device) -> torch.Tensor:
    return torch.rand(n, device=device) < p


def apply(x: torch.Tensor, cfg: AugmentConfig, generator: torch.Generator | None = None) -> torch.Tensor:
    """Augment a (B, C, L) batch. Returns a new tensor; input is left alone."""
    if not cfg.enabled:
        return x
    b, c, _ = x.shape
    dev = x.device
    x = x.clone()

    if cfg.gain_jitter_sd > 0:
        gain = torch.randn(b, c, 1, device=dev) * cfg.gain_jitter_sd
        x = x * gain.exp()

    if cfg.electrode_shift_p > 0 and cfg.electrode_shift_max > 0:
        # Electrodes sit in a ring around the forearm, so a donning rotation is a
        # circular shift of the channel axis -- the single most useful EMG
        # augmentation, because it is the dominant between-session nuisance.
        hit = _bernoulli(cfg.electrode_shift_p, b, dev)
        if hit.any():
            shifts = torch.randint(
                -cfg.electrode_shift_max, cfg.electrode_shift_max + 1, (b,), device=dev
            )
            idx = torch.arange(c, device=dev)[None, :].expand(b, c)
            idx = (idx - shifts[:, None] * hit[:, None].long()) % c
            x = torch.gather(x, 1, idx[:, :, None].expand(b, c, x.shape[2]))

    if cfg.channel_dropout_p > 0 and cfg.channel_dropout_max > 0:
        hit = _bernoulli(cfg.channel_dropout_p, b, dev)
        if hit.any():
            k = torch.randint(1, cfg.channel_dropout_max + 1, (b,), device=dev)
            # Rank random scores per item; the lowest `k` channels get zeroed.
            order = torch.rand(b, c, device=dev).argsort(dim=1).argsort(dim=1)
            mask = (order < k[:, None]) & hit[:, None]
            x = x.masked_fill(mask[:, :, None], 0.0)

    if cfg.channel_corrupt_p > 0:
        hit = _bernoulli(cfg.channel_corrupt_p, b, dev)
        if hit.any():
            victim = torch.randint(0, c, (b,), device=dev)
            onehot = torch.zeros(b, c, device=dev)
            onehot[torch.arange(b, device=dev), victim] = 1.0
            onehot = onehot * hit[:, None].float()
            # Heavy broadband noise plus a DC offset: a saturated or floating lead.
            junk = torch.randn_like(x) * 3.0 + torch.randn(b, c, 1, device=dev) * 2.0
            x = x * (1 - onehot[:, :, None]) + junk * onehot[:, :, None]

    if cfg.time_shift_max > 0:
        max_shift = int(x.shape[2] * cfg.time_shift_max)
        if max_shift > 0:
            shifts = torch.randint(-max_shift, max_shift + 1, (b,), device=dev)
            idx = torch.arange(x.shape[2], device=dev)[None, :].expand(b, x.shape[2])
            idx = (idx - shifts[:, None]) % x.shape[2]
            x = torch.gather(x, 2, idx[:, None, :].expand(b, c, x.shape[2]))

    if cfg.noise_sd > 0:
        x = x + torch.randn_like(x) * cfg.noise_sd

    return x
