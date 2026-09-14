"""1D convolutional encoder over multichannel EMG windows.

Shape contract: (B, C, L) raw windows in, (B, D) embeddings out.

The first layer is deliberately depthwise -- each electrode gets its own temporal
filters before any mixing across electrodes. Electrode-to-muscle correspondence
is what changes between people and between wearings, so keeping the temporal
feature extraction independent of the spatial layout gives the adapters a clean
place to correct placement without disturbing the learned waveform features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import FiLM


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, pool: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2, bias=False)
        self.norm = nn.BatchNorm1d(c_out)
        self.film = FiLM(c_out)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.conv(x))
        x = self.film(x)          # identity until personalization fits it
        x = F.relu(x, inplace=True)
        return self.drop(self.pool(x))


class EMGEncoder(nn.Module):
    """Depthwise temporal stem -> spatial mixing -> conv stack -> pooled embedding."""

    def __init__(
        self,
        n_channels: int = 12,
        width: int = 64,
        embed_dim: int = 128,
        depth: int = 4,
        dropout: float = 0.2,
        temporal_kernel: int = 15,
        per_channel_filters: int = 8,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.embed_dim = embed_dim

        # Per-electrode gain/offset: the cheapest possible correction for a
        # sensor that sits differently today than it did during pretraining.
        self.input_film = FiLM(n_channels)

        # Depthwise: `per_channel_filters` temporal filters per electrode, no mixing.
        self.stem = nn.Conv1d(
            n_channels,
            n_channels * per_channel_filters,
            temporal_kernel,
            padding=temporal_kernel // 2,
            groups=n_channels,
            bias=False,
        )
        self.stem_norm = nn.BatchNorm1d(n_channels * per_channel_filters)
        # Pointwise: the one place channels are allowed to mix.
        self.spatial = nn.Conv1d(n_channels * per_channel_filters, width, 1, bias=False)
        self.spatial_norm = nn.BatchNorm1d(width)
        self.spatial_film = FiLM(width)

        self.blocks = nn.ModuleList(
            [ConvBlock(width, width, kernel=5, pool=2, dropout=dropout) for _ in range(depth)]
        )
        # Mean and std over time: EMG class information lives in amplitude
        # distribution as much as in waveform shape, and std pooling captures the
        # variance features (RMS-like) that classical EMG pipelines rely on.
        self.project = nn.Linear(width * 2, embed_dim)
        self.embed_norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_film(x)
        x = F.relu(self.stem_norm(self.stem(x)), inplace=True)
        x = self.spatial_film(self.spatial_norm(self.spatial(x)))
        x = F.relu(x, inplace=True)
        for block in self.blocks:
            x = block(x)
        pooled = torch.cat([x.mean(dim=-1), x.std(dim=-1)], dim=-1)
        return self.embed_norm(self.project(self.drop(pooled)))

    # -- parameter groups used by the personalization conditions ---------------

    def adapter_parameters(self):
        """Only the FiLM parameters -- what 'lightweight adaptation' may touch."""
        for module in self.modules():
            if isinstance(module, FiLM):
                yield from module.parameters()

    def n_adapter_parameters(self) -> int:
        return sum(p.numel() for p in self.adapter_parameters())

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
