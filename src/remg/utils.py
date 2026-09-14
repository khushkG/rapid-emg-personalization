"""Device selection, seeding, and other small shared helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """Apple-silicon GPU when available, else CUDA, else CPU."""
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch.

    Note for the writeup: this does not make MPS kernels bit-deterministic, so
    repeated runs of the same configuration will differ slightly. Report results
    over several seeds rather than quoting a single run.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
