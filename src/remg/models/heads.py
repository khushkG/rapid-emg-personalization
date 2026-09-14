"""Classifier heads: a standard linear head and a prototype head.

The linear head is what the no-personalization baseline uses and what ordinary
fine-tuning updates. The prototype head is what rapid personalization uses: it
has no per-class weights to learn, so a new subject's classifier is defined by
averaging their calibration embeddings -- constant time, no gradient steps, and
nothing to overfit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearHead(nn.Module):
    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


class PrototypeHead(nn.Module):
    """Cosine-similarity classifier against per-class prototype embeddings.

    Cosine rather than Euclidean distance: a new subject's embeddings are often
    globally scaled relative to the pretraining cohort (weaker muscles, different
    electrode gain), which shifts Euclidean distances to every prototype at once
    while leaving direction -- the part that identifies the movement -- intact.
    """

    def __init__(self, temperature: float = 10.0, learn_temperature: bool = True):
        super().__init__()
        t = torch.tensor(float(temperature)).log()
        self.log_temperature = nn.Parameter(t, requires_grad=learn_temperature)
        self.register_buffer("prototypes", torch.empty(0), persistent=True)

    @staticmethod
    def compute_prototypes(z: torch.Tensor, y: torch.Tensor, n_classes: int) -> torch.Tensor:
        """Mean embedding per class, (K, D).

        Classes absent from the support set get a zero prototype, which cosine
        similarity scores as 0 for every query -- neutral rather than arbitrary.
        """
        d = z.shape[1]
        sums = torch.zeros(n_classes, d, device=z.device, dtype=z.dtype)
        counts = torch.zeros(n_classes, device=z.device, dtype=z.dtype)
        sums.index_add_(0, y, z)
        counts.index_add_(0, y, torch.ones_like(y, dtype=z.dtype))
        return sums / counts.clamp(min=1.0).unsqueeze(1)

    def set_prototypes(self, prototypes: torch.Tensor) -> None:
        self.prototypes = prototypes.detach().clone()

    def forward(self, z: torch.Tensor, prototypes: torch.Tensor | None = None) -> torch.Tensor:
        p = self.prototypes if prototypes is None else prototypes
        if p.numel() == 0:
            raise RuntimeError("prototype head has no prototypes -- call set_prototypes first")
        logits = F.normalize(z, dim=-1) @ F.normalize(p, dim=-1).T
        return logits * self.log_temperature.exp()


class EMGClassifier(nn.Module):
    """Encoder plus both heads.

    Both heads are trained jointly during pretraining so that all three
    experimental conditions start from one shared backbone. Without that, the
    comparison would confound the adaptation method with having trained a
    different network, and "rapid personalization wins" could just mean
    "episodic pretraining produces a better encoder".
    """

    def __init__(self, encoder: nn.Module, n_classes: int, temperature: float = 10.0):
        super().__init__()
        self.encoder = encoder
        self.n_classes = n_classes
        self.linear_head = LinearHead(encoder.embed_dim, n_classes)
        self.proto_head = PrototypeHead(temperature)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor, mode: str = "linear") -> torch.Tensor:
        z = self.embed(x)
        if mode == "linear":
            return self.linear_head(z)
        if mode == "proto":
            return self.proto_head(z)
        raise ValueError(f"mode must be 'linear' or 'proto', got {mode!r}")

    @torch.no_grad()
    def fit_prototypes(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Define this subject's classifier from their calibration windows."""
        was_training = self.training
        self.eval()
        z = self.embed(x)
        protos = PrototypeHead.compute_prototypes(z, y, self.n_classes)
        self.proto_head.set_prototypes(protos)
        self.train(was_training)
        return protos
