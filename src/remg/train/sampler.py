"""Episode sampling for meta-training.

An episode simulates, during pretraining, exactly the situation the model will
face at deployment: one subject, a handful of calibration repetitions as the
support set, and unseen repetitions of that same subject as the query set. The
model is scored on the query set using a classifier built only from the support
set, so it is directly optimised for "be personalizable from a few examples"
rather than for "classify the pretraining cohort well".

Support and query repetitions are always disjoint. Sampling support and query
windows from the same repetition would let overlapping windows of one muscle
contraction appear on both sides, and the episode would reward memorisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..data.windows import WindowSet


@dataclass
class EpisodeConfig:
    n_support_reps: int = 2     # repetitions acting as calibration data
    n_support: int = 8          # windows per class drawn from those repetitions
    n_query: int = 16           # windows per class drawn from held-out repetitions


class EpisodeSampler:
    """Pre-indexes a window set so episodes can be drawn without scanning it."""

    def __init__(self, ws: WindowSet, cfg: EpisodeConfig | None = None, seed: int = 0):
        self.ws = ws
        self.cfg = cfg or EpisodeConfig()
        self.rng = np.random.default_rng(seed)

        # (subject, rep, class) -> window indices
        self.index: dict[int, dict[int, dict[int, np.ndarray]]] = {}
        for s in np.unique(ws.subject):
            s_mask = ws.subject == s
            per_rep: dict[int, dict[int, np.ndarray]] = {}
            for r in np.unique(ws.rep[s_mask]):
                if r <= 0:
                    continue
                rs_mask = s_mask & (ws.rep == r)
                per_class = {
                    int(c): np.flatnonzero(rs_mask & (ws.y == c))
                    for c in np.unique(ws.y[rs_mask])
                }
                per_rep[int(r)] = per_class
            if per_rep:
                per_rep_reps = sorted(per_rep)
                # A subject is only usable if it has enough repetitions to split.
                if len(per_rep_reps) > self.cfg.n_support_reps:
                    self.index[int(s)] = per_rep
        if not self.index:
            raise ValueError(
                "no subject has more than n_support_reps repetitions -- "
                "lower EpisodeConfig.n_support_reps"
            )
        self.subjects = sorted(self.index)

    def _draw(self, pools: list[np.ndarray], n: int) -> np.ndarray | None:
        pool = np.concatenate(pools) if pools else np.empty(0, dtype=int)
        if pool.size == 0:
            return None
        # Sample with replacement when a class is thin rather than dropping it;
        # a missing class would silently change the episode's task.
        replace = pool.size < n
        return self.rng.choice(pool, size=n, replace=replace)

    def sample(self, subject: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Return (support_idx, support_y, query_idx, query_y, subject)."""
        s = int(subject) if subject is not None else int(self.rng.choice(self.subjects))
        per_rep = self.index[s]
        reps = sorted(per_rep)
        self.rng.shuffle(reps)
        sup_reps = reps[: self.cfg.n_support_reps]
        qry_reps = reps[self.cfg.n_support_reps :]

        sup_idx, sup_y, qry_idx, qry_y = [], [], [], []
        for c in range(self.ws.n_classes):
            sup_pool = [per_rep[r][c] for r in sup_reps if c in per_rep[r]]
            qry_pool = [per_rep[r][c] for r in qry_reps if c in per_rep[r]]
            s_take = self._draw(sup_pool, self.cfg.n_support)
            q_take = self._draw(qry_pool, self.cfg.n_query)
            if s_take is None or q_take is None:
                continue  # this subject never performed this movement
            sup_idx.append(s_take)
            sup_y.append(np.full(len(s_take), c))
            qry_idx.append(q_take)
            qry_y.append(np.full(len(q_take), c))

        return (
            np.concatenate(sup_idx),
            np.concatenate(sup_y),
            np.concatenate(qry_idx),
            np.concatenate(qry_y),
            s,
        )


def to_tensors(ws: WindowSet, idx: np.ndarray, y: np.ndarray, device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(ws.X[idx]).to(device, non_blocking=True)
    t = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(device, non_blocking=True)
    return x, t
