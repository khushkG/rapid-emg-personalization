"""Synthetic EMG that mimics the NinaPro DB2/DB3 recording structure.

This exists so the entire pipeline -- preprocessing, windowing, leave-one-
subject-out splits, pretraining, few-shot adaptation, robustness evaluation --
can be built and tested before the real recordings are in hand, and so the unit
tests have a fast, deterministic data source afterwards.

It is NOT a model of human physiology and results on it are not findings. What
it does reproduce is the structure the method has to cope with:

  * each movement has a characteristic spatial activation pattern across channels
  * every subject sees that pattern through their own electrode placement
    (neighbouring-channel leakage) and their own per-channel gains
  * a new session shifts the placement again, by less than a new subject does
  * amputee subjects have weaker, noisier, more variable activations
  * signal is bursty: activation ramps up and down within each repetition

Those are exactly the nuisances personalization is supposed to absorb, so a
method that cannot beat the no-personalization baseline here is broken.
"""

from __future__ import annotations

import numpy as np

from .movements import DEFAULT_SUBSET
from .records import SubjectRecord

# Fixed seed for the population-level structure, so the "true" movement patterns
# are identical no matter which subjects get generated in which order.
_POPULATION_SEED = 20260914


def _population_patterns(movement_ids: list[int], n_channels: int) -> dict[int, np.ndarray]:
    """The canonical activation pattern per movement, shared by all subjects."""
    rng = np.random.default_rng(_POPULATION_SEED)
    patterns = {}
    for m in movement_ids:
        if m == 0:  # rest: no activation
            patterns[m] = np.zeros(n_channels, dtype=np.float32)
            continue
        # Sparse-ish: a couple of dominant muscles plus low background activity.
        base = rng.gamma(shape=0.7, scale=1.0, size=n_channels).astype(np.float32)
        base /= base.max() + 1e-8
        patterns[m] = base
    return patterns


def _placement_matrix(rng: np.random.Generator, n_channels: int, spread: float) -> np.ndarray:
    """Near-identity mixing matrix modelling electrode shift around the arm.

    `spread` controls how much a channel's signal leaks into its ring
    neighbours; the ring wraps because the electrodes sit around the forearm.
    """
    idx = np.arange(n_channels)
    # Circular distance between every pair of electrodes.
    d = np.abs(idx[:, None] - idx[None, :])
    d = np.minimum(d, n_channels - d)
    shift = rng.normal(0.0, spread)  # rotation of the whole array around the arm
    m = np.exp(-((d - shift) ** 2) / (2 * max(spread, 1e-3) ** 2))
    m += rng.normal(0.0, 0.05 * spread, size=m.shape)
    m = np.clip(m, 0.0, None)
    m /= m.sum(axis=1, keepdims=True) + 1e-8
    return m.astype(np.float32)


def _burst_envelope(rng: np.random.Generator, n: int, rise_frac: float = 0.15) -> np.ndarray:
    """Smooth ramp-up / hold / ramp-down envelope for one repetition."""
    rise = max(int(n * rise_frac), 1)
    env = np.ones(n, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, rise, dtype=np.float32)
    env[:rise] = ramp
    env[-rise:] = ramp[::-1]
    # Slow fatigue/tremor-like wobble so repetitions are not carbon copies.
    wobble = rng.normal(0.0, 1.0, size=n // 200 + 2)
    wobble = np.interp(np.linspace(0, len(wobble) - 1, n), np.arange(len(wobble)), wobble)
    return env * (1.0 + 0.12 * wobble.astype(np.float32))


def make_subject(
    subject: int,
    *,
    dataset: str = "synthetic",
    amputee: bool = False,
    session: int = 0,
    movement_ids: list[int] | None = None,
    n_channels: int = 12,
    fs: int = 2000,
    n_repetitions: int = 6,
    move_s: float = 5.0,
    rest_s: float = 3.0,
    seed: int | None = None,
) -> SubjectRecord:
    """Generate one subject's continuous recording in NinaPro layout."""
    movement_ids = sorted(movement_ids if movement_ids is not None else DEFAULT_SUBSET)
    active = [m for m in movement_ids if m != 0]

    # Subject identity is seeded by subject id alone, so session 1 of subject 3
    # is the *same person* as session 0 of subject 3, just re-instrumented.
    ident = np.random.default_rng(
        seed if seed is not None else hash((dataset, subject)) % (2**32)
    )
    patterns = _population_patterns(movement_ids, n_channels)

    # Between-subject variation is large; between-session variation is smaller.
    placement_spread = 1.6 if not amputee else 2.4
    placement = _placement_matrix(ident, n_channels, placement_spread)
    gains = np.exp(ident.normal(0.0, 0.45, size=n_channels)).astype(np.float32)

    sess_rng = np.random.default_rng((hash((dataset, subject, session)) % (2**32)))
    if session != 0:
        placement = 0.75 * placement + 0.25 * _placement_matrix(sess_rng, n_channels, placement_spread)
        placement /= placement.sum(axis=1, keepdims=True) + 1e-8
        gains = gains * np.exp(sess_rng.normal(0.0, 0.25, size=n_channels)).astype(np.float32)

    # Amputees: weaker signal, worse separability, more variable repetitions.
    amp_scale = 0.55 if amputee else 1.0
    noise_sd = 0.16 if amputee else 0.08
    rep_jitter = 0.28 if amputee else 0.15

    n_move = int(move_s * fs)
    n_rest = int(rest_s * fs)

    emg_chunks, label_chunks, rep_chunks = [], [], []

    def _emit(pattern: np.ndarray, n: int, label: int, rep: int, rng: np.random.Generator):
        # Interference-pattern EMG is well approximated by amplitude-modulated
        # noise; the spatial pattern sets each channel's amplitude.
        env = _burst_envelope(rng, n) if label != 0 else np.full(n, 0.0, dtype=np.float32)
        carrier = rng.normal(0.0, 1.0, size=(n, n_channels)).astype(np.float32)
        sig = carrier * (env[:, None] * pattern[None, :] + noise_sd)
        emg_chunks.append(sig)
        label_chunks.append(np.full(n, label, dtype=np.int16))
        rep_chunks.append(np.full(n, rep, dtype=np.int16))

    rest_pattern = np.zeros(n_channels, dtype=np.float32)
    _emit(rest_pattern, n_rest, 0, 0, sess_rng)

    for rep in range(1, n_repetitions + 1):
        for m in active:
            rng = np.random.default_rng((hash((dataset, subject, session, m, rep)) % (2**32)))
            # Channel pattern as this subject's electrodes actually see it.
            observed = placement @ patterns[m]
            observed = observed * gains * amp_scale
            observed = observed * np.exp(rng.normal(0.0, rep_jitter, size=n_channels)).astype(np.float32)
            _emit(observed.astype(np.float32), n_move, m, rep, rng)
            _emit(rest_pattern, n_rest, 0, 0, rng)

    return SubjectRecord(
        emg=np.concatenate(emg_chunks, axis=0).astype(np.float32),
        label=np.concatenate(label_chunks),
        repetition=np.concatenate(rep_chunks),
        subject=subject,
        dataset=dataset,
        fs=fs,
        session=session,
        meta={"synthetic": True, "amputee": amputee},
    )


def make_cohort(
    n_subjects: int,
    *,
    dataset: str = "synthetic",
    amputee: bool = False,
    sessions: int = 1,
    **kwargs,
) -> list[SubjectRecord]:
    """Generate a whole cohort, optionally with repeated sessions per subject."""
    return [
        make_subject(s, dataset=dataset, amputee=amputee, session=sess, **kwargs)
        for s in range(1, n_subjects + 1)
        for sess in range(sessions)
    ]
