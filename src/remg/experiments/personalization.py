"""The main experiment: does rapid personalization beat the alternatives?

One run produces a tidy table with one row per
(seed, subject, shots, condition, evaluation) so every figure in the writeup is a
group-by on the same file and no number is computed twice in two places.

The protocol, stated once:

  * The encoder is pretrained on the source cohort only. A target subject
    contributes nothing to pretraining.
  * Each target subject is calibrated on `shots` repetitions and evaluated on the
    repetitions calibration never touched.
  * Every condition receives byte-identical calibration windows and is evaluated
    on byte-identical test windows.
  * Normalization statistics come from the calibration windows alone, so the
    no-personalization baseline is not handicapped by scaling it was denied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data import fit_normalizer
from ..data.normalize import normalize_per_subject
from ..data.splits import calibration_split, recency_partition, subjects
from ..data.windows import WindowSet
from ..evaluate.metrics import rejection_curve, rest_metrics, score
from ..evaluate.temporal import debounce
from ..evaluate.robustness import channel_failure_sweep
from ..train.adapt import CONDITIONS, AdaptConfig, adapt
from ..train.classic_gate import (
    ClassicGateConfig,
    classic_gate_scores,
    fit_classic_gate,
)
from ..train.pretrain import PretrainConfig, pretrain
from ..train.twostage import (
    GateConfig,
    _argmax_excluding_rest,
    apply_threshold,
    fit_gate,
    gate_scores,
)
from ..utils import pick_device


@dataclass
class ExperimentConfig:
    shots: tuple[int, ...] = (1, 2, 3)
    conditions: tuple[str, ...] = CONDITIONS
    norm_mode: str = "calib"
    seeds: tuple[int, ...] = (0,)
    # Sensor-failure sweep. Only run at the largest shot count by default -- it
    # multiplies evaluation cost by the number of electrode combinations.
    failure_counts: tuple[int, ...] = (1, 2)
    failure_kind: str = "drop"
    failure_max_combinations: int = 12
    failures_at_max_shots_only: bool = True
    rejection: bool = True
    # Cross-repetition session proxy: additionally score the held-out repetitions
    # split by how far after calibration they were recorded. Free -- it regroups
    # predictions already computed -- so it is on by default.
    recency_split: bool = True
    recency_bins: int = 2
    # Rest-gate (two-stage) evaluation. Scored alongside the single-stage
    # prediction from the same adapted model, so the two differ only in the
    # decision rule and the comparison costs one extra forward pass. Thresholds
    # must come from held-out source subjects -- see scripts/tune_gate.py.
    two_stage: bool = True
    gate_thresholds: dict[str, float] = field(default_factory=dict)
    gate: GateConfig = field(default_factory=GateConfig)
    # Debounce: emit a movement only after it is predicted N windows running.
    # Per condition, selected on held-out source subjects (scripts/tune_rules.py).
    # Absent or 1 means no debounce.
    debounce_n: dict[str, int] = field(default_factory=dict)
    # Classic rest gate: td_rf on time-domain features as stage 1, the adapted
    # network's movement argmax as stage 2. Thresholds also from held-out source.
    classic_gate: bool = True
    classic_gate_thresholds: dict[str, float] = field(default_factory=dict)
    classic_gate_cfg: ClassicGateConfig = field(default_factory=ClassicGateConfig)
    # Window stride, needed to turn window counts into wall-clock for the
    # false-activation event rate. Must match the WindowConfig used to segment.
    stride_samples: int = 100
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    adapt: AdaptConfig = field(default_factory=AdaptConfig)
    device: str | None = None


def run(
    source: WindowSet,
    target: WindowSet,
    cfg: ExperimentConfig | None = None,
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Run the full leave-one-subject-out personalization experiment.

    `source` is the pretraining cohort (e.g. intact DB2 subjects); `target` is the
    evaluation cohort (e.g. DB3 amputees). They must be disjoint sets of people --
    pass the same window set twice only for a within-cohort sanity check, and
    note that doing so makes the source subjects their own test subjects.
    """
    cfg = cfg or ExperimentConfig()
    device = pick_device(cfg.device)
    rows: list[dict] = []
    curves: list[dict] = []

    source_n = normalize_per_subject(source)

    for seed in cfg.seeds:
        if verbose:
            print(f"[seed {seed}] pretraining on {len(subjects(source))} source subjects "
                  f"({len(source_n)} windows) on {device}", flush=True)
        pcfg = PretrainConfig(**{**cfg.pretrain.__dict__, "seed": seed})
        base, history = pretrain(source_n, val=None, cfg=pcfg, verbose=verbose)

        for subj in subjects(target):
            for shots in cfg.shots:
                try:
                    split = calibration_split(target, subj, shots=shots)
                except ValueError as exc:
                    if verbose:
                        print(f"  skip S{subj} shots={shots}: {exc}", flush=True)
                    continue

                stats = fit_normalizer(split.calib, cfg.norm_mode)
                calib_n = stats.apply(split.calib)
                test_n = stats.apply(split.test)

                # Stage 1 of the classic gate reads raw calibration windows, not
                # the network, so it is fitted once per calibration set rather than
                # refitted identically for each condition.
                cgate = None
                p_move_classic = None
                if cfg.classic_gate:
                    cgate = fit_classic_gate(calib_n, cfg.classic_gate_cfg)
                    if not cgate.degenerate:
                        p_move_classic = classic_gate_scores(test_n, cgate)

                for condition in cfg.conditions:
                    acfg = AdaptConfig(**{**cfg.adapt.__dict__, "seed": seed})
                    res = adapt(base, calib_n, condition, device, acfg, split.calib_reps)

                    proba = res.predict_proba(test_n, device)
                    pred = proba.argmax(axis=1)
                    sc = score(test_n.y, pred, test_n.class_names)
                    # One group per (subject, session, repetition) so a false
                    # activation is never counted as running across a seam.
                    rest_group = (test_n.subject.astype(np.int64) * 1_000_000
                                  + test_n.session.astype(np.int64) * 1000
                                  + test_n.rep.astype(np.int64))
                    rm = rest_metrics(test_n.y, pred, start=test_n.start,
                                      group=rest_group,
                                      stride_samples=cfg.stride_samples, fs=test_n.fs)
                    base_row = {
                        "seed": seed,
                        "subject": subj,
                        "shots": shots,
                        "condition": condition,
                        "evaluation": "clean",
                        "calib_windows": res.calib_windows,
                        "calib_reps": ",".join(map(str, res.calib_reps)),
                        "adapt_seconds": res.seconds,
                        "updated_params": res.updated_params,
                        "stage": "single",
                        "gate_threshold": float("nan"),
                        "debounce_n": 1,
                        **sc.as_row(),
                        **rm.as_row(),
                    }
                    rows.append(base_row)
                    if verbose:
                        print(
                            f"  S{subj:<3} shots={shots} {condition:<13} "
                            f"bal_acc {sc.balanced_accuracy:.3f}  macroF1 {sc.macro_f1:.3f}  "
                            f"({res.seconds:.1f}s, {res.updated_params} params)",
                            flush=True,
                        )

                    # Every decision rule below is a post-hoc transform of the
                    # predictions already computed, so the whole comparison costs
                    # two extra forward passes rather than one run per rule. All of
                    # them are causal except the logistic/classic gates' reliance on
                    # per-window scores, which are themselves causal.
                    rest_i = cfg.gate.rest_index
                    n_deb = int(cfg.debounce_n.get(condition, 1))
                    variants: dict[str, np.ndarray] = {}

                    def _deb(p):
                        return debounce(p, start=test_n.start, group=rest_group,
                                        n=n_deb, stride_samples=cfg.stride_samples,
                                        rest_index=rest_i)

                    if n_deb > 1:
                        variants["debounce"] = _deb(pred)

                    if cfg.two_stage:
                        tau = cfg.gate_thresholds.get(condition, cfg.gate.threshold)
                        gate = fit_gate(res.model, calib_n, device, cfg.gate)
                        if not gate.degenerate:
                            p_move = gate_scores(res.model, test_n, device, gate)
                            mv = _argmax_excluding_rest(proba, rest_i)
                            variants["two_stage"] = apply_threshold(p_move, mv, tau, rest_i)
                            if n_deb > 1:
                                variants["two_stage+debounce"] = _deb(variants["two_stage"])
                        elif verbose:
                            print(f"      two-stage skipped: {gate.describe()}", flush=True)

                    if p_move_classic is not None:
                        ctau = cfg.classic_gate_thresholds.get(
                            condition, cfg.classic_gate_cfg.threshold)
                        mv = _argmax_excluding_rest(proba, rest_i)
                        variants["classic_gate"] = apply_threshold(
                            p_move_classic, mv, ctau, rest_i)
                        if n_deb > 1:
                            variants["classic_gate+debounce"] = _deb(variants["classic_gate"])

                    for name, vpred in variants.items():
                        vsc = score(test_n.y, vpred, test_n.class_names)
                        vrm = rest_metrics(test_n.y, vpred, start=test_n.start,
                                           group=rest_group,
                                           stride_samples=cfg.stride_samples,
                                           fs=test_n.fs)
                        extra = {}
                        if "two_stage" in name:
                            extra["gate_threshold"] = float(
                                cfg.gate_thresholds.get(condition, cfg.gate.threshold))
                        if "classic_gate" in name:
                            extra["gate_threshold"] = float(
                                cfg.classic_gate_thresholds.get(
                                    condition, cfg.classic_gate_cfg.threshold))
                        if "debounce" in name:
                            extra["debounce_n"] = n_deb
                        rows.append({**base_row, "stage": name, **extra,
                                     **vsc.as_row(), **vrm.as_row()})
                        if verbose:
                            print(f"      {name:<22} bal {vsc.balanced_accuracy:.3f} "
                                  f"F1 {vsc.macro_f1:.3f}  "
                                  f"FA/min {vrm.false_activations_per_min:5.1f}  "
                                  f"burst {vrm.mean_false_burst_ms:6.1f}ms  "
                                  f"(single bal {sc.balanced_accuracy:.3f} "
                                  f"FA/min {rm.false_activations_per_min:.1f})",
                                  flush=True)

                    if cfg.recency_split:
                        # Same predictions, regrouped by how long after calibration
                        # each repetition was recorded. A gap between `early` and
                        # `late` is within-session drift.
                        for name, mask in recency_partition(
                            test_n, split.calib_reps, cfg.recency_bins
                        ):
                            if not mask.any():
                                continue
                            rsc = score(
                                test_n.y[mask],
                                proba[mask].argmax(axis=1),
                                test_n.class_names,
                            )
                            rows.append({
                                **base_row,
                                "stage": "single",
                                "evaluation": f"clean_{name}",
                                "n_windows": int(mask.sum()),
                                **rsc.as_row(),
                            })

                    if cfg.rejection:
                        for r in rejection_curve(test_n.y, proba):
                            curves.append({**{k: base_row[k] for k in
                                              ("seed", "subject", "shots", "condition")},
                                           "curve": "rejection", **r})

                    run_failures = (not cfg.failures_at_max_shots_only) or shots == max(cfg.shots)
                    if run_failures:
                        for n_failed in cfg.failure_counts:
                            vals = []
                            for combo, degraded in channel_failure_sweep(
                                test_n,
                                n_failed,
                                kind=cfg.failure_kind,
                                max_combinations=cfg.failure_max_combinations,
                                seed=seed,
                            ):
                                dsc = score(
                                    degraded.y,
                                    res.predict(degraded, device),
                                    degraded.class_names,
                                )
                                vals.append((combo, dsc))
                            if not vals:
                                continue
                            bal = np.array([v[1].balanced_accuracy for v in vals])
                            worst = int(bal.argmin())
                            rows.append({
                                **base_row,
                                "evaluation": f"{cfg.failure_kind}_{n_failed}ch",
                                "balanced_accuracy": float(bal.mean()),
                                "balanced_accuracy_worst": float(bal.min()),
                                "worst_channels": ",".join(map(str, vals[worst][0])),
                                "macro_f1": float(np.mean([v[1].macro_f1 for v in vals])),
                                "accuracy": float(np.mean([v[1].accuracy for v in vals])),
                                "n_combinations": len(vals),
                            })
                            if verbose:
                                print(
                                    f"      {cfg.failure_kind} {n_failed}ch: "
                                    f"mean {bal.mean():.3f}  worst {bal.min():.3f} "
                                    f"(channels {vals[worst][0]})",
                                    flush=True,
                                )
    return rows, curves


def to_dataframe(rows: list[dict]):
    import pandas as pd

    return pd.DataFrame(rows)


def headline(rows: list[dict]) -> str:
    """Compact summary: mean +/- between-subject SD of balanced accuracy."""
    import pandas as pd

    df = pd.DataFrame(rows)
    clean = df[df["evaluation"] == "clean"]
    if clean.empty:
        return "(no clean-evaluation rows)"
    g = clean.groupby(["shots", "condition"])["balanced_accuracy"]
    out = ["shots  condition        bal_acc (mean +/- SD across subjects)"]
    for (shots, cond), vals in g:
        out.append(f"{shots:>5}  {cond:<15} {vals.mean():.3f} +/- {vals.std():.3f}  (n={len(vals)})")
    return "\n".join(out)


def drift_summary(rows: list[dict]) -> str:
    """The cross-repetition proxy: accuracy near calibration vs far from it.

    A negative `drift` means the model is worse on the repetitions recorded
    furthest after calibration -- the calibration going stale within a single
    session. This is a lower bound on the cross-session problem, not an answer
    to it; the electrodes were never re-donned.
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    if "evaluation" not in df or df.empty:
        return "(no rows)"
    early = df[df["evaluation"] == "clean_early"]
    late = df[df["evaluation"] == "clean_late"]
    if early.empty or late.empty:
        return "(no recency rows -- too few held-out repetitions, or recency_split off)"

    keys = ["shots", "condition"]
    merged = early.merge(late, on=keys + ["seed", "subject"], suffixes=("_early", "_late"))
    merged["drift"] = merged["balanced_accuracy_late"] - merged["balanced_accuracy_early"]

    out = ["shots  condition        early    late    drift (late - early)"]
    for (shots, cond), grp in merged.groupby(keys):
        out.append(
            f"{shots:>5}  {cond:<15} {grp['balanced_accuracy_early'].mean():.3f}  "
            f"{grp['balanced_accuracy_late'].mean():.3f}  "
            f"{grp['drift'].mean():+.3f} +/- {grp['drift'].std():.3f}  (n={len(grp)})"
        )
    return "\n".join(out)
