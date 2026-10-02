"""The real leave-one-subject-out personalization study: DB2 -> DB3.

`smoke.py` runs the same pipeline on synthetic data to prove it connects. This
runs it on the downloaded recordings and writes a tidy table to `results/`, one
row per (seed, subject, shots, condition, evaluation), so every figure in the
writeup is a group-by on one file.

    uv run python scripts/run_experiment.py                      # everything present
    uv run python scripts/run_experiment.py --source-subjects 1-10
    uv run python scripts/run_experiment.py --tag pilot --shots 1 3

Run `inspect_data.py` on both cohorts first. This script refuses to start if a
target subject is missing one of the movements in DEFAULT_SUBSET, because that
subject has no calibration example of it and the few-shot conditions assume one
-- see `--drop-incomplete` for the alternative.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

from remg.data import PreprocessConfig, WindowConfig, preprocess, segment
from remg.data.cohort import describe, excluded_subjects
from remg.data.movements import DEFAULT_SUBSET
from remg.data.ninapro import load_cohort
from remg.experiments import ExperimentConfig, drift_summary, headline, run, to_dataframe
from remg.train import AdaptConfig, PretrainConfig


def parse_subjects(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, _, hi = part.partition("-")
            out.update(range(int(lo), int(hi) + 1))
        elif part:
            out.add(int(part))
    return sorted(out)


def build(root: Path, dataset: str, subjects, cfg: PreprocessConfig, wcfg: WindowConfig,
          expected_channels: int | None):
    print(f"loading {dataset} from {root} ...", flush=True)
    recs = load_cohort(root, dataset, subjects=subjects,
                       expected_channels=expected_channels, verbose=True)
    t0 = time.time()
    # In place, so the raw and filtered copies of a ~2 GB cohort are never both
    # alive. Holding them together paged a 16 GB machine into swap during the
    # budget search and slowed it by more than an order of magnitude.
    n_recs = len(recs)
    for i in range(n_recs):
        recs[i] = preprocess(recs[i], cfg)
    print(f"  preprocessed {n_recs} subject(s) in {time.time() - t0:.0f}s", flush=True)
    ws = segment(recs, cfg=wcfg)
    recs.clear()
    gc.collect()
    print(f"  {ws.X.shape[0]:,} windows, {ws.X.shape[1]} channels, {ws.n_classes} classes",
          flush=True)
    return ws


def report_coverage(ws, name: str) -> list[int]:
    """Print per-subject class coverage and return subjects missing a class."""
    incomplete = []
    print(f"\n{name} class coverage:")
    for subject in sorted(set(int(s) for s in ws.subject)):
        present = set(int(c) for c in np.unique(ws.y[ws.subject == subject]))
        missing = [ws.class_names[i] for i in range(ws.n_classes) if i not in present]
        flag = "" if not missing else f"   MISSING {missing}"
        print(f"  S{subject:<3} {len(present)}/{ws.n_classes} classes{flag}")
        if missing:
            incomplete.append(subject)
    return incomplete


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-root", type=Path, default=Path("data/raw/DB2"))
    ap.add_argument("--target-root", type=Path, default=Path("data/raw/DB3"))
    ap.add_argument("--source", default="DB2")
    ap.add_argument("--target", default="DB3")
    ap.add_argument("--source-subjects", help='e.g. "1-10"')
    ap.add_argument("--target-subjects", help='e.g. "2,3,5"')
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=3000, help="pretraining steps")
    ap.add_argument("--target-fs", type=int, default=1000)
    ap.add_argument("--no-notch", action="store_true",
                    help="skip the 50 Hz notch (16x faster; the Delsys hardware "
                         "already band-passes, so this is often a no-op)")
    ap.add_argument("--drop-incomplete", action="store_true",
                    help="also exclude target subjects missing a subset movement that are "
                         "not already listed in remg.data.cohort")
    ap.add_argument("--include-excluded", action="store_true",
                    help="ignore the documented exclusions in remg.data.cohort and keep "
                         "every subject (for checking what an exclusion cost)")
    ap.add_argument("--no-failures", action="store_true",
                    help="skip the sensor-failure sweep (it dominates runtime)")
    ap.add_argument("--no-rejection", action="store_true",
                    help="skip the rejection curves")
    ap.add_argument("--unbalanced-calibration", action="store_true",
                    help="train the gradient baselines on unbalanced calibration windows "
                         "(the old, unfair behaviour -- for measuring what balancing changed)")
    ap.add_argument("--budget", type=Path,
                    help="JSON written by scripts/tune_budget.py holding the adaptation "
                         "step counts and learning rates selected on held-out SOURCE "
                         "subjects. Without it the hand-set defaults in AdaptConfig are "
                         "used, which were never selected by any procedure.")
    ap.add_argument("--gate", type=Path,
                    help="JSON written by scripts/tune_gate.py holding the rest-gate "
                         "threshold per condition, selected on held-out SOURCE subjects. "
                         "Without it the gate runs at 0.5, which is an unchosen default.")
    ap.add_argument("--no-two-stage", action="store_true",
                    help="skip the rest-gate evaluation")
    ap.add_argument("--rules", type=Path,
                    help="JSON written by scripts/tune_rules.py holding the debounce "
                         "length N and the classic-gate threshold per condition, both "
                         "selected on held-out SOURCE subjects.")
    ap.add_argument("--no-classic-gate", action="store_true",
                    help="skip the td_rf classic rest gate")
    ap.add_argument("--debounce-n", type=int,
                    help="Override the debounce length from --rules for every condition. "
                         "Use only to report a labelled operating point that the "
                         "pre-stated selection goal did not choose -- the override is "
                         "recorded in the meta file so a reader can tell the difference "
                         "between a selected value and a demonstration.")
    ap.add_argument("--tag", default="run", help="prefix for the output files")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    pcfg = PreprocessConfig(notch_hz=None if args.no_notch else 50.0, target_fs=args.target_fs)
    wcfg = WindowConfig()

    # Documented exclusions are applied by default and announced, so no run is
    # quietly missing a subject that another run had.
    pre_excluded: list[int] = []
    if not args.include_excluded:
        for dataset in (args.source, args.target):
            if excluded_subjects(dataset):
                print(describe(dataset) + "\n", flush=True)
        pre_excluded = excluded_subjects(args.target)

    source = build(args.source_root, args.source, parse_subjects(args.source_subjects),
                   pcfg, wcfg, expected_channels=12)
    target = build(args.target_root, args.target, parse_subjects(args.target_subjects),
                   pcfg, wcfg, expected_channels=12)

    if pre_excluded:
        keep = ~np.isin(target.subject, pre_excluded)
        if keep.any():
            target = target.select(keep)

    report_coverage(source, args.source)
    incomplete = report_coverage(target, args.target)
    if incomplete:
        if not args.drop_incomplete:
            raise SystemExit(
                f"\ntarget subjects {incomplete} are missing at least one movement in "
                f"DEFAULT_SUBSET. A subject with no calibration example of a class cannot "
                f"be personalized to it.\nEither re-run with --drop-incomplete to exclude "
                f"them (and say so in the writeup), or revise DEFAULT_SUBSET in "
                f"remg/data/movements.py to movements the whole cohort performed."
            )
        keep = np.isin(target.subject, [s for s in set(int(x) for x in target.subject)
                                        if s not in incomplete])
        target = target.select(keep)
        print(f"\nexcluded target subjects {incomplete} (--drop-incomplete)")

    # The adaptation budget, if it was selected rather than guessed. Only the
    # keys AdaptConfig actually has are accepted, so a stale or hand-edited file
    # fails loudly here instead of being silently ignored.
    budget: dict = {}
    if args.budget:
        raw = json.loads(args.budget.read_text())
        allowed = set(AdaptConfig().__dict__)
        for cond, entry in raw.items():
            for k, v in entry.items():
                if k in allowed:
                    budget[k] = v
        unknown = [k for e in raw.values() for k in e
                   if k not in allowed and k not in
                   ("config", "val_balanced_accuracy", "best_in_grid", "tie_band_sem",
                    "mean_adapt_seconds", "per_shot_best_diagnostic")]
        if unknown:
            raise SystemExit(f"--budget file has unrecognized keys: {sorted(set(unknown))}")
        print(f"\nadaptation budget from {args.budget}:")
        for cond, entry in raw.items():
            print(f"  {cond:<13} {entry.get('config', '?')}")
        print("  selected on held-out source subjects; DB3 played no part\n", flush=True)

    gate_thresholds: dict[str, float] = {}
    if args.gate:
        raw = json.loads(args.gate.read_text())
        gate_thresholds = {k: float(v["threshold"]) for k, v in raw.items()
                           if isinstance(v, dict) and "threshold" in v}
        print("rest-gate thresholds from", args.gate)
        for cond, entry in raw.items():
            if isinstance(entry, dict) and "threshold" in entry:
                met = entry.get("budget_met")
                note = "" if met is None else ("" if met else "  (FAR budget not met on DB2)")
                print(f"  {cond:<13} tau={entry['threshold']:.2f}{note}")
        print("  selected on held-out source subjects; DB3 played no part\n", flush=True)

    debounce_n: dict[str, int] = {}
    classic_tau: dict[str, float] = {}
    if args.rules:
        raw = json.loads(args.rules.read_text())
        for cond, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            if "debounce_n" in entry:
                debounce_n[cond] = int(entry["debounce_n"])
            if isinstance(entry.get("classic_gate"), dict):
                classic_tau[cond] = float(entry["classic_gate"]["threshold"])
        print("decision rules from", args.rules)
        for cond in sorted(set(debounce_n) | set(classic_tau)):
            n = debounce_n.get(cond, 1)
            lat = (n - 1) * wcfg.stride_ms
            print(f"  {cond:<13} debounce N={n} (+{lat:.0f} ms onset latency)"
                  f"   classic gate tau={classic_tau.get(cond, float('nan')):.2f}")
        print("  selected on held-out source subjects; DB3 played no part", flush=True)
    if args.debounce_n:
        for cond in ("none", "linear_probe", "finetune", "rapid"):
            debounce_n[cond] = args.debounce_n
        print(f"  debounce N OVERRIDDEN to {args.debounce_n} for every condition "
              f"(+{(args.debounce_n - 1) * wcfg.stride_ms:.0f} ms latency) -- a labelled "
              f"operating point, not the one the selection goal chose", flush=True)
    print("", flush=True)

    cfg = ExperimentConfig(
        shots=tuple(args.shots),
        seeds=tuple(args.seeds),
        pretrain=PretrainConfig(steps=args.steps),
        adapt=AdaptConfig(class_balanced=not args.unbalanced_calibration, **budget),
        failure_counts=() if args.no_failures else (1, 2),
        rejection=not args.no_rejection,
        two_stage=not args.no_two_stage,
        gate_thresholds=gate_thresholds,
        debounce_n=debounce_n,
        classic_gate=not args.no_classic_gate,
        classic_gate_thresholds=classic_tau,
        stride_samples=int(round(wcfg.stride_ms * pcfg.target_fs / 1000)),
    )

    t0 = time.time()
    rows, curves = run(source, target, cfg)
    elapsed = time.time() - t0

    args.out.mkdir(parents=True, exist_ok=True)
    rows_path = args.out / f"{args.tag}_rows.csv"
    to_dataframe(rows).to_csv(rows_path, index=False)
    if curves:
        to_dataframe(curves).to_csv(args.out / f"{args.tag}_curves.csv", index=False)

    meta = {
        "source": args.source, "target": args.target,
        "source_subjects": sorted(set(int(s) for s in source.subject)),
        "target_subjects": sorted(set(int(s) for s in target.subject)),
        "excluded_target_subjects": sorted(set(pre_excluded) | set(incomplete)),
        "documented_exclusions": {
            f"{args.target}-S{s_}": describe(args.target) for s_ in pre_excluded
        },
        "classes": list(source.class_names),
        "movement_ids": sorted(DEFAULT_SUBSET),
        "shots": args.shots, "seeds": args.seeds, "pretrain_steps": args.steps,
        "preprocess": {"notch_hz": pcfg.notch_hz, "target_fs": pcfg.target_fs},
        "calibration_class_balanced": cfg.adapt.class_balanced,
        "adapt_budget_source": str(args.budget) if args.budget else "AdaptConfig defaults (not selected)",
        "adapt_config": {k: v for k, v in cfg.adapt.__dict__.items()},
        "window": {"length_ms": wcfg.length_ms, "stride_ms": wcfg.stride_ms},
        "two_stage": cfg.two_stage,
        "gate_threshold_source": str(args.gate) if args.gate
            else ("GateConfig default 0.5 (not selected)" if cfg.two_stage else None),
        "gate_thresholds": gate_thresholds,
        "rules_source": str(args.rules) if args.rules else None,
        "debounce_n": debounce_n,
        "debounce_n_overridden": bool(args.debounce_n),
        "debounce_n_selected_on_source": (
            {k: int(v["debounce_n"]) for k, v in json.loads(args.rules.read_text()).items()
             if isinstance(v, dict) and "debounce_n" in v} if args.rules else {}),
        "debounce_added_latency_ms": {k: (v - 1) * wcfg.stride_ms
                                      for k, v in debounce_n.items()},
        "classic_gate": cfg.classic_gate,
        "classic_gate_thresholds": classic_tau,
        "elapsed_seconds": round(elapsed, 1),
    }
    (args.out / f"{args.tag}_meta.json").write_text(json.dumps(meta, indent=2))

    print("\n" + headline(rows))
    print("\ncross-repetition session proxy (within-session drift):")
    print(drift_summary(rows))
    print(f"\n{len(rows)} rows -> {rows_path}   ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
