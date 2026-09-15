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
    recs = [preprocess(r, cfg) for r in recs]
    print(f"  preprocessed {len(recs)} subject(s) in {time.time() - t0:.0f}s", flush=True)
    ws = segment(recs, cfg=wcfg)
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

    cfg = ExperimentConfig(
        shots=tuple(args.shots),
        seeds=tuple(args.seeds),
        pretrain=PretrainConfig(steps=args.steps),
        adapt=AdaptConfig(class_balanced=not args.unbalanced_calibration),
        failure_counts=() if args.no_failures else (1, 2),
        rejection=not args.no_rejection,
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
        "window": {"length_ms": wcfg.length_ms, "stride_ms": wcfg.stride_ms},
        "elapsed_seconds": round(elapsed, 1),
    }
    (args.out / f"{args.tag}_meta.json").write_text(json.dumps(meta, indent=2))

    print("\n" + headline(rows))
    print("\ncross-repetition session proxy (within-session drift):")
    print(drift_summary(rows))
    print(f"\n{len(rows)} rows -> {rows_path}   ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
