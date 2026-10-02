"""The control that was missing: td_rf given the same few calibration repetitions.

Every comparison so far has been unfair to one side or the other. `td_rf` looked
excellent in the literature benchmark, but that gave it **four** repetitions of the
subject's own data; the few-shot conditions get one, two or three. So the honest
question -- is a shallow model on hand-crafted features actually better than a
pretrained network *in the regime this project is about* -- has never been asked.

This asks it. Identical subjects, identical calibration repetitions, identical test
repetitions, identical normalization, identical windows. The only difference is the
model: 300 trees on MAV/RMS/WL/ZC/SSC against a DB2-pretrained network fine-tuned on
the same windows.

Two things make this cheap. It needs no pretraining, because there is nothing to
pretrain -- which is itself part of the comparison. And it needs no seeds: the
repetition split is deterministic and the forest's random_state is fixed, so the
result is seed-invariant by construction rather than averaged over noise. The
numbers it writes are therefore directly comparable to the LOSO study's per-subject
rows at the same shot count.

    uv run python scripts/classic_fewshot.py --tag classic_fewshot
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from remg.data import PreprocessConfig, WindowConfig, fit_normalizer, preprocess, segment
from remg.data.cohort import describe, excluded_subjects
from remg.data.movements import DEFAULT_SUBSET
from remg.data.ninapro import load_cohort
from remg.data.splits import calibration_split, subjects
from remg.evaluate.metrics import rest_metrics, score
from remg.train.classic_gate import ClassicGateConfig, fit_td_rf, predict_td_rf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-root", type=Path, default=Path("data/raw/DB3"))
    ap.add_argument("--target", default="DB3")
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--target-fs", type=int, default=1000)
    ap.add_argument("--trees", type=int, default=ClassicGateConfig().n_estimators)
    ap.add_argument("--class-balanced", action="store_true",
                    help="balance the forest's classes (off by default, which matches "
                         "the literature benchmark's td_rf and keeps the rest prior)")
    ap.add_argument("--include-excluded", action="store_true")
    ap.add_argument("--tag", default="classic_fewshot")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    pcfg = PreprocessConfig(notch_hz=None, target_fs=args.target_fs)
    wcfg = WindowConfig()
    stride = int(round(wcfg.stride_ms * pcfg.target_fs / 1000))

    pre_excluded: list[int] = []
    if not args.include_excluded and excluded_subjects(args.target):
        print(describe(args.target) + "\n", flush=True)
        pre_excluded = excluded_subjects(args.target)

    print(f"loading {args.target} from {args.target_root} ...", flush=True)
    recs = load_cohort(args.target_root, args.target, subjects=None,
                       expected_channels=12, verbose=True)
    for i in range(len(recs)):
        recs[i] = preprocess(recs[i], pcfg)
    target = segment(recs, cfg=wcfg)
    recs.clear(); del recs; gc.collect()
    if pre_excluded:
        target = target.select(~np.isin(target.subject, pre_excluded))
    print(f"\n{len(target):,} windows, {target.n_classes} classes, "
          f"subjects {subjects(target)}\nDB2 is never loaded: there is nothing to "
          f"pretrain, which is part of the comparison.\n", flush=True)

    rows: list[dict] = []
    t0 = time.time()
    for subj in subjects(target):
        for shots in args.shots:
            try:
                split = calibration_split(target, subj, shots=shots)
            except ValueError as exc:
                print(f"  skip S{subj} shots={shots}: {exc}", flush=True)
                continue
            stats = fit_normalizer(split.calib, "calib")
            calib_n, test_n = stats.apply(split.calib), stats.apply(split.test)
            model = fit_td_rf(calib_n, n_estimators=args.trees,
                              class_balanced=args.class_balanced, binary=False)
            if model.degenerate:
                print(f"  skip S{subj} shots={shots}: {model.describe()}", flush=True)
                continue
            pred = predict_td_rf(test_n, model)
            sc = score(test_n.y, pred, test_n.class_names)
            group = (test_n.subject.astype(np.int64) * 1_000_000
                     + test_n.session.astype(np.int64) * 1000
                     + test_n.rep.astype(np.int64))
            rm = rest_metrics(test_n.y, pred, start=test_n.start, group=group,
                              stride_samples=stride, fs=test_n.fs)
            rows.append({
                "subject": subj, "shots": shots, "condition": "td_rf_fewshot",
                "stage": "single", "evaluation": "clean",
                "calib_reps": ",".join(map(str, split.calib_reps)),
                "calib_windows": len(split.calib), "fit_seconds": model.seconds,
                **sc.as_row(), **rm.as_row(),
            })
            print(f"  S{subj:<3} shots={shots} reps {split.calib_reps}  "
                  f"bal {sc.balanced_accuracy:.3f}  F1 {sc.macro_f1:.3f}  "
                  f"FA/min {rm.false_activations_per_min:6.2f}  "
                  f"burst {rm.mean_false_burst_ms:6.1f}ms  ({model.seconds:.1f}s)",
                  flush=True)

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / f"{args.tag}_rows.csv", index=False)

    meta = {
        "purpose": ("td_rf given exactly the calibration repetitions the few-shot "
                    "conditions get, so the shallow and deep models are compared in "
                    "the same regime"),
        "target": args.target,
        "target_subjects": subjects(target),
        "excluded_target_subjects": sorted(pre_excluded),
        "documented_exclusions": {f"{args.target}-S{s}": describe(args.target)
                                  for s in pre_excluded},
        "classes": list(target.class_names),
        "movement_ids": sorted(DEFAULT_SUBSET),
        "shots": args.shots,
        "model": (f"StandardScaler + RandomForestClassifier(n_estimators={args.trees}, "
                  f"random_state=0), multiclass, "
                  f"class_weight={'balanced' if args.class_balanced else None}"),
        "features": {"set": ["MAV", "RMS", "WL", "ZC", "SSC"], "per_channel": True,
                     "zc_ssc_threshold": "0.01 x per-channel RMS of the CALIBRATION windows",
                     "scaler": "StandardScaler fitted on calibration windows only"},
        "seeds": ("none -- the repetition split is deterministic and the forest's "
                  "random_state is fixed, so this result is seed-invariant by "
                  "construction"),
        "pretraining": "none; DB2 is not loaded",
        "window": {"length_ms": wcfg.length_ms, "stride_ms": wcfg.stride_ms},
        "preprocess": {"notch_hz": pcfg.notch_hz, "target_fs": pcfg.target_fs},
        "normalization": "fitted on the calibration windows only, as in the LOSO study",
        "no_tuning_on_test": ("tree count and random_state fixed a priori from the "
                              "literature benchmark; no search of any kind"),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (args.out / f"{args.tag}_meta.json").write_text(json.dumps(meta, indent=2))

    if not df.empty:
        print("\nmean over subjects:")
        print(df.groupby("shots")[["balanced_accuracy", "macro_f1",
                                   "false_activations_per_min",
                                   "mean_false_burst_ms"]].mean().round(3).to_string())
    print(f"\n{len(rows)} rows -> {args.out / f'{args.tag}_rows.csv'}"
          f"   ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
