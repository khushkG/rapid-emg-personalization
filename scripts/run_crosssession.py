"""The cross-session study: calibrate on day 1, control the hand on day 5.

This is the one scenario in which the proposed method has a case that the rest of
the project has not already closed. Every earlier result is within-session: the
electrodes were never removed. Fine-tuning 109,317 parameters to one session's exact
electrode placement is precisely what should overfit when the sleeve is re-donned,
and 664 adapter parameters is precisely the kind of constraint that should survive
it. DB6 is the dataset that can answer it -- 10 intact subjects, 5 days, 7 grasps.

DB6 is not DB2 with different numbers, so this is a self-contained study rather than
an extension of the first one: 14 electrodes in a different placement, a different
movement set, intact subjects only. A DB2-pretrained encoder cannot even be loaded
(12 input channels against 14), so pretraining happens **within DB6**, leave one
subject out.

The protocol:

  * Pretrain on the other nine DB6 subjects. The held-out subject contributes
    nothing -- not one window.
  * Calibrate on repetitions of **day 1** only.
  * Evaluate twice from the same calibration: on the held-out repetitions of day 1,
    and on **day 5**. The first is the within-session number, the second is the
    cross-session number, and the difference between them is the quantity the whole
    study is about. Reporting day 5 alone would not distinguish a method that retains
    well from one that was never good.
  * The normalizer is fitted on **day-1 calibration windows only**. This is the crux
    of the design: re-fitting it on day 5 would quietly correct the amplitude shift
    that re-donning causes, which is most of what is being measured.
  * Hyperparameters come from the files selected on held-out DB2 subjects. Nothing is
    tuned on DB6.

Success criterion, fixed before the data existed. `rapid` wins only if BOTH hold:

  1. its day-1 -> day-5 drop in balanced accuracy is smaller than `finetune`'s by
     more than the seed noise measured in this study, with a bootstrap 95% CI on the
     paired difference excluding zero; and
  2. its day-5 balanced accuracy is not significantly worse than `finetune`'s
     (Wilcoxon p > 0.05, CI containing zero).

Condition 2 is there because "degrades less" is trivially satisfied by a model that
starts bad and stays bad -- `none` would pass condition 1 alone. `rapid` already lost
this way once, on the sensor-failure sweep: it met its robustness threshold and still
lost, because `finetune` never surrendered the lead.

    uv run python scripts/run_crosssession.py --budget results/budget_chosen.json \\
        --gate results/gate_chosen.json --rules results/rules_chosen.json
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
from remg.data.cohort import partial_recordings
from remg.data.movements import DB6_MOVEMENT_IDS, DB6_SUBSET
from remg.data.ninapro import find_files, load_subject
from remg.data.normalize import normalize_per_subject
from remg.data.splits import calibration_split, subjects
from remg.evaluate.metrics import rest_metrics, score
from remg.train import AdaptConfig, PretrainConfig
from remg.train.adapt import adapt, predict_proba
from remg.train.classic_gate import (
    ClassicGateConfig,
    classic_gate_scores,
    fit_classic_gate,
    fit_td_rf,
    predict_td_rf,
)
from remg.train.pretrain import pretrain
from remg.train.twostage import _argmax_excluding_rest, apply_threshold
from remg.utils import pick_device

NEURAL_CONDITIONS = ("none", "linear_probe", "finetune", "rapid")


def load_selected(path: Path | None, keys: set[str]) -> dict:
    if not path:
        return {}
    raw = json.loads(path.read_text())
    return {k: v for e in raw.values() if isinstance(e, dict)
            for k, v in e.items() if k in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/raw/DB6"))
    ap.add_argument("--dataset", default="DB6")
    ap.add_argument("--subjects", type=int, nargs="+", default=None)
    ap.add_argument("--calib-day", type=int, default=1)
    ap.add_argument("--test-day", type=int, default=5)
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--expected-channels", type=int, default=14)
    ap.add_argument("--budget", type=Path)
    ap.add_argument("--gate", type=Path)
    ap.add_argument("--rules", type=Path)
    ap.add_argument("--no-classic-gate", action="store_true")
    ap.add_argument("--no-td-rf", action="store_true")
    ap.add_argument("--tag", default="crosssession")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    device = pick_device(None)
    pcfg = PreprocessConfig(notch_hz=None, target_fs=1000)
    wcfg = WindowConfig()
    stride = int(round(wcfg.stride_ms * pcfg.target_fs / 1000))
    calib_sess, test_sess = args.calib_day - 1, args.test_day - 1

    budget = load_selected(args.budget, set(AdaptConfig().__dict__))
    gate_tau = {k: float(v["threshold"]) for k, v in
                (json.loads(args.gate.read_text()).items() if args.gate else [])
                if isinstance(v, dict) and "threshold" in v}
    classic_tau: dict[str, float] = {}
    if args.rules:
        for cond, entry in json.loads(args.rules.read_text()).items():
            if isinstance(entry, dict) and isinstance(entry.get("classic_gate"), dict):
                classic_tau[cond] = float(entry["classic_gate"]["threshold"])

    print(f"adaptation budget: {budget or 'AdaptConfig defaults'}")
    print(f"logistic gate tau: {gate_tau or 'default'}")
    print(f"classic gate tau:  {classic_tau or 'default'}")
    print("all selected on held-out DB2 subjects; nothing tuned on DB6\n", flush=True)

    for f, why in partial_recordings(args.dataset).items():
        print(f"note: {f} is a known-partial recording -- {why[:110]}...", flush=True)

    # Only the two days the study uses are loaded. Loading all five would cost ~8 GB
    # of windows for data no condition ever sees.
    keys = sorted(find_files(args.root, args.dataset))
    want = sorted({s for s, _ in keys}) if args.subjects is None else sorted(args.subjects)
    print(f"\nloading {args.dataset} days {args.calib_day} and {args.test_day} "
          f"for subjects {want} ...", flush=True)
    recs = []
    for subj in want:
        for sess in (calib_sess, test_sess):
            if (subj, sess) not in keys:
                print(f"  S{subj} day{sess + 1}: MISSING", flush=True)
                continue
            rec = load_subject(args.root, args.dataset, subj, sess,
                               expected_channels=args.expected_channels)
            recs.append(preprocess(rec, pcfg))
            print(f"  S{subj} day{sess + 1}: {rec.duration_s:.0f}s "
                  f"{rec.n_channels}ch  dropped {rec.meta['dropped_channels']}", flush=True)
    ws = segment(recs, cfg=wcfg, subset=DB6_SUBSET)
    recs.clear(); del recs; gc.collect()
    print(f"\n{ws.X.shape[0]:,} windows, {ws.X.shape[1]} channels, {ws.n_classes} classes",
          flush=True)

    cohort = subjects(ws)
    for s in cohort:
        for sess in (calib_sess, test_sess):
            m = (ws.subject == s) & (ws.session == sess)
            present = sorted({int(c) for c in np.unique(ws.y[m])}) if m.any() else []
            missing = [ws.class_names[i] for i in range(ws.n_classes) if i not in present]
            flag = "" if not missing else f"   MISSING {missing}"
            print(f"  S{s} day{sess + 1}: {int(m.sum()):>6} windows, "
                  f"{len(present)}/{ws.n_classes} classes{flag}")

    rows: list[dict] = []
    rows_path = args.out / f"{args.tag}_rows.csv"
    args.out.mkdir(parents=True, exist_ok=True)
    if rows_path.exists():
        rows_path.unlink()
    wrote_header = False

    def flush(batch: list[dict]) -> None:
        nonlocal wrote_header
        if batch:
            pd.DataFrame(batch).to_csv(rows_path, mode="a", index=False,
                                       header=not wrote_header)
            wrote_header = True

    t_start = time.time()
    for seed in args.seeds:
        for held in cohort:
            # select() copies, so the source cohort can be normalized in place --
            # a second 2.8 GB array here is what pages this machine.
            src = ws.select(ws.subject != held)
            src_n = normalize_per_subject(src, inplace=True)
            print(f"\n[seed {seed}] hold out S{held}: pretraining on "
                  f"{len(subjects(src_n))} subjects ({len(src_n):,} windows)", flush=True)
            base, _ = pretrain(src_n, val=None,
                               cfg=PretrainConfig(steps=args.steps, seed=seed),
                               verbose=True)
            del src, src_n
            gc.collect()

            tgt = ws.select(ws.subject == held)
            for shots in args.shots:
                try:
                    sp_same = calibration_split(tgt, held, shots=shots, session=calib_sess)
                    sp_cross = calibration_split(tgt, held, shots=shots,
                                                 session=calib_sess, test_session=test_sess)
                except ValueError as exc:
                    print(f"  skip S{held} shots={shots}: {exc}", flush=True)
                    continue
                assert sp_same.calib_reps == sp_cross.calib_reps, (
                    "the two evaluations must share one calibration set"
                )
                stats = fit_normalizer(sp_same.calib, "calib")
                calib_n = stats.apply(sp_same.calib)
                tests = {"day1": stats.apply(sp_same.test),
                         "day5": stats.apply(sp_cross.test)}

                cgate = fit_classic_gate(calib_n, ClassicGateConfig()) \
                    if not args.no_classic_gate else None
                td = fit_td_rf(calib_n, n_estimators=ClassicGateConfig().n_estimators,
                               binary=False) if not args.no_td_rf else None

                def emit(cond: str, stage: str, which: str, pred, test_ws, extra=None):
                    sc = score(test_ws.y, pred, test_ws.class_names)
                    grp = (test_ws.subject.astype(np.int64) * 1_000_000
                           + test_ws.session.astype(np.int64) * 1000
                           + test_ws.rep.astype(np.int64))
                    rm = rest_metrics(test_ws.y, pred, start=test_ws.start, group=grp,
                                      stride_samples=stride, fs=test_ws.fs)
                    rows.append({"seed": seed, "subject": held, "shots": shots,
                                 "condition": cond, "stage": stage, "evaluation": which,
                                 "calib_reps": ",".join(map(str, sp_same.calib_reps)),
                                 "calib_windows": len(calib_n),
                                 **(extra or {}), **sc.as_row(), **rm.as_row()})
                    return sc

                before = len(rows)
                for cond in NEURAL_CONDITIONS:
                    acfg = AdaptConfig(**{**AdaptConfig().__dict__, **budget, "seed": seed})
                    res = adapt(base, calib_n, cond, device, acfg, sp_same.calib_reps)
                    got = {}
                    for which, tws in tests.items():
                        proba = predict_proba(res.model, tws, device, mode=res.mode)
                        got[which] = emit(cond, "single", which, proba.argmax(axis=1), tws)
                        if cond == "finetune" and cgate is not None and not cgate.degenerate:
                            pm = classic_gate_scores(tws, cgate)
                            mv = _argmax_excluding_rest(proba, 0)
                            tau = classic_tau.get("finetune", ClassicGateConfig().threshold)
                            emit(cond, "classic_gate", which,
                                 apply_threshold(pm, mv, tau), tws,
                                 {"gate_threshold": tau})
                    drop = got["day1"].balanced_accuracy - got["day5"].balanced_accuracy
                    print(f"  S{held:<3} shots={shots} {cond:<13} "
                          f"day1 {got['day1'].balanced_accuracy:.3f} -> "
                          f"day5 {got['day5'].balanced_accuracy:.3f}  "
                          f"drop {drop:+.3f}", flush=True)

                if td is not None and not td.degenerate:
                    got = {w: emit("td_rf_fewshot", "single", w, predict_td_rf(t, td), t)
                           for w, t in tests.items()}
                    print(f"  S{held:<3} shots={shots} {'td_rf_fewshot':<13} "
                          f"day1 {got['day1'].balanced_accuracy:.3f} -> "
                          f"day5 {got['day5'].balanced_accuracy:.3f}  drop "
                          f"{got['day1'].balanced_accuracy - got['day5'].balanced_accuracy:+.3f}",
                          flush=True)
                flush(rows[before:])
            del tgt, base
            gc.collect()

    meta = {
        "study": "cross-session: calibrate on day 1, evaluate on day 5",
        "dataset": args.dataset,
        "protocol": "leave-one-subject-out within DB6; no DB2 pretraining",
        "why_not_db2": ("DB6 has 14 electrodes against DB2's 12, a different placement "
                        "and a different movement set, so DB2 weights cannot be loaded "
                        "let alone transferred"),
        "subjects": cohort,
        "calib_day": args.calib_day, "test_day": args.test_day,
        "days_segmented": [args.calib_day, args.test_day],
        "shots": args.shots, "seeds": args.seeds, "pretrain_steps": args.steps,
        "classes": list(ws.class_names),
        "db6_movement_ids": list(DB6_MOVEMENT_IDS),
        "movement_names_provisional": True,
        "expected_channels": args.expected_channels,
        "window": {"length_ms": wcfg.length_ms, "stride_ms": wcfg.stride_ms},
        "preprocess": {"notch_hz": pcfg.notch_hz, "target_fs": pcfg.target_fs},
        "normalizer": "fitted on day-1 calibration windows only; never re-fitted on day 5",
        "adapt_budget_source": str(args.budget) if args.budget else None,
        "gate_threshold_source": str(args.gate) if args.gate else None,
        "rules_source": str(args.rules) if args.rules else None,
        "hyperparameters_selected_on": "held-out DB2 subjects; nothing tuned on DB6",
        "partial_recordings": partial_recordings(args.dataset),
        "success_criterion": (
            "rapid wins only if BOTH: (1) its day1->day5 balanced-accuracy drop is "
            "smaller than finetune's by more than the seed noise measured in this "
            "study, bootstrap 95% CI on the paired difference excluding zero; and "
            "(2) its day-5 balanced accuracy is not significantly worse than "
            "finetune's, Wilcoxon p > 0.05 with the CI containing zero. Stated "
            "before the data existed."
        ),
        "elapsed_seconds": round(time.time() - t_start, 1),
        "device": str(device),
    }
    (args.out / f"{args.tag}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n{len(rows)} rows -> {rows_path}   ({(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
