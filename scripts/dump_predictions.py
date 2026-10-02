"""Dump per-window predictions for the figures, from the same protocol as the studies.

The charts can be rebuilt from the saved result tables, but the hand animation and the
rest timeline need the actual decoded label *sequence*, which no run saved and which no
kept checkpoint can regenerate -- by standing instruction this project keeps no model
weights. So this reproduces the exact protocol once more and writes the predictions
themselves, which are small, so every figure becomes reproducible from files in the
repository without storing a single weight.

No EMG is written. The output is: window index, repetition, true label, and one
predicted label per method. Nothing here allows a signal to be reconstructed, which
matters because NinaPro publishes a citation requirement and no redistribution
licence.

Two modes, run separately so a 3 GB cohort is never resident alongside a 2 GB one:

    --mode db6   LOSO within DB6; calibrate day 1, predict day 1 held-out and day 5
    --mode db3   DB2-pretrained; calibrate on day-1-equivalent reps, predict held-out

Hyperparameters come from the files selected on held-out DB2 subjects. Nothing is
tuned on anything these predictions are drawn from.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from remg.data import PreprocessConfig, WindowConfig, fit_normalizer, preprocess, segment
from remg.data.cohort import excluded_subjects
from remg.data.movements import DB6_SUBSET
from remg.data.ninapro import find_files, load_cohort, load_subject
from remg.data.normalize import normalize_per_subject
from remg.data.splits import calibration_split, subjects
from remg.evaluate.metrics import score
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

CONDITIONS = ("none", "linear_probe", "finetune", "rapid")


def selected(path: Path | None, keys: set[str]) -> dict:
    if not path:
        return {}
    raw = json.loads(path.read_text())
    return {k: v for e in raw.values() if isinstance(e, dict)
            for k, v in e.items() if k in keys}


def classic_threshold(path: Path | None, cond: str = "finetune") -> float:
    if not path:
        return ClassicGateConfig().threshold
    raw = json.loads(path.read_text())
    e = raw.get(cond, {})
    if isinstance(e.get("classic_gate"), dict):
        return float(e["classic_gate"]["threshold"])
    return ClassicGateConfig().threshold


def predict_all(base, calib_n, tests: dict, device, budget, seed, ctau):
    """Every method's predictions on every test set, from one calibration set."""
    out: dict[str, dict[str, np.ndarray]] = {}
    accs: dict[str, dict[str, float]] = {}
    cg = fit_classic_gate(calib_n, ClassicGateConfig())
    for cond in CONDITIONS:
        acfg = AdaptConfig(**{**AdaptConfig().__dict__, **budget, "seed": seed})
        res = adapt(base, calib_n, cond, device, acfg, ())
        for which, tws in tests.items():
            proba = predict_proba(res.model, tws, device, mode=res.mode)
            out.setdefault(cond, {})[which] = proba.argmax(axis=1)
            accs.setdefault(cond, {})[which] = score(
                tws.y, out[cond][which], tws.class_names).balanced_accuracy
            if cond == "finetune" and not cg.degenerate:
                pm = classic_gate_scores(tws, cg)
                mv = _argmax_excluding_rest(proba, 0)
                pred = apply_threshold(pm, mv, ctau)
                out.setdefault("finetune_classic_gate", {})[which] = pred
                accs.setdefault("finetune_classic_gate", {})[which] = score(
                    tws.y, pred, tws.class_names).balanced_accuracy
    td = fit_td_rf(calib_n, n_estimators=ClassicGateConfig().n_estimators, binary=False)
    if not td.degenerate:
        for which, tws in tests.items():
            pred = predict_td_rf(tws, td)
            out.setdefault("td_rf", {})[which] = pred
            accs.setdefault("td_rf", {})[which] = score(
                tws.y, pred, tws.class_names).balanced_accuracy
    return out, accs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["db6", "db3"])
    ap.add_argument("--db6-root", type=Path, default=Path("data/raw/DB6"))
    ap.add_argument("--db2-root", type=Path, default=Path("data/raw/DB2"))
    ap.add_argument("--db3-root", type=Path, default=Path("data/raw/DB3"))
    ap.add_argument("--targets", type=int, nargs="+", required=True)
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--budget", type=Path, default=Path("results/budget_chosen.json"))
    ap.add_argument("--rules", type=Path, default=Path("results/rules_chosen.json"))
    ap.add_argument("--out", type=Path, default=Path("results/visuals"))
    args = ap.parse_args()

    device = pick_device(None)
    pcfg = PreprocessConfig(notch_hz=None, target_fs=1000)
    wcfg = WindowConfig()
    budget = selected(args.budget, set(AdaptConfig().__dict__))
    ctau = classic_threshold(args.rules)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"budget {budget}\nclassic gate tau {ctau}\nseed {args.seed}, "
          f"{args.shots} shots\n", flush=True)

    frames: list[pd.DataFrame] = []
    summary: list[dict] = []

    if args.mode == "db6":
        keys = sorted(find_files(args.db6_root, "DB6"))
        cohort = sorted({s for s, _ in keys})
        recs = []
        for s in cohort:
            for sess in (0, 4):
                if (s, sess) in keys:
                    recs.append(preprocess(load_subject(args.db6_root, "DB6", s, sess,
                                                        expected_channels=14), pcfg))
        ws = segment(recs, cfg=wcfg, subset=DB6_SUBSET)
        recs.clear(); del recs; gc.collect()
        print(f"DB6: {len(ws):,} windows, {ws.n_classes} classes", flush=True)

        for target in args.targets:
            src = ws.select(ws.subject != target)
            src_n = normalize_per_subject(src, inplace=True)
            print(f"\n[S{target}] pretraining on {len(subjects(src_n))} subjects", flush=True)
            base, _ = pretrain(src_n, val=None,
                               cfg=PretrainConfig(steps=args.steps, seed=args.seed),
                               verbose=True)
            del src, src_n; gc.collect()
            tgt = ws.select(ws.subject == target)
            sp1 = calibration_split(tgt, target, shots=args.shots, session=0)
            sp5 = calibration_split(tgt, target, shots=args.shots, session=0,
                                    test_session=4)
            stats = fit_normalizer(sp1.calib, "calib")
            calib_n = stats.apply(sp1.calib)
            tests = {"day1": stats.apply(sp1.test), "day5": stats.apply(sp5.test)}
            preds, accs = predict_all(base, calib_n, tests, device, budget,
                                      args.seed, ctau)
            for which, tws in tests.items():
                df = pd.DataFrame({"dataset": "DB6", "subject": target, "day": which,
                                   "rep": tws.rep, "start": tws.start, "y_true": tws.y})
                for meth, byday in preds.items():
                    df[f"pred_{meth}"] = byday[which]
                frames.append(df)
                for meth in preds:
                    summary.append({"dataset": "DB6", "subject": target, "day": which,
                                    "method": meth,
                                    "balanced_accuracy": accs[meth][which]})
                print(f"  S{target} {which}: " + "  ".join(
                    f"{m} {accs[m][which]:.3f}" for m in preds), flush=True)
            del tgt, base, calib_n, tests; gc.collect()
        classes = list(ws.class_names)
    else:
        src_ws = segment([preprocess(r, pcfg) for r in
                          load_cohort(args.db2_root, "DB2", subjects=None,
                                      expected_channels=12, verbose=False)], cfg=wcfg)
        tgt_ws = segment([preprocess(r, pcfg) for r in
                          load_cohort(args.db3_root, "DB3", subjects=args.targets,
                                      expected_channels=12, verbose=False)], cfg=wcfg)
        excl = excluded_subjects("DB3")
        if excl:
            tgt_ws = tgt_ws.select(~np.isin(tgt_ws.subject, excl))
        print(f"DB2 source {len(src_ws):,} windows; DB3 target {len(tgt_ws):,}", flush=True)
        base, _ = pretrain(normalize_per_subject(src_ws, inplace=True), val=None,
                           cfg=PretrainConfig(steps=args.steps, seed=args.seed),
                           verbose=True)
        del src_ws; gc.collect()
        for target in subjects(tgt_ws):
            sp = calibration_split(tgt_ws, target, shots=args.shots)
            stats = fit_normalizer(sp.calib, "calib")
            calib_n = stats.apply(sp.calib)
            tests = {"test": stats.apply(sp.test)}
            preds, accs = predict_all(base, calib_n, tests, device, budget,
                                      args.seed, ctau)
            tws = tests["test"]
            df = pd.DataFrame({"dataset": "DB3", "subject": target, "day": "test",
                               "rep": tws.rep, "start": tws.start, "y_true": tws.y})
            for meth, byday in preds.items():
                df[f"pred_{meth}"] = byday["test"]
            frames.append(df)
            for meth in preds:
                summary.append({"dataset": "DB3", "subject": target, "day": "test",
                                "method": meth, "balanced_accuracy": accs[meth]["test"]})
            print(f"  S{target}: " + "  ".join(
                f"{m} {accs[m]['test']:.3f}" for m in preds), flush=True)
        classes = list(tgt_ws.class_names)

    out = pd.concat(frames, ignore_index=True)
    pred_path = args.out / f"preds_{args.mode}.csv"
    out.to_csv(pred_path, index=False)
    pd.DataFrame(summary).to_csv(args.out / f"preds_{args.mode}_summary.csv", index=False)
    (args.out / f"preds_{args.mode}_meta.json").write_text(json.dumps({
        "mode": args.mode, "targets": args.targets, "shots": args.shots,
        "seed": args.seed, "pretrain_steps": args.steps,
        "classes": classes,
        "protocol": ("LOSO within DB6, calibrate day 1, predict day-1 held-out "
                     "repetitions and day 5" if args.mode == "db6"
                     else "DB2-pretrained, calibrate on the first repetitions, "
                          "predict the held-out repetitions"),
        "contains_emg": False,
        "columns": ("dataset, subject, day, rep, start (sample index), y_true, and one "
                    "pred_<method> column per method -- labels only, no signal"),
        "hyperparameters": {"budget": str(args.budget), "classic_gate_tau": ctau,
                            "selected_on": "held-out DB2 subjects; nothing tuned on test"},
        "window": {"length_ms": wcfg.length_ms, "stride_ms": wcfg.stride_ms},
    }, indent=2))
    print(f"\n{len(out):,} rows -> {pred_path}", flush=True)


if __name__ == "__main__":
    main()
