"""Choose the debounce length N and the classic-gate threshold on held-out DB2 only.

Two decision rules, both operating points, both selected the same way the
adaptation budget and the logistic gate's threshold were: DB2 split by person, the
early subjects pretraining the encoder and the last few held out and treated as
unseen targets. DB3 is not loaded by this script.

Goals stated in advance, not chosen after seeing the curves:

  debounce N        minimise false activations per minute, subject to balanced
                    accuracy staying within --bal-tolerance (default 0.03) of the
                    undebounced model. N trades latency for safety -- N windows at a
                    100 ms stride is (N-1)*100 ms of added onset delay -- so the
                    constraint has to be on accuracy and the objective on events,
                    not the reverse.

  classic gate tau  maximise balanced accuracy subject to false-activation rate
                    <= --far-budget (default 0.10), the same objective already used
                    for the logistic gate, so the two gates are comparable.

The encoder is loaded from the checkpoint written by tune_gate.py when present, so
this costs neither a fresh pretrain nor holding the source cohort in memory. The
checkpoint is rejected if its pretraining subjects overlap the held-out set.

    uv run python scripts/tune_rules.py --budget results/budget_chosen.json
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from remg.data import PreprocessConfig, WindowConfig, fit_normalizer, preprocess, segment
from remg.data.ninapro import load_cohort
from remg.data.splits import calibration_split, subjects
from remg.evaluate.metrics import rest_metrics, score
from remg.evaluate.temporal import debounce
from remg.models import build_model
from remg.train import AdaptConfig
from remg.train.adapt import adapt, predict_proba
from remg.train.classic_gate import (
    ClassicGateConfig,
    classic_gate_scores,
    fit_classic_gate,
)
from remg.train.twostage import _argmax_excluding_rest, apply_threshold
from remg.utils import pick_device

DEBOUNCE_N = (1, 2, 3, 4, 5, 6, 8)
GATE_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def load_budget(path: Path | None) -> dict:
    if not path:
        return {}
    raw = json.loads(path.read_text())
    allowed = set(AdaptConfig().__dict__)
    return {k: v for e in raw.values() if isinstance(e, dict)
            for k, v in e.items() if k in allowed}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/raw/DB2"))
    ap.add_argument("--dataset", default="DB2")
    ap.add_argument("--heldout", type=int, nargs="+", default=[12, 13, 14, 15])
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--conditions", nargs="+", default=["finetune", "linear_probe", "rapid"])
    ap.add_argument("--budget", type=Path)
    ap.add_argument("--encoder", type=Path, default=Path("results/db2_encoder.pt"))
    ap.add_argument("--bal-tolerance", type=float, default=0.03)
    ap.add_argument("--far-budget", type=float, default=0.10)
    ap.add_argument("--tag", default="rules")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    device = pick_device(None)
    pcfg, wcfg = PreprocessConfig(notch_hz=None, target_fs=1000), WindowConfig()
    stride = int(round(wcfg.stride_ms * pcfg.target_fs / 1000))
    budget = load_budget(args.budget)
    heldout = sorted(args.heldout)

    print(f"loading held-out subjects {heldout} from {args.root} ...", flush=True)
    recs = load_cohort(args.root, args.dataset, subjects=heldout,
                       expected_channels=12, verbose=True)
    for i in range(len(recs)):
        recs[i] = preprocess(recs[i], pcfg)
    val = segment(recs, cfg=wcfg)
    recs.clear(); del recs; gc.collect()
    print(f"\nselect N and tau on {heldout}; DB3 is never loaded by this script.\n",
          flush=True)

    if not args.encoder.exists():
        raise SystemExit(
            f"{args.encoder} not found. Run scripts/tune_gate.py first -- it writes the "
            f"source-only encoder this script reuses."
        )
    ck = torch.load(args.encoder, map_location="cpu", weights_only=False)
    pre = list(ck["pretrain_subjects"])
    overlap = sorted(set(pre) & set(heldout))
    if overlap:
        raise SystemExit(
            f"{args.encoder} was pretrained on {pre}, overlapping held-out {overlap}. "
            f"Selecting a rule with an encoder that saw those subjects is leakage."
        )

    rows: list[dict] = []
    t0 = time.time()
    for seed in args.seeds:
        base = build_model(val.n_channels, val.n_classes, **ck.get("model", {})).to(device)
        base.load_state_dict(ck["state_dict"])
        base.eval()
        print(f"[seed {seed}] encoder from {args.encoder} (pretrained on {pre})", flush=True)

        for subj in heldout:
            for shots in args.shots:
                try:
                    split = calibration_split(val, subj, shots=shots)
                except ValueError as exc:
                    print(f"  skip S{subj} shots={shots}: {exc}", flush=True)
                    continue
                stats = fit_normalizer(split.calib, "calib")
                calib_n, test_n = stats.apply(split.calib), stats.apply(split.test)
                group = (test_n.subject.astype(np.int64) * 1_000_000
                         + test_n.session.astype(np.int64) * 1000
                         + test_n.rep.astype(np.int64))

                # Stage 1 of the classic gate does not depend on the condition, so it
                # is fitted once per calibration set rather than once per condition.
                cg = fit_classic_gate(calib_n, ClassicGateConfig())
                p_move_classic = classic_gate_scores(test_n, cg)

                for cond in args.conditions:
                    acfg = AdaptConfig(**{**AdaptConfig().__dict__, **budget, "seed": seed})
                    res = adapt(base, calib_n, cond, device, acfg, split.calib_reps)
                    proba = predict_proba(res.model, test_n, device, mode=res.mode)
                    plain = proba.argmax(axis=1)
                    mv = _argmax_excluding_rest(proba, 0)

                    def record(rule, param, pred):
                        sc = score(test_n.y, pred, test_n.class_names)
                        rm = rest_metrics(test_n.y, pred, start=test_n.start, group=group,
                                          stride_samples=stride, fs=test_n.fs)
                        rows.append({"seed": seed, "subject": subj, "shots": shots,
                                     "condition": cond, "rule": rule, "param": param,
                                     "balanced_accuracy": sc.balanced_accuracy,
                                     "macro_f1": sc.macro_f1, "accuracy": sc.accuracy,
                                     **rm.as_row()})

                    for n in DEBOUNCE_N:
                        record("debounce", n,
                               debounce(plain, start=test_n.start, group=group, n=n,
                                        stride_samples=stride))
                    if not cg.degenerate:
                        for tau in GATE_THRESHOLDS:
                            record("classic_gate", tau,
                                   apply_threshold(p_move_classic, mv, tau))
                    print(f"  S{subj:<3} shots={shots} {cond:<13} "
                          f"{len(DEBOUNCE_N)} N x {len(GATE_THRESHOLDS)} tau  "
                          f"| {cg.describe()}", flush=True)

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / f"{args.tag}_rows.csv", index=False)

    chosen: dict[str, dict] = {}
    print("\n" + "=" * 86)
    for cond in args.conditions:
        chosen[cond] = {}

        # --- debounce N -----------------------------------------------------
        d = df[(df.condition == cond) & (df.rule == "debounce")]
        g = (d.groupby("param").agg(bal=("balanced_accuracy", "mean"),
                                    f1=("macro_f1", "mean"),
                                    far=("false_activation_rate", "mean"),
                                    fa=("false_activations_per_min", "mean"),
                                    burst=("mean_false_burst_ms", "mean"))
             .sort_index())
        baseline = float(g.loc[1, "bal"])
        print(f"\n{cond}  debounce (N=1 is undebounced, bal {baseline:.4f})")
        print(g.round(4).to_string())
        ok = g[g.bal >= baseline - args.bal_tolerance]
        n_pick = int(ok.fa.idxmin())
        chosen[cond]["debounce_n"] = n_pick
        chosen[cond]["debounce"] = {
            "n": n_pick, "bal_tolerance": args.bal_tolerance,
            "undebounced_balanced_accuracy": baseline,
            "val_balanced_accuracy": float(g.loc[n_pick, "bal"]),
            "val_false_activations_per_min": float(g.loc[n_pick, "fa"]),
            "val_false_activation_rate": float(g.loc[n_pick, "far"]),
            "added_latency_ms": (n_pick - 1) * wcfg.stride_ms,
        }
        print(f"  -> chosen N={n_pick}  (bal {g.loc[n_pick, 'bal']:.4f} vs {baseline:.4f} "
              f"undebounced, FA/min {g.loc[n_pick, 'fa']:.2f} vs {g.loc[1, 'fa']:.2f}, "
              f"+{(n_pick - 1) * wcfg.stride_ms:.0f} ms latency)")

        # --- classic gate threshold -----------------------------------------
        c = df[(df.condition == cond) & (df.rule == "classic_gate")]
        if c.empty:
            print(f"\n{cond}  classic gate: no rows (gate degenerate everywhere)")
            continue
        gc_ = (c.groupby("param").agg(bal=("balanced_accuracy", "mean"),
                                      f1=("macro_f1", "mean"),
                                      far=("false_activation_rate", "mean"),
                                      rest=("rest_recall", "mean"),
                                      fa=("false_activations_per_min", "mean"))
               .sort_index())
        print(f"\n{cond}  classic gate (td_rf stage 1)")
        print(gc_.round(4).to_string())
        ok2 = gc_[gc_.far <= args.far_budget]
        if len(ok2):
            tau = float(ok2.bal.idxmax()); met = True
        else:
            tau = float(gc_.far.idxmin()); met = False
            print(f"  !! no tau reaches far <= {args.far_budget}; taking the lowest far")
        chosen[cond]["classic_gate"] = {
            "threshold": tau, "far_budget": args.far_budget, "budget_met": met,
            "val_balanced_accuracy": float(gc_.loc[tau, "bal"]),
            "val_false_activation_rate": float(gc_.loc[tau, "far"]),
            "val_rest_recall": float(gc_.loc[tau, "rest"]),
            "val_false_activations_per_min": float(gc_.loc[tau, "fa"]),
        }
        print(f"  -> chosen tau={tau:.2f}  (bal {gc_.loc[tau, 'bal']:.4f}, "
              f"far {gc_.loc[tau, 'far']:.4f}, FA/min {gc_.loc[tau, 'fa']:.2f})")

    meta = {
        "purpose": "debounce N and classic-gate threshold, selected on held-out DB2 only",
        "dataset": args.dataset, "pretrain_subjects": pre, "selection_subjects": heldout,
        "db3_loaded": False,
        "encoder_checkpoint": str(args.encoder),
        "debounce_grid": list(DEBOUNCE_N), "gate_threshold_grid": list(GATE_THRESHOLDS),
        "debounce_objective": (f"minimise false activations per minute subject to balanced "
                              f"accuracy >= undebounced - {args.bal_tolerance}"),
        "classic_gate_objective": (f"maximise balanced accuracy subject to "
                                   f"false_activation_rate <= {args.far_budget}"),
        "classic_gate_model": (f"StandardScaler + RandomForest("
                               f"n_estimators={ClassicGateConfig().n_estimators}, "
                               f"random_state={ClassicGateConfig().random_state}) on "
                               f"MAV/RMS/WL/ZC/SSC; natural class balance"),
        "debounce_is_causal": True,
        "window": {"length_ms": wcfg.length_ms, "stride_ms": wcfg.stride_ms},
        "shots": args.shots, "seeds": args.seeds,
        "adapt_budget_source": str(args.budget) if args.budget else "AdaptConfig defaults",
        "chosen": chosen,
        "elapsed_seconds": round(time.time() - t0, 1), "device": str(device),
    }
    (args.out / f"{args.tag}_meta.json").write_text(json.dumps(meta, indent=2))
    (args.out / f"{args.tag}_chosen.json").write_text(json.dumps(chosen, indent=2))
    print(f"\n{len(rows)} rows -> {args.out / f'{args.tag}_rows.csv'}"
          f"   ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
