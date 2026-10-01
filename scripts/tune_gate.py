"""Choose the rest-gate threshold on held-out DB2 subjects only.

The gate threshold is an operating point, not an accuracy knob: raising it buys
rest recall and spends movement recall. Choosing it on the amputee cohort would be
choosing where to sit on that trade-off using the data the result is reported on,
so it is chosen here, the same way the adaptation budget was -- DB2 split by
person, subjects 1-11 pretraining, 12-15 held out and treated as unseen targets.
DB3 is not loaded by this script.

The objective is stated in advance rather than picked after seeing the curve:

    maximise balanced accuracy subject to false_activation_rate <= --far-budget

A prosthesis that moves unbidden in more than one rest window in ten is not
usable, so the constraint comes first and accuracy is maximised within it. If no
threshold on the grid meets the budget, the one with the lowest false-activation
rate is chosen and the shortfall is recorded rather than hidden.

    uv run python scripts/tune_gate.py --budget results/budget_chosen.json
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
from remg.data.normalize import normalize_per_subject
from remg.data.ninapro import load_cohort
from remg.data.splits import calibration_split, subjects
from remg.models import build_model
from remg.evaluate.metrics import rest_metrics, score
from remg.train import AdaptConfig, PretrainConfig
from remg.train.adapt import adapt, predict_proba
from remg.train.pretrain import pretrain
from remg.train.twostage import (
    GateConfig,
    _argmax_excluding_rest,
    apply_threshold,
    fit_gate,
    gate_scores,
)
from remg.utils import pick_device

THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


def load_budget(path: Path | None) -> dict:
    if not path:
        return {}
    raw = json.loads(path.read_text())
    allowed = set(AdaptConfig().__dict__)
    return {k: v for entry in raw.values() if isinstance(entry, dict)
            for k, v in entry.items() if k in allowed}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/raw/DB2"))
    ap.add_argument("--dataset", default="DB2")
    ap.add_argument("--heldout", type=int, nargs="+", default=[12, 13, 14, 15])
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="one seed is enough for a threshold, which is a coarse "
                         "operating point rather than an accuracy knob; the grid "
                         "already averages over subjects and shot counts")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--conditions", nargs="+",
                    default=["finetune", "linear_probe", "rapid"])
    ap.add_argument("--budget", type=Path, help="adaptation budget from tune_budget.py")
    ap.add_argument("--far-budget", type=float, default=0.10,
                    help="maximum acceptable false-activation rate (default 0.10)")
    ap.add_argument("--encoder", type=Path, default=Path("results/db2_encoder.pt"),
                    help="Checkpoint of the source-only encoder (pretrained on the "
                         "non-held-out source subjects). Loaded if present, written if "
                         "not. One ~0.5 MB file, so the threshold sweep pays neither for "
                         "a fresh 3000-step pretrain nor for holding the whole source "
                         "cohort in memory to do it.")
    ap.add_argument("--repretrain", action="store_true",
                    help="ignore an existing --encoder checkpoint and pretrain again")
    ap.add_argument("--tag", default="gate")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    device = pick_device(None)
    pcfg, wcfg = PreprocessConfig(notch_hz=None, target_fs=1000), WindowConfig()
    budget = load_budget(args.budget)
    stride_samples = int(round(wcfg.stride_ms * pcfg.target_fs / 1000))

    def load_windows(subject_list, label):
        print(f"loading {label} from {args.root} ...", flush=True)
        recs = load_cohort(args.root, args.dataset, subjects=subject_list,
                           expected_channels=12, verbose=True)
        for i in range(len(recs)):
            recs[i] = preprocess(recs[i], pcfg)
        out = segment(recs, cfg=wcfg)
        recs.clear(); del recs; gc.collect()
        return out

    heldout = sorted(args.heldout)
    # The source cohort is only loaded if the encoder actually has to be trained.
    # Loading it otherwise costs ~2 GB for nothing, and on a memory-pressured host
    # that is the difference between a run that finishes and one that does not.
    have_ckpt = bool(args.encoder and args.encoder.exists() and not args.repretrain)
    val = load_windows(heldout, f"held-out subjects {heldout}")
    print(f"\nselect the threshold on {heldout}; "
          f"DB3 is never loaded by this script.\n", flush=True)

    # Rows are appended to disk as they are produced, not held until the end. On a
    # memory-pressured host this script can be killed mid-run, and a selection that
    # loses two hours of adaptations because the results existed only in RAM is a
    # self-inflicted wound. The partial file is a valid input to the same selection.
    rows_path = args.out / f"{args.tag}_rows.csv"
    args.out.mkdir(parents=True, exist_ok=True)
    if rows_path.exists():
        rows_path.unlink()
    written = {"header": False}

    def flush(batch: list[dict]) -> None:
        if not batch:
            return
        pd.DataFrame(batch).to_csv(rows_path, mode="a", index=False,
                                   header=not written["header"])
        written["header"] = True

    rows: list[dict] = []
    t_start = time.time()
    pre: list[int] = []
    for seed in args.seeds:
        if have_ckpt:
            ck = torch.load(args.encoder, map_location="cpu", weights_only=False)
            pre = list(ck["pretrain_subjects"])
            overlap = sorted(set(pre) & set(heldout))
            if overlap:
                raise SystemExit(
                    f"{args.encoder} was pretrained on {pre}, which overlaps the "
                    f"held-out subjects {overlap}. Choosing a threshold with an encoder "
                    f"that already saw those subjects is leakage. Delete the checkpoint "
                    f"or pass --repretrain."
                )
            if int(ck.get("seed", -1)) != seed:
                print(f"  note: checkpoint was trained with seed {ck.get('seed')}, "
                      f"this pass asked for {seed}; the encoder is reused as-is so the "
                      f"threshold is selected against one encoder, not {len(args.seeds)}",
                      flush=True)
            base = build_model(val.n_channels, val.n_classes, **ck.get("model", {})).to(device)
            base.load_state_dict(ck["state_dict"])
            base.eval()
            print(f"[seed {seed}] encoder loaded from {args.encoder} "
                  f"(pretrained on {pre}, {ck.get('steps')} steps)", flush=True)
        else:
            src_all = load_windows(None, "the full source cohort (to pretrain)")
            pre = [s for s in subjects(src_all) if s not in heldout]
            src = src_all.select(np.isin(src_all.subject, pre))
            del src_all; gc.collect()
            src_n = normalize_per_subject(src); del src; gc.collect()
            print(f"[seed {seed}] pretraining on {len(pre)} subjects", flush=True)
            base, _ = pretrain(src_n, val=None,
                               cfg=PretrainConfig(steps=args.steps, seed=seed), verbose=True)
            del src_n; gc.collect()
            if args.encoder:
                args.encoder.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": base.state_dict(), "pretrain_subjects": pre,
                            "held_out": heldout, "steps": args.steps, "seed": seed,
                            "dataset": args.dataset, "model": {}}, args.encoder)
                print(f"  saved the encoder to {args.encoder}", flush=True)
                have_ckpt = True
        for subj in heldout:
            for shots in args.shots:
                try:
                    split = calibration_split(val, subj, shots=shots)
                except ValueError as exc:
                    print(f"  skip S{subj} shots={shots}: {exc}", flush=True)
                    continue
                stats = fit_normalizer(split.calib, "calib")
                calib_n, test_n = stats.apply(split.calib), stats.apply(split.test)
                group = (test_n.subject.astype(np.int64) * 1000
                         + test_n.session.astype(np.int64) * 100 + test_n.rep)

                for cond in args.conditions:
                    acfg = AdaptConfig(**{**AdaptConfig().__dict__, **budget, "seed": seed})
                    res = adapt(base, calib_n, cond, device, acfg, split.calib_reps)
                    # One forward pass for stage 2 and one for stage 1, then every
                    # threshold is a comparison on cached arrays.
                    proba = predict_proba(res.model, test_n, device, mode=res.mode)
                    gate = fit_gate(res.model, calib_n, device, GateConfig())
                    p_move = gate_scores(res.model, test_n, device, gate)
                    mv = _argmax_excluding_rest(proba, 0)

                    def record(tau, pred, kind):
                        sc = score(test_n.y, pred, test_n.class_names)
                        rm = rest_metrics(test_n.y, pred, start=test_n.start, group=group,
                                          stride_samples=stride_samples, fs=test_n.fs)
                        rows.append({"seed": seed, "subject": subj, "shots": shots,
                                     "condition": cond, "stage": kind, "threshold": tau,
                                     "balanced_accuracy": sc.balanced_accuracy,
                                     "macro_f1": sc.macro_f1, "accuracy": sc.accuracy,
                                     **rm.as_row()})

                    before = len(rows)
                    record(float("nan"), proba.argmax(axis=1), "single")
                    if not gate.degenerate:
                        for tau in THRESHOLDS:
                            record(tau, apply_threshold(p_move, mv, tau), "two_stage")
                    flush(rows[before:])
                    print(f"  S{subj:<3} shots={shots} {cond:<13} "
                          f"single bal {rows[-len(THRESHOLDS) - 1]['balanced_accuracy']:.3f} "
                          f"far {rows[-len(THRESHOLDS) - 1]['false_activation_rate']:.3f} "
                          f"| {gate.describe()}", flush=True)

    df = pd.DataFrame(rows)

    chosen: dict[str, dict] = {}
    print("\n" + "=" * 78)
    single = df[df.stage == "single"]
    two = df[df.stage == "two_stage"]
    for cond in args.conditions:
        s0 = single[single.condition == cond]
        g = (two[two.condition == cond].groupby("threshold")
             .agg(bal=("balanced_accuracy", "mean"), f1=("macro_f1", "mean"),
                  far=("false_activation_rate", "mean"),
                  rest=("rest_recall", "mean"),
                  fa_min=("false_activations_per_min", "mean"))
             .sort_index())
        print(f"\n{cond}  (single stage: bal {s0.balanced_accuracy.mean():.4f}, "
              f"far {s0.false_activation_rate.mean():.4f}, "
              f"rest recall {s0.rest_recall.mean():.4f})")
        print(g.round(4).to_string())
        ok = g[g.far <= args.far_budget]
        if len(ok):
            tau = float(ok.bal.idxmax()); met = True
        else:
            tau = float(g.far.idxmin()); met = False
            print(f"  !! no threshold reaches far <= {args.far_budget}; "
                  f"taking the lowest far instead")
        chosen[cond] = {
            "threshold": tau, "far_budget": args.far_budget, "budget_met": met,
            "val_balanced_accuracy": float(g.loc[tau, "bal"]),
            "val_false_activation_rate": float(g.loc[tau, "far"]),
            "val_rest_recall": float(g.loc[tau, "rest"]),
            "single_stage_balanced_accuracy": float(s0.balanced_accuracy.mean()),
            "single_stage_false_activation_rate": float(s0.false_activation_rate.mean()),
        }
        print(f"  -> chosen threshold {tau:.2f}  "
              f"(bal {g.loc[tau, 'bal']:.4f}, far {g.loc[tau, 'far']:.4f}, "
              f"rest recall {g.loc[tau, 'rest']:.4f})")

    meta = {
        "purpose": "rest-gate threshold selected on held-out DB2 subjects only",
        "dataset": args.dataset, "pretrain_subjects": pre, "selection_subjects": heldout,
        "encoder_checkpoint": str(args.encoder) if args.encoder else None,
        "encoder_reused_across_seeds": bool(have_ckpt and len(args.seeds) > 1),
        "db3_loaded": False,
        "objective": (f"maximise balanced accuracy subject to false_activation_rate "
                      f"<= {args.far_budget}; stated before the sweep was run"),
        "thresholds": list(THRESHOLDS),
        "shots": args.shots, "seeds": args.seeds, "pretrain_steps": args.steps,
        "adapt_budget_source": str(args.budget) if args.budget else "AdaptConfig defaults",
        "gate": {"model": "LogisticRegression on adapted embeddings", "C": GateConfig().C,
                 "class_balanced": GateConfig().class_balanced,
                 "note": "stage 1 keeps the natural rest prior on purpose"},
        "chosen": chosen,
        "elapsed_seconds": round(time.time() - t_start, 1), "device": str(device),
    }
    (args.out / f"{args.tag}_meta.json").write_text(json.dumps(meta, indent=2))
    (args.out / f"{args.tag}_chosen.json").write_text(json.dumps(chosen, indent=2))
    print(f"\n{len(rows)} rows -> {rows_path}"
          f"   ({(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
