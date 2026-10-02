"""Choose the adaptation budget using held-out DB2 subjects only.

The benchmark showed the few-shot path was adapting with 100 steps at lr 1e-4
while per-subject training on the same encoder used 600 steps at 3e-4. That is a
budget choice, not a property of the few-shot regime, and it was never selected
by any procedure -- it was a default.

This selects it properly:

  * DB2 is split by *person*. The early subjects pretrain the encoder; the last
    few are held out and treated exactly like unseen target subjects -- same
    calibration-by-repetition protocol, same normalization rule, same scoring.
  * The grid is searched on those held-out DB2 people and nothing else. DB3 is
    not loaded by this script at all, so it cannot leak in even by accident.
  * All three adaptive conditions get the same treatment. Raising only
    `finetune`'s budget would swap one unfair comparison for its mirror image.

The chosen values are written to results/<tag>_chosen.json, which the LOSO study
then reads. The grid itself is in <tag>_rows.csv so the choice is auditable.

    uv run python scripts/tune_budget.py --tag budget

Caveat worth stating in the writeup: held-out DB2 subjects are intact, and the
real targets are amputees. Selecting on intact people is the correct protocol --
the alternative touches the test set -- but it is not the same distribution, so
the chosen budget is principled rather than optimal.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from remg.data import PreprocessConfig, WindowConfig, fit_normalizer, preprocess, segment
from remg.data.normalize import normalize_per_subject
from remg.data.ninapro import load_cohort
from remg.data.splits import calibration_split, subjects
from remg.evaluate.metrics import score
from remg.train import AdaptConfig, PretrainConfig
from remg.train.adapt import adapt
from remg.train.pretrain import pretrain
from remg.utils import pick_device

# The grids. Each brackets the current default on both sides so the search can
# say "the default was already right" rather than only ever moving upward.
#
# A first pass included a 2400-step cell in the finetune and linear_probe grids to
# check the optimum was not pinned to the top of the range. It was not: 2400 steps
# scored below 1200 for every learning rate (results/budget_firstpass_rows.csv), so
# the optimum is interior and the 2400 cells only cost time -- 51s each against 6s
# for the cheapest. They are dropped on that evidence, which is DB2 evidence.
GRIDS: dict[str, list[dict]] = {
    "finetune": [
        {"finetune_steps": s, "finetune_lr": lr}
        for s in (100, 300, 600, 1200)
        for lr in (1e-4, 3e-4, 1e-3)
    ],
    "linear_probe": [
        {"probe_steps": s, "probe_lr": lr}
        for s in (200, 600, 1200)
        for lr in (1e-3, 3e-3)
    ],
    "rapid": [
        {"rapid_steps": s, "rapid_lr": lr}
        for s in (50, 200, 600, 1200)
        for lr in (1e-3, 5e-3, 1e-2)
    ],
}


def _peak_rss_gb() -> float:
    """PEAK resident set size in GB -- a high-water mark, not current usage.

    Here to make the swap-thrash that cost this script two hours visible. Peak is
    reached while segmenting and never falls, so a large value after freeing is
    expected; what matters is that current RSS (check with ps) stays well under
    physical RAM for the rest of the run.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3


def _config_lr(cond: str, over: dict) -> float:
    key = {"finetune": "finetune_lr", "linear_probe": "probe_lr",
           "rapid": "rapid_lr"}[cond]
    return float(over[key])


def config_steps(cond: str, over: dict) -> int:
    """The step count a config spends -- the tie-break's measure of cost.

    Wall-clock was the obvious choice and was wrong. Measured seconds depend on
    whatever else the machine is doing: during the first budget search this host
    paged into swap and recorded 91s for a 600-step config against 21s for a
    1200-step one, which would have inverted the tie-break. Step count is the
    same quantity the grid varies, is exact, and is reproducible on any machine.
    """
    key = {"finetune": "finetune_steps", "linear_probe": "probe_steps",
           "rapid": "rapid_steps"}[cond]
    return int(over[key])


def config_label(cond: str, over: dict) -> str:
    if cond == "finetune":
        return f"steps={over['finetune_steps']},lr={over['finetune_lr']:g}"
    if cond == "linear_probe":
        return f"steps={over['probe_steps']},lr={over['probe_lr']:g}"
    return f"steps={over['rapid_steps']},lr={over['rapid_lr']:g}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/raw/DB2"))
    ap.add_argument("--dataset", default="DB2")
    ap.add_argument("--heldout", type=int, nargs="+", default=[12, 13, 14, 15],
                    help="DB2 subjects held out of pretraining and used to select")
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=3000, help="pretraining steps")
    ap.add_argument("--conditions", nargs="+",
                    default=["finetune", "linear_probe", "rapid"])
    ap.add_argument("--tag", default="budget")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    device = pick_device(None)
    pcfg = PreprocessConfig(notch_hz=None, target_fs=1000)
    wcfg = WindowConfig()

    print(f"loading {args.dataset} from {args.root} ...", flush=True)
    recs = load_cohort(args.root, args.dataset, subjects=None,
                       expected_channels=12, verbose=True)
    # Preprocess in place and segment, then drop every intermediate. Each of these
    # holds on the order of 2 GB of float32; keeping them all alive at once pushed
    # a 16 GB machine ~11 GB into swap and slowed the grid by about 16x. The
    # arrays are not needed again once `src_n` and `val` exist.
    for i in range(len(recs)):
        recs[i] = preprocess(recs[i], pcfg)
    ws = segment(recs, cfg=wcfg)
    recs.clear()
    del recs
    gc.collect()
    all_subj = subjects(ws)
    heldout = [s for s in args.heldout if s in all_subj]
    pretrain_subj = [s for s in all_subj if s not in heldout]
    if not heldout:
        raise SystemExit(f"none of --heldout {args.heldout} present; have {all_subj}")
    print(f"\n{len(all_subj)} {args.dataset} subjects: pretrain on {pretrain_subj}, "
          f"select on {heldout}\n{len(ws):,} windows, {ws.n_classes} classes "
          f"{list(ws.class_names)}\nDB3 is never loaded by this script.\n", flush=True)

    src = ws.select(np.isin(ws.subject, pretrain_subj))
    val = ws.select(np.isin(ws.subject, heldout))
    del ws
    gc.collect()
    src_n = normalize_per_subject(src)
    del src
    gc.collect()
    print(f"  peak resident so far: {_peak_rss_gb():.2f} GB "
          f"(high-water mark from segmenting, not current usage)", flush=True)

    rows: list[dict] = []
    t_start = time.time()
    for seed in args.seeds:
        print(f"[seed {seed}] pretraining on {len(pretrain_subj)} subjects "
              f"({len(src_n):,} windows) on {device}", flush=True)
        base, _ = pretrain(src_n, val=None,
                           cfg=PretrainConfig(steps=args.steps, seed=seed), verbose=True)

        for subj in heldout:
            for shots in args.shots:
                try:
                    split = calibration_split(val, subj, shots=shots)
                except ValueError as exc:
                    print(f"  skip S{subj} shots={shots}: {exc}", flush=True)
                    continue
                stats = fit_normalizer(split.calib, "calib")
                calib_n, test_n = stats.apply(split.calib), stats.apply(split.test)

                # Reference point: no adaptation at all, computed once.
                res0 = adapt(base, calib_n, "none", device,
                             AdaptConfig(seed=seed), split.calib_reps)
                sc0 = score(test_n.y, res0.predict(test_n, device), test_n.class_names)
                rows.append({"seed": seed, "subject": subj, "shots": shots,
                             "condition": "none", "config": "-",
                             "balanced_accuracy": sc0.balanced_accuracy,
                             "macro_f1": sc0.macro_f1, "accuracy": sc0.accuracy,
                             "seconds": res0.seconds})

                for cond in args.conditions:
                    for over in GRIDS[cond]:
                        acfg = AdaptConfig(**{**AdaptConfig().__dict__, **over, "seed": seed})
                        res = adapt(base, calib_n, cond, device, acfg, split.calib_reps)
                        sc = score(test_n.y, res.predict(test_n, device), test_n.class_names)
                        label = config_label(cond, over)
                        rows.append({"seed": seed, "subject": subj, "shots": shots,
                                     "condition": cond, "config": label,
                                     "balanced_accuracy": sc.balanced_accuracy,
                                     "macro_f1": sc.macro_f1, "accuracy": sc.accuracy,
                                     "seconds": res.seconds})
                        print(f"  S{subj:<3} shots={shots} {cond:<13} {label:<22} "
                              f"bal {sc.balanced_accuracy:.3f}  F1 {sc.macro_f1:.3f}  "
                              f"({res.seconds:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / f"{args.tag}_rows.csv", index=False)

    # Selection: mean balanced accuracy over held-out subjects, shots and seeds.
    # Ties broken toward the cheaper setting, so a budget is only spent if it buys
    # something measurable.
    chosen: dict[str, dict] = {}
    print("\n" + "=" * 78)
    for cond in args.conditions:
        sub = df[df.condition == cond]
        g = (sub.groupby("config")
                .agg(bal=("balanced_accuracy", "mean"), f1=("macro_f1", "mean"),
                     sec=("seconds", "mean"), n=("balanced_accuracy", "size"))
                .sort_values("bal", ascending=False))
        print(f"\n{cond}  (held-out DB2 subjects {heldout}, shots {args.shots}, "
              f"seeds {args.seeds})")
        print(g.round(4).to_string())
        best_bal = g.bal.max()
        # Within one SEM of the best, prefer the cheapest -- a budget that costs
        # 4x the work for a difference smaller than the noise is not a win. Cost is
        # measured in STEPS, not seconds: see config_steps for why timings here are
        # not trustworthy. Learning rate breaks a remaining tie toward the smaller
        # value, so the choice is fully determined by the data and not by dict order.
        sem = sub.groupby("config").balanced_accuracy.mean().std() / max(len(g) ** 0.5, 1)
        near = list(g[g.bal >= best_bal - sem].index)
        by_label = {config_label(cond, o): o for o in GRIDS[cond]}
        pick = min(near, key=lambda lab: (config_steps(cond, by_label[lab]),
                                          _config_lr(cond, by_label[lab])))
        over = by_label[pick]
        chosen[cond] = {"config": pick, **over,
                        "val_balanced_accuracy": float(g.loc[pick, "bal"]),
                        "best_in_grid": float(best_bal),
                        "tie_band_sem": float(sem),
                        "tie_break": "fewest steps among configs within one SEM of the best",
                        "n_within_tie_band": len(near),
                        "steps": config_steps(cond, over),
                        "mean_adapt_seconds_unreliable": float(g.loc[pick, "sec"])}
        print(f"  -> chosen: {pick}   (best in grid {best_bal:.4f}, "
              f"chosen {g.loc[pick, 'bal']:.4f}, within {sem:.4f})")
        # Diagnostic only -- one budget is applied everywhere. This says whether
        # that pooled choice hides a disagreement between shot counts.
        per_shot = (sub.groupby(["shots", "config"]).balanced_accuracy.mean()
                       .groupby(level=0).idxmax())
        for sh, key in per_shot.items():
            print(f"     best at shots={sh}: {key[1]}")
        chosen[cond]["per_shot_best_diagnostic"] = {
            int(sh): key[1] for sh, key in per_shot.items()}

    none_bal = df[df.condition == "none"].balanced_accuracy.mean()
    meta = {
        "purpose": "adaptation budget selected on held-out DB2 subjects only",
        "dataset": args.dataset,
        "pretrain_subjects": pretrain_subj,
        "selection_subjects": heldout,
        "db3_loaded": False,
        "shots": args.shots, "seeds": args.seeds, "pretrain_steps": args.steps,
        "grids": {k: [config_label(k, o) for o in v] for k, v in GRIDS.items()
                  if k in args.conditions},
        "selection_rule": ("highest mean balanced accuracy across held-out subjects x "
                           "shots x seeds; among configs within one SEM of the best, "
                           "the fewest STEPS wins, then the smaller learning rate. "
                           "Wall-clock is recorded but never used to choose -- it is "
                           "distorted by whatever else the machine is doing"),
        "no_adaptation_reference": float(none_bal),
        "chosen": chosen,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "device": str(device),
    }
    (args.out / f"{args.tag}_meta.json").write_text(json.dumps(meta, indent=2))
    (args.out / f"{args.tag}_chosen.json").write_text(json.dumps(chosen, indent=2))
    print(f"\nno-adaptation reference on the same held-out subjects: {none_bal:.4f}")
    print(f"\n{len(rows)} rows -> {args.out / f'{args.tag}_rows.csv'}"
          f"   ({(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
