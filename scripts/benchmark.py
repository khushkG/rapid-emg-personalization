"""Benchmark our model against the published NinaPro DB3 protocol.

Our headline numbers come from a deliberately hard few-shot setup: pretrain on
intact subjects, then personalize a *previously unseen* amputee from one to
three repetitions. The literature does something much easier -- train on most of
each subject's own repetitions and test on the rest -- and reports plain
accuracy. The two are not comparable, so "our 0.40 against their 0.66-0.85" does
not by itself say our model is weak.

This script settles that by running *our* model under *their* protocol:

    train on repetitions 1, 3, 4, 6      test on repetitions 2 and 5

Every method sees exactly the same windows. Nothing is fitted on the test
repetitions -- not a hyperparameter, not a normalization statistic, not the
zero-crossing threshold. Classifier settings are fixed a priori and recorded in
the run's meta file; there is no search anywhere in here, because a search
scored on repetitions 2 and 5 would quietly turn this benchmark into the thing
it is meant to check.

    uv run python scripts/benchmark.py --movements subset
    uv run python scripts/benchmark.py --movements full

Published reference points, for context rather than as a target: ~46% for SVM on
classic time-domain features, and 66-85% for deep learning, both on DB3 with
per-subject training and usually plain accuracy including rest.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from remg.data import PreprocessConfig, WindowConfig, preprocess, segment
from remg.data.cohort import excluded_subjects
from remg.data.movements import DEFAULT_SUBSET, OFFICIAL_MOVEMENT_NAMES
from remg.data.ninapro import load_cohort, load_subject
from remg.data.normalize import fit as fit_normalizer
from remg.data.windows import WindowSet
from remg.evaluate.metrics import rest_metrics
from remg.features import fit_zc_threshold, td_features

# --- the protocol, fixed --------------------------------------------------
TRAIN_REPS = (1, 3, 4, 6)
TEST_REPS = (2, 5)

# Majority-vote smoothing span, in windows. Chosen a priori as ~half a second of
# controller output at the 100 ms stride below -- not swept, because sweeping it
# against repetitions 2 and 5 is exactly the leak this script exists to avoid.
SMOOTH_WINDOWS = 5

# Classifier settings, fixed a priori. No grid search anywhere in this file.
SVM_C, SVM_GAMMA = 10.0, "scale"
RF_TREES = 300
SCRATCH_STEPS, FINETUNE_STEPS = 1200, 600
BATCH = 128
SCRATCH_LR, FINETUNE_LR = 1e-3, 3e-4


# Classic time-domain features live in remg.features so the literature baseline
# and the classic rest gate cannot drift apart.

# --- majority-vote smoothing ----------------------------------------------

def smooth_predictions(pred: np.ndarray, rep: np.ndarray, span: int) -> np.ndarray:
    """Majority vote over `span` consecutive windows, within one repetition.

    A controller emits a decision every stride; smoothing trades latency for
    stability and is what any real system would do. Votes never cross a
    repetition boundary, because those windows are not consecutive in time.
    Only the predictions are used -- never the labels.
    """
    out = pred.copy()
    half = span // 2
    for r in np.unique(rep):
        idx = np.flatnonzero(rep == r)
        if len(idx) == 0:
            continue
        block = pred[idx]
        voted = np.empty_like(block)
        for i in range(len(block)):
            lo, hi = max(0, i - half), min(len(block), i + half + 1)
            vals, counts = np.unique(block[lo:hi], return_counts=True)
            voted[i] = vals[counts.argmax()]
        out[idx] = voted
    return out


# --- metrics ---------------------------------------------------------------

def metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int,
            rest_index: int | None) -> dict:
    from sklearn.metrics import balanced_accuracy_score, f1_score

    labels = list(range(n_classes))
    out = {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                   zero_division=0)),
    }
    # Accuracy on the movement windows alone. Rest is most of the data, so
    # including it flatters every method; the literature usually does not say
    # which it reports, which is half of why the numbers look so far apart.
    if rest_index is not None:
        m = y_true != rest_index
        out["accuracy_no_rest"] = float((y_true[m] == y_pred[m]).mean()) if m.any() else float("nan")
    else:
        out["accuracy_no_rest"] = out["accuracy"]
    return out


# --- methods ---------------------------------------------------------------

def run_classic(name, clf, Xtr, ytr, Xte, thr):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ftr = td_features(Xtr, thr)
    fte = td_features(Xte, thr)
    pipe = make_pipeline(StandardScaler(), clf)      # scaler fitted on train only
    pipe.fit(ftr, ytr)
    return pipe.predict(fte)


def run_cnn(Xtr, ytr, Xte, n_classes, n_channels, device, steps, lr,
            pretrained_encoder=None, seed=0):
    """Train our CNN on one subject and predict the held-out repetitions.

    With `pretrained_encoder`, the DB2 backbone is transferred and a fresh head
    is sized to this subject's class count; without it the network starts from
    random init. Everything else is identical between the two.
    """
    import copy

    import torch
    import torch.nn.functional as F

    from remg.models import build_model
    from remg.utils import seed_everything

    seed_everything(seed)
    model = build_model(n_channels=n_channels, n_classes=n_classes).to(device)
    if pretrained_encoder is not None:
        model.encoder.load_state_dict(copy.deepcopy(pretrained_encoder))

    xtr = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)

    # Class-balanced sampling: rest is most of the windows, and plain sampling
    # converges to predicting it.
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    w = np.zeros(len(ytr))
    nz = counts > 0
    per = np.zeros_like(counts)
    per[nz] = 1.0 / counts[nz]
    w = per[ytr]
    w = w / w.sum()

    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(steps):
        idx = rng.choice(len(ytr), size=min(BATCH, len(ytr)), replace=True, p=w)
        sel = torch.from_numpy(idx).to(device)
        loss = F.cross_entropy(model(xtr[sel], mode="linear"), ytr_t[sel])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), 512):
            xb = torch.from_numpy(Xte[i:i + 512]).to(device)
            preds.append(model(xb, mode="linear").argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


# --- data ------------------------------------------------------------------

def subject_windows(root: Path, subject: int, movements: str, pcfg, wcfg):
    """Window one DB3 subject under the chosen movement set."""
    rec = load_subject(root, "DB3", subject, expected_channels=12)
    rec = preprocess(rec, pcfg)
    if movements == "subset":
        subset = DEFAULT_SUBSET
    else:
        present = sorted({int(v) for v in np.unique(rec.label) if v > 0})
        subset = {0: "rest"}
        subset.update({g: OFFICIAL_MOVEMENT_NAMES.get(g, f"movement_{g}") for g in present})
    return segment([rec], cfg=wcfg, subset=subset)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--movements", choices=["subset", "full"], required=True)
    ap.add_argument("--db3", type=Path, default=Path("data/raw/DB3"))
    ap.add_argument("--db2", type=Path, default=Path("data/raw/DB2"))
    ap.add_argument("--subjects", default=None, help='e.g. "2-11"')
    ap.add_argument("--pretrain-steps", type=int, default=3000)
    ap.add_argument("--length-ms", type=float, default=200.0)
    ap.add_argument("--stride-ms", type=float, default=100.0)
    ap.add_argument("--target-fs", type=int, default=1000)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    import torch
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    from remg.utils import pick_device

    tag = args.tag or f"bench_{args.movements}"
    device = pick_device(None)
    pcfg = PreprocessConfig(notch_hz=None, target_fs=args.target_fs)
    wcfg = WindowConfig(length_ms=args.length_ms, stride_ms=args.stride_ms)
    # Each window represents one stride of wall-clock time, which is what turns a
    # count of mistaken windows into a false-activation rate per minute.
    stride_samples = int(round(wcfg.stride_ms * pcfg.target_fs / 1000))

    # S1 is excluded from the 12-movement runs for the documented reason (it has
    # no tripod or lateral grasp). Under the full movement set that reason does
    # not apply -- every class it is scored on is one it actually performed -- so
    # it is kept there, and the comparison to the literature uses all 11.
    if args.subjects:
        lo, _, hi = args.subjects.partition("-")
        subjects = list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
    elif args.movements == "subset":
        subjects = [s for s in range(1, 12) if s not in excluded_subjects("DB3")]
    else:
        subjects = list(range(1, 12))

    print(f"benchmark: {args.movements} movement set, DB3 subjects {subjects}", flush=True)
    print(f"protocol: train reps {TRAIN_REPS}, test reps {TEST_REPS}", flush=True)
    print(f"windows: {args.length_ms:.0f} ms, {args.stride_ms:.0f} ms stride "
          f"({100 * (1 - args.stride_ms / args.length_ms):.0f}% overlap) @ {args.target_fs} Hz",
          flush=True)

    # --- one DB2-pretrained encoder, shared by every subject -----------------
    # DB3 subjects are not in DB2, so this leaks nothing. It is pretrained once
    # and reused, which is also what the real pipeline does.
    t0 = time.time()
    print(f"\npretraining the shared DB2 encoder ({args.pretrain_steps} steps) ...", flush=True)
    from remg.data.normalize import normalize_per_subject
    from remg.train.pretrain import PretrainConfig, pretrain

    db2 = load_cohort(args.db2, "DB2", expected_channels=12, verbose=False)
    db2 = [preprocess(r, pcfg) for r in db2]
    db2_ws = segment(db2, cfg=wcfg)
    base, _ = pretrain(normalize_per_subject(db2_ws), val=None,
                       cfg=PretrainConfig(steps=args.pretrain_steps), verbose=False)
    pretrained_encoder = {k: v.cpu().clone() for k, v in base.encoder.state_dict().items()}
    del db2, db2_ws, base
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    rows = []
    for subject in subjects:
        ws = subject_windows(args.db3, subject, args.movements, pcfg, wcfg)
        tr_mask = np.isin(ws.rep, TRAIN_REPS)
        te_mask = np.isin(ws.rep, TEST_REPS)
        train, test = ws.select(tr_mask), ws.select(te_mask)
        if len(train) == 0 or len(test) == 0:
            print(f"  S{subject}: no windows on one side, skipping", flush=True)
            continue

        rest_index = ws.class_names.index("rest") if "rest" in ws.class_names else None
        n_classes = ws.n_classes

        # Every statistic below is fitted on the training repetitions only.
        stats = fit_normalizer(train, "calib")
        tr_n, te_n = stats.apply(train), stats.apply(test)
        thr = fit_zc_threshold(train.X)

        print(f"\nS{subject}: {n_classes} classes, {len(train)} train / {len(test)} test windows",
              flush=True)

        preds = {}
        t = time.time()
        preds["td_lda"] = run_classic("lda", LinearDiscriminantAnalysis(),
                                      train.X, train.y, test.X, thr)
        preds["td_svm"] = run_classic("svm", SVC(C=SVM_C, gamma=SVM_GAMMA),
                                      train.X, train.y, test.X, thr)
        preds["td_rf"] = run_classic("rf", RandomForestClassifier(
            n_estimators=RF_TREES, random_state=0, n_jobs=-1), train.X, train.y, test.X, thr)
        print(f"  classic methods {time.time() - t:.0f}s", flush=True)

        t = time.time()
        preds["cnn_scratch"] = run_cnn(tr_n.X, tr_n.y, te_n.X, n_classes, ws.n_channels,
                                       device, SCRATCH_STEPS, SCRATCH_LR)
        preds["cnn_pretrained_ft"] = run_cnn(tr_n.X, tr_n.y, te_n.X, n_classes,
                                             ws.n_channels, device, FINETUNE_STEPS,
                                             FINETUNE_LR, pretrained_encoder)
        print(f"  CNN methods {time.time() - t:.0f}s", flush=True)

        # One group per repetition so a false-activation event is never counted as
        # continuing across the seam between two repetitions.
        rest_group = test.rep.astype(np.int64)
        for method, p in preds.items():
            for smoothing in (False, True):
                yp = smooth_predictions(p, test.rep, SMOOTH_WINDOWS) if smoothing else p
                m = metrics(test.y, yp, n_classes, rest_index)
                rm = rest_metrics(test.y, yp, start=test.start, group=rest_group,
                                  stride_samples=stride_samples, fs=test.fs,
                                  rest_index=0 if rest_index is None else rest_index)
                rows.append({
                    "subject": subject, "method": method, "smoothed": smoothing,
                    "movements": args.movements, "n_classes": n_classes,
                    "n_train": len(train), "n_test": len(test), **m, **rm.as_row(),
                })
                mark = " (smoothed)" if smoothing else ""
                print(f"    {method:<18}{mark:<12} acc {m['accuracy']:.3f}  "
                      f"acc_no_rest {m['accuracy_no_rest']:.3f}  "
                      f"bal {m['balanced_accuracy']:.3f}  F1 {m['macro_f1']:.3f}  "
                      f"rest_rec {rm.rest_recall:.3f}  "
                      f"false_act {rm.false_activation_rate:.3f}  "
                      f"fa/min {rm.false_activations_per_min:.1f}", flush=True)

    import pandas as pd

    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / f"{tag}_rows.csv", index=False)

    meta = {
        "protocol": {"train_reps": list(TRAIN_REPS), "test_reps": list(TEST_REPS)},
        "movement_set": args.movements,
        "subjects": subjects,
        "excluded_from_subset_runs": excluded_subjects("DB3"),
        "window": {
            "length_ms": args.length_ms, "stride_ms": args.stride_ms,
            "overlap_pct": 100 * (1 - args.stride_ms / args.length_ms),
            "purity": wcfg.purity, "keep_rest": wcfg.keep_rest,
        },
        "preprocess": {"notch_hz": pcfg.notch_hz, "band_hz": pcfg.band_hz,
                       "target_fs": pcfg.target_fs, "source_fs": 2000},
        "features": {"set": ["MAV", "RMS", "WL", "ZC", "SSC"],
                     "per_channel": True,
                     "zc_ssc_threshold": "0.01 x per-channel RMS of the TRAINING windows",
                     "scaler": "StandardScaler fitted on training windows only"},
        "classifiers": {
            "lda": "LinearDiscriminantAnalysis(solver='svd')",
            "svm": f"SVC(kernel='rbf', C={SVM_C}, gamma='{SVM_GAMMA}')",
            "rf": f"RandomForestClassifier(n_estimators={RF_TREES}, random_state=0)",
            "cnn_scratch": f"{SCRATCH_STEPS} steps, batch {BATCH}, AdamW lr {SCRATCH_LR}",
            "cnn_pretrained_ft": (f"DB2 encoder transferred, fresh head, {FINETUNE_STEPS} "
                                  f"steps, batch {BATCH}, AdamW lr {FINETUNE_LR}"),
            "class_balanced_sampling": True,
        },
        "pretrain_steps": args.pretrain_steps,
        "rest_behaviour": {
            "false_activation_rate": "fraction of true-rest windows predicted as a "
                                     "movement; exactly 1 - rest_recall",
            "false_activations_per_min": "maximal runs of movement prediction inside a "
                                         "true-rest stretch, per minute of rest; events, "
                                         "not windows, and not derivable from the rate",
            "event_grouping": "per repetition, so no event spans a seam between two",
        },
        "smoothing": {"kind": "majority vote", "span_windows": SMOOTH_WINDOWS,
                      "span_ms": SMOOTH_WINDOWS * args.stride_ms,
                      "crosses_repetition_boundary": False},
        "no_tuning_on_test": ("All classifier settings fixed a priori; no search. "
                              "Normalization, feature scaler and ZC/SSC thresholds "
                              "fitted on training repetitions only."),
        "device": str(device),
    }
    (args.out / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n{len(rows)} rows -> {args.out / f'{tag}_rows.csv'}")


if __name__ == "__main__":
    main()
