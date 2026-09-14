"""Replay one amputee's recorded EMG and animate the hand the model decodes.

This is the demonstration the README promises: muscle signal in, intended
movement out, drawn as a hand. No prosthetic hardware is involved and nothing
here is real-time -- it replays windows in the order they were recorded, at the
rate the controller would have produced them.

    uv run python scripts/demo_hand.py --subject 2 --condition rapid

What it shows, side by side, is the decoded hand and the movement the subject
was actually cued to perform. Showing only the decoded hand would let a fluent
animation stand in for an accurate one; the point of the pairing is that a
mistake is visible as a mistake.

The windows come from repetitions calibration never touched, so this is the
model working on movements it has not seen from this subject.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from remg.data import PreprocessConfig, WindowConfig, preprocess, segment
from remg.data.cohort import excluded_subjects
from remg.data.normalize import normalize_per_subject
from remg.data.ninapro import load_cohort
from remg.data.splits import calibration_split
from remg.train.adapt import AdaptConfig, adapt
from remg.train.pretrain import PretrainConfig, pretrain
from remg.utils import pick_device


def parse_subjects(spec: str | None) -> list[int] | None:
    """Parse "1-15", "1,3,5" or "1-4,7" into a subject list."""
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


def build(root: Path, dataset: str, subjects, pcfg, wcfg, verbose=True):
    recs = load_cohort(root, dataset, subjects=subjects, expected_channels=12,
                       verbose=verbose)
    recs = [preprocess(r, pcfg) for r in recs]
    return segment(recs, cfg=wcfg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-root", type=Path, default=Path("data/raw/DB2"))
    ap.add_argument("--target-root", type=Path, default=Path("data/raw/DB3"))
    ap.add_argument("--subject", type=int, default=2, help="target (amputee) subject")
    ap.add_argument("--condition", default="rapid",
                    choices=["none", "linear_probe", "finetune", "rapid"])
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--source-subjects", default=None, help='e.g. "1-15"')
    ap.add_argument("--seconds", type=float, default=30.0,
                    help="how much of the held-out recording to animate")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--no-notch", action="store_true")
    args = ap.parse_args()

    if args.subject in excluded_subjects("DB3"):
        raise SystemExit(
            f"DB3 S{args.subject} is excluded from the study (see remg/data/cohort.py); "
            "demonstrating on it would show a hand the model was never asked to decode."
        )

    pcfg = PreprocessConfig(notch_hz=None if args.no_notch else 50.0, target_fs=1000)
    wcfg = WindowConfig()
    device = pick_device(None)

    print("loading cohorts ...", flush=True)
    srcs = parse_subjects(args.source_subjects)
    source = build(args.source_root, "DB2", srcs, pcfg, wcfg, verbose=False)
    target = build(args.target_root, "DB3", [args.subject], pcfg, wcfg, verbose=False)
    print(f"  source {source.X.shape}, target {target.X.shape} on {device}", flush=True)

    print(f"pretraining ({args.steps} steps) ...", flush=True)
    base, _ = pretrain(normalize_per_subject(source), val=None,
                       cfg=PretrainConfig(steps=args.steps), verbose=True)

    split = calibration_split(target, args.subject, shots=args.shots)
    from remg.data import fit_normalizer

    stats = fit_normalizer(split.calib, "calib")
    calib_n, test_n = stats.apply(split.calib), stats.apply(split.test)

    print(f"adapting with '{args.condition}' on {args.shots} repetition(s) ...", flush=True)
    res = adapt(base, calib_n, args.condition, device, AdaptConfig(), split.calib_reps)
    proba = res.predict_proba(test_n, device)
    pred = proba.argmax(axis=1)

    # Windows come out of segmentation in recording order, so a contiguous slice
    # is a contiguous stretch of the recording.
    n_frames = int(min(len(pred), args.seconds * 1000 / wcfg.stride_ms))
    sl = slice(0, n_frames)
    names = list(test_n.class_names)
    timeline = [
        {
            "t": round(i * wcfg.stride_ms / 1000.0, 3),
            "true": names[int(test_n.y[sl][i])],
            "pred": names[int(pred[sl][i])],
            "confidence": round(float(proba[sl][i].max()), 4),
        }
        for i in range(n_frames)
    ]
    correct = sum(f["true"] == f["pred"] for f in timeline)
    print(f"  {correct}/{n_frames} frames correct in the animated stretch", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"demo_S{args.subject}_{args.condition}"
    (args.out / f"{tag}.json").write_text(json.dumps({
        "subject": args.subject, "condition": args.condition, "shots": args.shots,
        "calib_reps": list(split.calib_reps), "test_reps": list(split.test_reps),
        "classes": names, "stride_ms": wcfg.stride_ms,
        "frames_correct": correct, "frames": n_frames,
        "timeline": timeline,
    }, indent=2))

    path = render(timeline, test_n, sl, args.out / f"{tag}.gif", args.fps, args.subject,
                  args.condition)
    print(f"wrote {path} and {args.out / f'{tag}.json'}")


def render(timeline, ws, sl, path: Path, fps: int, subject: int, condition: str) -> Path:
    """Animate decoded vs cued hand, with the EMG that produced each frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from remg.viz.hand import POSES, pose_for
    from remg.viz.render import draw_hand

    X = ws.X[sl]
    envelope = np.abs(X).mean(axis=(1, 2))

    fig = plt.figure(figsize=(9.0, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.5, 1.0], hspace=0.05, wspace=0.02)
    ax_pred, ax_true = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_sig = fig.add_subplot(gs[1, :])

    ax_sig.plot(np.arange(len(envelope)) * 0.1, envelope, color="#64748b", linewidth=1.0)
    ax_sig.set_xlabel("seconds of replayed recording", fontsize=8)
    ax_sig.set_ylabel("EMG", fontsize=8)
    ax_sig.tick_params(labelsize=7)
    for side in ("top", "right"):
        ax_sig.spines[side].set_visible(False)
    marker = ax_sig.axvline(0.0, color="#dc2626", linewidth=1.4)

    # Smooth between poses so the hand moves rather than teleporting; the
    # underlying prediction is still per-window and unsmoothed.
    state = {"pose": pose_for(timeline[0]["pred"])}

    def frame(i):
        f = timeline[i]
        ax_pred.clear()
        ax_true.clear()
        state["pose"] = state["pose"].lerp(pose_for(f["pred"]), 0.45)
        hit = f["true"] == f["pred"]
        draw_hand(ax_pred, state["pose"],
                  label=f"decoded: {f['pred'].replace('_', ' ')}  ({f['confidence']:.2f})")
        draw_hand(ax_true, pose_for(f["true"]),
                  label=f"cued: {f['true'].replace('_', ' ')}")
        ax_pred.set_title(ax_pred.get_title(),
                          color="#15803d" if hit else "#b91c1c", fontsize=10)
        ax_true.set_title(ax_true.get_title(), color="#334155", fontsize=10)
        marker.set_xdata([f["t"], f["t"]])
        return []

    fig.suptitle(
        f"DB3 S{subject} - decoded from EMG with '{condition}' vs the cued movement\n"
        f"held-out repetitions; {sum(t['true'] == t['pred'] for t in timeline)}"
        f"/{len(timeline)} frames correct",
        fontsize=10,
    )
    anim = FuncAnimation(fig, frame, frames=len(timeline), interval=1000 / fps, blit=False)
    anim.save(str(path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
