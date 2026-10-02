"""The side-by-side hand animation, from saved predictions only.

Two methods decoding the same held-out recording, frame for frame: the cued movement
beside the hand each method actually commanded, green when it matches and red when it
does not. Nothing is re-run and no EMG is read -- the input is the prediction table
written by scripts/dump_predictions.py, which contains labels only.

A note on the hand shapes. DB6's seven grasps are identified in the files only by the
sparse ids {1, 3, 4, 6, 9, 10, 11}, and this project has deliberately not assigned
them official names without a source -- it recorded `lateral_grasp` as the wrong id
once already, from exactly that kind of inference. So the pose each id is drawn with
here is a consistent, distinguishable placeholder, not a claim about which grasp it
is, and the figure says so on its face. What the animation is actually about -- does
the decoded hand match the cued one -- does not depend on the shapes being the right
grasps, only on their being told apart.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Placeholder poses, chosen for maximum visual separation rather than for
# resemblance to any grasp. Since the id-to-grasp mapping is unverified (see the
# module docstring) no shape here can be correct, so the only thing a shape can
# usefully do is be unmistakably different from the other seven. The grasp poses
# were tried first and rejected: medium wrap, tripod and large diameter render as
# near-identical fists, so a viewer could not see the decoded hand change at all.
DB6_POSE = {
    "rest": "rest",
    "db6_movement_1": "close_hand",
    "db6_movement_3": "open_hand",
    "db6_movement_4": "point_index",
    "db6_movement_6": "wrist_flexion",
    "db6_movement_9": "wrist_extension",
    "db6_movement_10": "wrist_supination",
    "db6_movement_11": "wrist_pronation",
}
SHORT = {k: ("rest" if k == "rest" else f"movement {k.split('_')[-1]}")
         for k in DB6_POSE}

OK_GREEN, BAD_RED = "#1f8f5f", "#d4351c"
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def pick_window(y_true: np.ndarray, n: int) -> int:
    """The stretch showing the most distinct movements, so the clip is informative."""
    best, best_score = 0, -1.0
    for s in range(0, max(1, len(y_true) - n), 5):
        seg = y_true[s:s + n]
        if len(seg) < n:
            break
        distinct = len(np.unique(seg))
        rest_frac = float((seg == 0).mean())
        # Want several movements and some rest, not a single sustained grasp.
        score = distinct + (1.0 if 0.1 < rest_frac < 0.6 else 0.0)
        if score > best_score:
            best, best_score = s, score
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path, default=Path("results/visuals/preds_db6.csv"))
    ap.add_argument("--summary", type=Path,
                    default=Path("results/visuals/preds_db6_summary.csv"))
    ap.add_argument("--subject", type=int, default=2)
    ap.add_argument("--day", default="day5")
    ap.add_argument("--left", default="finetune")
    ap.add_argument("--right", default="rapid")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--classes", type=Path,
                    default=Path("results/visuals/preds_db6_meta.json"))
    ap.add_argument("--out", type=Path, default=Path("results/visuals/day5_hands.gif"))
    args = ap.parse_args()

    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from remg.viz.hand import POSES, pose_for
    from remg.viz.render import draw_hand

    classes = json.loads(args.classes.read_text())["classes"]
    d = pd.read_csv(args.preds)
    d = d[(d.subject == args.subject) & (d.day == args.day)].sort_values("start")
    d = d.reset_index(drop=True)
    acc = pd.read_csv(args.summary)
    acc = acc[(acc.subject == args.subject) & (acc.day == args.day)]
    bal = dict(zip(acc.method, acc.balanced_accuracy))

    n = int(round(args.seconds * args.fps))
    start = pick_window(d.y_true.to_numpy(), n)
    seg = d.iloc[start:start + n].reset_index(drop=True)
    print(f"S{args.subject} {args.day}: frames {len(seg)} from window {start}")

    sides = [(args.left, f"pred_{args.left}"), (args.right, f"pred_{args.right}")]
    for _, col in sides:
        if col not in seg.columns:
            raise SystemExit(f"{col} not in {args.preds}")

    frames = []
    # One pose object per side, blended towards the target so the hand moves rather
    # than snapping between frames.
    cur = {m: POSES["rest"] for m, _ in sides}
    cur_cue = POSES["rest"]
    for i in range(len(seg)):
        true_name = classes[int(seg.y_true[i])]
        cur_cue = cur_cue.lerp(pose_for(DB6_POSE[true_name]), 0.45)
        fig, axes = plt.subplots(1, 4, figsize=(9.6, 4.9), dpi=100)
        fig.patch.set_facecolor(SURFACE)
        for k, (meth, col) in enumerate(sides):
            pred_name = classes[int(seg[col][i])]
            correct = pred_name == true_name
            cur[meth] = cur[meth].lerp(pose_for(DB6_POSE[pred_name]), 0.45)
            ax_cue, ax_dec = axes[2 * k], axes[2 * k + 1]
            for ax in (ax_cue, ax_dec):
                ax.set_facecolor(SURFACE)
            draw_hand(ax_cue, cur_cue, color=None, linewidth=6.0)
            draw_hand(ax_dec, cur[meth], color=OK_GREEN if correct else BAD_RED,
                      linewidth=6.0)
            ax_cue.set_title("cued", fontsize=10, color=INK_2)
            ax_dec.set_title("decoded", fontsize=10,
                             color=OK_GREEN if correct else BAD_RED)
            x0 = 0.055 + 0.5 * k
            fig.text(x0, 0.955, {"finetune": "standard fine-tuning",
                                 "rapid": "rapid personalization"}.get(meth, meth),
                     fontsize=13, color=INK, va="top")
            fig.text(x0, 0.895, f"Day 5   ·   balanced accuracy "
                                f"{bal.get(meth, float('nan')):.3f}",
                     fontsize=10.5, color=INK_2, va="top")
            fig.text(x0, 0.255, SHORT[true_name], fontsize=10.5, color=INK_2, va="top")
            fig.text(x0 + 0.215, 0.255, SHORT[pred_name], fontsize=10.5,
                     color=OK_GREEN if correct else BAD_RED, va="top")
        fig.text(0.5, 0.155,
                 "NinaPro DB6 · calibrated on day 1 from 3 repetitions\n"
                 "DB6's seven grasps are named in Palermo et al., ICORR 2017, but no "
                 "source states which id is which —\nso these hand shapes are "
                 "arbitrary, distinguishable placeholders, not grasp identities",
                 fontsize=8.5, color=INK_2, ha="center", va="top", linespacing=1.5)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.33, wspace=0.05)
        fig.canvas.draw()
        frames.append(Image.fromarray(
            np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
        plt.close(fig)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    pal[0].save(args.out, save_all=True, append_images=pal[1:],
                duration=int(1000 / args.fps), loop=0, optimize=True)
    mb = args.out.stat().st_size / 1024 / 1024
    print(f"wrote {args.out}  {len(frames)} frames  {mb:.2f} MB")
    if mb > 5:
        print("WARNING: over the 5 MB budget")


if __name__ == "__main__":
    main()
