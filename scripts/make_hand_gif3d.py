"""Composite the 3D-rendered hands into the card layout, day 1 beside day 5.

Same model (standard fine-tuning), same subject, same single calibration from day 1 --
only the recording differs. That is the comparison the project's headline rests on, so
it is the one worth animating.

Window selection rule, fixed in advance and stated on the figure: for each day
independently, take the 10-second window containing the most distinct *intended*
grasps, ties broken by the earliest start. It reads only the cued labels and never the
predictions, so it cannot favour either day or either method.

Hands are rendered by three.js in headless Chromium (scripts/render_hands.py); this
module only arranges them. Frames are written to disk as they are produced and the
browser page is recycled between batches, because this host has stalled on memory
repeatedly and a page accumulating hundreds of framebuffers is that same shape of
workload.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from render_hands import BAD, GOOD, NEUTRAL, ORDER, POSES, HandRenderer, lerp_pose, static_server

BG = "#f7f7f5"
CARD, CARD_EDGE = "#ffffff", "#e4e4e0"
INK, INK_2, INK_3 = "#14140f", "#6b6a64", "#9b9a94"
GOOD_EDGE, BAD_EDGE = "#3f8f68", "#c4553f"
STRIP_IDLE = "#ececea"

WINDOW_RULE = ("10-second window with the most distinct intended grasps, earliest start "
               "wins ties — chosen from the cued labels only, never the predictions")


def pick_window(y_true: np.ndarray, n: int) -> int:
    """Most distinct intended grasps; earliest start breaks ties. Cues only."""
    best, best_n = 0, -1
    for s in range(0, max(1, len(y_true) - n + 1)):
        d = len(np.unique(y_true[s:s + n]))
        if d > best_n:
            best, best_n = s, d
    return best


def load_side(preds: Path, summary: Path, classes: list[str], subject: int, day: str,
              method: str, windows: int) -> dict:
    letters = {}
    g = iter(ORDER[1:])
    for i, c in enumerate(classes):
        letters[i] = "rest" if c == "rest" else next(g)
    d = pd.read_csv(preds)
    d = d[(d.subject == subject) & (d.day == day)].sort_values("start").reset_index(drop=True)
    start = pick_window(d.y_true.to_numpy(), windows)
    seg = d.iloc[start:start + windows].reset_index(drop=True)
    acc = pd.read_csv(summary)
    acc = acc[(acc.subject == subject) & (acc.day == day) & (acc.method == method)]
    return {
        "true": [letters[int(v)] for v in seg.y_true],
        "pred": [letters[int(v)] for v in seg[f"pred_{method}"]],
        "acc": float(acc.balanced_accuracy.iloc[0]) if len(acc) else float("nan"),
        "start_window": int(start),
        "distinct_cues": int(len(set(letters[int(v)] for v in seg.y_true))),
    }


def compose(frame: int, left: dict, right: dict, imgs: dict, fps: int, windows: int,
            subject: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    per_window = fps * 0.1
    pos = frame / per_window
    w0 = min(int(pos), windows - 1)

    fig = plt.figure(figsize=(12.0, 6.75), dpi=100)
    fig.patch.set_facecolor(BG)
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")

    bg.text(0.042, 0.936, "Same model, day 1 vs day 5", fontsize=26, color=INK,
            va="center")
    bg.text(0.042, 0.880,
            "Standard fine-tuning, calibrated once on day 1 from three repetitions. "
            "Nothing is re-calibrated in between.",
            fontsize=12.5, color=INK_2, va="center")

    for k, (side, title, sub) in enumerate((
            (left, "Day 1", "same session as the calibration"),
            (right, "Day 5", "sensors taken off and put back on"))):
        x0 = 0.042 + k * 0.479
        cw, ch, y0 = 0.437, 0.600, 0.190
        bg.add_patch(FancyBboxPatch((x0, y0), cw, ch,
                                    boxstyle="round,pad=0,rounding_size=0.018",
                                    linewidth=1.3, edgecolor=CARD_EDGE,
                                    facecolor=CARD, zorder=1))
        bg.text(x0 + 0.026, y0 + ch - 0.050, title, fontsize=17, color=INK,
                va="center", zorder=3)
        bg.text(x0 + 0.026, y0 + ch - 0.088, sub, fontsize=11, color=INK_3,
                va="center", zorder=3)
        bg.text(x0 + cw - 0.026, y0 + ch - 0.050, f"{side['acc']:.3f}", fontsize=17,
                color=INK, va="center", ha="right", zorder=3)
        bg.text(x0 + cw - 0.026, y0 + ch - 0.088, "balanced accuracy", fontsize=10.5,
                color=INK_3, va="center", ha="right", zorder=3)

        t_now, p_now = side["true"][w0], side["pred"][w0]
        ok = t_now == p_now
        edge = GOOD_EDGE if ok else BAD_EDGE
        lab = lambda s: "rest" if s == "rest" else f"Grasp {s}"       # noqa: E731
        bg.text(x0 + 0.112, y0 + ch - 0.138, f"Intended:  {lab(t_now)}", fontsize=12,
                color=INK_2, va="center", ha="center", zorder=3)
        bg.text(x0 + 0.325, y0 + ch - 0.138, f"Decoded:  {lab(p_now)}", fontsize=12,
                color=edge, va="center", ha="center", zorder=3)

        for j, key in enumerate(("true", "pred")):
            ax = fig.add_axes([x0 + 0.008 + j * 0.213, y0 + 0.040, 0.212, 0.395])
            ax.axis("off"); ax.set_facecolor(CARD)
            ax.imshow(imgs[(k, key)])

        sx, sw, sy, sh = x0 + 0.026, cw - 0.052, y0 - 0.055, 0.020
        bg.add_patch(Rectangle((sx, sy), sw, sh, facecolor=STRIP_IDLE,
                               edgecolor="none", zorder=2))
        cell = sw / windows
        for i in range(w0 + 1):
            c = GOOD_EDGE if side["true"][i] == side["pred"][i] else BAD_EDGE
            bg.add_patch(Rectangle((sx + i * cell, sy), cell * 1.04, sh,
                                   facecolor=c, edgecolor="none", zorder=3))
        bg.add_patch(Rectangle((sx + (w0 + 1) * cell - 0.0014, sy - 0.004), 0.0028,
                               sh + 0.008, facecolor=INK, edgecolor="none", zorder=4))
        bg.text(sx, sy - 0.032, "correct", fontsize=10, color=GOOD_EDGE, va="center")
        bg.text(sx + 0.058, sy - 0.032, "wrong", fontsize=10, color=BAD_EDGE, va="center")
        bg.text(sx + sw, sy - 0.032, f"{w0 + 1} / {windows} decisions", fontsize=10,
                color=INK_3, va="center", ha="right")

    bg.text(0.042, 0.062, f"NinaPro DB6 subject S{subject}  ·  window: {WINDOW_RULE}",
            fontsize=9.5, color=INK_3, va="center")
    bg.text(0.042, 0.036,
            "Grasps labelled A–G — the seven are named in Palermo et al., ICORR 2017, "
            "but no source states which recorded id is which.",
            fontsize=9, color=INK_3, va="center")
    bg.text(0.042, 0.014,
            "Hand model: WebXR Input Profiles, W3C Software and Document License  ·  "
            "rendered with three.js, MIT",
            fontsize=9, color=INK_3, va="center")
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", type=Path, default=Path("results/visuals/preds_db6.csv"))
    ap.add_argument("--summary", type=Path,
                    default=Path("results/visuals/preds_db6_summary.csv"))
    ap.add_argument("--meta", type=Path,
                    default=Path("results/visuals/preds_db6_meta.json"))
    ap.add_argument("--subject", type=int, default=2)
    ap.add_argument("--method", default="finetune")
    ap.add_argument("--windows", type=int, default=100)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--slowdown-factor", type=float, default=3.0)
    ap.add_argument("--still", type=int, default=None)
    ap.add_argument("--frames-dir", type=Path, default=Path("assets/render/frames"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/visuals/day1_vs_day5_hands.gif"))
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    classes = json.loads(args.meta.read_text())["classes"]
    left = load_side(args.preds, args.summary, classes, args.subject, "day1",
                     args.method, args.windows)
    right = load_side(args.preds, args.summary, classes, args.subject, "day5",
                      args.method, args.windows)
    print(f"day1: start window {left['start_window']}, {left['distinct_cues']} distinct cues,"
          f" acc {left['acc']:.3f}")
    print(f"day5: start window {right['start_window']}, {right['distinct_cues']} distinct cues,"
          f" acc {right['acc']:.3f}")

    n_frames = int(round(args.windows * 0.1 * args.fps))
    per_window = args.fps * 0.1
    args.frames_dir.mkdir(parents=True, exist_ok=True)

    def poses_for(frame: int):
        pos = frame / per_window
        w0 = min(int(pos), args.windows - 1)
        w1 = min(w0 + 1, args.windows - 1)
        raw = float(np.clip(pos - w0, 0.0, 1.0))
        u = float(np.clip((raw - 0.50) / 0.50, 0.0, 1.0))
        t = u * u * (3.0 - 2.0 * u)      # hold, then move, so hand and label agree
        out = {}
        for k, side in ((0, left), (1, right)):
            for key in ("true", "pred"):
                a, b = POSES[side[key][w0]], POSES[side[key][w1]]
                out[(k, key)] = lerp_pose(a, b, t)
        ok = {k: (side["true"][w0] == side["pred"][w0])
              for k, side in ((0, left), (1, right))}
        return out, ok

    frames = [args.still] if args.still is not None else list(range(n_frames))
    with static_server(Path("assets")) as origin:
        url = f"{origin}/render/hand.html"
        r = HandRenderer(url)
        times: list[float] = []
        baseline = None
        try:
            r.frame_all([POSES[k] for k in ORDER])
            for bi in range(0, len(frames), args.batch):
                chunk = frames[bi:bi + args.batch]
                for f in chunk:
                    t0 = time.time()
                    poses, ok = poses_for(f)
                    imgs = {}
                    for k in (0, 1):
                        for key in ("true", "pred"):
                            tint = NEUTRAL if key == "true" else (
                                GOOD if ok[k] else BAD)
                            imgs[(k, key)] = Image.open(
                                io.BytesIO(r.render(poses[(k, key)], tint)))
                    fig = compose(f, left, right, imgs, args.fps, args.windows,
                                  args.subject)
                    if args.still is not None:
                        p = args.out.with_name(f"still3d_{f:04d}.png")
                        fig.savefig(p, facecolor=BG); plt.close(fig)
                        print(f"wrote {p} ({p.stat().st_size/1024:.0f} KB)")
                        return
                    fig.savefig(args.frames_dir / f"f{f:04d}.png", facecolor=BG)
                    plt.close(fig)
                    dt = time.time() - t0
                    times.append(dt)
                    if baseline is None and len(times) == 5:
                        baseline = float(np.median(times))
                        print(f"  baseline {baseline:.2f} s/frame")
                    if baseline and dt > baseline * args.slowdown_factor:
                        raise SystemExit(
                            f"STOPPING: frame {f} took {dt:.1f}s against a baseline of "
                            f"{baseline:.2f}s ({dt/baseline:.1f}x). The host is "
                            f"degrading; {len(times)} frames are written to "
                            f"{args.frames_dir} and the run can be resumed."
                        )
                done = min(bi + args.batch, len(frames))
                print(f"  {done}/{len(frames)} frames  "
                      f"{np.median(times):.2f}s/frame median", flush=True)
                r.recycle()        # drop the page's buffers between batches
        finally:
            r.close()

    imgs = [Image.open(args.frames_dir / f"f{f:04d}.png").convert("RGB")
            for f in range(n_frames)]
    pal = [im.convert("P", palette=Image.ADAPTIVE, colors=128) for im in imgs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pal[0].save(args.out, save_all=True, append_images=pal[1:],
                duration=int(round(1000 / args.fps)), loop=0, optimize=True)
    mb = args.out.stat().st_size / 1024 / 1024
    print(f"wrote {args.out}  {n_frames} frames @ {args.fps} fps  "
          f"{n_frames/args.fps:.1f}s  {mb:.2f} MB")


if __name__ == "__main__":
    main()
