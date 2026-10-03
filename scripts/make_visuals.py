"""Figures for sharing: the cross-session slope chart and the rest timeline.

Reads only saved result tables and saved predictions. **No EMG is read or drawn.**
NinaPro publishes a citation requirement and no redistribution licence, so nothing
here contains, plots, or allows reconstruction of a signal -- only model predictions,
cued labels, and aggregate metrics.

Colour: the four categorical slots already validated for this project are kept in
their fixed order for the four neural conditions. `td_rf` is drawn in neutral grey
with a dashed line rather than a fifth hue -- a neutral is not a categorical slot, so
this needs no new validation, and it carries meaning: td_rf is the one method that is
not a neural network. Every series is labelled directly as well, so identity never
rests on colour alone.

    uv run python scripts/make_visuals.py --figure slope
    uv run python scripts/make_visuals.py --figure timeline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLORS = {
    "none": "#2a78d6",
    "linear_probe": "#eb6834",
    "finetune": "#1baf7a",
    "rapid": "#eda100",
    "td_rf_fewshot": "#52514e",      # neutral, not a categorical slot
}
STYLE = {k: "-" for k in COLORS} | {"td_rf_fewshot": "--"}
LABEL = {
    "none": "no personalization",
    "td_rf_fewshot": "classic features (random forest)",
    "linear_probe": "linear probe",
    "finetune": "standard fine-tuning",
    "rapid": "rapid personalization",
}
ORDER = ["none", "td_rf_fewshot", "linear_probe", "finetune", "rapid"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e6e3", "#fcfcfb"


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    mu = x[idx].mean(axis=1)
    return float(np.percentile(mu, 2.5)), float(np.percentile(mu, 97.5))


def _end_labels(ax, items, x, dy=0.012, side="left", pad=0.045):
    """Place labels at one x, nudged apart so they never overlap.

    Direct labels are the whole reason this chart needs no legend, so two of them
    printing on top of each other is not cosmetic -- it destroys the one thing that
    makes the series identifiable without relying on colour.
    """
    items = sorted(items, key=lambda t: t[0])
    ys = [y for y, _, _ in items]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < dy:
            ys[i] = ys[i - 1] + dy
    ha = "left" if side == "left" else "right"
    dx = pad if side == "left" else -pad
    for (orig, text, color), y in zip(items, ys):
        ax.annotate(text, xy=(x, orig), xytext=(x + dx, y),
                    color=color, fontsize=11.5, va="center", ha=ha,
                    annotation_clip=False)


def figure_slope(rows: Path, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pd.read_csv(rows)
    d = d[(d.stage == "single") & (d.shots == 3)]
    p = d.pivot_table(index=["seed", "subject", "condition"], columns="evaluation",
                      values="balanced_accuracy").reset_index()

    fig, ax = plt.subplots(figsize=(12.0, 6.75), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    labels, starts = [], []
    for cond in ORDER:
        q = p[p.condition == cond]
        if q.empty:
            continue
        m1, m5 = q.day1.mean(), q.day5.mean()
        lo1, hi1 = boot_ci(q.day1.values)
        lo5, hi5 = boot_ci(q.day5.values)
        c = COLORS[cond]
        ax.plot([0, 1], [m1, m5], STYLE[cond], color=c, linewidth=2.0,
                marker="o", markersize=9, markeredgecolor=SURFACE,
                markeredgewidth=2.0, zorder=3)
        for x, lo, hi in ((0, lo1, hi1), (1, lo5, hi5)):
            ax.plot([x, x], [lo, hi], color=c, linewidth=2.0, alpha=0.45,
                    solid_capstyle="butt", zorder=2)
        labels.append((m5, f"{LABEL[cond]}  {m5:.3f}", c))
        starts.append((m1, f"{m1:.3f}", c))

    _end_labels(ax, labels, 1.0)
    _end_labels(ax, starts, 0.0, side="right")

    ax.set_xlim(-0.30, 1.72)
    lo = min(p.day5.min(), p.day1.min())
    ax.set_ylim(max(0.0, 0.22), 0.47)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Day 1\n(same session as calibration)",
                        "Day 5\n(electrodes re-donned)"], fontsize=12, color=INK)
    ax.set_ylabel("Balanced accuracy  (8 movements, chance = 0.125)",
                  fontsize=11.5, color=INK_2)
    ax.tick_params(axis="y", labelcolor=INK_2, labelsize=10.5)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    fig.text(0.062, 0.945, "Accuracy from day 1 to day 5 after a single calibration",
             fontsize=19, color=INK, va="top")
    fig.text(0.062, 0.878,
             "NinaPro DB6 · 10 intact subjects · 3 calibration repetitions on day 1 · "
             "leave-one-subject-out · 3 seeds (n=30) · bars are bootstrap 95% CIs",
             fontsize=10.5, color=INK_2, va="top")
    fig.text(0.062, 0.085,
             "Fitting every weight to one session degrades most.",
             fontsize=12, color=INK, va="top")
    fig.text(0.062, 0.045,
             "Touching fewer parameters retains better — and a plain linear probe "
             "retains as well as the adapter method.",
             fontsize=12, color=INK_2, va="top")
    fig.subplots_adjust(left=0.115, right=0.995, top=0.815, bottom=0.185)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def figure_timeline(preds: Path, out: Path, seconds: float = 60.0,
                    stride_ms: float = 100.0, subject: int | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pd.read_csv(preds)
    if subject is not None:
        d = d[d.subject == subject]
    subject = int(d.subject.iloc[0])
    d = d.sort_values("start").reset_index(drop=True)

    n = int(round(seconds * 1000.0 / stride_ms))
    # Pick a stretch that actually shows the contrast: plenty of rest, but real
    # movements too. Maximising rest alone produces a near-empty band in which the
    # reader cannot see that the model is also being asked to do something.
    rest = (d.y_true == 0).to_numpy()
    if len(d) > n:
        best, best_score = 0, -1.0
        for st in range(0, len(d) - n, 5):
            seg_rest = rest[st:st + n]
            frac = float(seg_rest.mean())
            moves = len(np.unique(d.y_true.to_numpy()[st:st + n]))
            # Want roughly half rest and several distinct movements.
            score = moves - 8.0 * abs(frac - 0.5)
            if score > best_score:
                best, best_score = st, score
        start = best
    else:
        start = 0
    seg = d.iloc[start:start + n].reset_index(drop=True)
    t = np.arange(len(seg)) * stride_ms / 1000.0

    panes = [("pred_finetune", "standard fine-tuning"),
             ("pred_finetune_classic_gate", "standard fine-tuning + classic rest gate")]
    panes = [(c, lab) for c, lab in panes if c in seg.columns]

    fig, axes = plt.subplots(len(panes), 1, figsize=(12.0, 5.2), dpi=100, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    if len(panes) == 1:
        axes = [axes]

    true_rest = (seg.y_true == 0).to_numpy()
    for ax, (col, lab) in zip(axes, panes):
        ax.set_facecolor(SURFACE)
        pred_move = (seg[col] != 0).to_numpy()
        # true-rest intervals as a calm band
        for s, e in _runs(true_rest):
            ax.axvspan(t[s], t[e] + stride_ms / 1000.0, color="#f1f3f0",
                       linewidth=0, zorder=1)
        # A real movement is a clearly distinct neutral band -- the reader has to be
        # able to see that the model is also being asked to do something, or the
        # figure looks like a model misbehaving in a vacuum.
        for s, e in _runs(~true_rest):
            ax.axvspan(t[s], t[e] + stride_ms / 1000.0, color="#c3cedd",
                       linewidth=0, zorder=1)
        bad = true_rest & pred_move
        for s, e in _runs(bad):
            ax.axvspan(t[s], t[e] + stride_ms / 1000.0, color="#d4351c",
                       alpha=0.95, linewidth=0, zorder=3)
        n_events = len(_runs(bad))
        rest_minutes = true_rest.sum() * stride_ms / 1000.0 / 60.0
        rate = n_events / rest_minutes if rest_minutes > 0 else float("nan")
        ax.set_ylim(0, 1); ax.set_yticks([])
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        # Both labels sit *above* the band. Printed inside it they land on top of
        # the red marks, which are the one thing the figure exists to show.
        ax.text(0.0, 1.62, lab, transform=ax.transAxes, fontsize=13,
                color=INK, va="top")
        ax.text(1.0, 1.62, f"{rate:.1f} false activations per minute of rest",
                transform=ax.transAxes, fontsize=11.5, color="#a32b17",
                va="top", ha="right")
    axes[-1].set_xlabel("seconds", fontsize=11.5, color=INK_2)
    axes[-1].tick_params(axis="x", labelcolor=INK_2, labelsize=10.5)
    axes[-1].set_xlim(0, t[-1] + stride_ms / 1000.0)

    fig.suptitle("Unwanted movement while the user is at rest",
                 fontsize=19, color=INK, x=0.045, ha="left", y=0.985)
    fig.text(0.045, 0.905,
             f"NinaPro DB3 amputee participant · 3 calibration repetitions · "
             f"{seconds:.0f} s of held-out recording",
             fontsize=11, color=INK_2, va="top")
    fig.text(0.045, 0.862,
             "pale = user at rest   ·   blue-grey = a real movement   ·   "
             "red = the hand moved when it should have been still",
             fontsize=10.5, color=INK_2, va="top")
    fig.text(0.045, 0.035, "The gate removes most unwanted activations.",
             fontsize=11.5, color=INK)
    # Generous side margins: at 9 px of slack the right-hand labels read as clipped
    # once the image is scaled down to the page width.
    fig.subplots_adjust(left=0.045, right=0.955, top=0.70, bottom=0.17, hspace=0.95)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of every True run."""
    mask = np.asarray(mask)
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1))
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask) - 1]
    return list(zip(starts, ends))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", required=True, choices=["slope", "timeline", "all"])
    ap.add_argument("--rows", type=Path, default=Path("results/xs10_merged_rows.csv"))
    ap.add_argument("--preds", type=Path, default=Path("results/visuals/preds_db3.csv"))
    ap.add_argument("--subject", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", type=Path, default=Path("results/visuals"))
    args = ap.parse_args()

    if args.figure in ("slope", "all"):
        p = figure_slope(args.rows, args.out / "day1_to_day5.png")
        print(f"wrote {p}  ({p.stat().st_size / 1024:.0f} KB)")
    if args.figure in ("timeline", "all"):
        p = figure_timeline(args.preds, args.out / "rest_timeline.png",
                            seconds=args.seconds, subject=args.subject)
        print(f"wrote {p}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
