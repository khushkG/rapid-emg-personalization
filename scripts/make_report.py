"""Turn a run's CSVs into the tables and figures for the writeup.

    uv run python scripts/make_report.py --tag robust15

Reads `results/<tag>_rows.csv` (and `_curves.csv` if present) and writes
`results/<tag>_report.md` plus PNG figures beside it. Sections whose inputs are
absent are skipped with a line saying so, rather than silently omitted -- a
missing sensor-failure sweep should be visible in the report, not inferred from
its absence.

Every comparison is reported whichever way it falls. The tables carry the
numbers; the figures are there to make the shape obvious, not to make a case.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Categorical slots 1-4 of the reference palette, in fixed order, validated for
# this set: worst adjacent CVD deltaE 9.1, normal-vision 22.9, both above floor.
# Aqua and yellow fall below 3:1 against the light surface, so every figure
# carries direct labels and the report carries the table -- the relief the
# contrast WARN requires.
COLORS = {
    "none": "#2a78d6",           # blue
    "linear_probe": "#eb6834",   # orange
    "finetune": "#1baf7a",       # aqua
    "rapid": "#eda100",          # yellow
}
LABEL = {
    "none": "no personalization",
    "linear_probe": "linear probe",
    "finetune": "standard fine-tuning",
    "rapid": "rapid personalization",
}
ORDER = ["none", "linear_probe", "finetune", "rapid"]

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e6e3"
SURFACE = "#fcfcfb"


def _style(ax):
    """Recessive grid and axes; the data should be the only assertive thing."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_color(INK_2)


def _end_labels(ax, items, min_gap):
    """Label each series at its end, nudged apart so near-equal values stay legible.

    Two conditions landing on the same value is a finding here, not a nuisance,
    so the labels have to survive it rather than overprint into mush.
    """
    items = sorted(items, key=lambda t: t[1])          # by y
    placed = []
    for x, y, text, color in items:
        y_lab = y
        if placed and y_lab - placed[-1] < min_gap:
            y_lab = placed[-1] + min_gap
        placed.append(y_lab)
        ax.annotate(text, xy=(x, y), xytext=(8, 0), textcoords="offset points",
                    xycoords="data", va="center", ha="left", fontsize=9,
                    color=INK_2,
                    annotation_clip=False)
        if abs(y_lab - y) > 1e-9:                      # moved: re-place explicitly
            ax.texts[-1].remove()
            ax.annotate(text, xy=(x, y_lab), xytext=(8, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=9, color=INK_2, annotation_clip=False)


def fig_shots(clean: pd.DataFrame, out: Path, chance: float) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.4), facecolor=SURFACE)
    shots = sorted(clean.shots.unique())
    ends = []
    for cond in [c for c in ORDER if c in set(clean.condition)]:
        g = clean[clean.condition == cond].groupby("shots").balanced_accuracy
        m = g.mean()
        # Standard error of the cohort mean, not the between-subject spread:
        # the question these bars serve is "is the mean where it looks", and an
        # SD band here is wide enough to imply the conditions are inseparable
        # when the paired test says otherwise.
        se = g.std() / np.sqrt(g.count())
        ax.errorbar(shots, m.values, yerr=se.values, fmt="-o", color=COLORS[cond],
                    linewidth=2.0, markersize=8, capsize=3, elinewidth=1.2,
                    label=LABEL[cond], zorder=3)
        ends.append((shots[-1], m.values[-1], f"{m.values[-1]:.3f}", COLORS[cond]))

    ax.axhline(chance, color=INK_2, linestyle=":", linewidth=1.2, zorder=1)
    ax.annotate("chance", xy=(shots[0], chance), xytext=(0, 5),
                textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_xticks(shots)
    ax.set_xlabel("calibration repetitions per movement", fontsize=10, color=INK_2)
    ax.set_ylabel("balanced accuracy", fontsize=10, color=INK_2)
    ax.set_xlim(min(shots) - 0.12, max(shots) + 0.62)
    top = max(0.5, float(clean.groupby(["condition", "shots"]).balanced_accuracy
                         .mean().max()) * 1.35)
    ax.set_ylim(0, top)
    _end_labels(ax, ends, min_gap=top * 0.045)
    ax.set_title("Personalization from a handful of repetitions",
                 fontsize=12, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_failure(df: pd.DataFrame, out: Path) -> Path | None:
    """Accuracy as electrodes fail. Worst case matters more than the mean."""
    import matplotlib.pyplot as plt

    fails = df[df.evaluation.str.contains("ch", na=False)]
    if fails.empty:
        return None
    shots = fails.shots.max()
    clean = df[(df.evaluation == "clean") & (df.shots == shots)]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), facecolor=SURFACE, sharey=True)
    for ax, col, title in (
        (axes[0], "balanced_accuracy", "mean over electrode combinations"),
        (axes[1], "balanced_accuracy_worst", "worst single combination"),
    ):
        ends = []
        for cond in [c for c in ORDER if c in set(fails.condition)]:
            xs, ys = [0], [clean[clean.condition == cond].balanced_accuracy.mean()]
            for n in (1, 2):
                sub = fails[(fails.condition == cond) & (fails.shots == shots)
                            & (fails.evaluation == f"drop_{n}ch")]
                if sub.empty or col not in sub:
                    continue
                xs.append(n)
                ys.append(sub[col].mean())
            ax.plot(xs, ys, "-o", color=COLORS[cond], linewidth=2.0, markersize=8,
                    label=LABEL[cond], zorder=3)
            ends.append((xs[-1], ys[-1], f"{ys[-1]:.3f}", COLORS[cond]))
        _end_labels(ax, ends, min_gap=0.022)
        ax.set_xticks([0, 1, 2])
        ax.set_xlabel("electrodes failed", fontsize=10, color=INK_2)
        ax.set_title(title, fontsize=10, color=INK_2, loc="left")
        ax.set_xlim(-0.1, 2.5)
        _style(ax)
    axes[0].set_ylabel("balanced accuracy", fontsize=10, color=INK_2)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower left")
    fig.suptitle(f"Robustness to electrode failure ({shots} calibration repetitions)",
                 fontsize=12, color=INK, x=0.012, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_rejection(curves: pd.DataFrame, out: Path) -> Path | None:
    """Coverage against accuracy when the controller may decline to act.

    Only the points where every class survives the threshold are drawn: past
    that, balanced accuracy is averaging over a smaller class set and is no
    longer comparable to the full-coverage point.
    """
    import matplotlib.pyplot as plt

    if curves is None or curves.empty or "comparable" not in curves:
        return None
    ok = curves[curves.comparable.astype(bool)]
    if ok.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.6, 4.4), facecolor=SURFACE)
    rej_ends = []
    shots = ok.shots.max()
    for cond in [c for c in ORDER if c in set(ok.condition)]:
        g = (ok[(ok.condition == cond) & (ok.shots == shots)]
             .groupby("threshold")[["coverage", "balanced_accuracy"]].mean()
             .sort_values("coverage"))
        if g.empty:
            continue
        ax.plot(g.coverage, g.balanced_accuracy, "-o", color=COLORS[cond],
                linewidth=2.0, markersize=7, label=LABEL[cond], zorder=3)
        rej_ends.append((g.coverage.iloc[0], g.balanced_accuracy.iloc[0],
                         f"{g.balanced_accuracy.iloc[0]:.2f}", COLORS[cond]))
    _end_labels(ax, rej_ends, min_gap=0.025)
    ax.set_xlabel("coverage (fraction of windows acted on)", fontsize=10, color=INK_2)
    ax.set_ylabel("balanced accuracy on those windows", fontsize=10, color=INK_2)
    ax.set_title("Declining to act when unsure", fontsize=12, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_per_class(clean: pd.DataFrame, out: Path) -> Path:
    """Per-movement recall: where a headline number hides a collapse onto rest."""
    import matplotlib.pyplot as plt

    shots = clean.shots.max()
    sub = clean[clean.shots == shots]
    cols = [c for c in clean.columns if c.startswith("recall/")]
    names = [c.split("/", 1)[1] for c in cols]
    conds = [c for c in ORDER if c in set(sub.condition)]

    fig, ax = plt.subplots(figsize=(9.0, 5.6), facecolor=SURFACE)
    y = np.arange(len(names))
    h = 0.8 / len(conds)
    for i, cond in enumerate(conds):
        vals = [sub[sub.condition == cond][c].mean() for c in cols]
        ax.barh(y + i * h - 0.4 + h / 2, vals, height=h * 0.86, color=COLORS[cond],
                label=LABEL[cond], zorder=3, edgecolor=SURFACE, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("recall", fontsize=10, color=INK_2)
    ax.set_xlim(0, 1.0)
    ax.set_title(f"Per-movement recall ({shots} calibration repetitions)",
                 fontsize=12, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def table(clean: pd.DataFrame, metric: str) -> str:
    conds = [c for c in ORDER if c in set(clean.condition)]
    lines = ["| shots | " + " | ".join(LABEL[c] for c in conds) + " |",
             "| --- | " + " | ".join("---" for _ in conds) + " |"]
    for sh in sorted(clean.shots.unique()):
        s = clean[clean.shots == sh]
        cells = []
        for c in conds:
            v = s[s.condition == c][metric]
            cells.append(f"{v.mean():.3f} ± {v.std():.3f}")
        lines.append(f"| {sh} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def paired(clean: pd.DataFrame, a: str, b: str) -> str:
    from scipy import stats

    out = []
    for sh in sorted(clean.shots.unique()):
        p = (clean[clean.shots == sh]
             .pivot_table(index=["subject", "seed"], columns="condition",
                          values="balanced_accuracy"))
        if a not in p or b not in p:
            continue
        d = (p[a] - p[b]).dropna()
        if len(d) < 3:
            continue
        w = stats.wilcoxon(d) if d.abs().sum() > 0 else None
        ci = stats.bootstrap((d.values,), np.mean, confidence_level=0.95,
                             n_resamples=5000).confidence_interval
        out.append(f"| {sh} | {d.mean():+.3f} | [{ci.low:+.3f}, {ci.high:+.3f}] | "
                   f"{int((d > 0).sum())}/{len(d)} | "
                   f"{('%.2g' % w.pvalue) if w else 'n/a'} |")
    head = [f"| shots | {LABEL[a]} − {LABEL[b]} | 95% CI | wins | p |",
            "| --- | --- | --- | --- | --- |"]
    return "\n".join(head + out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    rows_path = args.results / f"{args.tag}_rows.csv"
    if not rows_path.exists():
        raise SystemExit(f"no such results file: {rows_path}")
    df = pd.read_csv(rows_path)
    clean = df[df.evaluation == "clean"]
    curves_path = args.results / f"{args.tag}_curves.csv"
    curves = pd.read_csv(curves_path) if curves_path.exists() else None

    n_classes = len([c for c in df.columns if c.startswith("recall/")])
    chance = 1.0 / max(n_classes, 1)
    subjects = sorted(clean.subject.unique())
    seeds = sorted(clean.seed.unique())

    figs = {}
    figs["shots"] = fig_shots(clean, args.results / f"{args.tag}_fig_shots.png", chance)
    figs["per_class"] = fig_per_class(clean, args.results / f"{args.tag}_fig_recall.png")
    figs["failure"] = fig_failure(df, args.results / f"{args.tag}_fig_failure.png")
    figs["rejection"] = fig_rejection(curves, args.results / f"{args.tag}_fig_rejection.png")

    md = [f"# Results: `{args.tag}`", ""]
    md += [f"{len(subjects)} amputees x {len(seeds)} seed(s) = "
           f"{len(subjects) * len(seeds)} runs per cell. {n_classes} classes, "
           f"chance = {chance:.3f}.", ""]
    md += ["Every comparison below is reported whichever way it falls.", ""]
    md += ["## Balanced accuracy", "", table(clean, "balanced_accuracy"), ""]
    md += ["## Macro F1", "", table(clean, "macro_f1"), ""]
    md += ["## Paired comparisons", ""]
    for a, b in [("finetune", "rapid"), ("rapid", "linear_probe"),
                 ("finetune", "linear_probe"), ("rapid", "none")]:
        if a in set(clean.condition) and b in set(clean.condition):
            md += [paired(clean, a, b), ""]

    md += ["## Figures", ""]
    for key, path in figs.items():
        if path is None:
            md += [f"- *{key}: not present in this run.*"]
        else:
            md += [f"![{key}]({Path(path).name})"]
    md += [""]

    if figs["failure"] is None:
        md += ["> The sensor-failure sweep was not enabled for this run.", ""]
    if figs["rejection"] is None:
        md += ["> No rejection curves in this run.", ""]

    out = args.results / f"{args.tag}_report.md"
    out.write_text("\n".join(md))
    print(f"wrote {out}")
    for k, v in figs.items():
        print(f"  {k}: {v if v else '(skipped)'}")


if __name__ == "__main__":
    main()
