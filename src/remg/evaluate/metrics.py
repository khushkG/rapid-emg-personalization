"""Scoring.

Balanced accuracy and macro F1 lead, plain accuracy is reported but never
headlined. The reason is in the data: rest occupies roughly six times as many
windows as any single movement, so a model that predicted rest for everything
would score around 40% plain accuracy while being useless as a controller.
Per-movement recall is kept because an average hides the failure mode that
matters -- one or two grasps being systematically unrecognised.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score


@dataclass
class Scores:
    balanced_accuracy: float
    macro_f1: float
    accuracy: float
    per_class_recall: dict[str, float]
    confusion: np.ndarray = field(repr=False)
    n: int

    def as_row(self) -> dict:
        row = {
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "accuracy": self.accuracy,
            "n": self.n,
        }
        row.update({f"recall/{k}": v for k, v in self.per_class_recall.items()})
        return row

    def worst_classes(self, k: int = 3) -> list[tuple[str, float]]:
        return sorted(self.per_class_recall.items(), key=lambda kv: kv[1])[:k]


def score(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> Scores:
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    support = cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(support > 0, np.diag(cm) / np.maximum(support, 1), np.nan)
    return Scores(
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        accuracy=float((y_true == y_pred).mean()) if len(y_true) else float("nan"),
        per_class_recall={n: float(r) for n, r in zip(class_names, recall)},
        confusion=cm,
        n=int(len(y_true)),
    )


def rejection_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[dict]:
    """Accuracy as a function of how much low-confidence output is suppressed.

    A prosthetic controller can decline to act: holding the current grip when the
    model is unsure is far better than executing the wrong grasp. This trades
    coverage (fraction of windows acted on) against balanced accuracy on the
    windows that were acted on, which is the operationally meaningful curve.

    One trap, which is why `classes_present` is reported alongside every point:
    balanced accuracy averages recall over the classes that appear in `y_true`,
    and raising the threshold removes whole classes from the surviving windows.
    A curve that climbs at high thresholds may simply be averaging over fewer,
    easier classes rather than getting better at the same task. Points are
    comparable to the full-coverage point only while `classes_present` equals
    the total, so plot the curve truncated where that stops holding, and read
    anything past it as a different question.
    """
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    n_classes = proba.shape[1]
    if thresholds is None:
        thresholds = np.linspace(0.0, 0.95, 20)
    rows = []
    for t in thresholds:
        keep = conf >= t
        n_kept = int(keep.sum())
        present = int(len(np.unique(y_true[keep]))) if n_kept else 0
        if n_kept == 0 or present < 2:
            # One class left makes balanced accuracy either 1.0 or 0.0 by
            # construction; reporting it as a number invites reading it as skill.
            rows.append({
                "threshold": float(t),
                "coverage": float(keep.mean()) if len(keep) else 0.0,
                "n_kept": n_kept,
                "classes_present": present,
                "n_classes": n_classes,
                "comparable": False,
                "balanced_accuracy": float("nan"),
                "accuracy": float((y_true[keep] == pred[keep]).mean()) if n_kept else float("nan"),
            })
            continue
        rows.append(
            {
                "threshold": float(t),
                "coverage": float(keep.mean()),
                "n_kept": n_kept,
                "classes_present": present,
                "n_classes": n_classes,
                "comparable": present == n_classes,
                "balanced_accuracy": float(balanced_accuracy_score(y_true[keep], pred[keep])),
                "accuracy": float((y_true[keep] == pred[keep]).mean()),
            }
        )
    return rows


def predictive_entropy(proba: np.ndarray) -> np.ndarray:
    """Per-window entropy in nats; the uncertainty signal used for rejection."""
    p = np.clip(proba, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def aggregate(rows: list[dict], by: str, metric: str) -> dict[str, tuple[float, float]]:
    """Mean and standard deviation of `metric`, grouped by `by`.

    Subject-level standard deviation is the number to report alongside the mean:
    with 11 amputee subjects, a two-point mean difference that is smaller than
    the between-subject spread is not yet evidence of anything.
    """
    groups: dict[str, list[float]] = {}
    for r in rows:
        if metric in r and r[metric] == r[metric]:  # skip NaN
            groups.setdefault(str(r[by]), []).append(float(r[metric]))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in groups.items()}


@dataclass
class RestScores:
    """How a controller behaves when the user is holding still.

    A prosthesis that misclassifies one grasp for another is annoying. One that
    moves when the user is at rest is unusable, and the headline metrics hide the
    difference: balanced accuracy counts rest as one class out of twelve, so a
    model can look good while spending half of every rest period commanding
    motion.

    Two rates, measuring different things:

    `false_activation_rate` is per window, and is exactly `1 - rest_recall`. It
    answers "what fraction of the time would the hand be moving when it should be
    still?".

    `false_activations_per_min` counts *events* -- maximal runs of consecutive
    movement predictions inside a true-rest stretch. It answers "how many separate
    unwanted movements would the user feel?", which is the complaint they would
    actually make. The two can diverge sharply: one sustained 10-second error and
    fifty scattered 200 ms twitches give the same per-window rate and are not the
    same device. Both are reported because neither alone is enough.
    """

    rest_recall: float
    false_activation_rate: float
    false_activations_per_min: float
    mean_false_burst_ms: float
    longest_false_burst_ms: float
    rest_windows: int
    rest_seconds: float

    def as_row(self) -> dict:
        return {
            "rest_recall": self.rest_recall,
            "false_activation_rate": self.false_activation_rate,
            "false_activations_per_min": self.false_activations_per_min,
            "mean_false_burst_ms": self.mean_false_burst_ms,
            "longest_false_burst_ms": self.longest_false_burst_ms,
            "rest_windows": self.rest_windows,
            "rest_seconds": self.rest_seconds,
        }


def rest_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    start: np.ndarray,
    group: np.ndarray,
    stride_samples: int,
    fs: int,
    rest_index: int = 0,
) -> RestScores:
    """Rest-period behaviour, counted per window and per event.

    `start` is each window's first-sample index and `group` identifies the
    recording it belongs to (subject, session and repetition combined), because an
    event must not be counted as continuing across a seam between two recordings.

    Windows that straddle a rest/movement boundary are dropped during
    segmentation, so two consecutive rows are not necessarily adjacent in time.
    Adjacency is therefore tested on `start`, not on row order: a gap larger than
    one stride ends the current burst instead of silently extending it across the
    hole. Getting this wrong would merge distinct events and *understate* the
    event rate, which is the direction that flatters the model.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    start = np.asarray(start)
    group = np.asarray(group)
    if not (len(y_true) == len(y_pred) == len(start) == len(group)):
        raise ValueError("y_true, y_pred, start and group must be the same length")

    is_rest = y_true == rest_index
    n_rest = int(is_rest.sum())
    if n_rest == 0:
        nan = float("nan")
        return RestScores(nan, nan, nan, nan, nan, 0, 0.0)

    stride_ms = 1000.0 * stride_samples / fs
    # Each rest window represents one stride of wall-clock time. Using the window
    # length instead would double-count the overlap between neighbours.
    rest_seconds = n_rest * stride_ms / 1000.0
    wrong = y_pred[is_rest] != rest_index
    recall = float(1.0 - wrong.mean())

    bursts: list[int] = []
    order = np.lexsort((start[is_rest], group[is_rest]))
    g = group[is_rest][order]
    s = start[is_rest][order]
    w = wrong[order]
    run = 0
    for i in range(len(w)):
        contiguous = (
            i > 0
            and g[i] == g[i - 1]
            and int(s[i] - s[i - 1]) <= stride_samples
        )
        if w[i] and run > 0 and contiguous:
            run += 1
            continue
        if run > 0:
            bursts.append(run)
            run = 0
        if w[i]:
            run = 1
    if run > 0:
        bursts.append(run)

    per_min = len(bursts) / (rest_seconds / 60.0) if rest_seconds > 0 else float("nan")
    burst_ms = np.array(bursts, dtype=float) * stride_ms
    return RestScores(
        rest_recall=recall,
        false_activation_rate=float(wrong.mean()),
        false_activations_per_min=float(per_min),
        mean_false_burst_ms=float(burst_ms.mean()) if len(burst_ms) else 0.0,
        longest_false_burst_ms=float(burst_ms.max()) if len(burst_ms) else 0.0,
        rest_windows=n_rest,
        rest_seconds=float(rest_seconds),
    )
