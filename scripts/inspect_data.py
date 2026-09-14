"""Verify what the downloaded NinaPro files actually contain.

Run this first, before any training. It checks the assumptions the rest of the
code makes -- channel counts, label numbering, repetition counts, movement
coverage -- and prints what it finds, so a wrong assumption shows up here as a
line of output rather than later as a mysteriously mediocre accuracy.

    uv run python scripts/inspect_data.py data/raw/DB2 --dataset DB2
    uv run python scripts/inspect_data.py data/raw/DB3 --dataset DB3
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from remg.data.movements import DEFAULT_SUBSET, EXERCISE_OFFSET, EXERCISE_SIZES
from remg.data.ninapro import _load_mat, find_files, load_file, parse_filename


def inspect_file(path: Path, dataset: str) -> dict:
    mat = _load_mat(path)
    subject, exercise, day = parse_filename(path)
    fields = sorted(k for k in mat if not k.startswith("__"))

    rec = load_file(path, dataset)
    local = np.asarray(mat["restimulus"]).ravel()
    local_max = int(local.max())
    reps = sorted(int(r) for r in np.unique(rec.repetition) if r > 0)

    expected = EXERCISE_SIZES.get(exercise) if exercise else None
    label_ok = expected is None or local_max == expected

    return {
        "file": path.name,
        "subject": subject,
        "exercise": exercise,
        "day": day,
        "fields": fields,
        "channels": rec.n_channels,
        "samples": rec.emg.shape[0],
        "duration_s": rec.duration_s,
        "local_label_max": local_max,
        "expected_label_max": expected,
        "label_ok": label_ok,
        "global_labels": sorted(int(v) for v in np.unique(rec.label) if v > 0),
        "repetitions": reps,
        "has_stimulus_and_restimulus": {"stimulus", "restimulus"}.issubset(fields),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="directory containing the .mat files")
    ap.add_argument("--dataset", default="DB2", choices=["DB1", "DB2", "DB3", "DB6"])
    ap.add_argument("--max-files", type=int, default=None, help="stop after N files")
    args = ap.parse_args()

    groups = find_files(args.root, args.dataset)
    paths = [p for key in sorted(groups) for p in sorted(groups[key])]
    if args.max_files:
        paths = paths[: args.max_files]

    print(f"{args.dataset}: {len(groups)} subject-session(s), {len(paths)} file(s) under {args.root}\n")

    channel_counts: Counter = Counter()
    problems: list[str] = []
    seen_global: set[int] = set()
    per_subject_channels: dict[int, set[int]] = {}

    for path in paths:
        info = inspect_file(path, args.dataset)
        channel_counts[info["channels"]] += 1
        per_subject_channels.setdefault(info["subject"], set()).add(info["channels"])
        seen_global.update(info["global_labels"])

        flag = "" if info["label_ok"] else "  <-- LABEL RANGE MISMATCH"
        print(
            f"  {info['file']:<20} S{info['subject']:<3} E{info['exercise']} "
            f"{info['channels']:>2}ch {info['duration_s']:>7.1f}s  "
            f"local 1..{info['local_label_max']} -> global "
            f"{min(info['global_labels'], default=0)}..{max(info['global_labels'], default=0)}  "
            f"{len(info['repetitions'])} reps{flag}"
        )
        if not info["label_ok"]:
            problems.append(
                f"{info['file']}: exercise {info['exercise']} has max local label "
                f"{info['local_label_max']}, expected {info['expected_label_max']} -- "
                f"the offsets in movements.EXERCISE_OFFSET ({EXERCISE_OFFSET}) may be wrong "
                f"for {args.dataset}"
            )
        if not info["has_stimulus_and_restimulus"]:
            problems.append(f"{info['file']}: missing stimulus/restimulus field")

    print(f"\nchannel counts across files: {dict(channel_counts)}")
    odd = {s: c for s, c in per_subject_channels.items() if c != {12}}
    if odd:
        print(f"subjects not at 12 channels: {odd}")

    print(f"distinct global movement ids seen: {len(seen_global)} "
          f"(range {min(seen_global, default=0)}..{max(seen_global, default=0)})")

    missing = sorted(set(DEFAULT_SUBSET) - {0} - seen_global)
    if missing:
        print(f"\nWARNING: movements in DEFAULT_SUBSET not found in this data: {missing}")
        print("  Either these files are a partial download, or the subset ids need revising.")
    else:
        print("all DEFAULT_SUBSET movements are present")

    if problems:
        print(f"\n{len(problems)} problem(s) to resolve before training:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("\nno structural problems found")


if __name__ == "__main__":
    main()
