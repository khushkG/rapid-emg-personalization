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

from remg.data.movements import (
    DEFAULT_SUBSET,
    EXERCISE_OFFSET,
    EXERCISE_SIZES,
    OFFICIAL_MOVEMENT_NAMES,
    labels_are_global,
)
from remg.data.ninapro import _load_mat, find_files, load_file, parse_filename


def inspect_file(path: Path, dataset: str) -> dict:
    mat = _load_mat(path)
    subject, exercise, day = parse_filename(path)
    fields = sorted(k for k in mat if not k.startswith("__"))

    rec = load_file(path, dataset)
    local = np.asarray(mat["restimulus"]).ravel()
    local_max = int(local.max())
    reps = sorted(int(r) for r in np.unique(rec.repetition) if r > 0)

    # Which numbering the file uses, and whether its labels land inside the
    # global range that exercise is allowed to occupy. A file may legitimately
    # cover only part of that range -- DB3 amputees do not all complete every
    # grasp -- so the check is containment, not an exact match.
    if exercise is None:
        is_global, label_ok, expected_range, covered = None, True, None, None
    else:
        is_global = labels_are_global(local, exercise)
        offset = EXERCISE_OFFSET[exercise]
        expected_range = (offset + 1, offset + EXERCISE_SIZES[exercise])
        seen = [int(v) for v in np.unique(rec.label) if v > 0]
        label_ok = bool(seen) and expected_range[0] <= min(seen) and max(seen) <= expected_range[1]
        covered = (len(seen), EXERCISE_SIZES[exercise])

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
        "expected_range": expected_range,
        "labels_were_global": is_global,
        "covered": covered,
        "label_ok": label_ok,
        "global_labels": sorted(int(v) for v in np.unique(rec.label) if v > 0),
        "repetitions": reps,
        "label_samples": Counter(int(v) for v in rec.label),
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
    incomplete: list[str] = []
    seen_global: set[int] = set()
    label_samples: Counter = Counter()
    per_subject_channels: dict[int, set[int]] = {}
    per_subject_labels: dict[int, set[int]] = {}

    for path in paths:
        info = inspect_file(path, args.dataset)
        channel_counts[info["channels"]] += 1
        per_subject_channels.setdefault(info["subject"], set()).add(info["channels"])
        seen_global.update(info["global_labels"])
        per_subject_labels.setdefault(info["subject"], set()).update(info["global_labels"])
        label_samples.update(info["label_samples"])

        flag = "" if info["label_ok"] else "  <-- LABEL RANGE MISMATCH"
        n_seen, n_total = info["covered"] or (0, 0)
        partial = "" if n_seen == n_total else f"  <-- only {n_seen}/{n_total} movements"
        numbering = {True: "global", False: "file-local", None: "n/a"}[info["labels_were_global"]]
        print(
            f"  {info['file']:<20} S{info['subject']:<3} E{info['exercise']} "
            f"{info['channels']:>2}ch {info['duration_s']:>7.1f}s  "
            f"{numbering:<10} global "
            f"{min(info['global_labels'], default=0)}..{max(info['global_labels'], default=0)}  "
            f"{len(info['repetitions'])} reps{flag}{partial}"
        )
        if not info["label_ok"]:
            lo, hi = info["expected_range"]
            problems.append(
                f"{info['file']}: exercise {info['exercise']} produced global labels "
                f"{min(info['global_labels'], default=0)}..{max(info['global_labels'], default=0)}, "
                f"outside the {lo}..{hi} this exercise may occupy -- the numbering "
                f"detection in movements.ensure_global_labels is wrong for {args.dataset}"
            )
        elif n_seen != n_total:
            incomplete.append(
                f"{info['file']}: exercise {info['exercise']} contains {n_seen} of "
                f"{n_total} movements (global "
                f"{min(info['global_labels'], default=0)}..{max(info['global_labels'], default=0)})"
            )
        if not info["has_stimulus_and_restimulus"]:
            problems.append(f"{info['file']}: missing stimulus/restimulus field")

    print(f"\nchannel counts across files: {dict(channel_counts)}")
    odd = {s: c for s, c in per_subject_channels.items() if c != {12}}
    if odd:
        print(f"subjects not at 12 channels: {odd}")

    print(f"distinct global movement ids seen: {len(seen_global)} "
          f"(range {min(seen_global, default=0)}..{max(seen_global, default=0)})")

    # Per-movement sample counts for the subset the experiments actually use.
    # A movement that is present but barely sampled is a different problem from
    # one that is absent, and neither is visible from the id list alone.
    print("\nDEFAULT_SUBSET movements in this data:")
    total_samples = sum(label_samples.values()) or 1
    for global_id, name in sorted(DEFAULT_SUBSET.items()):
        n = label_samples.get(global_id, 0)
        official = OFFICIAL_MOVEMENT_NAMES.get(global_id, "rest" if global_id == 0 else "?")
        mark = "" if n else "   <-- ABSENT"
        print(f"  {global_id:>3}  {name:<22} {n:>10,} samples "
              f"({100 * n / total_samples:5.1f}%)  [{official}]{mark}")

    # Which subset movements every subject has. A movement missing from even one
    # subject cannot be a class in a leave-one-subject-out study: that subject
    # would have no calibration example of it.
    if len(per_subject_labels) > 1:
        wanted = sorted(set(DEFAULT_SUBSET) - {0})
        print("\nper-subject coverage of the subset (. = present, X = missing):")
        header = "  subject  " + " ".join(f"{g:>3}" for g in wanted)
        print(header)
        for subject in sorted(per_subject_labels):
            have = per_subject_labels[subject]
            cells = " ".join(f"{'  .' if g in have else '  X'}" for g in wanted)
            n_missing = sum(1 for g in wanted if g not in have)
            note = "" if not n_missing else f"   ({n_missing} missing)"
            print(f"  S{subject:<7} {cells}{note}")

        universal = [g for g in wanted if all(g in v for v in per_subject_labels.values())]
        dropped = [g for g in wanted if g not in universal]
        print(f"\n  present in all {len(per_subject_labels)} subjects: {universal}")
        if dropped:
            print(f"  NOT in every subject: {dropped}")
            print("   -> these cannot stay in DEFAULT_SUBSET for a leave-one-subject-out")
            print("      study unless the subjects missing them are excluded.")

    missing = sorted(set(DEFAULT_SUBSET) - {0} - seen_global)
    if missing:
        print(f"\nWARNING: movements in DEFAULT_SUBSET not found in this data: {missing}")
        print("  Either these files are a partial download, or the subset ids need revising.")
    else:
        print("all DEFAULT_SUBSET movements are present")

    if incomplete:
        print(f"\n{len(incomplete)} file(s) do not contain their full movement set:")
        for p in incomplete:
            print(f"  - {p}")
        print("  This is a property of the recording, not a bug. It constrains which")
        print("  movements DEFAULT_SUBSET can use across the whole cohort.")

    if problems:
        print(f"\n{len(problems)} problem(s) to resolve before training:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("\nno structural problems found")


if __name__ == "__main__":
    main()
