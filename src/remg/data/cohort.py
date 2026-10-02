"""Subjects excluded from the study, and why.

An exclusion that lives in a command-line flag is an exclusion that eventually
gets forgotten, applied inconsistently between runs, or discovered by a reader
of the paper rather than stated by it. They live here instead: named, with the
reason attached, applied by default, and recorded in every run's metadata.

The bar for adding one is that the subject cannot answer the question being
asked -- not that the subject is inconvenient or scores badly. Anything that
looks like the latter belongs in the results, not in this file.
"""

from __future__ import annotations

# (dataset, subject) -> reason, stated precisely enough to go in a writeup.
EXCLUSIONS: dict[tuple[str, int], str] = {
    ("DB3", 1): (
        "performed only 12 of the 23 grasping movements (global ids 18-29), so the "
        "recording contains no tripod grasp (30) and no lateral grasp (34). The "
        "few-shot conditions require a calibration example of every movement in "
        "DEFAULT_SUBSET, and this subject can supply none for those two. Verified "
        "with scripts/inspect_data.py; the other 9 DB3 subjects checked carry all 12."
    ),
}


# (dataset, filename) -> reason. Separate from EXCLUSIONS on purpose: a defective
# *recording* is not a defective *subject*. Excluding the whole subject because one
# of ten files is short would throw away nine good recordings and fail this file's
# own stated bar -- the subject can still answer the question being asked. These are
# recorded so that any analysis which depends on the affected day says so, and so
# that nobody re-discovers the same truncation in six months and wonders if it is a
# loader bug.
PARTIAL_RECORDINGS: dict[tuple[str, str], str] = {
    ("DB6", "S2_D2_T2.mat"): (
        "truncated: 103 s of recording against ~715 s for every other DB6 trial, "
        "containing only movement 1 (with all 12 of its repetitions) instead of the "
        "seven movements every other file carries. The session appears to have been "
        "stopped after the first movement block. Day 1 and day 5 of this subject are "
        "complete, so the day-1 -> day-5 cross-session study is unaffected; only an "
        "analysis using day 2 is. Verified by direct inspection of all 20 files of "
        "DB6 S1 and S2."
    ),
    ("DB6", "S9_D1_T1.mat"): (
        "electrode 7 (0-indexed, and still index 7 after the padding columns 8 and 9 "
        "are removed) is identically zero for the whole of this trial, while in every "
        "other S9 file it carries a weak but nonzero signal -- RMS 4e-07 to 1e-05 "
        "against about 2e-05 for its neighbour. So this is a contact failure during "
        "trial 1 of day 1, not a dead electrode: concatenated with trial 2, day 1 has "
        "one channel that is flat for half the session and marginal for the other half. "
        "The loader reports it rather than dropping it, so the channel count stays 14 "
        "and every condition sees identical inputs; the comparison between conditions is "
        "therefore still fair. Worth knowing because day 1 is the calibration day: this "
        "subject calibrates from effectively 13 informative electrodes."
    ),
}


def partial_recordings(dataset: str) -> dict[str, str]:
    """Known-defective individual files for `dataset`, filename -> reason."""
    return {f: why for (ds, f), why in PARTIAL_RECORDINGS.items() if ds == dataset}


def partial_reason(dataset: str, filename: str) -> str | None:
    return PARTIAL_RECORDINGS.get((dataset, filename))


def excluded_subjects(dataset: str) -> list[int]:
    """Subject ids excluded from `dataset`, sorted."""
    return sorted(s for (ds, s) in EXCLUSIONS if ds == dataset)


def exclusion_reason(dataset: str, subject: int) -> str | None:
    return EXCLUSIONS.get((dataset, subject))


def describe(dataset: str) -> str:
    """Human-readable summary, for run logs and the writeup."""
    ids = excluded_subjects(dataset)
    if not ids:
        return f"{dataset}: no subjects excluded"
    lines = [f"{dataset}: {len(ids)} subject(s) excluded"]
    for subject in ids:
        lines.append(f"  S{subject}: {EXCLUSIONS[(dataset, subject)]}")
    return "\n".join(lines)
