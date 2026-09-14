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
