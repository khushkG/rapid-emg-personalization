"""Loader for NinaPro DB2 / DB3 / DB6 `.mat` recordings.

Expected layout under the data root (the zips unpack to roughly this; the loader
searches recursively, so an extra directory level per subject is fine):

    data/raw/DB2/S1_E1_A1.mat  S1_E2_A1.mat  S1_E3_A1.mat  S2_E1_A1.mat  ...
    data/raw/DB3/S1_E1_A1.mat  ...
    data/raw/DB6/S1_D1_T1.mat  ...

Filename convention: `S<subject>_E<exercise>_A1.mat` for DB2/DB3. DB6 instead
encodes day and trial as `S<subject>_D<day>_T<trial>.mat`, which is what makes it
the cross-session dataset.

Three things here are load-bearing and easy to get silently wrong:

1. `restimulus`, not `stimulus`. The raw `stimulus` is the cue the subject was
   shown; `restimulus` is the label after NinaPro's own movement-onset
   correction. Training on `stimulus` labels the subject's reaction time as
   movement, which costs several points of accuracy for no reason.

2. Label numbering differs between releases. The DB2/DB3 files served today
   already carry global ids -- exercise 2 holds 18..40, not 1..23 -- while the
   commonly-cited description of the format has them restarting at 1 per file.
   Offsetting an already-global file silently renames every movement in it, so
   `movements.ensure_global_labels` detects which convention the file uses and
   the verdict is recorded in `meta["labels_were_global"]`.

3. Channel count is not always 12 in DB3. Several amputees have a shorter stump
   than the electrode array needs, so their recordings carry fewer channels. The
   loader refuses to guess: it reports what it found and leaves the decision to
   `expected_channels`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .movements import ensure_global_labels
from .records import SubjectRecord

# Native sampling rates. DB1 is included for completeness but is not part of the
# planned experiments (different electrodes, different protocol, 100 Hz).
NATIVE_FS = {"DB1": 100, "DB2": 2000, "DB3": 2000, "DB6": 2000}

_DB23_NAME = re.compile(r"[Ss](\d+)_[Ee](\d+)_[Aa](\d+)\.mat$")
_DB6_NAME = re.compile(r"[Ss](\d+)_[Dd](\d+)_[Tt](\d+)\.mat$")


def _load_mat(path: Path) -> dict:
    """Read a .mat file written in either the v5 or the v7.3 (HDF5) format."""
    from scipy.io import loadmat

    try:
        return loadmat(str(path), squeeze_me=False)
    except NotImplementedError:
        import h5py  # v7.3 files are HDF5

        with h5py.File(path, "r") as f:
            # HDF5 .mat stores arrays transposed relative to the v5 reader.
            return {k: np.array(v).T for k, v in f.items() if not k.startswith("#")}


def _field(mat: dict, name: str, path: Path) -> np.ndarray:
    if name not in mat:
        present = sorted(k for k in mat if not k.startswith("__"))
        raise KeyError(f"{path.name} has no field {name!r}; fields present: {present}")
    return np.asarray(mat[name])


def parse_filename(path: Path) -> tuple[int, int | None, int | None]:
    """Return (subject, exercise, day) from a NinaPro filename.

    DB2/DB3 files give an exercise and no day; DB6 gives a day and no exercise.
    """
    m = _DB23_NAME.search(path.name)
    if m:
        return int(m.group(1)), int(m.group(2)), None
    m = _DB6_NAME.search(path.name)
    if m:
        return int(m.group(1)), None, int(m.group(2))
    raise ValueError(f"unrecognised NinaPro filename: {path.name}")


def load_file(path: Path, dataset: str) -> SubjectRecord:
    """Load one .mat file as a SubjectRecord with globally-numbered labels."""
    path = Path(path)
    mat = _load_mat(path)
    subject, exercise, day = parse_filename(path)

    emg = _field(mat, "emg", path).astype(np.float32)
    if emg.ndim != 2:
        raise ValueError(f"{path.name}: emg has shape {emg.shape}, expected (samples, channels)")
    if emg.shape[0] < emg.shape[1]:  # stored transposed
        emg = emg.T

    label = _field(mat, "restimulus", path).astype(np.int16).ravel()
    rep = _field(mat, "rerepetition", path).astype(np.int16).ravel()

    n = min(len(label), len(rep), emg.shape[0])
    emg, label, rep = emg[:n], label[:n], rep[:n]

    labels_were_global = None
    if exercise is not None:
        label, labels_were_global = ensure_global_labels(label, exercise)
        label = label.astype(np.int16)

    return SubjectRecord(
        emg=np.ascontiguousarray(emg),
        label=np.ascontiguousarray(label),
        repetition=np.ascontiguousarray(rep),
        subject=subject,
        dataset=dataset,
        fs=NATIVE_FS.get(dataset, 2000),
        session=day - 1 if day is not None else 0,
        meta={
            "source_file": path.name,
            "exercise": exercise,
            "day": day,
            "labels_were_global": labels_were_global,
        },
    )


def find_files(root: Path, dataset: str) -> dict[tuple[int, int], list[Path]]:
    """Group every .mat under `root` by (subject, session)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"data directory not found: {root}")
    groups: dict[tuple[int, int], list[Path]] = {}
    for path in sorted(root.rglob("*.mat")):
        try:
            subject, _, day = parse_filename(path)
        except ValueError:
            continue
        groups.setdefault((subject, 0 if day is None else day - 1), []).append(path)
    if not groups:
        raise FileNotFoundError(f"no NinaPro .mat files found under {root}")
    return groups


def load_subject(
    root: Path,
    dataset: str,
    subject: int,
    session: int = 0,
    expected_channels: int | None = None,
) -> SubjectRecord:
    """Concatenate one subject's exercise files into a single recording."""
    files = find_files(root, dataset).get((subject, session))
    if not files:
        raise FileNotFoundError(f"no files for {dataset} subject {subject} session {session}")

    parts = [load_file(p, dataset) for p in sorted(files)]
    counts = {p.n_channels for p in parts}
    if len(counts) > 1:
        raise ValueError(
            f"{dataset} S{subject}: exercise files disagree on channel count {sorted(counts)}"
        )
    n_ch = parts[0].n_channels
    if expected_channels is not None and n_ch != expected_channels:
        raise ValueError(
            f"{dataset} S{subject} has {n_ch} channels, expected {expected_channels}. "
            "Some DB3 amputees were recorded with fewer electrodes; either exclude this "
            "subject or set expected_channels=None and handle the mismatch explicitly."
        )

    return SubjectRecord(
        emg=np.concatenate([p.emg for p in parts], axis=0),
        label=np.concatenate([p.label for p in parts]),
        repetition=np.concatenate([p.repetition for p in parts]),
        subject=subject,
        dataset=dataset,
        fs=parts[0].fs,
        session=session,
        meta={
            "source_files": [p.meta["source_file"] for p in parts],
            "n_channels": n_ch,
            "labels_were_global": {
                p.meta["source_file"]: p.meta["labels_were_global"] for p in parts
            },
        },
    )


def load_cohort(
    root: Path,
    dataset: str,
    subjects: list[int] | None = None,
    expected_channels: int | None = 12,
    skip_mismatched: bool = True,
    verbose: bool = True,
) -> list[SubjectRecord]:
    """Load every subject-session under `root`.

    With `skip_mismatched`, subjects whose channel count differs from
    `expected_channels` are reported and skipped rather than aborting the load --
    but they are *named*, so an excluded amputee is a documented exclusion in the
    writeup rather than a silent one.
    """
    groups = find_files(root, dataset)
    keys = sorted(k for k in groups if subjects is None or k[0] in subjects)

    records, skipped = [], []
    for subject, session in keys:
        try:
            rec = load_subject(root, dataset, subject, session, expected_channels)
        except ValueError as exc:
            if not skip_mismatched:
                raise
            skipped.append((subject, session, str(exc)))
            continue
        records.append(rec)
        if verbose:
            print(f"  {rec.summary()}", flush=True)

    if skipped and verbose:
        print(f"\n  skipped {len(skipped)} subject-session(s):", flush=True)
        for subject, session, why in skipped:
            print(f"    S{subject} sess{session}: {why}", flush=True)
    if not records:
        raise RuntimeError(f"no usable recordings loaded from {root}")
    return records
