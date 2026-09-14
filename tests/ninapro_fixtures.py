"""Synthetic NinaPro `.mat` files, structured like the real downloads.

These builders emit files with the same field names, shapes, dtypes and
container formats as the real recordings, so the loader can be tested without
keeping 22 GB of NinaPro in the repository and without a network.

Shapes here were checked against real downloads (DB2 s1, DB3 s1) rather than
against the published description of the format, which turns out to disagree
with the files on the most important point: the label numbering. Both
conventions are therefore available via `numbering`, and the default is the one
the real files use.

What is faithful: field names (`emg`, `stimulus`, `restimulus`, `repetition`,
`rerepetition`), both label conventions, partial movement coverage as seen in
DB3 amputees, the `S<n>_E<n>_A1.mat` / `S<n>_D<n>_T<n>.mat` filename
conventions, both the v5 and v7.3/HDF5 containers, the MATLAB userblock a real
v7.3 file carries, and its transposed axis order. What is not: the EMG is shaped
noise, and recordings are seconds long rather than minutes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Movements per exercise block in DB2/DB3, matching remg.data.movements.
EXERCISE_SIZES = {1: 17, 2: 23, 3: 9}

# Short by design -- these files exist to exercise indexing, not signal processing.
SAMPLES_PER_MOVE = 40
SAMPLES_PER_REST = 20

# How late `stimulus` is relative to `restimulus`, standing in for the subject's
# reaction time. Any loader reading `stimulus` by mistake picks up these samples
# as movement, which is exactly what the corresponding test looks for.
REACTION_SAMPLES = 7


# Global id of the first movement in each exercise, mirroring
# remg.data.movements.EXERCISE_OFFSET.
EXERCISE_OFFSET = {1: 0, 2: 17, 3: 40}


def build_exercise(
    exercise: int,
    n_channels: int = 12,
    n_repetitions: int = 6,
    seed: int = 0,
    numbering: str = "global",
    n_movements: int | None = None,
) -> dict[str, np.ndarray]:
    """Build one exercise file's arrays: rest, movement, rest, movement, ...

    `numbering` selects which label convention the file uses:

      "global"  ids continue across exercises (exercise 2 holds 18..40). This is
                what the DB2/DB3 files served today actually contain, verified
                against real downloads, so it is the default.
      "local"   ids restart at 1 in every file. The convention the published
                description of the format describes, and what older releases and
                much third-party code assume.

    Both exist in the wild, and the loader has to tell them apart -- writing only
    one of them would leave that detection untested.

    `n_movements` truncates the movement set, standing in for a DB3 amputee who
    did not complete every grasp (subject 1 stops at global id 29).
    Rest carries label 0 and repetition 0.
    """
    if numbering not in ("global", "local"):
        raise ValueError(f"numbering must be 'global' or 'local', got {numbering!r}")

    rng = np.random.default_rng(seed)
    n_movements = n_movements or EXERCISE_SIZES[exercise]
    if n_movements > EXERCISE_SIZES[exercise]:
        raise ValueError(
            f"exercise {exercise} has only {EXERCISE_SIZES[exercise]} movements"
        )
    first = 1 if numbering == "local" else EXERCISE_OFFSET[exercise] + 1

    label_parts, rep_parts = [], []
    for rep in range(1, n_repetitions + 1):
        for movement in range(first, first + n_movements):
            label_parts.append(np.zeros(SAMPLES_PER_REST, dtype=np.int16))
            rep_parts.append(np.zeros(SAMPLES_PER_REST, dtype=np.int16))
            label_parts.append(np.full(SAMPLES_PER_MOVE, movement, dtype=np.int16))
            rep_parts.append(np.full(SAMPLES_PER_MOVE, rep, dtype=np.int16))
    label_parts.append(np.zeros(SAMPLES_PER_REST, dtype=np.int16))  # trailing rest
    rep_parts.append(np.zeros(SAMPLES_PER_REST, dtype=np.int16))

    restimulus = np.concatenate(label_parts)
    rerepetition = np.concatenate(rep_parts)
    n = len(restimulus)

    # Uncorrected cue: the same blocks shifted later by the reaction time.
    stimulus = np.roll(restimulus, REACTION_SAMPLES)
    stimulus[:REACTION_SAMPLES] = 0
    repetition = np.roll(rerepetition, REACTION_SAMPLES)
    repetition[:REACTION_SAMPLES] = 0

    # Amplitude tracks the corrected labels, so a loader reading `stimulus` also
    # ends up with labels misaligned against the signal.
    active = (restimulus > 0).astype(np.float32)[:, None]
    emg = rng.normal(0.0, 0.01, size=(n, n_channels)).astype(np.float32)
    emg += active * rng.normal(0.0, 0.08, size=(n, n_channels)).astype(np.float32)

    return {
        "emg": emg,
        "stimulus": stimulus.reshape(-1, 1),
        "restimulus": restimulus.reshape(-1, 1),
        "repetition": repetition.reshape(-1, 1),
        "rerepetition": rerepetition.reshape(-1, 1),
    }


def _matlab_v73_header() -> bytes:
    """The 128-byte MATLAB header that precedes the HDF5 payload in a v7.3 file.

    A bare HDF5 file is not a `.mat` file. MATLAB writes v7.3 files as HDF5
    behind a userblock carrying this header, and scipy sniffs it to decide the
    format -- without it, `loadmat` fails with a "version 0, 0" ValueError
    instead of the NotImplementedError that signals "use an HDF5 reader". Since
    that branch is precisely what `_load_mat` keys on, a fixture missing the
    header would test a path the real files never take.
    """
    text = (
        "MATLAB 7.3 MAT-file, Platform: synthetic, "
        "Created by remg test fixtures, HDF5 schema 1.00 ."
    )
    header = bytearray(b" " * 128)
    header[: len(text)] = text.encode("ascii")
    header[124:128] = b"\x00\x02IM"      # version 2.0, little-endian marker
    return bytes(header)


def write_mat(path: Path, arrays: dict[str, np.ndarray], fmt: str = "v5") -> Path:
    """Write arrays as a MATLAB file in either container format.

    v7.3 files are HDF5 and store arrays transposed relative to the v5 reader,
    which is the quirk `_load_mat` compensates for; writing them transposed here
    is what makes that compensation testable. They also carry a `#refs#` group
    that is not a dataset, which is why the loader filters `#`-prefixed keys.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "v5":
        from scipy.io import savemat

        savemat(str(path), arrays, do_compression=False)
    elif fmt == "v7.3":
        import h5py

        with h5py.File(path, "w", userblock_size=512) as f:
            f.create_group("#refs#")
            for key, value in arrays.items():
                f.create_dataset(key, data=np.asarray(value).T)
        with open(path, "r+b") as fh:
            fh.write(_matlab_v73_header())
    else:
        raise ValueError(f"fmt must be 'v5' or 'v7.3', got {fmt!r}")
    return path


def write_subject(
    root: Path,
    subject: int,
    n_channels: int = 12,
    n_repetitions: int = 6,
    fmt: str = "v5",
    exercises: tuple[int, ...] = (1, 2, 3),
    nested: bool = False,
    numbering: str = "global",
) -> list[Path]:
    """Write one DB2/DB3 subject's three exercise files.

    With `nested`, files go in a per-subject subdirectory -- the layout the real
    zips unpack to, and the reason `find_files` searches recursively.
    """
    out = root / f"s{subject}" if nested else root
    paths = []
    for exercise in exercises:
        arrays = build_exercise(
            exercise,
            n_channels=n_channels,
            n_repetitions=n_repetitions,
            seed=subject * 10 + exercise,
            numbering=numbering,
        )
        paths.append(write_mat(out / f"S{subject}_E{exercise}_A1.mat", arrays, fmt))
    return paths


def write_cohort(
    root: Path,
    subjects: int = 3,
    n_channels: int = 12,
    n_repetitions: int = 6,
    fmt: str = "v5",
    numbering: str = "global",
) -> Path:
    """Write a small DB2/DB3-style cohort under `root`."""
    for subject in range(1, subjects + 1):
        write_subject(root, subject, n_channels, n_repetitions, fmt, numbering=numbering)
    return root


def write_db6_subject(
    root: Path,
    subject: int,
    days: tuple[int, ...] = (1, 5),
    n_channels: int = 14,
    n_repetitions: int = 6,
    fmt: str = "v5",
) -> list[Path]:
    """Write DB6-style files, which encode day and trial instead of exercise.

    DB6 has its own electrode count and a much smaller movement set; the point
    here is the filename convention and the day -> session mapping.
    """
    paths = []
    for day in days:
        arrays = build_exercise(
            1, n_channels=n_channels, n_repetitions=n_repetitions, seed=subject * 100 + day
        )
        paths.append(write_mat(root / f"S{subject}_D{day}_T1.mat", arrays, fmt))
    return paths
