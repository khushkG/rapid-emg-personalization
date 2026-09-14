"""Movement label handling for NinaPro DB2 / DB3.

DB2 and DB3 share the same acquisition protocol: 49 movements + rest, recorded
in three exercise blocks and stored in three files per subject.

    Exercise B (file *_E1_*): 17 movements -> global ids 1..17
    Exercise C (file *_E2_*): 23 movements -> global ids 18..40
    Exercise D (file *_E3_*): 9 movements  -> global ids 41..49

The `restimulus` field inside each file restarts at 1 for every exercise, so a
per-file offset has to be added before labels from different files can be
concatenated. This is the single most common source of silently wrong labels
when working with NinaPro, so it lives here in one place.
"""

from __future__ import annotations

# Number of movements in each exercise block, in file order (E1, E2, E3).
EXERCISE_SIZES = {1: 17, 2: 23, 3: 9}

# Additive offset that turns a within-exercise label into a global 1..49 label.
EXERCISE_OFFSET = {1: 0, 2: 17, 3: 40}

N_MOVEMENTS = 49
REST = 0


def to_global_label(local_label, exercise: int):
    """Map a file-local `restimulus` array to global movement ids.

    Rest (0) stays 0; every non-rest label is shifted by the exercise offset.
    Works on numpy arrays or plain ints.
    """
    if exercise not in EXERCISE_OFFSET:
        raise ValueError(f"exercise must be one of {sorted(EXERCISE_OFFSET)}, got {exercise!r}")
    offset = EXERCISE_OFFSET[exercise]
    try:  # numpy path
        import numpy as np

        arr = np.asarray(local_label)
        return np.where(arr > 0, arr + offset, 0).astype(arr.dtype)
    except ImportError:  # pragma: no cover
        return local_label + offset if local_label > 0 else 0


# ---------------------------------------------------------------------------
# The practical movement subset used by the core experiments.
#
# NOTE: these names follow the published NinaPro movement figures. Re-check them
# against the official movement list once the real recordings are downloaded --
# `scripts/inspect_data.py` prints the per-class counts so a mismatch shows up
# immediately. Changing the subset means editing this dict and nothing else.
# ---------------------------------------------------------------------------
DEFAULT_SUBSET: dict[int, str] = {
    0: "rest",
    5: "open_hand",          # abduction of all fingers
    6: "close_hand",         # fingers flexed together in a fist
    7: "point_index",
    9: "wrist_supination",
    10: "wrist_pronation",
    13: "wrist_flexion",
    14: "wrist_extension",
    18: "large_diameter_grasp",
    22: "medium_wrap",
    30: "tripod_grasp",
    32: "lateral_grasp",
}


def subset_mapping(subset: dict[int, str] | None = None):
    """Return (global_id -> contiguous class index, class_names).

    Models need contiguous 0..K-1 targets; everything outside the subset is
    dropped rather than folded into rest, so the classifier never has to learn a
    grab-bag "other" class.
    """
    subset = subset or DEFAULT_SUBSET
    ids = sorted(subset)
    return {g: i for i, g in enumerate(ids)}, [subset[g] for g in ids]
