"""Movement label handling for NinaPro DB2 / DB3.

DB2 and DB3 share the same acquisition protocol: 49 movements + rest, recorded
in three exercise blocks and stored in three files per subject.

    Exercise B (file *_E1_*): 17 movements -> global ids 1..17
    Exercise C (file *_E2_*): 23 movements -> global ids 18..40
    Exercise D (file *_E3_*): 9 movements  -> global ids 41..49

Which numbering `restimulus` uses depends on the release, and getting it wrong
is the most common source of silently mislabelled NinaPro data:

  * The **DB2 and DB3 files served today already carry global ids.** Verified
    against real downloads: DB2 s1 holds 1..17, 18..40, 41..49 in its three
    files, and DB3 s1 holds 1..17, 18..29, 41..49. Adding a per-exercise offset
    to these corrupts every label outside exercise 1.
  * Other releases and the widely-copied description of the format have labels
    restarting at 1 in each file, which *does* need the offset.

So the offset is applied only when the labels are actually file-local.
`labels_are_global` decides, and `ensure_global_labels` is what callers use; the
decision is recorded on the record's metadata rather than made silently, because
a wrong answer here does not crash, it just renames every movement.
"""

from __future__ import annotations

# Number of movements in each exercise block, in file order (E1, E2, E3).
EXERCISE_SIZES = {1: 17, 2: 23, 3: 9}

# Additive offset that turns a within-exercise label into a global 1..49 label.
EXERCISE_OFFSET = {1: 0, 2: 17, 3: 40}

N_MOVEMENTS = 49
REST = 0


def labels_are_global(labels, exercise: int) -> bool:
    """Decide whether `labels` are already global ids rather than file-local ones.

    Two signals, either of which settles it:

      * a label above this exercise's movement count cannot be file-local, since
        local ids run 1..size;
      * every non-rest label sitting at or above this exercise's global start,
        when a file-local block would have to include its own id 1.

    Exercise 1 is the same under both conventions (its offset is zero), so it is
    reported as global and nothing is added either way.
    """
    import numpy as np

    if exercise not in EXERCISE_OFFSET:
        raise ValueError(f"exercise must be one of {sorted(EXERCISE_OFFSET)}, got {exercise!r}")

    offset = EXERCISE_OFFSET[exercise]
    if offset == 0:
        return True

    nz = np.asarray(labels)
    nz = nz[nz > 0]
    if nz.size == 0:
        return True                                  # nothing to shift
    return int(nz.max()) > EXERCISE_SIZES[exercise] or int(nz.min()) > offset


def ensure_global_labels(labels, exercise: int) -> tuple:
    """Return (global labels, was_already_global).

    The safe entry point: it offsets file-local labels and leaves already-global
    ones alone, instead of assuming one convention and mangling the other.
    """
    if labels_are_global(labels, exercise):
        import numpy as np

        return np.asarray(labels), True
    return to_global_label(labels, exercise), False


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
# The official movement list, in global 1..40 numbering.
#
# Transcribed from Table I of Atzori et al., "Building the NINAPRO Database: A
# Resource for the Biorobotics Community" (BioRob 2012), which enumerates the
# movements *within* each exercise; the global ids below are those plus
# EXERCISE_OFFSET. Cross-checked against the class numbers cited in Jung et al.,
# Front. Bioeng. Biotechnol. 9:548357 (2021), which uses this same global
# numbering and independently agrees on ids 18, 20, 22, 24, 26, 27, 30, 31, 33,
# 34 and 36.
#
# Exercise D (global 41..49) is nine force patterns rather than named postures.
# Nothing here uses them, so they are deliberately not enumerated rather than
# guessed at; `scripts/inspect_data.py` reports their counts if they appear.
# ---------------------------------------------------------------------------
OFFICIAL_MOVEMENT_NAMES: dict[int, str] = {
    # Exercise B -- hand postures (global 1..8)
    1: "Thumb up",
    2: "Flexion of ring and little finger, thumb flexed over middle and little",
    3: "Flexion of ring and little finger",
    4: "Thumb opposing base of little finger",
    5: "Abduction of the fingers",
    6: "Fingers flexed together",
    7: "Pointing index",
    8: "Fingers closed together",
    # Exercise B -- wrist movements (global 9..17)
    9: "Wrist supination (rotation axis through the middle finger)",
    10: "Wrist pronation (rotation axis through the middle finger)",
    11: "Wrist supination (rotation axis through the little finger)",
    12: "Wrist pronation (rotation axis through the little finger)",
    13: "Wrist flexion",
    14: "Wrist extension",
    15: "Wrist radial deviation",
    16: "Wrist ulnar deviation",
    17: "Wrist extension with closed hand",
    # Exercise C -- grasping and functional movements (global 18..40)
    18: "Large diameter grasp",
    19: "Small diameter grasp",
    20: "Fixed hook grasp",
    21: "Index finger extension grasp",
    22: "Medium wrap",
    23: "Ring grasp",
    24: "Prismatic four fingers grasp",
    25: "Stick grasp",
    26: "Writing tripod grasp",
    27: "Power sphere grasp",
    28: "Three finger sphere grasp",
    29: "Precision sphere grasp",
    30: "Tripod grasp",
    31: "Prismatic pinch grasp",
    32: "Tip pinch grasp",
    33: "Quadpod grasp",
    34: "Lateral grasp",
    35: "Parallel extension grasp",
    36: "Extension type grasp",
    37: "Power disk grasp",
    38: "Open a bottle with a tripod grasp",
    39: "Turn a screw",
    40: "Cut something",
}


# ---------------------------------------------------------------------------
# The practical movement subset used by the core experiments.
#
# Verified against OFFICIAL_MOVEMENT_NAMES above; `test_protocol.py` pins every
# entry to its official name, so a typo here cannot survive the suite. One
# correction came out of that check: `lateral_grasp` was previously id 32, which
# is Tip pinch grasp -- Lateral grasp is 34. Ids 32 and 34 are both real
# movements, so the mistake would have trained and evaluated cleanly on the
# wrong class rather than failing.
#
# The per-class counts from `scripts/inspect_data.py` remain the check that these
# movements are actually present in the downloaded recordings.
# ---------------------------------------------------------------------------
DEFAULT_SUBSET: dict[int, str] = {
    0: "rest",
    5: "open_hand",          # abduction of the fingers
    6: "close_hand",         # fingers flexed together in a fist
    7: "point_index",
    9: "wrist_supination",
    10: "wrist_pronation",
    13: "wrist_flexion",
    14: "wrist_extension",
    18: "large_diameter_grasp",
    22: "medium_wrap",
    30: "tripod_grasp",
    34: "lateral_grasp",     # key grip -- NOT 32, which is tip pinch
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
