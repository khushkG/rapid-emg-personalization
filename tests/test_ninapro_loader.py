"""Tests for the NinaPro loader, against synthetic files in the real format.

`remg.data.ninapro` names three things as load-bearing and easy to get silently
wrong: reading `restimulus` rather than `stimulus`, offsetting the per-exercise
label restart, and refusing to guess when an amputee's channel count differs.
Each has a test here that fails loudly if it regresses, plus coverage of the
container-format and filename mechanics that would otherwise first be exercised
on real data.

These do not prove the loader matches the real downloads -- only that it matches
the published layout. `scripts/inspect_data.py` remains the check that runs
against actual files.
"""

from __future__ import annotations

import numpy as np
import pytest

from remg.data.movements import EXERCISE_OFFSET, N_MOVEMENTS
from remg.data.ninapro import (
    find_files,
    load_cohort,
    load_file,
    load_subject,
    parse_filename,
)

from ninapro_fixtures import (
    EXERCISE_SIZES,
    REACTION_SAMPLES,
    build_exercise,
    write_cohort,
    write_db6_subject,
    write_mat,
    write_subject,
)


# --- the three load-bearing assumptions -------------------------------------

def test_labels_come_from_restimulus_not_stimulus(tmp_path):
    """`stimulus` is the cue shown; `restimulus` is onset-corrected truth.

    Reading the wrong one labels the subject's reaction time as movement. The
    fixture offsets `stimulus` by a reaction delay, so the two disagree.
    """
    arrays = build_exercise(1, seed=1)
    path = write_mat(tmp_path / "S1_E1_A1.mat", arrays)

    rec = load_file(path, "DB2")

    expected = arrays["restimulus"].ravel()
    wrong = arrays["stimulus"].ravel()
    assert np.array_equal(rec.label, expected)
    assert not np.array_equal(rec.label, wrong), "loader picked up the uncorrected cue"


def test_repetitions_come_from_rerepetition_not_repetition(tmp_path):
    """Same correction, applied to the repetition index."""
    arrays = build_exercise(1, seed=2)
    path = write_mat(tmp_path / "S1_E1_A1.mat", arrays)

    rec = load_file(path, "DB2")

    assert np.array_equal(rec.repetition, arrays["rerepetition"].ravel())
    assert not np.array_equal(rec.repetition, arrays["repetition"].ravel())


def test_movement_onset_aligns_with_signal_amplitude(tmp_path):
    """The labelled movement windows are the high-amplitude ones.

    A reaction-time misalignment would leave the first samples of each labelled
    block sitting on resting signal; this catches that without asserting on the
    label arrays themselves.
    """
    arrays = build_exercise(1, seed=3)
    rec = load_file(write_mat(tmp_path / "S1_E1_A1.mat", arrays), "DB2")

    amplitude = np.abs(rec.emg).mean(axis=1)
    moving = rec.label > 0
    assert amplitude[moving].mean() > 3 * amplitude[~moving].mean()

    # And specifically at the onsets, where a reaction-time shift would show.
    onsets = np.flatnonzero(moving & ~np.roll(moving, 1))
    early = np.concatenate([np.arange(o, o + REACTION_SAMPLES) for o in onsets[:-1]])
    assert amplitude[early].mean() > 2 * amplitude[~moving].mean()


# The bug these were written for: the real DB2/DB3 files already carry global
# ids, while the published description of the format says labels restart at 1
# per file. Applying the offset to an already-global file renames every movement
# in it -- exercise 2 becomes 35..57 -- and nothing crashes.

@pytest.mark.parametrize("numbering", ["global", "local"])
@pytest.mark.parametrize("exercise", [1, 2, 3])
def test_labels_end_up_global_under_either_convention(tmp_path, numbering, exercise):
    arrays = build_exercise(exercise, seed=20 + exercise, numbering=numbering)
    path = write_mat(tmp_path / f"S1_E{exercise}_A1.mat", arrays)

    rec = load_file(path, "DB2")

    offset = EXERCISE_OFFSET[exercise]
    size = EXERCISE_SIZES[exercise]
    seen = sorted(int(v) for v in np.unique(rec.label) if v > 0)
    assert seen == list(range(offset + 1, offset + size + 1)), (
        f"{numbering} labels in exercise {exercise} did not land in their global range"
    )


def test_already_global_labels_are_not_offset_twice(tmp_path):
    """The specific corruption: exercise 2 landing at 35..57 instead of 18..40."""
    arrays = build_exercise(2, seed=21, numbering="global")
    rec = load_file(write_mat(tmp_path / "S1_E2_A1.mat", arrays), "DB2")

    assert int(rec.label.max()) == 40
    assert rec.meta["labels_were_global"] is True


def test_file_local_labels_are_still_offset(tmp_path):
    """Older releases do restart at 1, and must keep working."""
    arrays = build_exercise(2, seed=22, numbering="local")
    rec = load_file(write_mat(tmp_path / "S1_E2_A1.mat", arrays), "DB2")

    assert sorted(int(v) for v in np.unique(rec.label) if v > 0) == list(range(18, 41))
    assert rec.meta["labels_were_global"] is False


def test_partial_movement_coverage_is_still_detected_as_global(tmp_path):
    """DB3 s1 stops at global id 29 -- 12 of exercise 2's 23 movements.

    Its maximum (29) is above the movement count (23), but an amputee who
    completed even fewer could fall below it, so detection cannot lean on the
    maximum alone.
    """
    for n in (12, 8, 3, 1):
        arrays = build_exercise(2, seed=23, numbering="global", n_movements=n)
        rec = load_file(write_mat(tmp_path / f"n{n}" / "S1_E2_A1.mat", arrays), "DB2")
        seen = sorted(int(v) for v in np.unique(rec.label) if v > 0)
        assert seen == list(range(18, 18 + n)), f"{n} movements were re-offset to {seen}"
        assert rec.meta["labels_were_global"] is True


def test_the_numbering_verdict_is_recorded_per_file(tmp_path):
    """An exclusion or a surprise has to be traceable to a specific file."""
    write_subject(tmp_path, subject=1, numbering="global")
    rec = load_subject(tmp_path, "DB2", subject=1)

    verdicts = rec.meta["labels_were_global"]
    assert set(verdicts) == {"S1_E1_A1.mat", "S1_E2_A1.mat", "S1_E3_A1.mat"}
    assert all(verdicts.values())


def test_exercise_labels_are_offset_into_disjoint_global_ranges(tmp_path):
    """Labels restart at 1 per exercise file and must not collide once merged."""
    write_subject(tmp_path, subject=1)
    rec = load_subject(tmp_path, "DB2", subject=1)

    movements = set(int(v) for v in np.unique(rec.label) if v > 0)
    assert movements == set(range(1, N_MOVEMENTS + 1)), "expected all 49 movements exactly once"

    for exercise, size in EXERCISE_SIZES.items():
        offset = EXERCISE_OFFSET[exercise]
        block = set(range(offset + 1, offset + size + 1))
        assert block <= movements


def test_rest_is_never_offset(tmp_path):
    """Rest stays 0 in every exercise, or it becomes a phantom movement."""
    for exercise in (1, 2, 3):
        arrays = build_exercise(exercise, seed=exercise)
        rec = load_file(write_mat(tmp_path / f"S1_E{exercise}_A1.mat", arrays), "DB2")
        rest = arrays["restimulus"].ravel() == 0
        assert rest.any()
        assert (rec.label[rest] == 0).all()


def test_channel_mismatch_is_refused_not_guessed(tmp_path):
    """A short-stump amputee has fewer electrodes; the loader must not improvise."""
    write_subject(tmp_path, subject=1, n_channels=10)

    with pytest.raises(ValueError, match="channels, expected 12"):
        load_subject(tmp_path, "DB3", subject=1, expected_channels=12)

    rec = load_subject(tmp_path, "DB3", subject=1, expected_channels=None)
    assert rec.n_channels == 10


def test_excluded_subjects_are_named_in_the_output(tmp_path, capsys):
    """An exclusion has to be documentable, so the skipped subject is printed."""
    write_subject(tmp_path, subject=1, n_channels=12)
    write_subject(tmp_path, subject=2, n_channels=10)   # shorter stump
    write_subject(tmp_path, subject=3, n_channels=12)

    records = load_cohort(tmp_path, "DB3", expected_channels=12, skip_mismatched=True)

    assert [r.subject for r in records] == [1, 3]
    out = capsys.readouterr().out
    assert "skipped 1 subject-session" in out
    assert "S2" in out


def test_cohort_load_can_be_made_strict(tmp_path):
    write_subject(tmp_path, subject=1, n_channels=12)
    write_subject(tmp_path, subject=2, n_channels=10)

    with pytest.raises(ValueError):
        load_cohort(tmp_path, "DB3", expected_channels=12, skip_mismatched=False, verbose=False)


# --- container formats and file layout --------------------------------------

@pytest.mark.parametrize("fmt", ["v5", "v7.3"])
def test_both_mat_container_formats_load_identically(tmp_path, fmt):
    """DB2/DB3 ship a mix of v5 and v7.3 files; v7.3 is HDF5 and transposed."""
    arrays = build_exercise(2, seed=5)
    path = write_mat(tmp_path / "S1_E2_A1.mat", arrays, fmt=fmt)

    rec = load_file(path, "DB2")

    assert rec.emg.shape == arrays["emg"].shape
    assert np.array_equal(rec.label, arrays["restimulus"].ravel())
    assert np.allclose(rec.emg, arrays["emg"], atol=1e-6)


def test_transposed_emg_is_corrected(tmp_path):
    """Some files store emg as (channels, samples); orientation is inferred."""
    arrays = build_exercise(1, seed=6)
    upright = arrays["emg"]
    arrays = {**arrays, "emg": upright.T}

    rec = load_file(write_mat(tmp_path / "S1_E1_A1.mat", arrays), "DB2")

    assert rec.emg.shape == upright.shape
    assert np.allclose(rec.emg, upright, atol=1e-6)


def test_filename_conventions(tmp_path):
    assert parse_filename(tmp_path / "S12_E3_A1.mat") == (12, 3, None)
    assert parse_filename(tmp_path / "s7_e1_a1.mat") == (7, 1, None)
    assert parse_filename(tmp_path / "S4_D5_T1.mat") == (4, None, 5)

    with pytest.raises(ValueError, match="unrecognised"):
        parse_filename(tmp_path / "subject1.mat")


def test_files_are_found_in_nested_per_subject_directories(tmp_path):
    """The zips unpack to a folder per subject; the search is recursive."""
    write_subject(tmp_path, subject=1, nested=True)
    write_subject(tmp_path, subject=2, nested=True)

    groups = find_files(tmp_path, "DB2")

    assert sorted(groups) == [(1, 0), (2, 0)]
    assert all(len(v) == 3 for v in groups.values())


def test_unrecognised_files_are_ignored_not_fatal(tmp_path):
    write_subject(tmp_path, subject=1)
    write_mat(tmp_path / "readme_notes.mat", build_exercise(1, seed=9))

    groups = find_files(tmp_path, "DB2")

    assert sorted(groups) == [(1, 0)]


def test_missing_directory_and_empty_directory_are_distinguished(tmp_path):
    with pytest.raises(FileNotFoundError, match="data directory not found"):
        find_files(tmp_path / "nope", "DB2")

    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no NinaPro .mat files"):
        find_files(tmp_path / "empty", "DB2")


def test_missing_field_names_what_was_actually_present(tmp_path):
    """The error has to be diagnosable without opening the file by hand."""
    arrays = build_exercise(1, seed=10)
    del arrays["restimulus"]
    path = write_mat(tmp_path / "S1_E1_A1.mat", arrays)

    with pytest.raises(KeyError) as exc:
        load_file(path, "DB2")

    message = str(exc.value)
    assert "restimulus" in message
    assert "stimulus" in message and "emg" in message


def test_ragged_field_lengths_are_truncated_to_the_shortest(tmp_path):
    """Real files occasionally carry one extra label sample; alignment must hold."""
    arrays = build_exercise(1, seed=11)
    arrays["emg"] = arrays["emg"][:-5]
    path = write_mat(tmp_path / "S1_E1_A1.mat", arrays)

    rec = load_file(path, "DB2")

    n = len(arrays["emg"])
    assert rec.emg.shape[0] == len(rec.label) == len(rec.repetition) == n
    assert np.array_equal(rec.label, arrays["restimulus"].ravel()[:n])


def test_exercise_files_disagreeing_on_channels_is_an_error(tmp_path):
    """A subject whose own files disagree is corrupt, not merely excluded."""
    write_subject(tmp_path, subject=1, exercises=(1,), n_channels=12)
    write_subject(tmp_path, subject=1, exercises=(2,), n_channels=10)

    with pytest.raises(ValueError, match="disagree on channel count"):
        load_subject(tmp_path, "DB3", subject=1, expected_channels=None)


# --- DB6 / cross-session ----------------------------------------------------

def test_db6_days_become_distinct_sessions(tmp_path):
    """The cross-session experiment depends on day -> session being right."""
    write_db6_subject(tmp_path, subject=1, days=(1, 5))

    groups = find_files(tmp_path, "DB6")
    assert sorted(groups) == [(1, 0), (1, 4)]

    day1 = load_subject(tmp_path, "DB6", subject=1, session=0, expected_channels=None)
    day5 = load_subject(tmp_path, "DB6", subject=1, session=4, expected_channels=None)
    assert day1.session == 0 and day5.session == 4
    assert day1.uid != day5.uid


def test_db6_labels_are_not_exercise_offset(tmp_path):
    """DB6 files carry no exercise, so no offset may be applied."""
    paths = write_db6_subject(tmp_path, subject=1, days=(1,))
    rec = load_file(paths[0], "DB6")

    assert rec.label.max() == EXERCISE_SIZES[1]     # 1..17, unshifted
    assert rec.meta["exercise"] is None
    assert rec.meta["day"] == 1


# --- integration with the rest of the pipeline ------------------------------

def test_loaded_records_satisfy_the_record_contract(tmp_path):
    """Whatever the loader emits has to be indistinguishable from synthetic."""
    write_cohort(tmp_path, subjects=2)
    records = load_cohort(tmp_path, "DB2", expected_channels=12, verbose=False)

    assert len(records) == 2
    for rec in records:
        assert rec.emg.dtype == np.float32
        assert rec.label.dtype == np.int16 and rec.repetition.dtype == np.int16
        assert rec.emg.shape[0] == len(rec.label) == len(rec.repetition)
        assert rec.emg.flags["C_CONTIGUOUS"]
        assert rec.fs == 2000 and rec.dataset == "DB2"
        assert rec.summary()


def test_rest_windows_get_attached_to_a_repetition(tmp_path):
    """NinaPro marks rest as repetition 0; splits by repetition would drop it."""
    write_subject(tmp_path, subject=1)
    rec = load_subject(tmp_path, "DB2", subject=1).with_rest_repetitions()

    assert (rec.repetition > 0).all()
    assert np.array_equal(np.unique(rec.repetition), np.arange(1, 7))


# --- DB6: padding columns and per-trial repetitions ------------------------

def test_db6_drops_the_two_empty_columns(tmp_path):
    """14 electrodes must reach the model, not 16 columns with two of them zero."""
    write_db6_subject(tmp_path, subject=1, days=(1,))
    rec = load_file(tmp_path / "S1_D1_T1.mat", "DB6")
    assert rec.n_channels == 14
    assert rec.meta["dropped_channels"] == [8, 9]
    assert rec.meta["channels_before_drop"] == 16
    assert rec.meta["other_flat_channels"] == []


def test_db6_refuses_to_drop_columns_that_carry_signal(tmp_path):
    """The registry is an assumption, so it has to be checked, not trusted.

    If a future release pads different columns, dropping 8 and 9 regardless would
    delete two real electrodes and quietly degrade every result.
    """
    write_db6_subject(tmp_path, subject=1, days=(1,), empty_channels=())
    with pytest.raises(ValueError, match="carry signal"):
        load_file(tmp_path / "S1_D1_T1.mat", "DB6")


def test_db6_reports_an_unexpected_flat_column_without_dropping_it(tmp_path):
    """A dead electrode is not padding; DB3 has several and they must stay."""
    write_db6_subject(tmp_path, subject=1, days=(1,), empty_channels=(8, 9, 3))
    rec = load_file(tmp_path / "S1_D1_T1.mat", "DB6")
    assert rec.meta["dropped_channels"] == [8, 9]
    assert rec.meta["other_flat_channels"] == [3]
    assert rec.n_channels == 14, "a dead electrode must not change the channel count"


def test_db6_trials_get_consecutive_repetitions(tmp_path):
    """Both trials of a day number their repetitions from 1 in the real files.

    Concatenated as-is, every (movement, repetition) pair occurs twice and a
    "1-shot" calibration set silently holds two repetitions -- so every reported
    shot count would be wrong by a factor of two.
    """
    write_db6_subject(tmp_path, subject=1, days=(1,), trials=(1, 2), n_repetitions=6)
    rec = load_subject(tmp_path, "DB6", subject=1, session=0, expected_channels=14)
    reps = sorted({int(r) for r in np.unique(rec.repetition) if r > 0})
    assert reps == list(range(1, 13)), f"expected 12 distinct repetitions, got {reps}"
    assert rec.meta["repetitions_renumbered_per_trial"] is True
    assert rec.meta["trials"] == [1, 2]


def test_db2_repetitions_are_never_renumbered(tmp_path):
    """DB2/DB3 files are exercises, not trials: renumbering would change every result."""
    write_subject(tmp_path, subject=1, n_repetitions=6)
    rec = load_subject(tmp_path, "DB2", subject=1, session=0, expected_channels=12)
    reps = sorted({int(r) for r in np.unique(rec.repetition) if r > 0})
    assert reps == list(range(1, 7)), (
        f"DB2 repetitions must stay 1..6 across its three exercise files, got {reps}"
    )
    assert rec.meta["repetitions_renumbered_per_trial"] is False


def test_db6_single_trial_day_is_not_renumbered(tmp_path):
    """With one file there is nothing to disambiguate, so leave the numbering alone."""
    write_db6_subject(tmp_path, subject=1, days=(1,), trials=(1,), n_repetitions=6)
    rec = load_subject(tmp_path, "DB6", subject=1, session=0, expected_channels=14)
    reps = sorted({int(r) for r in np.unique(rec.repetition) if r > 0})
    assert reps == list(range(1, 7))
    assert rec.meta["repetitions_renumbered_per_trial"] is False
