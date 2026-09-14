# Rapid Personalization of Prosthetic Hand Control Using Muscle Signals

Can a shared deep-learning model adapt to a new amputee from a few calibration
examples, and stay reliable across sessions and sensor failures?

Multichannel surface EMG goes in, an intended hand movement comes out. The model
is pretrained on a cohort of intact subjects, then personalized to a previously
unseen amputee using only a handful of movement repetitions. No prosthetic
hardware is involved; the demonstration replays recorded EMG and animates a
digital hand.

## Status

| Stage | State |
| --- | --- |
| Environment, package layout | done |
| Preprocessing, windowing, splits | done |
| Synthetic EMG source (for development and tests) | done |
| Encoder, adapters, prototype head | done |
| Pretraining with episodic meta-learning | done |
| Four personalization conditions | done |
| Metrics, sensor-failure sweep, rejection curves | done |
| Protocol test suite (145 tests) | passing |
| NinaPro DB2/DB3/DB6 loader | run against real DB2 and DB3 files; label-numbering bug found and fixed |
| Movement subset | verified against the official movement list (one id was wrong — see Scope notes) |
| Cross-repetition session proxy | done |
| Real-data pipeline (DB2 -> DB3, end to end) | runs; `scripts/run_experiment.py` |
| DB3 download | 10 of 11 subjects; cohort coverage verified |
| DB2 download | subject 1 only (pretraining cohort not yet fetched) |
| Cross-session experiment (day 1 → day 5) | needs DB6 (see Scope notes) |
| Digital hand visualisation | done: `scripts/demo_hand.py` |
| Real results | blocked on the DB2 pretraining cohort |

Everything runs end-to-end today on synthetic data:

```
uv run python scripts/smoke.py     # ~1 minute
```

Those numbers are plumbing checks, not findings. The data is simulated.

The tests are the other half of that: 145 of them, covering the protocol rules
below and the NinaPro loader against synthetic files written in the real `.mat`
format (both v5 and v7.3 containers, the per-exercise label restart, and the
reduced-channel amputee recordings).

## Setup

```
curl -LsSf https://astral.sh/uv/install.sh | sh     # once
uv sync --group dev
uv run --group dev pytest -q
```

Torch runs on the M-series GPU via MPS; `remg.utils.pick_device` selects it
automatically.

## Getting the data

The databases are served as one zip per subject, straight from
<https://ninapro.hevs.ch/> — no account, no login. (An earlier version of this
file said registration was required; that portal is gone, and following it leads
to a site with no sign-up button anywhere.) The papers remain the required
citation, so read the terms on the site before publishing.

The two URL patterns differ in more than the number — DB3 is lowercase and
carries a `_0` suffix:

```
https://ninapro.hevs.ch/files/DB2_Preproc/DB2_s<N>.zip     # N = 1..40, ~470 MB each
https://ninapro.hevs.ch/files/db3_Preproc/s<N>_0.zip       # N = 1..11, ~252 MB each
```

DB2 is ~19 GB in full and DB3 ~2.8 GB. `scripts/fetch_data.py` downloads and
unpacks them, resuming interrupted transfers and skipping subjects already on
disk:

```
uv run python scripts/fetch_data.py --dataset DB3                 # all 11 amputees
uv run python scripts/fetch_data.py --dataset DB2 --subjects 1-4  # a slice first
```

Fetch a few subjects and run `inspect_data.py` on them before committing to the
full 19 GB. The loader is tested against the published layout, not against the
real files, and a small sample is exactly what tells you where those differ.

1. Fetch **DB2** (40 intact subjects) and **DB3** (11 transradial amputees).
   DB6 is only needed for the cross-session experiment.
2. Files land under `data/raw/DB2/` and `data/raw/DB3/`. Nested per-subject
   folders are fine — the loader searches recursively.
3. Verify before training anything:

```
uv run python scripts/inspect_data.py data/raw/DB2 --dataset DB2
uv run python scripts/inspect_data.py data/raw/DB3 --dataset DB3
```

`inspect_data.py` checks the assumptions the code makes — channel counts, the
per-exercise label offsets, repetition counts, whether the chosen movement subset
is actually present — and prints any mismatch as a line of output. It is not
optional; two of those assumptions are ones that fail silently.

## Protocol

Four rules are enforced in code, each one blocking a specific way a result could
come out inflated while still looking correct. `tests/test_protocol.py` is the
enforcement, and a change that breaks one of them fails the suite.

**A test subject contributes nothing to pretraining.** Leave-one-subject-out,
with the pretraining cohort (DB2) and the evaluation cohort (DB3) disjoint by
construction.

**Calibration is sampled by repetition, never by random windows.** Consecutive
windows within one repetition overlap and share a single muscle contraction, so a
random split puts near-duplicates on both sides. "Three calibration examples"
means three repetitions — which is also what a user would actually be asked to
perform. Calibration sets are nested, so the 1-shot set is a subset of the 2-shot
set and the shots-vs-accuracy curve is not confounded by luckier repetitions.

**Normalization statistics come from the calibration windows alone.** Normalizing
a new subject with their own statistics is itself a form of personalization; if
the baseline were denied it, the reported gain would be partly just rescaling.
Every condition gets identical calibration windows. `remg.data.normalize` also
offers `source` (harsher baseline) and `oracle` (leaks test data — a diagnostic
ceiling, never a result).

**Adaptation hyperparameters are tuned on held-out source subjects, never on the
test subject.** Not enforceable in code; it is on us not to violate it.

Reported metrics are balanced accuracy and macro F1, with per-movement recall
alongside. Rest dominates the recordings: measured on the real files, it is
**27-111x** the samples of any single movement in the subset (DB2 s1: 36x the
largest, 111x the smallest; DB3 s2: 27x and 68x), and about 4-6x all eleven
movements put together. An earlier version of this file said "roughly six times
any single movement" -- that figure is rest against the *combined* movement
total, not against one movement, and it understated the imbalance by an order of
magnitude.

So plain accuracy rewards a model that predicts rest and nothing else, and even
balanced accuracy leaves training heavily skewed. Nothing currently caps the
rest windows (`WindowConfig.keep_rest` is all-or-nothing); whether to subsample
rest is an open decision, and it should be made before the headline run, not
after seeing which choice reports better.

## The four conditions

All four start from the identical pretrained checkpoint and see identical
calibration data. They differ only in what calibration may change:

| Condition | What it updates | Role |
| --- | --- | --- |
| `none` | nothing | the general model applied as-is |
| `linear_probe` | final linear layer | the cheap classical baseline a new method must beat |
| `finetune` | every weight | ordinary transfer learning |
| `rapid` | FiLM adapters + prototypes | the proposed method |

`rapid` updates a few hundred parameters against roughly 110k for full
fine-tuning. FiLM adapters are initialised to the exact identity, so before
calibration begins the adapted model and the baseline are literally the same
network — any difference is attributable to adaptation rather than to a different
architecture.

## The demonstration

```
uv run python scripts/demo_hand.py --subject 2 --condition rapid
```

Replays an amputee's recorded EMG from repetitions calibration never touched,
and animates a hand from what the model decodes -- the one part of this project
a non-specialist can read directly.

It draws the decoded hand **beside the movement the subject was cued to
perform**. Showing only the decoded hand would let a fluent animation pass for
an accurate one; paired, a mistake is visible as a mistake, and the frame title
turns red when they disagree.

The hand is articulated rather than a set of stock pictures: each finger is a
three-segment chain driven by a flexion parameter, so a grasp that needs the
thumb opposed cannot be faked with a nicer drawing, and the pose can be
interpolated when the prediction changes. Because it is drawn from the back,
flexion renders as foreshortening -- the sideways sweep that is the obvious
thing to implement makes every grasp look like the same hand waving.

`test_viz.py` checks all 66 pairs of poses are visually distinct. Two movements
drawn alike would make a correct prediction unreadable, and no accuracy number
would catch it.

The backbone is trained with a classification loss and an episodic prototype loss
*simultaneously*, so one checkpoint serves all four conditions. Training separate
backbones would confound "this adaptation method is better" with "episodic
pretraining produces a better encoder". Setting `episodic_weight=0` gives that
ablation.

## Layout

```
src/remg/
  data/        records, NinaPro loader, synthetic source, preprocessing,
               windowing, normalization, splits
  models/      encoder, FiLM adapters, linear + prototype heads
  train/       augmentation, episode sampler, pretraining, the four conditions
  evaluate/    metrics, rejection curves, sensor-failure sweeps
  experiments/ the leave-one-subject-out personalization study
  viz/         the articulated hand the demonstration animates
scripts/       fetch_data.py, inspect_data.py, smoke.py, run_experiment.py,
               demo_hand.py
tests/         protocol guards, loader tests, synthetic .mat fixtures
```

## Scope notes

**DB6 is not a drop-in cross-session test set.** DB2 and DB3 share an acquisition
protocol, so a model pretrained on DB2 applies to DB3 directly. DB6 differs — a
different electrode count and a much smaller movement set, recorded from intact
subjects only — so weights do not transfer from a DB2-pretrained model. Two ways
forward, to be decided once the files are in hand and `inspect_data.py` has
confirmed the specifics:

- *Self-contained DB6 study.* Pretrain on DB6 subjects, hold one out, calibrate on
  day 1 and evaluate on day 5. Answers the cross-session question properly, on
  fewer movements, as a second experiment rather than an extension of the first.
- *Cross-repetition proxy within DB2/DB3.* Calibrate on early repetitions and test
  on late ones. Weaker — it captures drift within a session, not re-donning
  between days — but needs no extra data and is worth reporting either way.

The cross-repetition proxy is implemented and runs as part of the main
experiment (`recency_split`, on by default). It costs nothing: it regroups
predictions already made, splitting the held-out repetitions into those recorded
soonest after calibration and those recorded last, and reports the gap.
`drift_summary()` prints it. On synthetic data the gap is ~0, which is correct —
the generator models no drift — so it is plumbing-verified but tells us nothing
yet.

Recommendation: report the proxy for DB2/DB3, add the DB6 study if time allows,
and state plainly in the writeup which question each one answers.

**One DB3 subject is excluded, and it is not the one the literature warns about.**

S1 performed only 12 of the 23 grasping movements (global ids 18-29), so its
recording contains no tripod grasp (30) and no lateral grasp (34). The few-shot
conditions need a calibration example of every movement in `DEFAULT_SUBSET`, and
S1 can supply none for those two. The alternative was to drop both movements for
the whole cohort; excluding one subject keeps 10 amputees x 12 movements, which
beats 11 x 10, and lateral grasp is among the more functionally useful classes
for prosthetic control.

The exclusion lives in `remg/data/cohort.py` with its reason attached, is applied
by default, and is recorded in each run's `_meta.json` -- not passed as a flag
someone can forget. `--include-excluded` reverses it if you want to measure what
it cost. `test_protocol.py` pins the registry so an exclusion cannot be added or
reworded unnoticed.

S10's exercise-3 file is also partial (2 of 9 force patterns), but those are ids
41-49 and no experiment here uses them, so S10 stays.

**The variable-electrode-count warning did not reproduce.** An earlier version of
this file said several DB3 amputees were recorded with fewer than 12 electrodes.
Across the 10 subjects downloaded so far every file carries 12 channels. The
loader's `expected_channels` guard is still there and still correct to keep --
it is cheap, and the claim may yet hold for a subject not checked -- but it has
not fired on real data, and nothing has been excluded for it.

**The movement subset has been verified** against Table I of Atzori et al.,
"Building the NINAPRO Database" (BioRob 2012), cross-checked against the class
numbers cited in Jung et al., Front. Bioeng. Biotechnol. 9:548357 (2021), which
uses the same global numbering and agrees on every id the two sources share.

That check found one error. `lateral_grasp` was id 32, which is **Tip pinch
grasp**; Lateral grasp is 34. Both are real movements, so the mistake would have
trained and reported cleanly on a movement other than the one named — no crash,
no warning, just a mislabelled result. The id is now 34, on the reading that the
names in that dict express the intent and the id was the typo. `movements.py`
now carries the full official 1..40 list, and `test_protocol.py` pins every
subset entry to its official name, so this cannot drift again.

`inspect_data.py` still prints the per-class counts: it is the check that these
movements are actually *present* in the downloaded recordings, which no amount
of reading the literature can establish.

## Next

1. Fetch DB2 and DB3 with `scripts/fetch_data.py`, then run `inspect_data.py`.
   No registration is needed; this is just a large download.
2. Fix whatever `inspect_data.py` reports — the loader is tested against the
   published layout, not against the actual files.
3. Full leave-one-subject-out run: DB2 → DB3, shots 1/2/3, four conditions.
4. Sensor-failure sweep; read off the cross-repetition proxy from the same run.
5. Digital hand visualisation.
