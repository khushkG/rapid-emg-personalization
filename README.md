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
| Metrics, sensor-failure sweep, rejection curves | done; **run on real data** (see Results) |
| Protocol test suite (155 tests) | passing |
| NinaPro DB2/DB3/DB6 loader | run against real DB2 and DB3 files; label-numbering bug found and fixed |
| Movement subset | verified against the official movement list (one id was wrong — see Scope notes) |
| Cross-repetition session proxy | done |
| Real-data pipeline (DB2 -> DB3, end to end) | runs; `scripts/run_experiment.py` |
| DB3 download | 11 of 11 subjects; cohort coverage verified |
| DB2 download | 15 of 40 subjects (enough for the study) |
| Cross-session experiment (day 1 → day 5) | needs DB6 (see Scope notes) |
| Digital hand visualisation | done: `scripts/demo_hand.py` |
| Real results | complete: clean, sensor-failure and abstention. **The proposed method does not beat the baselines on any axis measured** |
| Benchmark against the published DB3 protocol | done: classic baselines reproduce the literature, so the pipeline is sound (`scripts/benchmark.py`) |
| Adaptation budget selected on held-out DB2 | done: every hand-set default was near the bottom of its grid (`scripts/tune_budget.py`) |

Everything runs end-to-end today on synthetic data:

```
uv run python scripts/smoke.py     # ~1 minute
```

Those numbers are plumbing checks, not findings. The data is simulated.

The tests are the other half of that: 147 of them, covering the protocol rules
below and the NinaPro loader against synthetic files written in the real `.mat`
format (both v5 and v7.3 containers, the per-exercise label restart, and the
reduced-channel amputee recordings).

## Results

Leave-one-subject-out, 15 DB2 subjects pretraining, 10 DB3 amputees evaluated,
3000 steps, **3 seeds** (n = 30 subject-seed pairs). Balanced accuracy over 12
classes; chance is 0.083. Adaptation step counts and learning rates are selected
on **held-out DB2 subjects** (`scripts/tune_budget.py`); DB3 never participates.
Source: `results/tuned15_rows.csv`.

| shots | `none` | `linear_probe` | **`finetune`** | `rapid` |
| --- | --- | --- | --- | --- |
| 1 | 0.163 | 0.307 | **0.395** | 0.310 |
| 2 | 0.159 | 0.350 | **0.477** | 0.333 |
| 3 | 0.155 | 0.375 | **0.508** | 0.351 |

Paired per (subject, seed), with bootstrap 95% confidence intervals:

| comparison | shots=3 mean | 95% CI | wins | p |
| --- | --- | --- | --- | --- |
| `finetune` - `rapid` | **+0.157** | [+0.132, +0.181] | 28/30 | 9e-09 |
| `finetune` - `linear_probe` | **+0.133** | [+0.113, +0.150] | 29/30 | 4e-09 |
| `rapid` - `linear_probe` | **-0.024** | [-0.038, -0.010] | 7/30 | 0.0015 |

**The central hypothesis is not supported.** `rapid` -- FiLM adapters plus
prototypes, 664 parameters -- loses to the cheap linear-probe baseline, and loses
to ordinary full fine-tuning by 0.157 in 28 of 30 runs.

Seed-to-seed spread of the cohort mean is **0.007**, so the gap to `finetune` is
more than twenty times run-to-run noise. This is not a variance artifact.

Earlier runs used hand-set adaptation budgets that were never selected by any
procedure (`results/main15_rows.csv`, kept for provenance). Those budgets
understated every condition and understated `finetune` most, which flattered the
proposed method -- see *The adaptation budget was never selected* below. The
robustness, abstention and drift sections further down were measured at the old
budget and have not been rerun.

What *is* supported, clearly: personalization works. One calibration repetition
takes balanced accuracy from 0.16 to 0.31-0.40 -- roughly double to two and a
half times, against a general model sitting near twice chance. The question was never whether to
personalize; it is whether this way is better, and it is not.

### Benchmark against published DB3 results

Is the model weak, or is the few-shot setup simply harder than the published
protocols? Answered by running the standard NinaPro protocol -- per-subject
training on repetitions 1/3/4/6, testing on 2/5 -- with `scripts/benchmark.py`.
Mean over subjects, majority-vote smoothed over 5 windows (500 ms):

| movement set | method | acc (with rest) | acc (no rest) | bal acc | macro F1 |
| --- | --- | --- | --- | --- | --- |
| full, 39-50 classes | `td_lda` | 0.506 | 0.302 | 0.331 | 0.352 |
| | `td_svm` | 0.548 | 0.346 | 0.381 | 0.427 |
| | `td_rf` | **0.556** | 0.364 | 0.394 | 0.439 |
| | `cnn_scratch` | 0.303 | 0.442 | 0.439 | 0.361 |
| | `cnn_pretrained_ft` | 0.297 | 0.414 | 0.417 | 0.339 |
| 12-class subset | `td_svm` | 0.793 | 0.409 | 0.480 | 0.523 |
| | `cnn_scratch` | 0.596 | 0.568 | 0.578 | 0.480 |
| | `cnn_pretrained_ft` | 0.535 | **0.643** | **0.632** | 0.481 |

**The pipeline is sound.** Published classic-feature SVM on DB3 sits around 46%
plain accuracy; ours is 54.8% on the same full movement set under the same
protocol. That baseline shares this repository's loader, windowing, filtering,
label handling and split logic, so the agreement validates all of it
independently of any deep learning.

**The published 66-85% deep-learning band is not measuring what we measure.** Our
CNN gets 0.30 plain accuracy on the full set yet beats every classic method on
accuracy-excluding-rest and on balanced accuracy. Rest is ~79% of windows and we
train class-balanced, so the CNN forgoes a prior those numbers bank. On the
12-class subset the same model scores 0.535 plain and 0.632 balanced -- one model,
numbers 10 points apart, depending only on how rest is counted.

**Recording quality dominates DB3.** S7's EMG amplitude during attempted movement
is 1.00x its resting amplitude -- the signal does not change when the subject
tries to move -- and two of its twelve channels are flat. Across the 11 subjects,
active/rest amplitude ratio predicts balanced accuracy at Spearman rho = 0.84,
p = 0.0013 (`results/db3_signal_quality.csv`); four subjects sit below 1.5x. No
architecture change moves that ceiling.

### The adaptation budget was never selected

The benchmark exposed an asymmetry that has nothing to do with the few-shot
regime: per-subject training fine-tuned for 600 steps at lr 3e-4, while the
few-shot `finetune` condition used **100 steps at lr 1e-4** -- a default, not a
choice. `scripts/tune_budget.py` selects it properly, splitting DB2 by person:
subjects 1-11 pretrain, 12-15 are held out and treated exactly like unseen
targets. DB3 is never loaded (`db3_loaded: false` in the meta file).

Every hand-set default turned out to be at or near the bottom of its own grid:

| condition | old default | rank in its grid | selected | DB2 bal acc |
| --- | --- | --- | --- | --- |
| `finetune` | 100 @ 1e-4 | **12th of 12** | 300 @ 3e-4 | 0.618 -> 0.669 |
| `linear_probe` | 200 @ 1e-3 | **6th of 6** | 600 @ 3e-3 | 0.526 -> 0.568 |
| `rapid` | 50 @ 5e-3 | 11th of 12 | 600 @ 5e-3 | 0.511 -> 0.532 |

All three adaptive conditions were re-selected, not only `finetune`. Fixing the
winning condition's budget alone would have replaced one unfair comparison with
its mirror image.

The fix helps, and it helps the baselines most. On DB3 at 3 shots `finetune` gains
+0.105, `linear_probe` +0.047 and `rapid` +0.020, while `none` moves by -0.002,
which is the MPS run-to-run noise floor. The `finetune`-`rapid` gap doubled from
0.072 to 0.157, and `rapid` went from tying `linear_probe` (p = 0.98) to losing to
it (p = 0.0015).

**`rapid` is saturated.** A 24x increase in its adaptation steps buys +0.020, and
its DB2 grid spans 0.049 across twelve configurations with the top seven inside
0.007. Its 664 parameters are the binding constraint, not its training budget.

**How close does few-shot get to full per-subject training?** Three calibration
repetitions reach **86% of the unsmoothed per-subject reference** (0.508 against
0.590), up from 68% at the old budget; against the smoothed 0.632 it is 80%. The
two protocols hold out different repetitions, so this is close-but-not-exact.
Roughly half of what looked like "the few-shot regime is harder" was an untuned
default.

One caveat on the selection machinery itself. The tie-break prefers the cheapest
configuration within one SEM of the best, and the first implementation measured
cost in wall-clock seconds. The host paged into swap mid-search and recorded 91s
for a 600-step configuration against 21s for a 1200-step one, inverting the
ordering for `rapid`. Cost is now counted in **steps** (`config_steps`, guarded by
`tests/test_budget_selection.py`). The completed DB3 run used the wall-clock pick
of 1200 steps for `rapid`; the step-rule pick is 600 steps at 0.5318 against
0.5341 -- inside the SEM band, and in the direction of giving the proposed method
*more* budget -- so the run was not repeated. `results/budget_chosen.json` is what
that run used; `results/budget_chosen_steprule.json` is what the corrected rule
selects and what new runs should use.

### The rescue hypothesis, tested and rejected

`rapid` depends on episodic meta-learning, whose episodes are sampled across
source subjects, so the obvious objection to an earlier 7-subject run was that
the episodic objective had too little subject diversity to work with. Doubling
the pretraining cohort to 15 does not rescue it (both columns at the old
hand-set budget, which is the comparison that was available at the time):

| | 7 subjects | 15 subjects | gain |
| --- | --- | --- | --- |
| `rapid` | 0.307 | 0.331 | +0.024 |
| `linear_probe` | 0.305 | 0.328 | +0.023 |
| `finetune` | 0.389 | 0.403 | +0.014 |

More pretraining data helps everything, and helps `rapid` no more than it helps
a linear probe. The gap to `finetune` widened slightly rather than closing.

### The result that was wrong first

An earlier run had `rapid` ahead by 0.09, winning 8-9 of 9 subjects. That was an
artifact, and how it happened is worth recording.

`rapid` classifies with per-class prototypes, which weight every class equally
however many windows it has. `linear_probe` and `finetune` trained with plain
cross-entropy on calibration windows that are ~79% rest, so they learned the
rest prior and collapsed onto predicting it -- rest recall 0.98, plain accuracy
0.72, balanced accuracy 0.21. The proposed method was being measured against
baselines crippled by an imbalance it is immune to by construction.

Pretraining had always sampled class-balanced; adaptation did not.
`AdaptConfig.class_balanced` (on by default, `--unbalanced-calibration` to
reproduce the old behaviour) closes the gap entirely and lifts every baseline.

The headline metric looked like a clean, significant win. What caught it was the
**per-class recall table**, where 0.98 rest recall made the collapse obvious.

### Robustness to electrode failure: hypothesis met, claim still dead

`rapid` touches 664 parameters against 109,317, so the remaining live question
was whether it is less brittle when an electrode fails. **A threshold was fixed
before running**: `rapid` wins only if it degrades less than `finetune` by more
than 0.013, the measured seed noise.

It met that threshold:

| electrodes failed | `finetune` drops | `rapid` drops | difference | 95% CI |
| --- | --- | --- | --- | --- |
| 1 | 0.066 | 0.040 | **+0.026** | [+0.020, +0.032] |
| 2 | 0.112 | 0.069 | **+0.043** | [+0.033, +0.055] |

`rapid` degrades less in 28 of 30 runs. Fewer parameters really are less
brittle. Two further checks kill the claim anyway.

**`finetune` never loses the lead.** It starts far enough ahead that degrading
faster does not cost it the comparison:

| electrodes failed | `finetune` | `rapid` | `finetune` still ahead in |
| --- | --- | --- | --- |
| 0 | **0.401** | 0.329 | -- |
| 1 | **0.336** | 0.290 | 24/30 |
| 2 | **0.289** | 0.260 | 23/30 |

Same on the worst single electrode combination (0.202 against 0.182), which is
the number that matters for a controller: surviving a random electrode failure
while breaking on one particular electrode is not robustness.

**The robustness is not `rapid`'s.** Against the cheap baseline, the difference
in degradation is `-0.003` (1ch) and `+0.000` (2ch), both CIs spanning zero,
p ~ 0.6. So the finding is *"full fine-tuning is more brittle"*, and a plain
linear probe buys the same robustness with none of the machinery. `rapid`
matches `linear_probe` on clean accuracy **and** on robustness -- it has no
advantage on any axis measured.

### Abstention and drift

Both were run on the same predictions and neither separates the conditions in
`rapid`'s favour. The rejection curves are in `results/robust15_curves.csv`,
plotted only where every class survives the threshold -- past that point
balanced accuracy is averaging over a smaller class set and is not comparable to
the full-coverage number.

The cross-repetition proxy shows `finetune` losing the most between early and
late held-out repetitions (-0.018 at 3 shots against -0.011 for `linear_probe`),
consistent with the brittleness above. It is same-session drift, not re-donning,
and too small to carry a claim.

### Genuinely still open

* **Cross session.** Fine-tuning a whole backbone on one session's electrode
  placement is exactly what should overfit across a re-donning, and it is the
  one scenario where a 664-parameter method has a real case. Needs DB6, which is
  not downloaded and whose URL layout is unverified.

## Setup

```
curl -LsSf https://astral.sh/uv/install.sh | sh     # once
uv sync --group dev
```

## Checking it works

```
uv run python scripts/check.py            # ~4 min, needs no data
uv run python scripts/check.py --quick    # ~3 min, tests only
uv run python scripts/check.py --full     # also runs a real experiment
```

Four levels, cheapest first, each answering a different question, so a failure
says *where* the problem is rather than only that there is one:

| Level | Question | Needs |
| --- | --- | --- |
| 1. tests | Is the logic right? | nothing |
| 2. synthetic | Do the stages connect end to end? | nothing |
| 3. data | Do the real files match what the loader assumes? | a download |
| 4. experiment | Does it produce results on real amputee data? | both cohorts |

A green level 1 with a red level 3 means the code is fine and the data is not,
which is a different morning's work from the reverse. Levels whose inputs are
missing are reported as skipped rather than failed -- an undownloaded cohort is
work not yet done, not a broken project.

Level 2 does one thing beyond checking that the code runs: it fails if adapting
to a subject does not beat not adapting, even on simulated data. Every stage can
execute cleanly with the adaptation wired to nothing.

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
scripts/       check.py, fetch_data.py, inspect_data.py, smoke.py,
               run_experiment.py, demo_hand.py
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
Across all 11 subjects every file carries 12 channels. The
loader's `expected_channels` guard is still there and still correct to keep --
it is cheap -- but it never fired on the real cohort, and nothing has been
excluded for it.

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

The study is complete. What remains is one experiment and one decision.

1. **The cross-session study.** The only unanswered part of the original
   question, and the one scenario that could still favour a low-parameter
   method. Needs DB6 downloaded and its URL layout confirmed.
2. **Decide what this project reports.** The honest headline is that
   personalization from three repetitions works well -- 0.16 to 0.40 balanced
   accuracy over 12 classes -- and that ordinary fine-tuning is the best way to
   do it among those tested. That is a useful negative result about adapters,
   not a failed project, provided it is written up as what it is.

Reproduce the headline numbers with:

```
uv run python scripts/run_experiment.py --steps 3000 --shots 1 2 3 --seeds 0 1 2 \
    --tag robust15 --no-notch
uv run python scripts/make_report.py --tag robust15
```
