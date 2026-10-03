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
| Protocol test suite (185 tests) | passing |
| NinaPro DB2/DB3/DB6 loader | run against real DB2 and DB3 files; label-numbering bug found and fixed |
| Movement subset | verified against the official movement list (one id was wrong — see Scope notes) |
| Cross-repetition session proxy | done |
| Real-data pipeline (DB2 -> DB3, end to end) | runs; `scripts/run_experiment.py` |
| DB3 download | 11 of 11 subjects; cohort coverage verified |
| DB2 download | 15 of 40 subjects (enough for the study) |
| DB6 download | 10 of 10 subjects, 20 GB; format verified against all 100 files |
| Cross-session experiment (day 1 → day 5) | **done**: DB6, 10 subjects, 3 seeds. `rapid` passes its pre-stated criterion — and `linear_probe` retains better still (`scripts/run_crosssession.py`) |
| Digital hand visualisation | done: `scripts/demo_hand.py` |
| Real results | complete: clean, sensor-failure and abstention. **The proposed method does not beat the baselines on any axis measured** |
| Benchmark against the published DB3 protocol | done: classic baselines reproduce the literature, so the pipeline is sound (`scripts/benchmark.py`) |
| Adaptation budget selected on held-out DB2 | done: every hand-set default was near the bottom of its grid (`scripts/tune_budget.py`) |
| Rest handling / false-activation rate | measured for every method; two-stage gate halves the rate but not the event count (`remg/train/twostage.py`, `scripts/tune_gate.py`) |
| Decision rules: debounce, classic `td_rf` gate | done; debounce fails its own accuracy goal, the classic gate beats the logistic one (`remg/evaluate/temporal.py`, `scripts/tune_rules.py`) |
| Matched-data comparison against `td_rf` | done: the network's margin is +0.036 balanced accuracy for 4.5x the false activations (`scripts/classic_fewshot.py`) |

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

### Rest handling: the cost of the class-balancing fix

Balanced accuracy counts rest as one class in twelve, which hides the failure mode
a prosthesis user would actually complain about. Two numbers expose it:
`false_activation_rate` is the fraction of true-rest windows predicted as a
movement (exactly `1 - rest_recall`), and `false_activations_per_min` counts
*events* -- maximal runs of movement prediction inside a rest stretch. The second
is not derivable from the first, and the difference matters: one sustained
ten-second error and fifty scattered 200 ms twitches give the same rate.

Training adaptation class-balanced was a real fix -- without it cross-entropy
collapses onto rest. But balancing discards the rest prior, and that cost was
never priced. The classic baselines, which train on the natural class balance,
show what it bought and what it cost (12-class subset, per-subject training,
smoothed, n=10):

| method | bal acc | rest recall | false-activation rate | false activations/min |
| --- | --- | --- | --- | --- |
| `td_rf` | 0.465 | **0.982** | **0.018** | **1.8** |
| `td_svm` | 0.480 | 0.970 | 0.030 | 3.4 |
| `td_lda` | 0.451 | 0.951 | 0.049 | 4.2 |
| `cnn_scratch` | 0.578 | 0.608 | 0.392 | 27.6 |
| `cnn_pretrained_ft` | **0.638** | 0.505 | 0.495 | 28.8 |

Same data, same protocol, same windows: a 10-16x difference in unwanted movements,
caused by a sampling choice. The CNN wins balanced accuracy by spending rest
accuracy to get it.

**The two-stage gate.** Put the prior back where it belongs: stage 1 decides rest
or movement and trains on the *natural* balance, stage 2 is the existing balanced
head with rest removed from the argmax (`remg/train/twostage.py`). Stage 1 is a
logistic regression on already-adapted embeddings -- 129 parameters, no gradient
steps, nothing added to the headline parameter counts. The threshold is an
operating point, so it is selected on held-out DB2 with a goal fixed in advance:
maximise balanced accuracy subject to false-activation rate <= 0.10
(`scripts/tune_gate.py`). DB3 plays no part.

On DB3 at 3 shots, paired per (subject, seed), n=30:

| condition | bal acc | macro F1 | rest recall | false-act rate | FA/min | mean burst |
| --- | --- | --- | --- | --- | --- | --- |
| `finetune` | 0.510 -> 0.483 | 0.386 -> **0.444** | 0.450 -> **0.749** | 0.550 -> **0.251** | 56.0 -> 49.7 | 576 -> **286 ms** |
| `rapid` | 0.353 -> 0.331 | 0.283 -> 0.307 | 0.537 -> 0.791 | 0.463 -> 0.209 | 47.3 -> 39.9 | 552 -> 290 ms |
| `linear_probe` | 0.376 -> 0.368 | 0.283 -> 0.324 | 0.433 -> 0.669 | 0.567 -> 0.331 | 61.3 -> 39.0 | 576 -> 418 ms |

Every rate and F1 change has p <= 1e-5; the balanced-accuracy losses are
significant too (p <= 0.04). Macro F1 *rises* because the gate removes false
positives, which F1 counts and recall-based balanced accuracy does not.

**What the event count reveals that the rate hides.** The per-window rate halves
almost entirely because each unwanted movement is half as long (576 -> 286 ms),
not because there are half as many: FA/min goes 56.0 -> 49.7, p = 0.05, and for
`rapid` 47.3 -> 39.9, p = 0.11 -- not reliably improved. Reporting only
`1 - rest_recall` would have implied half as many unwanted movements. A user would
still feel roughly one spurious activation per second of rest.

**Verdict.** The gate is the right shape of fix and it is not enough. `td_rf` with
per-subject training emits 1.8 false activations per minute at 0.465 balanced
accuracy; gated few-shot `finetune` emits 49.7 at 0.483. Those are different
protocols and not directly comparable, but inside the benchmark, where the
protocol is identical, the classic method beats the deep model 16x on false
activations *and* has higher plain accuracy. On this evidence `td_rf` is what you
would ship.

Two caveats. The <= 0.10 budget was met on DB2 (0.092) and missed on DB3 (0.251):
a threshold chosen where the rate is 0.295 is systematically too permissive where
it is 0.550, which is the honest price of not touching the test cohort. And
`none` + gate is not "no personalization" -- the gate is fitted on calibration
windows -- so that row is a reference, not a control.

### Decision rules: debounce, and a classic gate

Two further attempts at the false-activation problem, both operating points, both
selected on held-out DB2 with goals fixed before the sweeps ran
(`scripts/tune_rules.py`). DB3 plays no part.

**Debounce** (`remg/evaluate/temporal.py`) emits a movement only after it has been
predicted N windows running, and rest otherwise. It is strictly causal -- window i
is decided from i-n+1..i and nothing later -- so unlike the centred majority vote
used in the benchmark it could run on a device. Return to rest is immediate; only
activation is delayed, because a hand that keeps moving is the hazard. Goal:
minimise false activations per minute subject to balanced accuracy staying within
0.03 of undebounced.

It failed that goal. No N > 1 qualified -- even N = 2 costs 0.069 balanced accuracy
on DB2 -- so **N = 1, no debounce, was selected**. The rule works mechanically but
charges roughly 0.04 balanced accuracy per 100 ms of added onset latency:

| N | bal acc | macro F1 | FA/min | added latency |
| --- | --- | --- | --- | --- |
| **1** | **0.665** | 0.560 | 47.4 | 0 ms |
| 2 | 0.597 | 0.584 | 36.9 | 100 ms |
| 3 | 0.541 | 0.582 | 18.2 | 200 ms |
| 8 | 0.367 | 0.467 | 2.5 | 700 ms |

**A classic gate** (`remg/train/classic_gate.py`) uses the benchmark's own `td_rf`
-- 300 trees on MAV/RMS/WL/ZC/SSC from `remg.features`, natural class balance --
as stage 1, with the adapted network's movement argmax as stage 2. The motivation
is the benchmark measurement: `td_rf` reaches 0.982 rest recall where the CNN
manages 0.505, so each model does the thing it is demonstrably good at.

It beats the logistic gate on every axis. One structural property worth noting:
because stage 1 does not depend on stage 2, a gate decouples rest behaviour from
the movement classifier entirely -- the rest metrics are identical across
conditions.

**Results, `finetune` at 3 shots.** Reported twice: all 10 subjects, and the 5 whose
active/rest EMG amplitude ratio is >= 2.0. That threshold was **chosen after seeing
the signal-quality data** (`results/db3_signal_quality.csv`), so it is post-hoc
stratification for reporting, not a pre-registered criterion; the measured values
split 1.00/1.14/1.21/1.33/1.48 against 3.03/4.09/4.12/4.21/5.17/6.29, so the cut
lands in a wide empty gap. Adequate: S2, S3, S8, S9, S11.

| cohort | rule | bal acc | macro F1 | FA/min | mean burst |
| --- | --- | --- | --- | --- | --- |
| all 10 | single | **0.506** | 0.383 | 54.3 | 597 ms |
| | debounce N=3 (not selected) | 0.387 | 0.392 | 33.0 | 375 ms |
| | two_stage (logistic) | 0.481 | 0.442 | 49.9 | 289 ms |
| | **classic_gate (td_rf)** | 0.484 | **0.455** | **33.5** | 384 ms |
| | classic_gate+debounce | 0.360 | 0.417 | **11.0** | 287 ms |
| adequate (5) | single | **0.616** | 0.478 | 52.9 | 469 ms |
| | debounce N=3 (not selected) | 0.484 | 0.493 | 28.1 | 420 ms |
| | two_stage (logistic) | 0.597 | 0.566 | 27.0 | 223 ms |
| | **classic_gate (td_rf)** | 0.598 | **0.585** | **11.6** | 305 ms |
| | classic_gate+debounce | 0.463 | 0.546 | **3.1** | 290 ms |

Against no gate on the adequate cohort, `classic_gate` cuts false activations
**52.9 -> 11.6 per minute (78%)** for 0.018 balanced accuracy, with macro F1 up
0.107. Against the logistic gate it is better on balanced accuracy (p = 0.03),
macro F1 (p = 7e-06) and events (p = 2e-06).

**Debounce is strictly the worse way to buy rest safety here.** At N = 3 it costs
0.119 balanced accuracy to reach the same event rate the classic gate reaches for
0.021.

**Filtering by signal quality changes every magnitude and no ranking.** The gate is
twice as effective on adequate subjects (78% event cut against 38%), because stage
1 cannot detect activation in a recording where attempting to move barely changes
amplitude -- S7's ratio is 1.00. Balanced accuracy rises 0.506 -> 0.616. Half this
cohort is recordings in which the question is close to unanswerable.

**An artifact to read carefully.** Mean burst length *rises* as events fall:
`classic_gate` has longer bursts than `two_stage` (384 vs 289 ms, p = 2e-09)
despite a third fewer of them. Any rule that suppresses brief activations leaves
the long ones behind, so burst length must be read next to the event count, never
alone.

### The control that was missing: td_rf on the same few repetitions

Every earlier comparison against the classic baseline was unfair in one direction
or the other. `td_rf` looked excellent in the literature benchmark, but that gave it
**four** repetitions of the subject's own data, while the few-shot conditions get
one, two or three. So the question this project actually turns on -- is a shallow
model on hand-crafted features better than a pretrained network *in the regime the
project is about* -- had never been asked. `scripts/classic_fewshot.py` asks it:
identical subjects, identical calibration repetitions, identical test repetitions,
identical normalization and windows, the model being the only difference.

**A correction to the previous section's conclusion.** After the decision-rule work
the summary here claimed that per-subject `td_rf` beat the best few-shot
configuration "on both axes". That compared `td_rf` on four repetitions against the
network on three. Matched properly it does not hold: at three repetitions each,
`td_rf` reaches 0.562 balanced accuracy against the gated network's 0.598.
Pretraining buys something real, and the earlier framing understated it.

3 shots, mean over subjects:

| cohort | method | bal acc | macro F1 | FA/min | mean burst |
| --- | --- | --- | --- | --- | --- |
| all 10 | `finetune` | **0.506** | 0.383 | 54.3 | 597 ms |
| | `finetune` + classic gate | 0.484 | **0.455** | 33.6 | 384 ms |
| | **`td_rf` (few-shot)** | 0.393 | 0.431 | **4.88** | 252 ms |
| adequate (5) | `finetune` | **0.616** | 0.478 | 52.9 | 469 ms |
| | `finetune` + classic gate | 0.598 | 0.585 | 11.6 | 305 ms |
| | **`td_rf` (few-shot)** | 0.562 | **0.602** | **2.57** | 223 ms |

Paired at 3 shots, `td_rf` against each variant (`td_rf` is seed-invariant by
construction -- deterministic split, fixed `random_state` -- so it is replicated
across the network's seeds):

| cohort | comparison | bal acc | macro F1 | FA/min |
| --- | --- | --- | --- | --- |
| all 10 | vs `finetune` | -0.113 (p=1e-06) | **+0.048** (p=0.01) | **-49.4** (p=2e-06) |
| all 10 | vs + gate | -0.092 (p=2e-06) | -0.024 (p=0.07, tied) | **-28.7** (p=2e-06) |
| adequate | vs `finetune` | -0.054 (p=0.01) | **+0.124** (p=6e-05) | **-50.3** (p=6e-05) |
| adequate | vs + gate | -0.036 (p=0.015) | +0.017 (p=0.25, tied) | **-9.0** (p=6e-04) |

What the matched comparison says.

**The network's advantage is narrow.** On the adequate cohort it is **+0.036
balanced accuracy** over `td_rf`, and on macro F1 the two are statistically
indistinguishable (0.585 vs 0.602, p = 0.25).

**And it is expensive.** The network emits **4.5x more false activations** -- 11.6
per minute against 2.57 -- with the best rest-handling we built attached. Without
the gate it is 20x.

**Macro F1 is where the shallow model quietly wins.** `td_rf` beats *ungated*
`finetune` on F1 at every shot count in both cohorts while losing on balanced
accuracy. Balanced accuracy is pure recall and never asks how often a predicted
movement is wrong; F1 does. The network finds more movements and is less
trustworthy when it claims one.

**The cost asymmetry.** `td_rf` fits in 1-5 seconds per subject on CPU, needs no
GPU, and loads no DB2 at all -- there is nothing to pretrain. The network needs 15
DB2 subject-recordings, 3000 pretraining steps per seed and about 70 minutes of GPU
time per study, for +0.036 balanced accuracy and 4.5x the unwanted movements.

So the project's question has two honest answers that have to be given together.
Yes, a shared model does adapt to a new amputee from a few examples, and it works
better than it appeared before the budget was selected properly -- 3 shots reaches
86% of full per-subject training, up from 68%. And no, on this evidence it is not
what you would ship: against the cheap alternative given the same data, the margin
is 0.036 balanced accuracy, no F1 advantage, 4.5x the false activations and three
orders of magnitude more compute.

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

### Cross-session (DB6): the one scenario where `rapid` passes

Every result above is within-session -- the electrodes were never removed. Fitting
109,317 parameters to one session's electrode placement is exactly what should
overfit across a re-donning, and 664 adapter parameters is exactly the kind of
constraint that should survive it. DB6 answers it: 10 intact subjects, 5 days, 7
grasps, 14 electrodes.

Self-contained study (`scripts/run_crosssession.py`), because DB6 is a different
experiment: 14 electrodes against DB2's 12, so a DB2-pretrained encoder cannot even
be loaded. Pretraining is **leave-one-subject-out within DB6**. Calibration uses
repetitions of **day 1** only; each condition is then scored twice from that same
calibration -- on day 1's held-out repetitions and on **day 5** -- because day 5
alone cannot separate a method that retains well from one that was never good. The
normalizer is fitted on day-1 calibration windows and never re-fitted on day 5;
re-fitting would quietly correct the amplitude shift that re-donning causes, which
is most of what is being measured. Every hyperparameter comes from the files
selected on held-out DB2 subjects; nothing is tuned on DB6.

**The criterion was fixed before the data existed.** `rapid` wins only if BOTH: (1)
its day-1 -> day-5 drop is smaller than `finetune`'s by more than the seed noise
measured in this study, with a bootstrap 95% CI on the paired difference excluding
zero; and (2) its day-5 accuracy is not significantly worse than `finetune`'s.
Condition 2 exists because "degrades less" is trivially satisfied by a model that
starts bad and stays bad -- `none` passes condition 1 alone.

3 shots, mean over 10 subjects x 3 seeds (n = 30), balanced accuracy:

| method | day 1 | day 5 | drop | macro F1 (day 5) | FA/min (day 5) |
| --- | --- | --- | --- | --- | --- |
| `none` | 0.329 | 0.304 | **+0.025** | 0.262 | 24.2 |
| `td_rf` (few-shot) | 0.386 | 0.288 | +0.098 | 0.252 | **18.1** |
| `linear_probe` | 0.398 | 0.335 | +0.063 | 0.296 | 23.3 |
| `finetune` | 0.389 | 0.281 | +0.108 | 0.236 | 40.3 |
| `finetune` + classic gate | 0.393 | 0.284 | +0.109 | 0.237 | 36.1 |
| **`rapid`** | **0.412** | **0.339** | +0.073 | **0.300** | 20.3 |

**Both conditions pass.**

* Condition 1: `finetune` drops 0.1082, `rapid` drops 0.0731, difference
  **+0.0351**, bootstrap 95% CI **[+0.0193, +0.0506]** excluding zero. Seed noise
  measured here is **0.0053**, so the effect is 6.6x run-to-run variability.
  Wilcoxon p = 3.8e-04; `rapid` degrades less in 21 of 30.
* Condition 2: `rapid`'s day-5 accuracy is **0.3387** against `finetune`'s 0.2811 --
  not merely "not worse" but **+0.0576 better**, CI [+0.0352, +0.0828],
  p = 1.8e-05, better in 25 of 30.

This is the only axis in the entire project on which the proposed method wins, and
it wins on a criterion written down in advance.

**And the pre-registered failure mode fires too.** Before running, two outcomes were
named as fatal to the broader claim: `linear_probe` matching `rapid`'s retention, or
few-shot `td_rf` matching it. The first happened:

| comparison (3 shots, n=30) | `linear_probe` | `rapid` | difference | 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| day-5 balanced accuracy | 0.3346 | 0.3387 | +0.0041 | [-0.0055, +0.0127] | 0.17 |
| drop (day 1 -> day 5) | **0.0631** | 0.0731 | +0.0100 (rapid worse) | [+0.0032, +0.0171] | 0.023 |

A plain linear probe -- 1,548 parameters, no adapters, no episodic meta-learning --
**retains significantly better** than `rapid` and is statistically tied with it on
day-5 accuracy. So the mechanism is parameter count, not the adapters: what the data
supports is *"fitting the whole backbone to one session's electrode placement is
brittle, and touching fewer parameters is more robust"*. `rapid` is one way to touch
fewer parameters and not the best one here. That is the same shape of result as the
sensor-failure sweep, where `rapid` met its robustness threshold and `linear_probe`
got the same robustness for free.

`finetune` + classic gate does not help: the gate addresses rest behaviour, not
cross-session drift, and the drop is unchanged (0.109 against 0.108). Few-shot
`td_rf`, the strongest practical competitor within-session, degrades nearly as badly
as `finetune` (+0.098) -- hand-crafted amplitude features are themselves sensitive to
electrode placement.

One caveat on the hyperparameters: they were selected on held-out DB2 subjects for a
12-class, 12-channel problem, and DB6 is 8-class and 14-channel. That satisfies
"nothing tuned on DB6" and is the honest choice, but it is not the same as being
well-matched to DB6, and it cuts against every condition except `td_rf`, which has no
selected hyperparameters at all.

## The figures

`results/visuals/` holds the shareable output, and `results/project_page.html` is a
single self-contained page carrying all of it. None of it contains EMG: NinaPro asks
that its papers be cited and publishes no licence permitting redistribution of the
recordings, so every figure is built from model predictions, cued labels and summary
numbers, from which no signal can be reconstructed.

| file | what it shows |
| --- | --- |
| `day1_to_day5.png` | the cross-session result: five methods, day 1 to day 5, with 95% CIs |
| `rest_timeline.png` | a minute of recording with every false activation marked, gate on and off |
| `day1_vs_day5_hands.gif` | the same fine-tuned model decoding day 1 beside day 5 |
| `preds_db6.csv`, `preds_db3.csv` | the per-window predictions every figure is built from -- **not tracked in git** |

The two prediction tables are kept out of git: they are a few megabytes, they are
regenerable, and their `y_true` column is NinaPro's own annotation, which this project
cites but has no licence to redistribute in bulk. Regenerate them with:

```
uv run python scripts/dump_predictions.py --mode db6 --targets 1 2 6 --shots 3 --seed 0
uv run python scripts/dump_predictions.py --mode db3 --targets 3  --shots 3 --seed 0
```

The animation uses a rendered 3D hand (three.js in headless Chromium) rather than a
drawing. A 2D hand was tried first and abandoned: a closed hand loses its fingers
behind the palm, and a pinch and a fist share a silhouette, so the decoded hand did not
visibly change. See `assets/README.md`.

**Hand model:** the `generic-hand` profile from
[WebXR Input Profiles](https://github.com/immersive-web/webxr-input-profiles), by the
W3C Immersive Web Working Group, used under the
[W3C Software and Document License](https://www.w3.org/Consortium/Legal/copyright-software),
notice preserved in `assets/LICENSE.webxr-input-profiles.md`. Rendered with three.js
(MIT). The npm package for the model declares no licence field; the licence above comes
from the project repository.

**Grasps are labelled A-G, not named.** DB6's seven grasps are Large Diameter, Adducted
Thumb, Index Finger Extension, Medium Wrap, Writing Tripod, Power Sphere and Precision
Sphere (Palermo et al., IEEE ICORR 2017), but no source consulted states which recorded
id is which, so the poses are arbitrary and distinguishable rather than depictions --
see the DB6 note in `remg/data/movements.py`.

**The animation's window is chosen by a rule fixed in advance**, printed on the figure:
for each day independently, the 10-second window containing the most distinct
*intended* grasps, earliest start breaking ties. It reads the cued labels only and
never the predictions, so it cannot favour either day.

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
