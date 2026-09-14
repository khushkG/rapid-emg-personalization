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
| Protocol test suite (13 tests) | passing |
| NinaPro DB2/DB3/DB6 loader | written, **not yet run against real files** |
| Cross-session experiment | needs DB6 (see Scope notes) |
| Digital hand visualisation | not started |
| Real results | blocked on the NinaPro download |

Everything runs end-to-end today on synthetic data:

```
uv run python scripts/smoke.py     # ~1 minute
```

Those numbers are plumbing checks, not findings. The data is simulated.

## Setup

```
curl -LsSf https://astral.sh/uv/install.sh | sh     # once
uv sync --group dev
uv run --group dev pytest -q
```

Torch runs on the M-series GPU via MPS; `remg.utils.pick_device` selects it
automatically.

## Getting the data

NinaPro requires a (free) account and acceptance of its licence, so the download
cannot be automated:

1. Register at <https://ninapro.hevs.ch/> and accept the data licence.
2. Download **DB2** (40 intact subjects) and **DB3** (11 transradial amputees).
   DB6 is only needed for the cross-session experiment.
3. Unpack so the `.mat` files sit under `data/raw/DB2/` and `data/raw/DB3/`.
   Nested per-subject folders are fine — the loader searches recursively.
4. Verify before training anything:

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
alongside. Rest occupies roughly six times as many windows as any single
movement, so plain accuracy rewards a model that predicts rest for everything.

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
scripts/       smoke.py, inspect_data.py
tests/         protocol guards
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

The cross-repetition proxy already works with the current code. Recommendation:
do it now, add the DB6 study if time allows, and state plainly in the writeup
which question each one answers.

**Some DB3 amputees were recorded with fewer than 12 electrodes** (shorter stumps).
The loader reports and skips them by name rather than silently guessing, so any
exclusion is a documented one.

**The movement subset in `movements.py` needs verifying** against the official
NinaPro movement figures. The ids there are the intended 12 practical movements,
but `inspect_data.py` prints the per-class counts so a mismatch surfaces
immediately.

## Next

1. Register for NinaPro, download DB2 and DB3, run `inspect_data.py`.
2. Confirm the movement subset ids against the official movement list.
3. Full leave-one-subject-out run: DB2 → DB3, shots 1/2/3, four conditions.
4. Sensor-failure sweep and the cross-repetition session proxy.
5. Digital hand visualisation.
