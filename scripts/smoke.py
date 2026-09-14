"""Fast end-to-end run on synthetic data.

Purpose is plumbing, not science: it proves every stage connects and that the
conditions produce sensible relative behaviour, in a couple of minutes on a
laptop. Numbers printed here are NOT results -- the data is simulated.

    uv run python scripts/smoke.py
"""

from __future__ import annotations

import time

from remg.data import PreprocessConfig, WindowConfig, preprocess, segment
from remg.data.synthetic import make_cohort
from remg.experiments import ExperimentConfig, drift_summary, headline, run
from remg.train import AdaptConfig, PretrainConfig
from remg.train.sampler import EpisodeConfig


def build(n_subjects: int, amputee: bool, seed_offset: int = 0):
    recs = make_cohort(
        n_subjects,
        dataset="amp" if amputee else "intact",
        amputee=amputee,
        fs=2000,
        n_repetitions=6,
        move_s=3.0,
        rest_s=1.5,
    )
    recs = [preprocess(r, PreprocessConfig(target_fs=1000)) for r in recs]
    return segment(recs, cfg=WindowConfig(length_ms=200, stride_ms=100))


def main() -> None:
    t0 = time.time()
    print("building synthetic cohorts ...", flush=True)
    source = build(8, amputee=False)
    target = build(4, amputee=True)
    print(f"  source {source.X.shape}  target {target.X.shape}  "
          f"classes {source.n_classes}  ({time.time() - t0:.0f}s)", flush=True)

    cfg = ExperimentConfig(
        shots=(1, 3),
        seeds=(0,),
        failure_counts=(1,),
        failure_max_combinations=4,
        rejection=False,
        pretrain=PretrainConfig(
            steps=400,
            batch_size=128,
            episode=EpisodeConfig(n_support_reps=2, n_support=4, n_query=8),
        ),
        adapt=AdaptConfig(finetune_steps=40, probe_steps=60, rapid_steps=25),
    )
    rows, _ = run(source, target, cfg)

    print("\n" + headline(rows))
    print("\ncross-repetition session proxy (within-session drift):")
    print(drift_summary(rows))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
