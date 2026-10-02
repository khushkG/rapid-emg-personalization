"""Hand-crafted EMG features.

Here because two very different things need the identical implementation: the
literature benchmark (`scripts/benchmark.py`), where classic features are the
baseline that validates the pipeline, and the classic rest gate
(`remg/train/classic_gate.py`), where they are stage one of a hybrid controller. A
second copy would eventually drift from the first, and then the gate and the
baseline would no longer be the same method under two names.
"""

from .td import TD_FEATURE_NAMES, fit_zc_threshold, td_features

__all__ = ["TD_FEATURE_NAMES", "fit_zc_threshold", "td_features"]
