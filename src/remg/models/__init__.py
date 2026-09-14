from .adapters import FiLM, adapter_drift, reset_adapters
from .backbone import EMGEncoder
from .heads import EMGClassifier, LinearHead, PrototypeHead

__all__ = [
    "FiLM",
    "adapter_drift",
    "reset_adapters",
    "EMGEncoder",
    "EMGClassifier",
    "LinearHead",
    "PrototypeHead",
]


def build_model(n_channels: int, n_classes: int, **kwargs) -> EMGClassifier:
    """Standard construction path used by the experiment scripts."""
    temperature = kwargs.pop("temperature", 10.0)
    encoder = EMGEncoder(n_channels=n_channels, **kwargs)
    return EMGClassifier(encoder, n_classes=n_classes, temperature=temperature)
