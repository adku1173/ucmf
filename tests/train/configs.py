"""Small training configurations shared by the training tests (``train`` extra required)."""

from ucmf.train import TrainingConfig

DEFAULTS = {
    'model_name': 'AdaLFISTASW',
    'stages': 2,
    'steps_per_stage': 2,
    'batch_size': 4,
    'device': 'cpu',
    'precision': 'float64',
    'compile': False,  # eager for speed; test_compiled_training_matches_eager covers compile
}


def training_config(**overrides):
    """Return a ``TrainingConfig`` with small eager float64 CPU defaults; keywords override them."""
    return TrainingConfig(**{**DEFAULTS, **overrides})
