"""Incremental training of unfolded solvers; requires the optional ``train`` extra (Ray)."""

from .config import TrainingConfig
from .incremental import train_model
from .result import TrainingResult

__all__ = ['TrainingConfig', 'TrainingResult', 'train_model']
