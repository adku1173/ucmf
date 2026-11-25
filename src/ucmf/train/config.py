"""Configuration of incremental unfolded-solver training."""

from dataclasses import dataclass, field

import torch

MODEL_NAMES = (
    'AdaLFISTA',
    'AdaLFISTAR',
    'AdaLFISTASW',
    'AdaLFISTASWR',
    'NA_AdaLFISTA',
    'NA_AdaLFISTAR',
    'NA_AdaLFISTASW',
    'NA_AdaLFISTASWR',
)
LOSSES = ('mse', 'nmse', 'relative_residual')
SUPERVISED_LOSSES = ('mse', 'nmse')
OPERATOR_NAMES = ('MatrixMult', 'CMFDictionary', 'CMFPropagation')
PRECISIONS = ('float32', 'float64', 'bfloat16')


@dataclass
class TrainingConfig:
    """Configuration of :func:`ucmf.train.train_model`.

    Attributes
    ----------
    model_name : str
        Trainable unfolded solver, one of ``MODEL_NAMES``.
    model_kwargs : dict
        Keyword arguments of the solver constructor.
    operator_name : str
        'MatrixMult', 'CMFDictionary' or 'CMFPropagation'; built by ``build_operator``.
    operator_kwargs : dict
        Keyword arguments of the operator constructor, e.g. ``{'matrix': A}``.
    stages : int
        Number of incremental stages; stage ``t`` trains at solver depth ``t``.
    steps_per_stage : int
        Optimizer steps per stage.
    batch_size : int
        Training batch size (default 256); batches are drawn without replacement per epoch.
    learning_rate : float
        Adam learning rate; a fresh optimizer starts at every stage.
    loss : str
        'mse' or 'nmse' (supervised, need source targets) or 'relative_residual'.
    val_every : int
        Validation/reporting interval in steps within a stage.
    seed : int
        Seed of the batch-sampling generator on the configured device.
    grad_clip_norm : float or None
        Max gradient norm; None (default) disables clipping.
    device : str
        Torch device string ('cpu', 'cuda', ...); model and data are moved there.
    precision : str
        One of ``PRECISIONS`` (default 'float32'); the operator must already match.
    """

    model_name: str
    operator_name: str
    model_kwargs: dict = field(default_factory=dict)
    operator_kwargs: dict = field(default_factory=dict)
    stages: int = 15
    steps_per_stage: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    loss: str = 'mse'
    val_every: int = 1
    seed: int = 0
    grad_clip_norm: float | None = None
    device: str = 'cuda'
    precision: str = 'float32'

    def __post_init__(self):
        """Validate the configuration."""
        if self.model_name not in MODEL_NAMES:
            raise ValueError(f'model_name must be one of {MODEL_NAMES}, got {self.model_name!r}.')
        if self.operator_name not in OPERATOR_NAMES:
            raise ValueError(f'operator_name must be one of {OPERATOR_NAMES}, got {self.operator_name!r}.')
        if self.loss not in LOSSES:
            raise ValueError(f'loss must be one of {LOSSES}, got {self.loss!r}.')
        for name in ('stages', 'steps_per_stage', 'batch_size', 'val_every'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive, got {getattr(self, name)}.')
        if self.learning_rate <= 0:
            raise ValueError(f'learning_rate must be positive, got {self.learning_rate}.')
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError(f'grad_clip_norm must be positive or None, got {self.grad_clip_norm}.')
        if self.precision not in PRECISIONS:
            raise ValueError(f'precision must be one of {PRECISIONS}, got {self.precision!r}.')
        try:
            torch.device(self.device)
        except RuntimeError as error:
            raise ValueError(f'device must be a valid torch device, got {self.device!r}.') from error

    @property
    def torch_device(self):
        """Torch device of ``device``."""
        return torch.device(self.device)

    @property
    def torch_dtype(self):
        """Torch dtype of ``precision``."""
        return getattr(torch, self.precision)

    @property
    def supervised(self):
        """Whether the loss needs source-strength targets ``q_train``/``q_val``."""
        return self.loss in SUPERVISED_LOSSES
