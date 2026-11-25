"""Configuration of incremental unfolded-solver training."""

import inspect
from dataclasses import dataclass, field

from .objectives import OBJECTIVES, SPLITS, SUPPLIED

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
LOSSES = tuple(OBJECTIVES)
SUPERVISED_LOSSES = ('nmse',)
DATA_SCALINGS = ('none', 'equireg')
BATCH_SAMPLINGS = ('random', 'per_dictionary')
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
        Keyword arguments of the operator constructor, e.g. ``{'matrix': A}``. A leading
        dimension ``K`` (``matrix`` ``(K, m, n)``, ``H`` ``(K, c, n)``) makes the operator a
        dictionary bank, see ``batch_sampling``.
    stages : int
        Number of incremental stages; stage ``t`` trains at solver depth ``t``.
    steps_per_stage : int
        Optimizer steps per stage.
    batch_size : int
        Training batch size (default 256); batches are drawn without replacement per epoch.
    learning_rate : float
        Adam learning rate; a fresh optimizer starts at every stage.
    loss : str
        Objective of :mod:`ucmf.train.objectives`; 'nmse' (default) needs source targets.
    loss_kwargs : dict
        Keyword-only parameters of the objective besides ``w``, e.g. ``{'lam': 0.1}`` for 'lasso'.
    batch_sampling : str
        How training batches are drawn, 'random' (default) or 'per_dictionary'.

        Data layout: without a bank, one operator serves all samples. With a bank of ``K``
        dictionaries (e.g. one per frequency bin), :func:`~ucmf.train.train_model` gets
        ``index_train`` ``(N,)`` and ``index_val`` ``(M,)``: the bank entry of every row of
        ``y``/``q``. The rows may be in any order (e.g. grouped by frequency). How often an
        entry occurs is the sample distribution that training sees, so a weighting such as
        log-uniform frequencies belongs into the data (number of rows per entry), not here.

        'random': every epoch permutes all samples and splits them into full batches. A
        batch mixes entries in proportion to their row counts; every sample is solved with
        its own dictionary (a per-problem operator ``(B, ...)``). Works with and without a
        bank.

        'per_dictionary': every epoch groups the shuffled samples by entry, splits each
        group into full batches, and shuffles the batch order. All samples of a batch share
        one dictionary (a shared operator, which is cheaper per step). An entry contributes
        ``count // batch_size`` batches: its incomplete batch is dropped and entries with
        fewer than ``batch_size`` rows are never trained, which skews small datasets toward
        frequent entries. Requires a bank with at least one entry of ``batch_size`` rows.

        Validation ignores this setting and always uses mixed batches (one dictionary per
        sample), see :func:`~ucmf.train.utils.validation_batches`.
    w : float
        Weight base in ``[0, 1]`` of earlier iterates: iterate ``k`` of depth ``t`` enters
        with ``w ** (t - k)``; 0 (default) uses only the final iterate.
    data_scaling : str
        'equireg' (default) divides measurements and targets per sample by
        ``lambda_max`` of the normalized operator (nonnegative models only); 'none' keeps them.
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
    compile : bool
        If True (default), solve with :func:`~ucmf.optim.compile.compile_unrolled` of
        depth ``stages``: one graph serves all stages, which still compute all ``stages``
        iterations.
    """

    model_name: str
    operator_name: str
    model_kwargs: dict = field(default_factory=dict)
    operator_kwargs: dict = field(default_factory=dict)
    stages: int = 15
    steps_per_stage: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    loss: str = 'nmse'
    loss_kwargs: dict = field(default_factory=dict)
    batch_sampling: str = 'random'
    w: float = 0.0
    data_scaling: str = 'equireg'
    val_every: int = 1
    seed: int = 0
    grad_clip_norm: float | None = None
    device: str = 'cuda'
    precision: str = 'float32'
    compile: bool = True

    def __post_init__(self):
        """Validate the configuration."""
        if self.model_name not in MODEL_NAMES:
            raise ValueError(f'model_name must be one of {MODEL_NAMES}, got {self.model_name!r}.')
        if self.operator_name not in OPERATOR_NAMES:
            raise ValueError(f'operator_name must be one of {OPERATOR_NAMES}, got {self.operator_name!r}.')
        if self.loss not in LOSSES:
            raise ValueError(f'loss must be one of {LOSSES}, got {self.loss!r}.')
        parameters = inspect.signature(OBJECTIVES[self.loss]).parameters.values()
        keywords = {p.name for p in parameters if p.kind is p.KEYWORD_ONLY} - set(SUPPLIED)
        required = {p.name for p in parameters if p.kind is p.KEYWORD_ONLY and p.default is p.empty}
        required -= set(SUPPLIED)
        if not required <= set(self.loss_kwargs) <= keywords:
            raise ValueError(
                f'loss {self.loss!r} needs loss_kwargs {sorted(required)}, got {sorted(self.loss_kwargs)}.'
            )
        if not 0 <= self.w <= 1:
            raise ValueError(f'w must lie in [0, 1], got {self.w}.')
        if self.batch_sampling not in BATCH_SAMPLINGS:
            raise ValueError(f'batch_sampling must be one of {BATCH_SAMPLINGS}, got {self.batch_sampling!r}.')
        if self.data_scaling not in DATA_SCALINGS:
            raise ValueError(f'data_scaling must be one of {DATA_SCALINGS}, got {self.data_scaling!r}.')
        if self.loss == 'cv' and self.loss_kwargs.get('split', 'rows') not in SPLITS:  # 'rows': cv_loss default
            raise ValueError(f'split must be one of {tuple(SPLITS)}, got {self.loss_kwargs["split"]!r}.')
        if self.data_scaling == 'equireg' and self.model_kwargs.get('nonnegative', True) is False:
            raise ValueError("data_scaling='equireg' requires a nonnegative model.")
        if self.loss == 'cv' and self.model_kwargs.get('nonnegative', True) is False:
            raise ValueError("loss='cv' requires a nonnegative model (lambda_max).")
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
