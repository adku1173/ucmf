"""Result of incremental unfolded-solver training."""

from dataclasses import dataclass

from .config import TrainingConfig


@dataclass
class TrainingResult:
    """Outcome of :func:`ucmf.train.train_model`.

    Attributes
    ----------
    config : TrainingConfig
        Configuration of the run.
    history : list of dict
        Every validation point exactly as reported to Ray: ``stage``, global ``step``,
        ``train_loss``, ``val_loss``, and ``stage_best_val_loss``.
    best_val_loss : float
        Best validation loss of the last stage, i.e. of the returned model.
    state_dict : dict
        Trained model state on the CPU.
    """

    config: TrainingConfig
    history: list
    best_val_loss: float
    state_dict: dict
