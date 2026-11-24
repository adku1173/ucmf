"""Two-weight reduced Ada-LFISTA (Ada-LFISTA-R) for CMF measurements.

Like ``AdaLFISTA``, but both ``m x m`` auxiliary weights are congruences with
learned complex ``c x c`` matrices ``Br1`` (recurrence ``R``) and ``Br2``
(measurement branch ``M``), as in ``AdaLFISTASWR``. Neither weight is
materialized. ``Br1 = Br2 = I`` recovers FISTA.
"""

from .ista import FISTA
from .learned_params import LearnedParams
from .neural_augmentation import NeuralAugmentation
from .weights import make_propagation_weight_operator, validate_cmf_operator

import torch
from torch import nn


class _ReducedTwoWeights(FISTA):
    """Update direction ``(M.T @ y - R.T @ (R @ z)) / m``; ``Br1`` defines ``R``, ``Br2`` ``M``."""

    def __init__(self, *, num_mics, nonnegative=True, callbacks=None, r_diag=False):
        super().__init__(nonnegative=nonnegative, callbacks=callbacks)
        if num_mics <= int(r_diag):
            raise ValueError(f'num_mics must exceed {int(r_diag)} for r_diag={r_diag}, got {num_mics}.')
        self.num_mics, self.r_diag = num_mics, r_diag
        self.num_measurements = num_mics * (num_mics - int(r_diag))
        self.Br1_re = nn.Parameter(torch.eye(num_mics))
        self.Br1_im = nn.Parameter(torch.zeros(num_mics, num_mics))
        self.Br2_re = nn.Parameter(torch.eye(num_mics))
        self.Br2_im = nn.Parameter(torch.zeros(num_mics, num_mics))

    def _prepare_update(self, y, x0, operator):
        """Check CMF coordinates, detach the operator and bind the two-weight direction.

        For a microphone subset, align ``y`` with ``operator.rows``.
        """
        validate_cmf_operator(operator, self.num_mics, self.r_diag)
        y, x0, operator = self._prepare_operator_inputs(y, x0, operator)
        recurrent_operator = make_propagation_weight_operator(operator, self.Br1_re, self.Br1_im)
        weight_operator = make_propagation_weight_operator(operator, self.Br2_re, self.Br2_im)
        measurement_term = weight_operator.adjoint(y)
        m = float(y.shape[1])

        def gradient(x):
            feedback = recurrent_operator.adjoint(recurrent_operator.forward(x))
            return (measurement_term - feedback) / m

        def residual_gradient(x):
            # The residual is only needed as a controller feature: one extra forward per call.
            return y - operator.forward(x), gradient(x)

        callback_metadata = {'A': operator, 'W': weight_operator, 'W_recurrent': recurrent_operator, 'y': y}
        return y, x0, gradient, residual_gradient, callback_metadata


class AdaLFISTAR(LearnedParams, _ReducedTwoWeights):
    """Ada-LFISTA with two independent complex auxiliary propagation matrices.

    ``Br1`` defines the recurrence ``R``, and ``Br2`` the measurement branch
    ``M``. Both start at identity and stay factored. The update direction is
    ``(M.T @ y - R.T @ (R @ z)) / m``. With a removed diagonal, both weights act
    on zero-diagonal CSMs and keep only the off-diagonal output. All arguments
    are keyword-only.

    Parameters
    ----------
    num_iterations : int
        Number of unfolded iterations (layers).
    gamma_init : float
        Initial applied step size of every iteration.
    theta_init : float
        Initial applied soft-threshold of every iteration.
    num_mics : int
        Number of microphones of the full array.
    nonnegative : bool, optional
        If True (default), constrain the solution to ``x >= 0``.
    callbacks : list of IterationCallback, optional
        Solver callbacks; end callbacks get the operators ``A``, ``W = M``, and ``W_recurrent = R``.
    r_diag : bool, optional
        Whether the CSM diagonal is omitted (default False); must match the operator.
    """


class NA_AdaLFISTAR(NeuralAugmentation, _ReducedTwoWeights):
    """NA-Ada-LFISTA-R: trainable ``Br1_*`` and ``Br2_*``; an LSTM predicts the step parameters.

    Same update direction as :class:`AdaLFISTAR`, but the step size and threshold
    are predicted per sample and iteration from the residual and direction norms,
    see :class:`~ucmf.optim.neural_augmentation.NeuralAugmentation`. Any ``t >= 0``
    is allowed. Call ``init_normalizers`` once before solving. All arguments are
    keyword-only.

    Parameters
    ----------
    gamma_init : float
        Initial applied step size of the controller readout.
    theta_init : float
        Initial applied soft-threshold of the controller readout.
    lstm_dim : int, optional
        Hidden size of the LSTM controller (default 128).
    num_mics : int
        Number of microphones of the full array.
    nonnegative : bool, optional
        If True (default), constrain the solution to ``x >= 0``.
    callbacks : list of IterationCallback, optional
        Solver callbacks; end callbacks get the operators ``A``, ``W = M``, and ``W_recurrent = R``.
    r_diag : bool, optional
        Whether the CSM diagonal is omitted (default False); must match the operator.
    """
