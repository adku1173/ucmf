"""Single-weight reduced Ada-LFISTA (Ada-LFISTA-SWR) for CMF measurements.

Like ``AdaLFISTASW``, but the ``m x m`` auxiliary weight ``B`` is not stored.
It is the congruence with a learned complex ``Br = Br_re + i Br_im`` (``c x c``,
``c`` microphones) in the Frobenius-isometric vectorization ``hvec``:

    B v = hvec(Br hvec*(v) Br^H),     B^T v = hvec(Br^H hvec*(v) Br).

With ``r_diag=True``, ``hvec*`` restores a zero diagonal before weighting and
``hvec`` drops it afterwards. Neither ``B`` nor ``W = B @ A`` is materialized.
``Br = I`` gives ``B = I``, recovering FISTA. The general update is:

    x_next = shrink(z + gamma_k * A^T B^T (y - A z) / m, theta_k)
"""

from .ista import FISTA
from .learned_params import LearnedParams
from .neural_augmentation import NeuralAugmentation
from .weights import make_propagation_weight_operator, validate_cmf_operator

import torch
from torch import nn


class _ReducedSingleWeight(FISTA):
    """Update direction ``A^T B^T (y - A z) / m`` with ``B`` the congruence with ``Br``."""

    def __init__(self, *, num_mics, nonnegative=True, callbacks=None, r_diag=False):
        super().__init__(nonnegative=nonnegative, callbacks=callbacks)
        if num_mics <= int(r_diag):
            raise ValueError(f'num_mics must exceed {int(r_diag)} for r_diag={r_diag}, got {num_mics}.')
        self.num_mics = num_mics
        self.r_diag = r_diag
        self.num_measurements = num_mics * (num_mics - int(r_diag))
        self.Br_re = nn.Parameter(torch.eye(num_mics))
        self.Br_im = nn.Parameter(torch.zeros(num_mics, num_mics))

    def _prepare_update(self, y, x0, operator):
        """Check CMF coordinates, detach the operator and bind the weighted direction.

        For a microphone subset, ``operator.mics`` must index the full array
        of ``Br`` and ``y`` must follow ``operator.rows`` in the full vectorization.
        """
        validate_cmf_operator(operator, self.num_mics, self.r_diag)
        y, x0, operator = self._prepare_operator_inputs(y, x0, operator)
        weight_operator = make_propagation_weight_operator(operator, self.Br_re, self.Br_im)
        gradient = self._make_gradient(y, operator, weight_operator)
        residual_gradient = self._make_residual_gradient(y, operator, weight_operator)
        return y, x0, gradient, residual_gradient, {'A': operator, 'W': weight_operator, 'y': y}


class AdaLFISTASWR(LearnedParams, _ReducedSingleWeight):
    """Ada-LFISTA-SWR with trainable ``Br_re``, ``Br_im``, ``gamma`` and ``theta``.

    All arguments are keyword-only.

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
        Solver callbacks; end callbacks get the operators ``A`` and ``W``.
    r_diag : bool, optional
        Whether the CSM diagonal is omitted (default False); must match the operator.
    """


class NA_AdaLFISTASWR(NeuralAugmentation, _ReducedSingleWeight):
    """NA-Ada-LFISTA-SWR: trainable ``Br_re`` and ``Br_im``; an LSTM predicts the step parameters.

    Same update direction as :class:`AdaLFISTASWR`, but the step size and threshold
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
        Solver callbacks; end callbacks get the operators ``A`` and ``W``.
    r_diag : bool, optional
        Whether the CSM diagonal is omitted (default False); must match the operator.
    """
