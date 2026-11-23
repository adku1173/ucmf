"""Two-weight reduced Ada-LFISTA (Ada-LFISTA-R) for CMF measurements."""

from .ista import FISTA
from .learned_params import LearnedParams
from .weights import make_propagation_weight_operator, validate_cmf_operator

import torch
from torch import nn


class _ReducedTwoWeights(FISTA):
    """Update direction ``(M.T @ y - R.T @ (R @ z)) / m`` with congruences ``Br1`` (R) and ``Br2`` (M)."""

    def __init__(self, *, num_mics, positive=True, callbacks=None, r_diag=False):
        super().__init__(positive=positive, callbacks=callbacks)
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
        y, x0, operator, _ = self._prepare_operator_inputs(y, x0, operator)
        recurrent_operator = make_propagation_weight_operator(operator, self.Br1_re, self.Br1_im)
        weight_operator = make_propagation_weight_operator(operator, self.Br2_re, self.Br2_im)
        measurement_term = weight_operator.adjoint(y)
        m = float(y.shape[1])

        def gradient(x):
            feedback = recurrent_operator.adjoint(recurrent_operator.forward(x))
            return (measurement_term - feedback) / m

        callback_metadata = {'A': operator, 'W': weight_operator, 'W_recurrent': recurrent_operator, 'y': y}
        return y, x0, operator.shape[1], gradient, callback_metadata


class AdaLFISTAR(LearnedParams, _ReducedTwoWeights):
    """Ada-LFISTA with two independent complex auxiliary propagation matrices.

    ``Br1`` defines the recurrence ``R``, and ``Br2`` the measurement branch
    ``M``. Both start at identity and stay factored. The update direction is
    ``(M.T @ y - R.T @ (R @ z)) / m``. All arguments are keyword-only.

    ``num_mics`` is the full array size, and ``r_diag`` must match the CMF
    operator. With a removed diagonal, both weights act on zero-diagonal CSMs
    and keep only the off-diagonal output. End callbacks get operators ``A``,
    ``W = M`` and ``W_recurrent = R``.
    """
