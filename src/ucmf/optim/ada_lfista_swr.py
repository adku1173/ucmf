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
from .weights import make_propagation_weight_operator, validate_cmf_operator

import torch
from torch import nn


class _ReducedSingleWeight(FISTA):
    """Update direction ``A^T B^T (y - A z) / m`` with ``B`` the congruence with ``Br``."""

    def __init__(self, *, num_mics, positive=True, callbacks=None, r_diag=False):
        super().__init__(positive=positive, callbacks=callbacks)
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
        y, x0, operator, _ = self._prepare_operator_inputs(y, x0, operator)
        weight_operator = make_propagation_weight_operator(operator, self.Br_re, self.Br_im)
        gradient = self._make_gradient(y, operator, weight_operator)
        return y, x0, operator.shape[1], gradient, {'A': operator, 'W': weight_operator, 'y': y}


class AdaLFISTASWR(LearnedParams, _ReducedSingleWeight):
    """Ada-LFISTA-SWR with trainable ``Br_re``, ``Br_im``, ``gamma`` and ``theta``.

    All arguments are keyword-only.

    Args:
        num_iterations (int): number of unfolded iterations/layers.
        num_mics (int): number of microphones in the full array.
        gamma_init (float): initial applied step size for every iteration.
        theta_init (float): initial applied soft-threshold for every iteration.
        positive (bool): enforce non-negativity via positive shrinkage.
        callbacks (list | None): optional callbacks; ``A`` and ``W`` are operators.
        r_diag (bool): whether the CSM diagonal is omitted; must match the operator.
    """
