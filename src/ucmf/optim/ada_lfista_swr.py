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
from .utils import softplus_inverse, validate_unfolded_iterations
from .weights import make_propagation_weight_operator, validate_cmf_operator

import torch
import torch.nn.functional as F
from torch import nn


class AdaLFISTASWR(FISTA):
    """Ada-LFISTA-SWR with trainable ``Br_re``, ``Br_im``, ``gamma`` and ``theta``.

    Args:
        n_iterations (int): number of unfolded iterations/layers.
        n_mics (int): number of microphones in the full array.
        gamma_init (float): initial applied step size for every iteration.
        theta_init (float): initial applied soft-threshold for every iteration.
        positive (bool): enforce non-negativity via positive shrinkage.
        callbacks (list | None): optional callbacks; ``A`` and ``W`` are operators.
        r_diag (bool): whether the CSM diagonal is omitted; must match the operator.
    """

    def __init__(self, n_iterations, n_mics, gamma_init, theta_init, positive=True, callbacks=None, r_diag=False):
        super().__init__(positive=positive, callbacks=callbacks)
        if n_iterations <= 0:
            raise ValueError(f'n_iterations must be positive, got {n_iterations}')
        if n_mics <= int(r_diag):
            raise ValueError(f'n_mics must exceed {int(r_diag)} for r_diag={r_diag}, got {n_mics}.')
        self.n_iterations = n_iterations
        self.n_mics = n_mics
        self.r_diag = r_diag
        self.n_measurements = n_mics * (n_mics - int(r_diag))
        self.Br_re = nn.Parameter(torch.eye(n_mics))
        self.Br_im = nn.Parameter(torch.zeros(n_mics, n_mics))
        self._gamma = nn.Parameter(torch.full((n_iterations,), float(softplus_inverse(gamma_init))))
        self._theta = nn.Parameter(torch.full((n_iterations,), float(softplus_inverse(theta_init))))

    @property
    def gamma(self):
        """Per-iteration step sizes actually applied (``softplus(_gamma)``)."""
        return F.softplus(self._gamma)

    @property
    def theta(self):
        """Per-iteration soft-thresholds actually applied (``softplus(_theta)``)."""
        return F.softplus(self._theta)

    def forward(self, y, x0=None, t=1, operator=None):
        """Solve with a fixed CMF operator and a learned auxiliary propagation matrix.

        For a microphone subset, ``operator.mics`` must index the full array
        of ``Br`` and ``y`` must follow ``operator.rows`` in the full vectorization.
        """
        validate_unfolded_iterations(t, self.n_iterations)
        validate_cmf_operator(operator, self.n_mics, self.r_diag)
        y, x0, operator, _ = self._prepare_operator_inputs(y, x0, operator)
        weight_operator = make_propagation_weight_operator(operator, self.Br_re, self.Br_im)
        gammas, thetas = self.gamma.unbind(), self.theta.unbind()
        return self._iterate_accelerated(
            y, x0, operator.shape[1], t,
            step_params_fn=lambda k: (gammas[k], thetas[k]),
            gradient=self._make_gradient(y, operator, weight_operator),
            callback_metadata={'A': operator, 'W': weight_operator, 'y': y},
        )
