"""Two-weight reduced Ada-LFISTA (Ada-LFISTA-R) for CMF measurements."""

from .ista import FISTA
from .utils import softplus_inverse, validate_unfolded_iterations
from .weights import make_propagation_weight_operator, validate_cmf_operator

import torch
import torch.nn.functional as F
from torch import nn


class AdaLFISTAR(FISTA):
    """Ada-LFISTA with two independent complex auxiliary propagation matrices.

    ``Br1`` defines the recurrence ``R``, and ``Br2`` the measurement branch
    ``M``. Both start at identity and stay factored. The update direction is
    ``(M.T @ y - R.T @ (R @ z)) / m``.

    ``n_mics`` is the full array size, and ``r_diag`` must match the CMF
    operator. With a removed diagonal, both weights act on zero-diagonal CSMs
    and keep only the off-diagonal output. End callbacks get operators ``A``,
    ``W = M`` and ``W_recurrent = R``.
    """

    def __init__(self, n_iterations, n_mics, gamma_init, theta_init, positive=True, callbacks=None, r_diag=False):
        super().__init__(positive=positive, callbacks=callbacks)
        if n_iterations <= 0:
            raise ValueError(f'n_iterations must be positive, got {n_iterations}')
        if n_mics <= int(r_diag):
            raise ValueError(f'n_mics must exceed {int(r_diag)} for r_diag={r_diag}, got {n_mics}.')
        self.n_iterations, self.n_mics, self.r_diag = n_iterations, n_mics, r_diag
        self.n_measurements = n_mics * (n_mics - int(r_diag))
        self.Br1_re = nn.Parameter(torch.eye(n_mics))
        self.Br1_im = nn.Parameter(torch.zeros(n_mics, n_mics))
        self.Br2_re = nn.Parameter(torch.eye(n_mics))
        self.Br2_im = nn.Parameter(torch.zeros(n_mics, n_mics))
        self._gamma = nn.Parameter(torch.full((n_iterations,), float(softplus_inverse(gamma_init))))
        self._theta = nn.Parameter(torch.full((n_iterations,), float(softplus_inverse(theta_init))))

    @property
    def gamma(self):
        """Per-iteration step sizes (``softplus(_gamma)``)."""
        return F.softplus(self._gamma)

    @property
    def theta(self):
        """Per-iteration soft-thresholds (``softplus(_theta)``)."""
        return F.softplus(self._theta)

    def forward(self, y, x0=None, t=1, operator=None):
        """Solve with a fixed CMF operator; align subset ``y`` with ``operator.rows``."""
        validate_unfolded_iterations(t, self.n_iterations)
        validate_cmf_operator(operator, self.n_mics, self.r_diag)
        y, x0, operator, _ = self._prepare_operator_inputs(y, x0, operator)
        recurrent_operator = make_propagation_weight_operator(operator, self.Br1_re, self.Br1_im)
        weight_operator = make_propagation_weight_operator(operator, self.Br2_re, self.Br2_im)
        measurement_term = weight_operator.adjoint(y)
        m = float(y.shape[1])

        def gradient(x):
            feedback = recurrent_operator.adjoint(recurrent_operator.forward(x))
            return (measurement_term - feedback) / m

        gammas, thetas = self.gamma.unbind(), self.theta.unbind()
        return self._iterate_accelerated(
            y, x0, operator.shape[1], t,
            step_params_fn=lambda k: (gammas[k], thetas[k]),
            gradient=gradient,
            callback_metadata={'A': operator, 'W': weight_operator, 'W_recurrent': recurrent_operator, 'y': y},
        )
