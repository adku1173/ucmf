"""Per-iteration learned step sizes and thresholds for unfolded models."""

from .utils import softplus_inverse, validate_unfolded_iterations

import torch
import torch.nn.functional as F
from torch import nn


class LearnedParams:
    """Mixin learning one step size and one threshold per unfolded iteration.

    Combine with a weight base that implements ``_prepare_update``; the mixin
    owns the step parameters and ``forward``. All arguments are keyword-only.
    The operator must have unit column norms (unchecked precondition), see
    :func:`ucmf.scaling.normalize_operator`.

    Parameters
    ----------
    num_iterations : int
        Number of unfolded iterations (layers).
    gamma_init : float
        Initial applied step size of every iteration.
    theta_init : float
        Initial applied soft-threshold of every iteration.
    **kwargs
        Arguments of the weight base.

    Attributes
    ----------
    gamma, theta : Tensor
        Applied per-iteration values of shape ``(num_iterations,)``, the
        softplus of the unconstrained parameters ``_gamma`` and ``_theta``.
    """

    def __init__(self, *, num_iterations, gamma_init, theta_init, **kwargs):
        super().__init__(**kwargs)
        if num_iterations <= 0:
            raise ValueError(f'num_iterations must be positive, got {num_iterations}')
        self.num_iterations = num_iterations
        self._gamma = nn.Parameter(torch.full((num_iterations,), float(softplus_inverse(gamma_init))))
        self._theta = nn.Parameter(torch.full((num_iterations,), float(softplus_inverse(theta_init))))

    @property
    def gamma(self):
        """Per-iteration step sizes actually applied (``softplus(_gamma)``)."""
        return F.softplus(self._gamma)

    @property
    def theta(self):
        """Per-iteration soft-thresholds actually applied (``softplus(_theta)``)."""
        return F.softplus(self._theta)

    def forward(self, y, x0=None, t=1, operator=None, active_iterations=None):
        """Run ``t`` unfolded iterations with a fixed operator and the learned parameters.

        Inputs and model parameters must already share ``y``'s dtype and device.
        For a subset, ``y`` must follow ``operator.rows`` in the full measurement
        coordinates of the auxiliary weights. Row indices must be in range.

        Parameters
        ----------
        y : Tensor
            Measurements of shape ``(B, m)`` or ``(m,)``.
        x0 : Tensor, optional
            Warm start of shape ``(B, n)`` or ``(n,)``; zeros if omitted.
        t : int, optional
            Number of iterations, ``0 <= t <= num_iterations`` (default 1).
        operator : LinearOperator
            Measurement operator; treated as fixed.
        active_iterations : Tensor, optional
            0-d integer tensor ``s <= t``: iterations ``s + 1, ..., t`` keep the iterate, so the
            result is ``x_s``. One compiled graph of depth ``t`` thus serves every depth ``s``,
            see :func:`~ucmf.optim.compile.compile_unrolled`. Default: all ``t`` iterations.

        Returns
        -------
        Tensor
            Solution of shape ``(B, n)``.
        """
        validate_unfolded_iterations(t, self.num_iterations)
        y, x0, gradient, _, callback_metadata = self._prepare_update(y, x0, operator)
        # gamma = softplus(_gamma), theta = softplus(_theta): applied once here,
        # then unbound into per-layer scalars so the loop only does tuple indexing
        # (no per-layer tensor __getitem__ dispatch). unbind keeps the autograd graph.
        gammas, thetas = self.gamma.unbind(), self.theta.unbind()
        return self._iterate_accelerated(
            y, x0, t,
            step_params_fn=lambda k: (gammas[k], thetas[k]),
            gradient=gradient,
            callback_metadata=callback_metadata,
            active_iterations=active_iterations,
        )
