"""Two-matrix adaptive LFISTA (Ada-LFISTA, Aberdam et al., Algorithm 1).

``AdaLFISTA`` learns two ``m x m`` auxiliary weight matrices, numbered as the
paper's ``W1`` / ``W2`` in Algorithm 1, plus per-iteration ``gamma`` / ``theta``:

    R = B1 @ A    recurrence factor (W1)
    M = B2 @ A    measurement branch (W2)
    x_next = shrink(z + gamma_k * (M.T @ y - R.T @ (R @ z)) / m, theta_k)

with Nesterov momentum. This is not ``M.T @ (y - R @ z)``. With
``B1 = B2 = I`` the update reduces to FISTA; the source-space Gram ``R.T @ R``
is never formed. ``gamma`` / ``theta`` are the applied softplus values of the
unconstrained ``_gamma`` / ``_theta``; ``lam`` and ``gamma`` are no inputs.
"""

from .ista import FISTA
from .utils import softplus_inverse, validate_unfolded_iterations
from .weights import make_weight_operator

import torch
import torch.nn.functional as F
from torch import nn


class AdaLFISTA(FISTA):
    """Two-matrix Ada-LFISTA with trainable ``B1``, ``B2``, ``gamma`` and ``theta``.

    Args:
        n_iterations (int): number of unfolded iterations/layers.
        n_measurements (int): full measurement dimension ``m`` spanned by ``B1``/``B2``.
        gamma_init (float): initial applied step size for every iteration.
        theta_init (float): initial applied soft-threshold for every iteration.
        positive (bool): enforce non-negativity via positive shrinkage.
        callbacks (list | None): optional solver callbacks. End callbacks get the
            operators ``A``, ``W = M`` and ``W_recurrent = R``.
        B1, B2 (Tensor | None): optional ``(m, m)`` inits; default identity.
        weight_mode (str): ``'factored'`` leaves ``R`` as a composition.
            ``'precomputed'`` forms ``R`` once per solve and requires a dense
            measurement operator. ``M`` stays factored in either mode.
    """

    def __init__(
        self,
        n_iterations,
        n_measurements,
        gamma_init,
        theta_init,
        positive=True,
        callbacks=None,
        B1=None,
        B2=None,
        *,
        weight_mode='factored',
    ):
        super().__init__(positive=positive, callbacks=callbacks)
        if weight_mode not in ('factored', 'precomputed'):
            raise ValueError("weight_mode must be 'factored' or 'precomputed'.")
        self.weight_mode = weight_mode
        if n_iterations <= 0:
            raise ValueError(f'n_iterations must be positive, got {n_iterations}')
        self.n_iterations = n_iterations
        self.n_measurements = int(n_measurements)
        if self.n_measurements <= 0:
            raise ValueError(f'n_measurements must be positive, got {n_measurements}')
        self.B1 = nn.Parameter(self._initial_weight(B1, 'B1'))
        self.B2 = nn.Parameter(self._initial_weight(B2, 'B2'))
        self._gamma = nn.Parameter(torch.full((n_iterations,), float(softplus_inverse(gamma_init))))
        self._theta = nn.Parameter(torch.full((n_iterations,), float(softplus_inverse(theta_init))))

    def _initial_weight(self, B, name):
        B_tensor = torch.eye(self.n_measurements) if B is None else B.detach().clone()
        expected_shape = (self.n_measurements, self.n_measurements)
        if B_tensor.ndim != 2 or tuple(B_tensor.shape) != expected_shape:
            raise ValueError(f'{name} must have shape {expected_shape}, got {tuple(B_tensor.shape)}.')
        return B_tensor

    @property
    def gamma(self):
        """Per-iteration step sizes actually applied (``softplus(_gamma)``)."""
        return F.softplus(self._gamma)

    @property
    def theta(self):
        """Per-iteration soft-thresholds actually applied (``softplus(_theta)``)."""
        return F.softplus(self._theta)

    def forward(self, y, x0=None, t=1, operator=None):
        """Run two-matrix Ada-LFISTA with a fixed operator and learned weights.

        Inputs and model parameters must already share ``y``'s dtype and device.
        For a subset, ``y`` must follow ``operator.rows`` in the full measurement
        coordinates of both weights. Row indices must be in range.
        """
        validate_unfolded_iterations(t, self.n_iterations)
        y, x0, operator, _ = self._prepare_operator_inputs(y, x0, operator)
        recurrent_operator = make_weight_operator(operator, self.B1, self.weight_mode)
        weight_operator = make_weight_operator(operator, self.B2)
        # M.T @ y does not depend on the iteration; no need to form M.
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
