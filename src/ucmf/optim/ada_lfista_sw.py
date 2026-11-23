"""Single-weight adaptive LFISTA (Ada-LFISTA-SW).

``AdaLFISTASW`` augments FISTA with three sets of trainable parameters:

    - ``B``: a single ``m x m`` auxiliary weight matrix. The effective weight
      ``W = B @ A`` is factored by default. With ``B = I`` the update reduces to FISTA.
    - ``gamma``: per-iteration step sizes actually applied at each iteration.
    - ``theta``: per-iteration soft-thresholds actually applied at each iteration.

``gamma`` and ``theta`` are the effective values; they are kept positive via
softplus and stored unconstrained as ``_gamma`` / ``_theta``. The iteration is

    x_next = shrink(z + gamma_k * W.T @ (y - A z) / m, theta_k)

with Nesterov momentum. Neither ``lam`` nor ``gamma`` are passed to
``forward``; the model learns ``gamma`` and ``theta`` directly.
"""

from .ista import FISTA
from .utils import softplus_inverse, validate_unfolded_iterations
from .weights import make_weight_operator

import torch
import torch.nn.functional as F
from torch import nn


class AdaLFISTASW(FISTA):
    """Single-weight Ada-LFISTA with trainable ``B``, ``gamma`` and ``theta``.

    Args:
        n_iterations (int): number of unfolded iterations/layers.
        n_measurements (int): measurement dimension ``m`` spanned by ``B``.
        gamma_init (float): initial applied step size for every iteration.
        theta_init (float): initial applied soft-threshold for every iteration.
        positive (bool): enforce non-negativity via positive shrinkage.
        callbacks (list | None): optional solver callbacks.
        B (Tensor | None): optional ``(m, m)`` init for the auxiliary weight matrix;
            defaults to the identity, reproducing classical FISTA.
        weight_mode (str): ``'factored'`` applies ``A.T @ (B.T @ r)`` without
            forming ``W``. ``'precomputed'`` forms ``W = B @ A`` once per solve
            and requires a dense ``MatrixMult`` or ``CMFDictionary`` operator.
    """

    def __init__(
        self,
        n_iterations,
        n_measurements,
        gamma_init,
        theta_init,
        positive=True,
        callbacks=None,
        B=None,
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

        B_tensor = torch.eye(self.n_measurements) if B is None else B.detach().clone()
        expected_shape = (self.n_measurements, self.n_measurements)
        if B_tensor.ndim != 2 or tuple(B_tensor.shape) != expected_shape:
            raise ValueError(f'B must have shape {expected_shape}, got {tuple(B_tensor.shape)}.')
        self.B = nn.Parameter(B_tensor)

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
        """Run Ada-LFISTA-SW with a fixed measurement operator and learned weights.

        Inputs and model parameters must already share ``y``'s dtype and device.
        For a subset, ``y`` must follow ``operator.rows`` in the full measurement
        coordinates of ``B``. Row indices must be in range.
        """
        validate_unfolded_iterations(t, self.n_iterations)
        y, x0, operator, _ = self._prepare_operator_inputs(y, x0, operator)
        weight_operator = make_weight_operator(operator, self.B, self.weight_mode)
        # gamma = softplus(_gamma), theta = softplus(_theta): applied once here,
        # then unbound into per-layer scalars so the loop only does tuple indexing
        # (no per-layer tensor __getitem__ dispatch). unbind keeps the autograd graph.
        gammas, thetas = self.gamma.unbind(), self.theta.unbind()
        return self._iterate_accelerated(
            y, x0, operator.shape[1], t,
            step_params_fn=lambda k: (gammas[k], thetas[k]),
            gradient=self._make_gradient(y, operator, weight_operator),
            callback_metadata={'A': operator, 'W': weight_operator, 'y': y},
        )
