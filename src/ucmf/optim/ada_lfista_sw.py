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
from .learned_params import LearnedParams
from .weights import make_weight_operator

import torch
from torch import nn


class _SingleWeight(FISTA):
    """Update direction ``W.T @ (y - A z) / m`` with ``W = B_active @ A``."""

    def __init__(self, *, num_measurements, positive=True, callbacks=None, B=None, weight_mode='factored'):
        super().__init__(positive=positive, callbacks=callbacks)
        if weight_mode not in ('factored', 'precomputed'):
            raise ValueError("weight_mode must be 'factored' or 'precomputed'.")
        self.weight_mode = weight_mode
        self.num_measurements = int(num_measurements)
        if self.num_measurements <= 0:
            raise ValueError(f'num_measurements must be positive, got {num_measurements}')

        B_tensor = torch.eye(self.num_measurements) if B is None else B.detach().clone()
        expected_shape = (self.num_measurements, self.num_measurements)
        if B_tensor.ndim != 2 or tuple(B_tensor.shape) != expected_shape:
            raise ValueError(f'B must have shape {expected_shape}, got {tuple(B_tensor.shape)}.')
        self.B = nn.Parameter(B_tensor)

    def _prepare_update(self, y, x0, operator):
        """Detach the operator and bind the weighted update direction for one solve."""
        y, x0, operator, _ = self._prepare_operator_inputs(y, x0, operator)
        weight_operator = make_weight_operator(operator, self.B, self.weight_mode)
        gradient = self._make_gradient(y, operator, weight_operator)
        return y, x0, operator.shape[1], gradient, {'A': operator, 'W': weight_operator, 'y': y}


class AdaLFISTASW(LearnedParams, _SingleWeight):
    """Single-weight Ada-LFISTA with trainable ``B``, ``gamma`` and ``theta``.

    All arguments are keyword-only.

    Args:
        num_iterations (int): number of unfolded iterations/layers.
        num_measurements (int): measurement dimension ``m`` spanned by ``B``.
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
