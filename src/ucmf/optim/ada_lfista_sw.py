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
from .neural_augmentation import NeuralAugmentation
from .weights import make_weight_operator

import torch
from torch import nn


class _SingleWeight(FISTA):
    """Update direction ``W.T @ (y - A z) / m`` with ``W = B_active @ A``."""

    def __init__(self, *, num_measurements, nonnegative=True, callbacks=None, B=None, weight_mode='factored'):
        super().__init__(nonnegative=nonnegative, callbacks=callbacks)
        if weight_mode not in ('factored', 'precomputed'):
            raise ValueError("weight_mode must be 'factored' or 'precomputed'.")
        self.weight_mode = weight_mode
        self.num_measurements = num_measurements
        if self.num_measurements <= 0:
            raise ValueError(f'num_measurements must be positive, got {num_measurements}')

        B_tensor = torch.eye(self.num_measurements) if B is None else B.detach().clone()
        expected_shape = (self.num_measurements, self.num_measurements)
        if B_tensor.ndim != 2 or tuple(B_tensor.shape) != expected_shape:
            raise ValueError(f'B must have shape {expected_shape}, got {tuple(B_tensor.shape)}.')
        self.B = nn.Parameter(B_tensor)

    def _prepare_update(self, y, x0, operator):
        """Detach the operator and bind the weighted update direction for one solve."""
        y, x0, operator = self._prepare_operator_inputs(y, x0, operator)
        weight_operator = make_weight_operator(operator, self.B, self.weight_mode)
        gradient = self._make_gradient(y, operator, weight_operator)
        residual_gradient = self._make_residual_gradient(y, operator, weight_operator)
        return y, x0, gradient, residual_gradient, {'A': operator, 'W': weight_operator, 'y': y}


class AdaLFISTASW(LearnedParams, _SingleWeight):
    """Single-weight Ada-LFISTA with trainable ``B``, ``gamma`` and ``theta``.

    All arguments are keyword-only.

    Parameters
    ----------
    num_iterations : int
        Number of unfolded iterations (layers).
    gamma_init : float
        Initial applied step size of every iteration.
    theta_init : float
        Initial applied soft-threshold of every iteration.
    num_measurements : int
        Full measurement dimension ``m`` spanned by ``B``.
    nonnegative : bool, optional
        If True (default), constrain the solution to ``x >= 0``.
    callbacks : list of IterationCallback, optional
        Solver callbacks; end callbacks get the operators ``A`` and ``W``.
    B : Tensor, optional
        Initial ``(m, m)`` auxiliary weight; defaults to the identity, which reproduces FISTA.
    weight_mode : {'factored', 'precomputed'}, optional
        ``'factored'`` (default) applies ``A.T @ (B.T @ r)`` without forming ``W``.
        ``'precomputed'`` forms ``W = B @ A`` once per solve and requires a dense
        ``MatrixMult`` or ``CMFDictionary`` operator.
    """


class NA_AdaLFISTASW(NeuralAugmentation, _SingleWeight):
    """NA-Ada-LFISTA-SW: trainable ``B``; an LSTM predicts the step size and threshold.

    Same update direction as :class:`AdaLFISTASW`, but the step size and threshold
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
    num_measurements : int
        Full measurement dimension ``m`` spanned by ``B``.
    nonnegative : bool, optional
        If True (default), constrain the solution to ``x >= 0``.
    callbacks : list of IterationCallback, optional
        Solver callbacks; end callbacks get the operators ``A`` and ``W``.
    B : Tensor, optional
        Initial ``(m, m)`` auxiliary weight; defaults to the identity, which reproduces FISTA.
    weight_mode : {'factored', 'precomputed'}, optional
        ``'factored'`` (default) applies ``A.T @ (B.T @ r)`` without forming ``W``.
        ``'precomputed'`` forms ``W = B @ A`` once per solve and requires a dense
        ``MatrixMult`` or ``CMFDictionary`` operator.
    """
