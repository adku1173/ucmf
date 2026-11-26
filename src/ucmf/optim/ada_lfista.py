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
from .learned_params import LearnedParams
from .neural_augmentation import NeuralAugmentation
from .weights import make_weight_operator

import torch
from torch import nn


class _TwoWeights(FISTA):
    """Update direction ``(M.T @ y - R.T @ (R @ z)) / m`` with ``R = B1 @ A``, ``M = B2 @ A``."""

    def __init__(
        self, *, num_measurements, nonnegative=True, callbacks=None, B1=None, B2=None, weight_mode='factored',
    ):
        super().__init__(nonnegative=nonnegative, callbacks=callbacks)
        if weight_mode not in ('factored', 'precomputed'):
            raise ValueError("weight_mode must be 'factored' or 'precomputed'.")
        self.weight_mode = weight_mode
        self.num_measurements = num_measurements
        if self.num_measurements <= 0:
            raise ValueError(f'num_measurements must be positive, got {num_measurements}')
        self.B1 = nn.Parameter(self._initial_weight(B1, 'B1'))
        self.B2 = nn.Parameter(self._initial_weight(B2, 'B2'))

    def _initial_weight(self, B, name):
        B_tensor = torch.eye(self.num_measurements) if B is None else B.detach().clone()
        expected_shape = (self.num_measurements, self.num_measurements)
        if B_tensor.ndim != 2 or tuple(B_tensor.shape) != expected_shape:
            raise ValueError(f'{name} must have shape {expected_shape}, got {tuple(B_tensor.shape)}.')
        return B_tensor

    def _prepare_update(self, y, x0, operator):
        """Detach the operator and bind the two-matrix update direction for one solve."""
        y, x0, operator = self._prepare_operator_inputs(y, x0, operator)
        recurrent_operator = make_weight_operator(operator, self.B1, self.weight_mode)
        weight_operator = make_weight_operator(operator, self.B2)
        # M.T @ y does not depend on the iteration; no need to form M.
        measurement_term = weight_operator.adjoint(y)
        m = float(y.shape[1])

        def gradient(x):
            feedback = recurrent_operator.adjoint(recurrent_operator.forward(x))
            return (measurement_term - feedback) / m

        def residual_gradient(x):
            # The residual is only needed as a controller feature: one extra forward per call.
            return y - operator.forward(x), gradient(x)

        callback_metadata = {'A': operator, 'W': weight_operator, 'W_recurrent': recurrent_operator, 'y': y}
        return y, x0, gradient, residual_gradient, callback_metadata


class AdaLFISTA(LearnedParams, _TwoWeights):
    """Two-matrix Ada-LFISTA with trainable ``B1``, ``B2``, ``gamma`` and ``theta``.

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
        Full measurement dimension ``m`` spanned by ``B1`` and ``B2``.
    nonnegative : bool, optional
        If True (default), constrain the solution to ``x >= 0``.
    callbacks : list of IterationCallback, optional
        Solver callbacks; end callbacks get the operators ``A``, ``W = M``, and ``W_recurrent = R``.
    B1, B2 : Tensor, optional
        Initial ``(m, m)`` weights of the recurrence and the measurement branch;
        default identity.
    weight_mode : {'factored', 'precomputed'}, optional
        ``'factored'`` (default) leaves ``R`` as a composition. ``'precomputed'``
        forms ``R`` once per solve and requires a dense measurement operator.
        ``M`` stays factored in either mode.
        Prefer ``'precomputed'`` with one dense operator per batch and many measurement rows
        (about 25 % faster training at 64 microphones); keep ``'factored'`` for per-problem
        operators (one ``W`` per problem costs time and memory) and for ``CMFPropagation``.
    """


class NA_AdaLFISTA(NeuralAugmentation, _TwoWeights):
    """NA-Ada-LFISTA: trainable ``B1`` and ``B2``; an LSTM predicts the step size and threshold.

    Same update direction as :class:`AdaLFISTA`, but the step size and threshold
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
        Full measurement dimension ``m`` spanned by ``B1`` and ``B2``.
    nonnegative : bool, optional
        If True (default), constrain the solution to ``x >= 0``.
    callbacks : list of IterationCallback, optional
        Solver callbacks; end callbacks get the operators ``A``, ``W = M``, and ``W_recurrent = R``.
    B1, B2 : Tensor, optional
        Initial ``(m, m)`` weights of the recurrence and the measurement branch;
        default identity.
    weight_mode : {'factored', 'precomputed'}, optional
        ``'factored'`` (default) leaves ``R`` as a composition. ``'precomputed'``
        forms ``R`` once per solve and requires a dense measurement operator.
        ``M`` stays factored in either mode.
        Prefer ``'precomputed'`` with one dense operator per batch and many measurement rows
        (about 25 % faster training at 64 microphones); keep ``'factored'`` for per-problem
        operators (one ``W`` per problem costs time and memory) and for ``CMFPropagation``.
    """
