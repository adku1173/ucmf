"""Iterative shrinkage-thresholding algorithms (ISTA and FISTA).

Both solve the normalized LASSO problem ``min_x ||y - A x||_2^2 / (2 m) + lam ||x||_1``
with ``m`` measurements, optionally constrained to ``x >= 0``. Callbacks can
monitor the iterations and stop them early.
"""

import math

from .operator import LinearOperator
from .utils import (
    default_gamma_from_operator,
    resolve_batch_param,
    validate_tensor,
)

import torch
import torch.nn.functional as F
from torch import nn


class ISTA(nn.Module):
    """Iterative shrinkage-thresholding algorithm (ISTA) for the normalized LASSO.

    Minimizes ``||y - A x||_2^2 / (2 m) + lam ||x||_1``. Each step moves along the
    negative gradient of the data term, ``A.T @ (y - A x) / m``, and applies the
    proximal threshold ``theta = gamma * lam``.

    Iterates are local variables in an explicit loop. Subclasses can reuse input
    preparation, gradient/proximal steps, and callback helpers without a state
    container or iteration framework; see ``docs/architecture.md``.

    Parameters
    ----------
    nonnegative : bool, optional
        If True (default), constrain the solution to ``x >= 0``.
    callbacks : list of IterationCallback, optional
        Callbacks run at the beginning and the end of every iteration.

    Attributes
    ----------
    iter : int
        Zero-based index of the current iteration, set during a solve.
    stop : bool
        Set to True by a callback to stop the current solve.
    """

    def __init__(
        self,
        nonnegative=True,
        callbacks=None,
    ):
        super().__init__()

        self.nonnegative = nonnegative
        self.callbacks = callbacks if callbacks is not None else []

    @staticmethod
    def shrink(x, thresh):
        """Soft-threshold ``x`` element-wise.

        Parameters
        ----------
        x : Tensor
            Values to shrink.
        thresh : float or Tensor
            Non-negative threshold, broadcastable to ``x``.

        Returns
        -------
        Tensor
            ``sign(x) * max(|x| - thresh, 0)``.
        """
        return x - x.clamp(min=-thresh, max=thresh)

    @staticmethod
    def shrink_nonnegative(x, thresh):
        """Soft-threshold ``x`` element-wise and keep only non-negative values.

        Parameters
        ----------
        x : Tensor
            Values to shrink.
        thresh : float or Tensor
            Non-negative threshold, broadcastable to ``x``.

        Returns
        -------
        Tensor
            ``max(x - thresh, 0)``, not ``max(|x| - thresh, 0)``.
        """
        return F.relu(x - thresh)

    def _make_gradient(self, y, operator, weight_operator):
        """Bind the data and operators for the negative LS direction at ``x``."""
        m = float(y.shape[1])

        def gradient(x):
            return weight_operator.adjoint(y - operator.forward(x)) / m

        return gradient

    def _make_residual_gradient(self, y, operator, weight_operator):
        """Like ``_make_gradient``, but also return the residual ``y - A x`` the direction uses."""
        m = float(y.shape[1])

        def residual_gradient(x):
            residual = y - operator.forward(x)
            return residual, weight_operator.adjoint(residual) / m

        return residual_gradient

    def get_shrink_func(self):
        """Return the shrinkage function for the configured constraint.

        Returns
        -------
        callable
            :meth:`shrink_nonnegative` if :attr:`nonnegative`, otherwise :meth:`shrink`.
        """
        if self.nonnegative:
            return self.shrink_nonnegative
        return self.shrink

    def step(self, x, gradient, shrink_func, gamma, theta):
        """Perform one proximal gradient step.

        Parameters
        ----------
        x : Tensor
            Current point of shape ``(B, n)``.
        gradient : callable
            Prepared negative-gradient direction ``gradient(x)`` of shape ``(B, n)``.
        shrink_func : callable
            Shrinkage function, see :meth:`get_shrink_func`.
        gamma, theta : float or Tensor
            Step size and threshold, scalar or per sample with shape ``(B, 1)``.

        Returns
        -------
        Tensor
            ``shrink_func(x + gamma * gradient(x), theta)``.
        """
        return shrink_func(x + gamma * gradient(x), theta)

    def _prepare_operator_inputs(self, y, x0, operator):
        """Validate ``y``, ``x0`` and the operator, add batch axes, and detach the operator.

        A missing warm start becomes zeros of shape ``(B, n)``, so the returned
        ``x0`` is always a tensor.
        """
        if not torch.is_tensor(y) or y.ndim not in (1, 2) or not torch.is_floating_point(y):
            raise ValueError('y must be a real floating-point tensor with shape (m,) or (B, m).')
        if y.ndim == 1:
            y = y.unsqueeze(0)
        if x0 is not None:
            validate_tensor(x0, name='x0', dtype=y.dtype, device=y.device)
            if x0.ndim == 1:
                x0 = x0.unsqueeze(0)
        if not isinstance(operator, LinearOperator):
            raise ValueError('operator must be a LinearOperator; wrap dense matrices in MatrixMult.')
        if operator.dtype != y.dtype:
            raise ValueError(f'operator must have dtype {y.dtype}, got {operator.dtype}.')
        if operator.device != y.device:
            raise ValueError(f'operator must be on device {y.device}, got {operator.device}.')
        m, n = operator.shape
        if m <= 0 or n <= 0:
            raise ValueError('operator must have non-empty measurement and source dimensions.')
        batch_size = y.shape[0]
        if y.shape[1] != m or operator.batch_size not in (None, batch_size):
            msg = f'y has shape {tuple(y.shape)} but operator has shape {(m, n)} and batch size {operator.batch_size}.'
            raise ValueError(msg)
        if x0 is None:
            x0 = y.new_zeros((batch_size, n))
        elif x0.shape != (batch_size, n):
            raise ValueError(f'x0 has shape {tuple(x0.shape)} but expected {(batch_size, n)}')
        return y, x0, operator.detached()

    def _resolve_lam_value(self, lam, batch_size, operator):
        """Resolve regularization strength for scalar or per-batch input."""
        if torch.is_tensor(lam):
            validate_tensor(lam, name='lam', dtype=operator.dtype, device=operator.device)
        lam_value = resolve_batch_param(lam, batch_size, name='lam')
        if lam_value is None:
            raise ValueError('lam must be provided.')
        return lam_value

    def _resolve_gamma_theta(self, lam_value, gamma, batch_size, operator):
        """Resolve step size and shrinkage threshold for fixed-step solvers."""
        if gamma is None:
            gamma = default_gamma_from_operator(operator)
        if torch.is_tensor(gamma):
            validate_tensor(gamma, name='gamma', dtype=operator.dtype, device=operator.device)
        gamma_value = resolve_batch_param(gamma, batch_size, name='gamma')

        # Normalized LASSO convention:
        #   1 / (2 m) * ||y - A x||_2^2 + lam * ||x||_1
        # The step uses the normalized negative gradient A.T @ (y - A x) / m,
        # so the proximal threshold is gamma * lam.
        theta_value = lam_value * gamma_value
        return gamma_value, theta_value

    def _start_iterations(self):
        """Clear the stop flag; callbacks get ``y`` through the end-callback metadata."""
        self.stop = False

    def _on_iteration_begin(self, k, x):
        """Run begin callbacks and report whether iteration should continue."""
        self.iter = k
        for cb in self.callbacks:
            cb.on_step_begin(self, x)
        return not self.stop

    def _on_iteration_end(self, x, gamma, theta, **callback_metadata):
        """Run end callbacks and report whether iteration should continue."""
        for cb in self.callbacks:
            cb.on_step_end(self, x, gamma=gamma, theta=theta, **callback_metadata)
        return not self.stop

    def _iterate(self, y, x0, gamma, theta, t, *, gradient, callback_metadata):
        """Run fixed-step ISTA with the solution local to this solve."""
        shrink_func = self.get_shrink_func()
        x = x0.clone()

        self._start_iterations()
        for k in range(t):
            if not self._on_iteration_begin(k, x):
                break
            x = self.step(x, gradient, shrink_func, gamma, theta)
            if not self._on_iteration_end(x, gamma, theta, **callback_metadata):
                break
        return x

    def forward(self, y, x0=None, t=1, operator=None, lam=None, gamma=None):
        """Solve with a fixed measurement operator.

        All tensor inputs must already share ``y``'s dtype and device. Convert
        arrays and move or cast tensors beforehand. Subclass-specific inputs
        belong in an overridden ``forward``.

        Parameters
        ----------
        y : Tensor
            Measurements of shape ``(B, m)`` or ``(m,)``.
        x0 : Tensor, optional
            Warm start of shape ``(B, n)`` or ``(n,)``; zeros if omitted.
        t : int, optional
            Number of iterations (default 1).
        operator : LinearOperator
            Shared or per-problem measurement operator; treated as fixed.
        lam : float or Tensor
            Regularization strength, scalar or per sample with shape ``(B,)`` or ``(B, 1)``.
        gamma : float or Tensor, optional
            Step size, scalar or per sample. Defaults to ``1 / L`` with
            ``L = 1.05 * ||A||_2**2 / m``, see :func:`.default_gamma_from_operator`.

        Returns
        -------
        Tensor
            Solution of shape ``(B, n)``.

        Raises
        ------
        ValueError
            If an input has an invalid type, shape, dtype, or device, or ``lam`` is missing.
        """
        if t < 0:
            msg = 't must be non-negative.'
            raise ValueError(msg)
        y, x0, operator = self._prepare_operator_inputs(y, x0, operator)
        batch_size = y.shape[0]
        lam_value = self._resolve_lam_value(lam, batch_size, operator)
        gamma_value, theta_value = self._resolve_gamma_theta(lam_value, gamma, batch_size, operator)
        return self._iterate(
            y, x0, gamma_value, theta_value, t,
            gradient=self._make_gradient(y, operator, operator),
            callback_metadata={'A': operator, 'W': operator, 'y': y},
        )


class FISTA(ISTA):
    """Fast iterative shrinkage-thresholding algorithm (FISTA) for the normalized LASSO.

    Same objective, parameters, and interface as :class:`ISTA`, with Nesterov
    momentum for faster convergence.
    """

    def _nesterov_momentum(self, tk, x, x_next):
        """Apply the standard Nesterov momentum recurrence; return ``(t_next, z_next)``."""
        t_next = 0.5 + math.sqrt(1.0 + 4.0 * tk**2) / 2.0
        z_next = x_next + (tk - 1.0) / t_next * (x_next - x)
        return t_next, z_next

    def _iterate_accelerated(self, y, x0, t, step_params_fn, *, gradient, callback_metadata):
        """Run FISTA with local momentum and caller-provided step parameters."""
        shrink_func = self.get_shrink_func()
        # Separate copies: begin callbacks may modify x in place, and the output must not alias x0.
        x = x0.clone()
        z = x0.clone()
        tk = 1.0

        self._start_iterations()
        for k in range(t):
            if not self._on_iteration_begin(k, x):
                break
            gamma_k, theta_k = step_params_fn(k)
            x_next = self.step(z, gradient, shrink_func, gamma_k, theta_k)
            tk, z = self._nesterov_momentum(tk, x, x_next)
            x = x_next
            if not self._on_iteration_end(x, gamma_k, theta_k, **callback_metadata):
                break
        return x

    def _iterate(self, y, x0, gamma, theta, t, *, gradient, callback_metadata):
        """Run FISTA iterations with Nesterov momentum and return the solution."""
        return self._iterate_accelerated(
            y, x0, t,
            step_params_fn=lambda _k: (gamma, theta),
            gradient=gradient,
            callback_metadata=callback_metadata,
        )
