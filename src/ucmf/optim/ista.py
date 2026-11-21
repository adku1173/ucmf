"""Iterative Shrinkage-Thresholding Algorithms (ISTA and FISTA).

This module provides PyTorch implementations of:
    - ISTA: Iterative Shrinkage-Thresholding Algorithm
    - FISTA: Fast Iterative Shrinkage-Thresholding Algorithm

Both solve the LASSO problem:
    min_x 1/(2 m) ||y - A x||_2^2 + λ ||x||_1,
    where m is the length of y / number of measurements.

With support for:
    - Non-negativity constraints (positive shrinkage)
    - Normalized least-squares data fitting
    - Callback-based early stopping and monitoring
"""

import math

from .utils import (
    apply_adjoint_operator,
    apply_forward_operator,
    default_gamma_from_A,
    prepare_sensing_matrix,
    resolve_batch_param,
    validate_shapes,
    validate_tensor,
)
from .weights import prepare_gradient_weight

import torch
import torch.nn.functional as F
from torch import nn


class ISTA(nn.Module):
    """
    Iterative Shrinkage-Thresholding Algorithm (ISTA) for LASSO.

    Solves the objective:
        min_x 1/(2 m) ||y - A x||_2^2 + λ ||x||_1

    The ISTA/FISTA step uses the negative gradient of the normalized data-fit
    term, ``A.T @ (y - A x) / m``. The matching proximal threshold is therefore
    ``theta = gamma * λ``.

    Args:
        positive (bool): if True, enforces non-negativity on the solution
        callbacks (list | None): optional solver callbacks

    Iterates are local variables in an explicit loop. Subclasses can reuse
    input preparation, gradient/proximal steps, and callback helpers without
    a state container or iteration framework; see ``docs/architecture.md``.
    """

    def __init__(
        self,
        positive=True,
        callbacks=None,
    ):
        super().__init__()

        self.positive = positive
        self.callbacks = callbacks if callbacks is not None else []

    @staticmethod
    def shrink(x, thresh):
        """Element-wise soft-thresholding."""
        return torch.sign(x) * F.relu(x.abs() - thresh)

    @staticmethod
    def shrink_positive(x, thresh):
        """Element-wise positive soft-thresholding.

        For positive shrinkage, we want max(x - theta, 0), not max(|x| - theta, 0).
        This enforces non-negativity on the output while applying the threshold.
        """
        return F.relu(x - thresh)

    def ls_grad(self, y, x, A, Op=None, get_residual=False):
        """Compute gradient using least-squares mode."""
        if Op is None:
            Op = A
        res = y - apply_forward_operator(A, x)
        grad = apply_adjoint_operator(res, Op) / float(y.shape[1])
        if get_residual:
            return grad, res
        return grad

    def get_grad_func(self):
        """Return the normalized LS gradient hook."""
        return self.ls_grad

    def get_shrink_func(self):
        """Get shrinkage function based on positivity."""
        if self.positive:
            return self.shrink_positive
        return self.shrink

    def step(self, y, x, A, shrink_func, grad_func, gamma, theta, Op=None):
        """Perform a single ISTA step."""
        grad = grad_func(y, x, A, Op=Op)
        x = x + gamma * grad
        return shrink_func(x, theta)

    def _prepare_forward_inputs(self, y, x0, A):
        """Validate prepared tensors and add batch axes without dtype/device conversion."""
        if not torch.is_tensor(y) or y.ndim not in (1, 2) or not torch.is_floating_point(y):
            msg = 'y must be a real floating-point tensor with shape (m,) or (B, m).'
            raise ValueError(msg)
        if y.ndim == 1:
            y = y.unsqueeze(0)
        batch_size = y.shape[0]

        A_tensor, l2_norms = prepare_sensing_matrix(A)
        validate_tensor(A_tensor, name='A', dtype=y.dtype, device=y.device)
        if x0 is not None:
            validate_tensor(x0, name='x0', dtype=y.dtype, device=y.device)
            if x0.ndim == 1:
                x0 = x0.unsqueeze(0)

        self.l2_norms = l2_norms
        validate_shapes(y, x0, A_tensor)
        return y, x0, A_tensor, l2_norms, batch_size

    def _resolve_lam_value(self, lam, batch_size, A_tensor, missing_message='lam must be provided.'):
        """Resolve regularization strength for scalar or per-batch input."""
        if torch.is_tensor(lam):
            validate_tensor(lam, name='lam', dtype=A_tensor.dtype, device=A_tensor.device)
        lam_value = resolve_batch_param(lam, batch_size, name='lam')
        if lam_value is None:
            raise ValueError(missing_message)
        return lam_value

    def _resolve_gamma_theta(
        self,
        lam_value,
        gamma,
        batch_size,
        A_tensor,
    ):
        """Resolve step size and shrinkage threshold for fixed-step solvers."""
        gamma_value = gamma
        if gamma_value is None:
            gamma_value = default_gamma_from_A(A_tensor)
        if torch.is_tensor(gamma_value):
            validate_tensor(gamma_value, name='gamma', dtype=A_tensor.dtype, device=A_tensor.device)
        gamma_value = resolve_batch_param(gamma_value, batch_size, name='gamma')

        # Normalized LASSO convention:
        #   1 / (2 m) * ||y - A x||_2^2 + lam * ||x||_1
        # The step uses the normalized negative gradient A.T @ (y - A x) / m,
        # so the proximal threshold is gamma * lam.
        theta_value = lam_value * gamma_value
        return gamma_value, theta_value

    def _setup_forward(self, y, x0, A, lam, gamma, W=None):
        """Protected preparation for classical and subclass-owned weighted forwards.

        The classical forward never supplies W. A subclass can expose W in its
        own forward and use this hook without duplicating tensor preparation.
        """
        y, x0, A_tensor, l2_norms, batch_size = self._prepare_forward_inputs(y, x0, A)
        W_tensor = prepare_gradient_weight(W, A_tensor, y)
        lam_value = self._resolve_lam_value(lam, batch_size, A_tensor)
        gamma_value, theta_value = self._resolve_gamma_theta(
            lam_value,
            gamma,
            batch_size,
            A_tensor,
        )
        return y, x0, A_tensor, l2_norms, gamma_value, theta_value, W_tensor

    def _start_iterations(self, y):
        """Initialize callback-visible solver state."""
        self.y = y
        self.stop = False

    def _on_iteration_begin(self, k, x):
        """Run begin callbacks and report whether iteration should continue."""
        self.iter = k
        for cb in self.callbacks:
            cb.on_step_begin(self, x)
        return not self.stop

    def _on_iteration_end(self, x, gamma, theta, A_tensor, y, W_tensor=None):
        """Run end callbacks and report whether iteration should continue."""
        if W_tensor is None:
            W_tensor = A_tensor
        for cb in self.callbacks:
            cb.on_step_end(self, x, gamma=gamma, theta=theta, A=A_tensor, W=W_tensor, y=y)
        return not self.stop

    def _iteration_functions(self):
        """Resolve gradient and shrinkage functions once per solve."""
        return self.get_grad_func(), self.get_shrink_func()

    def _init_solver_vector(self, y, x0, A_tensor):
        """Initialize one solution-shaped vector."""
        return y.new_zeros((y.shape[0], A_tensor.shape[-1])) if x0 is None else x0.clone()

    def _iterate(self, y, x0, A_tensor, gamma_value, theta_value, t, Op_tensor=None):
        """Run fixed-step ISTA with the solution local to this solve."""
        if Op_tensor is None:
            Op_tensor = A_tensor
        grad_func, shrink_func = self._iteration_functions()
        x = self._init_solver_vector(y, x0, A_tensor)

        self._start_iterations(y)
        for k in range(t):
            if not self._on_iteration_begin(k, x):
                break
            x = self.step(
                y,
                x,
                A_tensor,
                shrink_func,
                grad_func,
                gamma=gamma_value,
                theta=theta_value,
                Op=Op_tensor,
            )
            if not self._on_iteration_end(x, gamma_value, theta_value, A_tensor, y, W_tensor=Op_tensor):
                break
        return x

    def forward(self, y, x0=None, t=1, A=None, lam=None, gamma=None):
        """Solve using A as the classical adjoint, without an independent weight.

        All tensor inputs must already share y's dtype and device; mismatches
        raise ValueError. Convert arrays and move/cast tensors in preprocessing.
        Subclass-specific inputs belong in an overridden forward.

        Args:
            y (Tensor): measurements of shape ``(B, m)`` or ``(m,)``
            x0 (Tensor | None): optional init of shape ``(B, n)`` or ``(n,)``
            t (int): number of iterations
            A (Tensor | None): real floating-point sensing matrix of shape
                ``(m, n)`` or ``(B, m, n)``; treated as fixed (detached).
            lam (float | Tensor | None): required regularization
                strength, scalar or per-sample with shape ``(B,)`` / ``(B, 1)``.
            gamma (float | Tensor | None): scalar or per-sample
                step size. Defaults to ``m / ||A||_2**2`` for LS.

        Returns
        -------
            Tensor: solution of shape ``(B, n)``
        """
        y, x0, A_tensor, l2_norms, gamma_value, theta_value, Op_tensor = self._setup_forward(y, x0, A, lam, gamma)
        del l2_norms
        return self._iterate(y, x0, A_tensor, gamma_value, theta_value, t, Op_tensor=Op_tensor)


class FISTA(ISTA):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) for LASSO.

    Solves the objective:
        min_x 1/(2 m) ||y - A x||_2^2 + λ ||x||_1

    Uses Nesterov momentum acceleration for faster convergence than ISTA.
    """

    def _nesterov_momentum(self, tk, x, x_next):
        """Apply the standard Nesterov momentum recurrence."""
        t_next = 0.5 + math.sqrt(1.0 + 4.0 * tk**2) / 2.0
        z_next = x_next + (tk - 1.0) / t_next * (x_next - x)
        return t_next, z_next, x_next

    def _iterate_accelerated(self, y, x0, A_tensor, t, step_params_fn, Op_tensor=None):
        """Run FISTA with local momentum and caller-provided step parameters."""
        if Op_tensor is None:
            Op_tensor = A_tensor
        grad_func, shrink_func = self._iteration_functions()
        x = self._init_solver_vector(y, x0, A_tensor)
        z = self._init_solver_vector(y, x0, A_tensor)
        tk = 1.0

        self._start_iterations(y)
        for k in range(t):
            if not self._on_iteration_begin(k, x):
                break
            gamma_k, theta_k = step_params_fn(k)
            x_next = self.step(
                y,
                z,
                A_tensor,
                shrink_func,
                grad_func,
                gamma=gamma_k,
                theta=theta_k,
                Op=Op_tensor,
            )
            tk, z, x = self._nesterov_momentum(tk, x, x_next)
            if not self._on_iteration_end(x, gamma_k, theta_k, A_tensor, y, W_tensor=Op_tensor):
                break
        return x

    def _iterate(self, y, x0, A_tensor, gamma_value, theta_value, t, Op_tensor=None):
        """Run FISTA iterations with Nesterov momentum and return the solution."""
        return self._iterate_accelerated(
            y,
            x0,
            A_tensor,
            t,
            step_params_fn=lambda _k: (gamma_value, theta_value),
            Op_tensor=Op_tensor,
        )
