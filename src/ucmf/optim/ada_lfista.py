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
from .utils import (
    apply_adjoint_operator,
    apply_forward_operator,
    softplus_inverse,
    validate_unfolded_iterations,
)
from .weights import apply_auxiliary_weight, measurement_subset_weight, prepare_gradient_weight

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
            physical ``A``, ``W = M`` and ``W_recurrent = R``.
        B1, B2 (Tensor | None): optional ``(m, m)`` inits; default identity.
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
    ):
        super().__init__(positive=positive, callbacks=callbacks)
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

    def _effective_weight(self, B, A_tensor, msm_subset):
        """``B @ A`` on the full set, ``B[I, I] @ A_sub`` on a reduced set ``I``."""
        if msm_subset is None:
            if A_tensor.shape[-2] != self.n_measurements:
                raise ValueError(f'A has {A_tensor.shape[-2]} measurements but B expects {self.n_measurements}.')
            return apply_auxiliary_weight(B, A_tensor)
        return measurement_subset_weight(B, A_tensor, msm_subset)

    def ls_grad(self, y, x, A, Op=None):
        """Two-matrix direction ``(M.T @ y - R.T @ (R @ x)) / m`` with ``Op = (M, M.T @ y, R)``."""
        _, Mty, R = Op
        feedback = apply_adjoint_operator(apply_forward_operator(R, x), R)
        return (Mty - feedback) / float(y.shape[1])

    def _on_iteration_end(self, x, gamma, theta, A_tensor, y, W_tensor=None):
        """Report the physical ``A`` and both effective operators to end callbacks."""
        M, _, R = W_tensor
        for cb in self.callbacks:
            cb.on_step_end(self, x, gamma=gamma, theta=theta, A=A_tensor, W=M, W_recurrent=R, y=y)
        return not self.stop

    def forward(self, y, x0=None, t=1, A=None, msm_subset=None):
        """Run two-matrix Ada-LFISTA with learned ``gamma``/``theta``.

        All tensor inputs and the model's parameters must already share ``y``'s
        dtype and device. ``msm_subset`` (``(m_sub,)`` or per-sample
        ``(B, m_sub)``) names the original rows of the already-subsetted
        ``y``/``A`` and restricts each factor first: ``B1[I, I]``, ``B2[I, I]``.
        """
        validate_unfolded_iterations(t, self.n_iterations)
        y, x0, A_tensor, _ = self._prepare_forward_inputs(y, x0, A)
        # R = B1 @ A (recurrence), M = B2 @ A (measurement branch); validated, autograd kept.
        R = prepare_gradient_weight(self._effective_weight(self.B1, A_tensor, msm_subset), A_tensor, y)
        M = prepare_gradient_weight(self._effective_weight(self.B2, A_tensor, msm_subset), A_tensor, y)
        # M.T @ y does not depend on the iteration: computed once per solve, so each
        # iteration needs only R @ z and R.T @ (.), as many products as Ada-LFISTA-SW.
        Mty = apply_adjoint_operator(y, M)
        gammas, thetas = self.gamma.unbind(), self.theta.unbind()
        return self._iterate_accelerated(
            y,
            x0,
            A_tensor,
            t,
            step_params_fn=lambda k: (gammas[k], thetas[k]),
            Op_tensor=(M, Mty, R),
        )
