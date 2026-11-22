"""Single-weight adaptive LFISTA (Ada-LFISTA-SW).

``AdaLFISTASW`` augments FISTA with three sets of trainable parameters:

    - ``B``: a single ``m x m`` auxiliary weight matrix forming the effective adjoint-side
      gradient weight ``W = B @ A``. With ``B = I`` the update reduces to FISTA.
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
from .weights import apply_auxiliary_weight, measurement_subset_weight, prepare_gradient_weight

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
    ):
        super().__init__(positive=positive, callbacks=callbacks)
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

    def _effective_weight(self, A_tensor, msm_subset):
        """Return the effective adjoint-side weight for this solve.

        The forward operator stays ``A``; only the adjoint ``A.T`` is replaced by
        ``W.T``. ``B = I`` gives ``W = A``, i.e. classical FISTA. For a reduced
        measurement set ``I`` the full ``B`` is restricted to the active rows and
        columns, so the gradient only couples active measurements.
        """
        if msm_subset is None:
            # W = B @ A on the full measurement set.
            if A_tensor.shape[-2] != self.n_measurements:
                raise ValueError(f'A has {A_tensor.shape[-2]} measurements but B expects {self.n_measurements}.')
            return apply_auxiliary_weight(self.B, A_tensor)
        # W_sub = B[I, I] @ A_sub on the reduced measurement set I = msm_subset.
        return measurement_subset_weight(self.B, A_tensor, msm_subset)

    def forward(self, y, x0=None, t=1, A=None, msm_subset=None):
        """Run single-weight Ada-LFISTA with learned ``gamma``/``theta`` and ``W = B @ A``.

        All tensor inputs and the model's parameters must already share ``y``'s
        dtype and device. ``msm_subset`` is an optional integer index tensor
        (``(m_sub,)`` or per-sample ``(B, m_sub)``) naming the original rows of
        the already-subsetted ``y``/``A``; it restricts ``B`` to ``B[I, I]``.
        """
        validate_unfolded_iterations(t, self.n_iterations)
        # The shared hook also returns the batch size for per-sample lam/gamma in
        # classical solvers; this model learns gamma/theta, so it is discarded.
        y, x0, A_tensor, _ = self._prepare_forward_inputs(y, x0, A)
        # W = B @ A, validated against A's shape and returned unchanged so the
        # autograd graph through B is preserved. The layer update then is
        #   x_{k+1} = shrink(z_k + gamma_k * W.T @ (y - A @ z_k) / m, theta_k).
        W_tensor = prepare_gradient_weight(self._effective_weight(A_tensor, msm_subset), A_tensor, y)
        # gamma = softplus(_gamma), theta = softplus(_theta): applied once here,
        # then unbound into per-layer scalars so the loop only does tuple indexing
        # (no per-layer tensor __getitem__ dispatch). unbind keeps the autograd graph.
        gammas, thetas = self.gamma.unbind(), self.theta.unbind()
        return self._iterate_accelerated(
            y,
            x0,
            A_tensor,
            t,
            step_params_fn=lambda k: (gammas[k], thetas[k]),
            Op_tensor=W_tensor,
        )
