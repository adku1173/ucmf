"""Single-weight reduced Ada-LFISTA (Ada-LFISTA-SWR) for CMF measurements.

Like ``AdaLFISTASW``, but the ``m x m`` auxiliary weight ``B`` is not stored.
It is the congruence with a learned complex ``Br = Br_re + i Br_im`` (``c x c``,
``c`` microphones) in the Frobenius-isometric vectorization ``hvec``:

    B hvec(S) = hvec(Br S Br^H),     B^T hvec(S) = hvec(Br^H S Br).

That is ``2 c**2`` instead of ``c**4`` parameters. ``W = B @ A`` is formed once
per solve column by column (``hvec* -> congruence -> hvec``) with real tensors
only; neither ``B`` nor complex tensors are materialized. ``Br = I`` gives
``W = A``, i.e. FISTA. A and y must use ``hvec`` (sqrt(2)-weighted, see
``operator.HermitianVectorization``). The iteration is the one of AdaLFISTASW:

    x_next = shrink(z + gamma_k * W.T @ (y - A z) / m, theta_k)
"""

from .ista import FISTA
from .operator import HermitianVectorization
from .utils import softplus_inverse, validate_unfolded_iterations
from .weights import prepare_gradient_weight, validate_measurement_subset

import torch
import torch.nn.functional as F
from torch import nn


class AdaLFISTASWR(FISTA):
    """Ada-LFISTA-SWR with trainable ``Br_re``, ``Br_im``, ``gamma`` and ``theta``.

    Args:
        n_iterations (int): number of unfolded iterations/layers.
        n_mics (int): number of microphones ``c``.
        gamma_init (float): initial applied step size for every iteration.
        theta_init (float): initial applied soft-threshold for every iteration.
        positive (bool): enforce non-negativity via positive shrinkage.
        callbacks (list | None): optional solver callbacks; they get ``W = B @ A``.
        r_diag (bool): CSM diagonal removed from A and y, ``m = c (c - 1)``;
            otherwise ``m = c**2``.
    """

    def __init__(self, n_iterations, n_mics, gamma_init, theta_init, positive=True, callbacks=None, r_diag=False):
        super().__init__(positive=positive, callbacks=callbacks)
        if n_iterations <= 0:
            raise ValueError(f'n_iterations must be positive, got {n_iterations}')
        self.n_iterations = n_iterations
        self.n_mics = n_mics
        self.hvec = HermitianVectorization(n_mics, r_diag=r_diag)
        self.n_measurements = self.hvec.n_measurements
        if self.n_measurements <= 0:
            raise ValueError(f'n_measurements must be positive, got {self.n_measurements} for n_mics={n_mics}')
        self.Br_re = nn.Parameter(torch.eye(n_mics))
        self.Br_im = nn.Parameter(torch.zeros(n_mics, n_mics))
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

    def _apply_auxiliary_weight(self, A_tensor):
        """``W = B @ A`` column by column: ``hvec(Br hvec*(a) Br^H)``, shape of A."""
        P, Q = self.hvec.adjoint(A_tensor.mT)  # S = P + iQ per column, (..., n, c, c)
        U, V = self.Br_re, self.Br_im
        # Br S = L_re + i L_im;  Br S Br^H = (L_re U^T + L_im V^T) + i (L_im U^T - L_re V^T)
        L_re = U @ P - V @ Q
        L_im = U @ Q + V @ P
        return self.hvec(L_re @ U.T + L_im @ V.T, L_im @ U.T - L_re @ V.T).mT

    def _effective_weight(self, A_tensor, msm_subset):
        """``B @ A`` on the full set; ``B[I, I] @ A_sub`` on a reduced set ``I``.

        The reduced weight zero-fills the missing rows, applies B and keeps rows I:
        ``(B @ A_full)[I] = B[I, I] @ A_sub`` as ``A_full`` vanishes outside I.
        Indices must be unique and in ``[0, m)`` (guaranteed by the index generator).
        """
        if msm_subset is None:
            if A_tensor.shape[-2] != self.n_measurements:
                raise ValueError(f'A has {A_tensor.shape[-2]} measurements but B expects {self.n_measurements}.')
            return self._apply_auxiliary_weight(A_tensor)
        idx = validate_measurement_subset(msm_subset, A_tensor, self.Br_re.device)
        rows = idx.unsqueeze(-1).expand_as(A_tensor)
        A_full = A_tensor.new_zeros((*A_tensor.shape[:-2], self.n_measurements, A_tensor.shape[-1]))
        return self._apply_auxiliary_weight(A_full.scatter(-2, rows, A_tensor)).gather(-2, rows)

    def forward(self, y, x0=None, t=1, A=None, msm_subset=None):
        """Run Ada-LFISTA-SWR; same inputs and contracts as ``AdaLFISTASW.forward``."""
        validate_unfolded_iterations(t, self.n_iterations)
        y, x0, A_tensor, _ = self._prepare_forward_inputs(y, x0, A)
        W_tensor = prepare_gradient_weight(self._effective_weight(A_tensor, msm_subset), A_tensor, y)
        gammas, thetas = self.gamma.unbind(), self.theta.unbind()
        return self._iterate_accelerated(
            y,
            x0,
            A_tensor,
            t,
            step_params_fn=lambda k: (gammas[k], thetas[k]),
            Op_tensor=W_tensor,
        )
