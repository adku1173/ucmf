"""Frobenius-isometric vectorization of Hermitian matrices (CMF measurements)."""

import math

import torch
from torch import nn


class HermitianVectorization(nn.Module):
    """Linear operator ``hvec`` from Hermitian ``c x c`` matrices to ``R^m``.

    A Hermitian matrix ``S`` is given by its real (symmetric) and imaginary
    (antisymmetric) parts of shape ``(..., c, c)``. ``hvec`` keeps the lower
    triangle of ``S.T`` in row-major order, real parts first, then imaginary
    off-diagonal parts: Acoular's CMF ordering and signs. Off-diagonal entries
    are weighted by ``sqrt(2)``, so ``hvec`` is an isometry for the Frobenius
    inner product ``<S, T> = <Re S, Re T> + <Im S, Im T>``:

        ||hvec(S)||_2 = ||S||_F,   adjoint(hvec(S)) = S,   hvec(adjoint(v)) = v.

    With ``r_diag=True`` the diagonal is dropped (``m = c (c - 1)``); the
    adjoint then returns a zero diagonal. Without it, ``m = c**2``.
    Index buffers move with ``.to(device)``; weights use the input dtype.
    """

    def __init__(self, n_mics, r_diag=False):
        super().__init__()
        self.n_mics = n_mics
        # Lower triangle of S.T at (row, col) holds S[col, row]: i = col, j = row.
        rows, cols = torch.tril_indices(n_mics, n_mics)
        off_diagonal = rows != cols
        keep_real = off_diagonal if r_diag else torch.ones_like(off_diagonal)
        self.register_buffer('real_i', cols[keep_real], persistent=False)
        self.register_buffer('real_j', rows[keep_real], persistent=False)
        self.register_buffer('imag_i', cols[off_diagonal], persistent=False)
        self.register_buffer('imag_j', rows[off_diagonal], persistent=False)
        self.n_real = int(keep_real.sum())
        self.n_measurements = self.n_real + int(off_diagonal.sum())

    def _real_weights(self, reference):
        """Weights of the real entries: 1 on the diagonal, sqrt(2) off it, in reference's dtype."""
        diagonal = (self.real_i == self.real_j).to(device=reference.device)
        return torch.where(diagonal, reference.new_tensor(1.0), reference.new_tensor(math.sqrt(2.0)))

    def row_weights(self, reference):
        """Weights relating an unweighted (current Acoular) vector to ``hvec``: ``hvec = w * vec``."""
        imag = reference.new_full((self.n_measurements - self.n_real,), math.sqrt(2.0))
        return torch.cat((self._real_weights(reference), imag))

    def forward(self, real, imag):
        """``hvec``: ``(..., c, c)`` real/imaginary parts -> ``(..., m)``. Hermitian symmetry is assumed."""
        return torch.cat(
            (
                real[..., self.real_i, self.real_j] * self._real_weights(real),
                imag[..., self.imag_i, self.imag_j] * math.sqrt(2.0),
            ),
            dim=-1,
        )

    def adjoint(self, vector):
        """``hvec*``: ``(..., m)`` -> Hermitian real/imaginary parts ``(..., c, c)``.

        Off-diagonal entries ``v / sqrt(2)`` are placed at ``(i, j)`` and mirrored to
        ``(j, i)`` (imaginary part with flipped sign); diagonal entries are ``v``.
        """
        shape = (*vector.shape[:-1], self.n_mics, self.n_mics)
        real = vector.new_zeros(shape)
        imag = vector.new_zeros(shape)
        # Symmetrization below adds the diagonal twice, hence the extra factor 2 there.
        scale = self._real_weights(vector) * (1 + (self.real_i == self.real_j).to(vector))
        real[..., self.real_i, self.real_j] = vector[..., : self.n_real] / scale
        imag[..., self.imag_i, self.imag_j] = vector[..., self.n_real :] / math.sqrt(2.0)
        return real + real.mT, imag - imag.mT
