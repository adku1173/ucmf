"""Linear operators for the solvers and the Hermitian vectorization of CMF measurements.

Operators are batched: one operator for all problems (``batch_size is None``)
or one per problem (``batch_size = B``). ``forward`` maps ``(B, n) -> (B, m)``,
``adjoint`` maps ``(B, m) -> (B, n)``. Subsets are new operators; ``rows``
holds their measurement indices relative to the full, original measurement
set (``None`` = all rows), shared ``(m,)`` or per problem ``(B, m)``.
Selection methods retain the original number of measurements (and microphones
for CMF) as shape metadata. An externally sliced matrix may not know that size.
Selection indices are a precondition (unique, in range): they are only checked
for dtype, device and shape, since value checks would force a GPU sync.
"""

import math
from copy import copy

from .utils import apply_adjoint_operator, apply_forward_operator, validate_tensor

import torch
from torch import nn


def _validate_index(index, name, device, batch_size):
    """Check a shared ``(k,)`` or per-problem ``(B, k)`` integer index; no value checks."""
    if not torch.is_tensor(index) or index.dtype not in (torch.int32, torch.int64):
        raise ValueError(f'{name} must be an integer tensor.')
    if index.device != device:
        raise ValueError(f'{name} must be on device {device}, got {index.device}.')
    if index.ndim not in (1, 2):
        raise ValueError(f'{name} must be 1D (shared) or 2D (per problem), got shape {tuple(index.shape)}.')
    if index.ndim == 2 and batch_size is not None and index.shape[0] != batch_size:
        raise ValueError(f'{name} has batch size {index.shape[0]} but the operator has {batch_size}.')
    return index.long()


def _take(tensor, index, dim, batched):
    """Select ``index`` along ``dim`` of a shared or batched tensor; per-problem index -> batched."""
    if index.ndim == 1:
        return tensor.index_select(dim, index)
    if not batched:  # shared tensor, per-problem index: (B, ...) via advanced indexing
        return tensor.movedim(dim, 0)[index].movedim(1, dim + 1)
    shape = list(tensor.shape)
    shape[dim] = index.shape[1]
    view = [index.shape[0]] + [1] * (tensor.ndim - 1)
    view[dim] = index.shape[1]
    return tensor.gather(dim, index.view(view).expand(shape))


class LinearOperator:
    """Batched real linear operator ``A: R^n -> R^m`` used by the solvers.

    Subclasses set ``shape = (m, n)``, ``batch_size``, ``rows``, ``dtype``,
    ``device`` and implement ``forward`` / ``adjoint``.
    """

    _num_msms_total = None

    def forward(self, x):
        """``A x`` for ``x`` of shape ``(B, n)``; returns ``(B, m)``."""
        raise NotImplementedError

    def adjoint(self, r):
        """``A^T r`` for ``r`` of shape ``(B, m)``; returns ``(B, n)``."""
        raise NotImplementedError

    def detached(self):
        """Return an operator with detached data sharing storage and unchanged subset metadata."""
        raise NotImplementedError

    def _subset_rows(self, index):
        """Original measurement indices of a row subset ``index`` of this operator."""
        if self.rows is None:
            return index
        return _take(self.rows, index, self.rows.ndim - 1, self.rows.ndim == 2)


class MatrixMult(LinearOperator):
    """Operator given by a stored real matrix ``(m, n)`` (shared) or ``(B, m, n)``.

    The matrix is used as given (no copy, no detach): ``matrix`` exposes it.
    """

    def __init__(self, matrix, rows=None):
        validate_tensor(matrix, name='matrix')
        if matrix.ndim not in (2, 3):
            raise ValueError(f'matrix must be 2D or 3D, got shape {tuple(matrix.shape)}')
        self.matrix = matrix
        self.rows = rows
        self.shape = tuple(matrix.shape[-2:])
        self._num_msms_total = self.shape[0] if rows is None else None
        self.batch_size = matrix.shape[0] if matrix.ndim == 3 else None
        self.dtype = matrix.dtype
        self.device = matrix.device

    def forward(self, x):
        return apply_forward_operator(self.matrix, x)

    def adjoint(self, r):
        return apply_adjoint_operator(r, self.matrix)

    def _with_matrix(self, matrix, rows):
        op = MatrixMult(matrix, rows=rows)
        op._num_msms_total = self._num_msms_total
        return op

    def detached(self):
        """Reuse the matrix storage without tracking gradients through it."""
        return self._with_matrix(self.matrix.detach(), rows=self.rows)

    def select_rows(self, index):
        """Operator of the measurement rows ``index``: ``A[index, :]`` (per problem: ``A_b[index_b, :]``)."""
        index = _validate_index(index, 'row index', self.device, self.batch_size)
        matrix = _take(self.matrix, index, self.matrix.ndim - 2, self.batch_size is not None)
        return self._with_matrix(matrix, rows=self._subset_rows(index))

    def select_columns(self, index):
        """Operator of the sources ``index``: ``A[:, index]`` (per problem: ``A_b[:, index_b]``)."""
        index = _validate_index(index, 'column index', self.device, self.batch_size)
        matrix = _take(self.matrix, index, self.matrix.ndim - 1, self.batch_size is not None)
        return self._with_matrix(matrix, rows=self.rows)  # measurement rows are unchanged


class _AuxiliaryOperator(LinearOperator):
    """Composition ``W = B @ A`` without materializing the product."""

    def __init__(self, operator, weight):
        self.operator, self.weight = operator, weight
        self.shape, self.rows = operator.shape, operator.rows
        self._num_msms_total = operator._num_msms_total
        self.dtype, self.device = operator.dtype, operator.device
        self.batch_size = weight.batch_size if weight.batch_size is not None else operator.batch_size

    def forward(self, x):
        return self.weight.forward(self.operator.forward(x))

    def adjoint(self, r):
        return self.operator.adjoint(self.weight.adjoint(r))

    def detached(self):
        return _AuxiliaryOperator(self.operator.detached(), self.weight.detached())


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


class _AuxiliaryPropagationOperator(LinearOperator):
    """Apply ``hvec(Br hvec*(v) Br^H)`` in the active CMF coordinates.

    ``Br`` is complex, shared ``(c, c)`` or per problem ``(B, c, c)``;
    ``hvec`` must match its microphone dimensions and device. For a reduced
    vectorization, ``hvec*`` zero-fills the diagonal before the congruence,
    and ``hvec`` discards it again after the congruence.
    """

    def __init__(self, Br, hvec):
        self.Br, self.hvec = Br, hvec
        m = hvec.n_measurements
        self.shape = (m, m)
        self.dtype, self.device = Br.real.dtype, Br.device
        self.batch_size = Br.shape[0] if Br.ndim == 3 else None
        self.rows = None

    def _apply(self, vector, Br):
        csm = torch.complex(*self.hvec.adjoint(vector))
        transformed = Br @ csm @ Br.mH
        return self.hvec(transformed.real, transformed.imag)

    def forward(self, x):
        return self._apply(x, self.Br)

    def adjoint(self, r):
        return self._apply(r, self.Br.mH)

    def detached(self):
        return _AuxiliaryPropagationOperator(self.Br.detach(), self.hvec)


class CMFPropagation(LinearOperator):
    """Matrix-free CMF using a complex propagation matrix ``H``, ``(c, n)`` or ``(B, c, n)``.

    Inputs and outputs of the operator are real. ``mics`` and ``rows`` map a
    subset to the original microphone and measurement coordinates.
    """

    def __init__(self, H, r_diag=False):
        if not torch.is_tensor(H) or not H.is_complex() or H.ndim not in (2, 3):
            raise ValueError('H must be a complex tensor with shape (c, n) or (B, c, n).')
        c, n = H.shape[-2:]
        if c <= int(r_diag) or n == 0:
            raise ValueError('H must define nonempty measurement and source dimensions.')
        self.H, self.r_diag = H, r_diag
        self.hvec = HermitianVectorization(c, r_diag=r_diag).to(H.device)
        self.shape = (self.hvec.n_measurements, n)
        self._num_msms_total, self._num_mics_total = self.shape[0], c
        self.dtype, self.device = H.real.dtype, H.device
        self.batch_size = H.shape[0] if H.ndim == 3 else None
        self.rows = self.mics = None

    def forward(self, x):
        """``hvec(H diag(x) H^H)`` without forming ``diag(x)`` or the dictionary."""
        csm = (self.H * x.unsqueeze(-2)) @ self.H.mH
        return self.hvec(csm.real, csm.imag)

    def adjoint(self, r):
        """``Re(diag(H^H hvec*(r) H))`` via column-wise inner products."""
        real, imag = self.hvec.adjoint(r)
        csm = torch.complex(real, imag)
        return (self.H.conj() * (csm @ self.H)).real.sum(-2)

    def detached(self):
        """Detach ``H`` while reusing the vectorization and subset metadata."""
        op = copy(self)
        op.H = self.H.detach()
        return op

    def _with_propagation(self, H):
        op = CMFPropagation(H, r_diag=self.r_diag)
        op._num_msms_total, op._num_mics_total = self._num_msms_total, self._num_mics_total
        op.rows, op.mics = self.rows, self.mics
        return op

    def select_columns(self, index):
        """Select source columns, shared or per problem; measurement rows stay fixed."""
        index = _validate_index(index, 'column index', self.device, self.batch_size)
        return self._with_propagation(_take(self.H, index, self.H.ndim - 1, self.H.ndim == 3))

    def select_mics(self, index):
        """Keep the CSM block of selected microphones in ascending local index order."""
        index = _validate_index(index, 'microphone index', self.device, self.batch_size).sort(dim=-1).values
        op = self._with_propagation(_take(self.H, index, self.H.ndim - 2, self.H.ndim == 3))
        op.mics = index if self.mics is None else _take(self.mics, index, self.mics.ndim - 1, self.mics.ndim == 2)
        i, j = index[..., op.hvec.real_i], index[..., op.hvec.real_j]
        # Offset of row j in the packed lower triangle of CSM.T, plus column i.
        real_rows = j * (j + (-1 if self.r_diag else 1)) // 2 + i
        i, j = index[..., op.hvec.imag_i], index[..., op.hvec.imag_j]
        imag_rows = self.hvec.n_real + j * (j - 1) // 2 + i
        op.rows = self._subset_rows(torch.cat((real_rows, imag_rows), dim=-1))
        return op

    def transformed(self, Br):
        """Build the operator of ``Br @ H`` (complex ``Br`` in the current microphone coordinates).

        With ``r_diag=True``, discard the diagonal *after* the transformation.
        This is not a congruence of the zero-diagonal CSM.
        """
        return self._with_propagation(Br @ self.H)


class CMFDictionary(MatrixMult):
    """Explicit CMF dictionary with columns ``hvec(h_j h_j^H)``.

    Microphone and source selection preserve the CMF representation. Arbitrary
    row selection returns a ``MatrixMult`` with the original row indices.
    """

    def __init__(self, H, r_diag=False):
        self._init_from_propagation(CMFPropagation(H, r_diag=r_diag))

    def _init_from_propagation(self, propagation, matrix=None):
        self._propagation = propagation
        self.H, self.hvec, self.r_diag = propagation.H, propagation.hvec, propagation.r_diag
        self.mics = propagation.mics
        if matrix is None:
            H, hvec = self.H, self.hvec
            # Form only the packed entries of h_j h_j^H, not the full CSM per source.
            real = (H[..., hvec.real_i, :] * H[..., hvec.real_j, :].conj()).real
            imag = (H[..., hvec.imag_i, :] * H[..., hvec.imag_j, :].conj()).imag
            entries = torch.cat((real, imag), dim=-2)
            matrix = entries * hvec.row_weights(entries).unsqueeze(-1)
        super().__init__(matrix, rows=propagation.rows)
        self._num_msms_total, self._num_mics_total = propagation._num_msms_total, propagation._num_mics_total

    def _derived(self, propagation, matrix=None):
        # Skip __init__: the propagation operator already carries the selected rows and microphones.
        op = CMFDictionary.__new__(CMFDictionary)
        op._init_from_propagation(propagation, matrix)
        return op

    def detached(self):
        """Detach both ``H`` and the cached dictionary without rebuilding it."""
        return self._derived(self._propagation.detached(), self.matrix.detach())

    def select_columns(self, index):
        """Select source columns from both ``H`` and the stored dictionary."""
        propagation = self._propagation.select_columns(index)
        # MatrixMult selects the same columns from the cached dictionary. Reuse its
        # matrix so the new CMFDictionary need not rebuild it from the selected H.
        return self._derived(propagation, super().select_columns(index).matrix)

    def select_mics(self, index):
        """Build the dictionary of the selected microphone block."""
        return self._derived(self._propagation.select_mics(index))

    def transformed(self, Br):
        """Build the dictionary of ``Br @ H``, dropping the diagonal afterwards if requested."""
        return self._derived(self._propagation.transformed(Br))
