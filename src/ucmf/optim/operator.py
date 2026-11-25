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

from .utils import validate_tensor

import torch
import torch.nn.functional as F


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


def _validate_scale(scale, operator):
    """Check a real shared ``(n,)`` or per-problem ``(B, n)`` column scale; no value checks."""
    validate_tensor(scale, name='scale', dtype=operator.dtype, device=operator.device)
    if scale.ndim not in (1, 2) or scale.shape[-1] != operator.shape[1]:
        raise ValueError(f'scale must have shape (n,) or (B, n) with n={operator.shape[1]}, got {tuple(scale.shape)}.')
    if scale.ndim == 2 and operator.batch_size is not None and scale.shape[0] != operator.batch_size:
        raise ValueError(f'scale has batch size {scale.shape[0]} but the operator has {operator.batch_size}.')
    return scale


def _take(tensor, index, dim, batched):
    """Select ``index`` along ``dim`` of a shared or batched tensor; a per-problem index batches."""
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

    Subclasses set the attributes below and implement :meth:`forward`,
    :meth:`adjoint`, and :meth:`detached`.

    Attributes
    ----------
    shape : tuple of int
        ``(m, n)``: active measurement rows and sources.
    batch_size : int or None
        Number of problems of a per-problem operator; None for a shared one.
    rows : Tensor or None
        Indices of the active rows in the full measurement set, shared ``(m,)``
        or per problem ``(B, m)``; None for all rows.
    dtype : torch.dtype
        Real dtype of inputs and outputs.
    device : torch.device
        Device of the operator data.
    """

    _num_measurements_total = None

    def forward(self, x):
        """Apply ``A``.

        Parameters
        ----------
        x : Tensor
            Source vectors of shape ``(B, n)``.

        Returns
        -------
        Tensor
            ``A x`` of shape ``(B, m)``.
        """
        raise NotImplementedError

    def adjoint(self, r):
        """Apply ``A^T``.

        Parameters
        ----------
        r : Tensor
            Measurement vectors of shape ``(B, m)``.

        Returns
        -------
        Tensor
            ``A^T r`` of shape ``(B, n)``.
        """
        raise NotImplementedError

    def detached(self):
        """Return the operator with detached data.

        The data share storage with this operator; the subset metadata are unchanged.

        Returns
        -------
        LinearOperator
            Operator of the same type through which no gradients flow.
        """
        raise NotImplementedError

    def _subset_rows(self, index):
        """Original measurement indices of a row subset ``index`` of this operator."""
        if self.rows is None:
            return index
        return _take(self.rows, index, self.rows.ndim - 1, self.rows.ndim == 2)


class MatrixMult(LinearOperator):
    """Operator given by a stored real matrix.

    The matrix is used as given, without copy or detach.

    Parameters
    ----------
    matrix : Tensor
        Real floating-point matrix, shared ``(m, n)`` or per problem ``(B, m, n)``.
    rows : Tensor, optional
        Indices of the matrix rows in the full measurement set if ``matrix`` is
        already a row subset. Such an operator does not know the full size.

    Attributes
    ----------
    matrix : Tensor
        The stored matrix.
    """

    def __init__(self, matrix, rows=None):
        validate_tensor(matrix, name='matrix')
        if matrix.ndim not in (2, 3):
            raise ValueError(f'matrix must be 2D or 3D, got shape {tuple(matrix.shape)}')
        self.matrix = matrix
        self.rows = rows
        self.shape = tuple(matrix.shape[-2:])
        self._num_measurements_total = self.shape[0] if rows is None else None
        self.batch_size = matrix.shape[0] if matrix.ndim == 3 else None
        self.dtype = matrix.dtype
        self.device = matrix.device

    def forward(self, x):
        """Multiply by the matrix, see :meth:`LinearOperator.forward`."""
        if self.matrix.ndim == 2:
            return F.linear(x, self.matrix)
        return torch.bmm(self.matrix, x.unsqueeze(-1)).squeeze(-1)

    def adjoint(self, r):
        """Multiply by the transposed matrix, see :meth:`LinearOperator.adjoint`."""
        if self.matrix.ndim == 2:
            return r @ self.matrix
        return torch.bmm(r.unsqueeze(1), self.matrix).squeeze(1)

    def column_norms(self):
        """Return the L2 norms of the matrix columns.

        Returns
        -------
        Tensor
            Norms of shape ``(n,)`` for a shared, ``(B, n)`` for a per-problem matrix.
        """
        return torch.linalg.vector_norm(self.matrix, dim=-2)

    def scale_columns(self, scale):
        """Return the operator ``A diag(scale)``; the subset metadata are unchanged.

        Parameters
        ----------
        scale : Tensor
            Positive column factors, shared ``(n,)`` or per problem ``(B, n)``.

        Returns
        -------
        MatrixMult
            Operator of the scaled matrix, per problem if ``scale`` is.
        """
        scale = _validate_scale(scale, self)
        return self._with_matrix(self.matrix * scale.unsqueeze(-2), rows=self.rows)

    def _with_matrix(self, matrix, rows):
        op = MatrixMult(matrix, rows=rows)
        op._num_measurements_total = self._num_measurements_total
        return op

    def detached(self):
        """Return the operator of the detached matrix, sharing its storage."""
        return self._with_matrix(self.matrix.detach(), rows=self.rows)

    def select_rows(self, index):
        """Select measurement rows.

        Parameters
        ----------
        index : Tensor
            Local row indices, shared ``(k,)`` or per problem ``(B, k)``.

        Returns
        -------
        MatrixMult
            ``A[index, :]`` (per problem ``A_b[index_b, :]``) with ``rows`` in the full
            measurement coordinates.
        """
        index = _validate_index(index, 'row index', self.device, self.batch_size)
        matrix = _take(self.matrix, index, self.matrix.ndim - 2, self.batch_size is not None)
        return self._with_matrix(matrix, rows=self._subset_rows(index))

    def select_columns(self, index):
        """Select sources; the measurement rows stay unchanged.

        Parameters
        ----------
        index : Tensor
            Source indices, shared ``(k,)`` or per problem ``(B, k)``.

        Returns
        -------
        MatrixMult
            ``A[:, index]`` (per problem ``A_b[:, index_b]``).
        """
        index = _validate_index(index, 'column index', self.device, self.batch_size)
        matrix = _take(self.matrix, index, self.matrix.ndim - 1, self.batch_size is not None)
        return self._with_matrix(matrix, rows=self.rows)  # measurement rows are unchanged

    def select_problems(self, index):
        """Select problems of a per-problem operator (a dictionary bank).

        Preconditions: the operator is per problem, ``rows`` is None or shared, and a
        tensor ``index`` is an integer tensor on the operator's device.

        Parameters
        ----------
        index : Tensor or int
            Problem indices ``(B,)`` for a per-problem result, or one index (int or 0-dim) for a
            shared one.

        Returns
        -------
        MatrixMult
            Operator of ``matrix[index]``.
        """
        return self._with_matrix(self.matrix[index], rows=self.rows)


class _AuxiliaryOperator(LinearOperator):
    """Composition ``W = B @ A`` without materializing the product."""

    def __init__(self, operator, weight):
        self.operator, self.weight = operator, weight
        self.shape, self.rows = operator.shape, operator.rows
        self._num_measurements_total = operator._num_measurements_total
        self.dtype, self.device = operator.dtype, operator.device
        self.batch_size = weight.batch_size if weight.batch_size is not None else operator.batch_size

    def forward(self, x):
        return self.weight.forward(self.operator.forward(x))

    def adjoint(self, r):
        return self.operator.adjoint(self.weight.adjoint(r))

    def detached(self):
        return _AuxiliaryOperator(self.operator.detached(), self.weight.detached())


class HermitianVectorization:
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
    The index tensors live on ``device``; weights use the input dtype. It is a plain
    object, not an ``nn.Module``: compiled graphs with CUDA graphs treat module buffers
    as static inputs and re-record whenever a derived operator brings new ones.
    Plain tensors are ordinary inputs, copied per replay.

    Parameters
    ----------
    num_mics : int
        Number of microphones ``c``.
    r_diag : bool, optional
        If True, drop the diagonal (default False).
    device : torch.device or str, optional
        Device of the index tensors (default CPU).

    Attributes
    ----------
    num_measurements : int
        Length ``m`` of the vectorization.
    num_real : int
        Number of leading real entries.
    """

    def __init__(self, num_mics, r_diag=False, device=None):
        self.num_mics = num_mics
        # Lower triangle of S.T at (row, col) holds S[col, row]: i = col, j = row.
        rows, cols = torch.tril_indices(num_mics, num_mics, device=device)
        off_diagonal = rows != cols
        keep_real = off_diagonal if r_diag else torch.ones_like(off_diagonal)
        self.real_i, self.real_j = cols[keep_real], rows[keep_real]
        self.imag_i, self.imag_j = cols[off_diagonal], rows[off_diagonal]
        self.num_real = int(keep_real.sum())
        self.num_measurements = self.num_real + int(off_diagonal.sum())

    def _real_weights(self, reference):
        """Weights of the real entries: 1 on the diagonal, sqrt(2) off it, in reference's dtype."""
        diagonal = (self.real_i == self.real_j).to(device=reference.device)
        return torch.where(diagonal, reference.new_tensor(1.0), reference.new_tensor(math.sqrt(2.0)))

    def row_weights(self, reference):
        """Return the weights ``w`` relating Acoular's unweighted vector to ``hvec = w * vec``.

        Parameters
        ----------
        reference : Tensor
            Tensor whose dtype and device the weights take.

        Returns
        -------
        Tensor
            Weights of shape ``(m,)``: 1 for diagonal entries, ``sqrt(2)`` otherwise.
        """
        imag = reference.new_full((self.num_measurements - self.num_real,), math.sqrt(2.0))
        return torch.cat((self._real_weights(reference), imag))

    def __call__(self, real, imag):
        """Vectorize Hermitian matrices; Hermitian symmetry is assumed, not checked.

        Parameters
        ----------
        real, imag : Tensor
            Real (symmetric) and imaginary (antisymmetric) parts of shape ``(..., c, c)``.

        Returns
        -------
        Tensor
            ``hvec(S)`` of shape ``(..., m)``.
        """
        return torch.cat(
            (
                real[..., self.real_i, self.real_j] * self._real_weights(real),
                imag[..., self.imag_i, self.imag_j] * math.sqrt(2.0),
            ),
            dim=-1,
        )

    def adjoint(self, vector):
        """Apply the adjoint ``hvec*``, the inverse on Hermitian matrices.

        Off-diagonal entries ``v / sqrt(2)`` are placed at ``(i, j)`` and mirrored to
        ``(j, i)`` (imaginary part with flipped sign); diagonal entries are ``v``.

        Parameters
        ----------
        vector : Tensor
            Vectors of shape ``(..., m)``.

        Returns
        -------
        real, imag : Tensor
            Real and imaginary parts of shape ``(..., c, c)``.
        """
        shape = (*vector.shape[:-1], self.num_mics, self.num_mics)
        real = vector.new_zeros(shape)
        imag = vector.new_zeros(shape)
        # Symmetrization below adds the diagonal twice, hence the extra factor 2 there.
        scale = self._real_weights(vector) * (1 + (self.real_i == self.real_j).to(vector))
        real[..., self.real_i, self.real_j] = vector[..., : self.num_real] / scale
        imag[..., self.imag_i, self.imag_j] = vector[..., self.num_real :] / math.sqrt(2.0)
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
        m = hvec.num_measurements
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
    """Matrix-free CMF operator ``x -> hvec(H diag(x) H^H)`` of a propagation matrix.

    Inputs and outputs of the operator are real. ``mics`` and ``rows`` map a
    subset to the original microphone and measurement coordinates.

    Parameters
    ----------
    H : Tensor
        Complex propagation matrix, shared ``(c, n)`` or per problem ``(B, c, n)``.
    r_diag : bool, optional
        If True, omit the CSM diagonal (default False).

    Attributes
    ----------
    hvec : HermitianVectorization
        Vectorization of the active microphones.
    mics : Tensor or None
        Indices of the active microphones in the full array; None for all.
    """

    def __init__(self, H, r_diag=False):
        if not torch.is_tensor(H) or not H.is_complex() or H.ndim not in (2, 3):
            raise ValueError('H must be a complex tensor with shape (c, n) or (B, c, n).')
        c, n = H.shape[-2:]
        if c <= int(r_diag) or n == 0:
            raise ValueError('H must define nonempty measurement and source dimensions.')
        self.H, self.r_diag = H, r_diag
        self.hvec = HermitianVectorization(c, r_diag=r_diag, device=H.device)
        self.shape = (self.hvec.num_measurements, n)
        self._num_measurements_total, self._num_mics_total = self.shape[0], c
        self.dtype, self.device = H.real.dtype, H.device
        self.batch_size = H.shape[0] if H.ndim == 3 else None
        self.rows = self.mics = None

    def forward(self, x):
        """Apply ``A x = hvec(H diag(x) H^H)`` without forming ``diag(x)`` or the dictionary."""
        csm = (self.H * x.unsqueeze(-2)) @ self.H.mH
        return self.hvec(csm.real, csm.imag)

    def adjoint(self, r):
        """Apply ``A^T r = Re(diag(H^H hvec*(r) H))`` via column-wise inner products."""
        real, imag = self.hvec.adjoint(r)
        csm = torch.complex(real, imag)
        return (self.H.conj() * (csm @ self.H)).real.sum(-2)

    def column_norms(self):
        """Return the column norms ``||hvec(h_j h_j^H)||_2`` without forming the dictionary.

        Since ``hvec`` is an isometry, the norm is ``||h_j h_j^H||_F = ||h_j||_2^2``.
        With ``r_diag``, the diagonal entries ``|h_ij|^2`` are missing:
        ``sqrt(||h_j||_2^4 - sum_i |h_ij|^4)``.

        Returns
        -------
        Tensor
            Norms of shape ``(n,)`` for a shared, ``(B, n)`` for a per-problem ``H``.
        """
        power = self.H.abs().square()
        squared_norms = power.sum(-2).square()
        if self.r_diag:
            squared_norms = squared_norms - power.square().sum(-2)
        return squared_norms.sqrt()

    def scale_columns(self, scale):
        """Return the operator ``A diag(scale)`` by scaling ``H`` with ``sqrt(scale)``.

        Column ``j`` is quadratic in ``h_j``. ``rows`` and ``mics`` are unchanged.

        Parameters
        ----------
        scale : Tensor
            Positive column factors, shared ``(n,)`` or per problem ``(B, n)``.

        Returns
        -------
        CMFPropagation
            Operator of the scaled propagation matrix, per problem if ``scale`` is.
        """
        scale = _validate_scale(scale, self)
        return self._with_propagation(self.H * scale.sqrt().unsqueeze(-2))

    def detached(self):
        """Detach ``H`` while reusing the vectorization and subset metadata."""
        op = copy(self)
        op.H = self.H.detach()
        return op

    def _with_propagation(self, H):
        op = CMFPropagation(H, r_diag=self.r_diag)
        op._num_measurements_total, op._num_mics_total = self._num_measurements_total, self._num_mics_total
        op.rows, op.mics = self.rows, self.mics
        return op

    def select_columns(self, index):
        """Select sources; the measurement rows stay unchanged.

        Parameters
        ----------
        index : Tensor
            Source indices, shared ``(k,)`` or per problem ``(B, k)``.

        Returns
        -------
        CMFPropagation
            Operator of the selected columns of ``H``.
        """
        index = _validate_index(index, 'column index', self.device, self.batch_size)
        return self._with_propagation(_take(self.H, index, self.H.ndim - 1, self.H.ndim == 3))

    def select_mics(self, index):
        """Keep the CSM block of the selected microphones.

        The block keeps ascending order; ``mics`` and ``rows`` refer to the full array.

        Parameters
        ----------
        index : Tensor
            Local microphone indices, shared ``(k,)`` or per problem ``(B, k)``.

        Returns
        -------
        CMFPropagation
            Operator of the subarray.
        """
        index = _validate_index(index, 'microphone index', self.device, self.batch_size).sort(dim=-1).values
        op = self._with_propagation(_take(self.H, index, self.H.ndim - 2, self.H.ndim == 3))
        op.mics = index if self.mics is None else _take(self.mics, index, self.mics.ndim - 1, self.mics.ndim == 2)
        i, j = index[..., op.hvec.real_i], index[..., op.hvec.real_j]
        # Offset of row j in the packed lower triangle of CSM.T, plus column i.
        real_rows = j * (j + (-1 if self.r_diag else 1)) // 2 + i
        i, j = index[..., op.hvec.imag_i], index[..., op.hvec.imag_j]
        imag_rows = self.hvec.num_real + j * (j - 1) // 2 + i
        op.rows = self._subset_rows(torch.cat((real_rows, imag_rows), dim=-1))
        return op

    def select_problems(self, index):
        """Select problems of a per-problem ``H``, see :meth:`MatrixMult.select_problems`.

        Preconditions as there; ``rows`` and ``mics`` are None or shared and stay unchanged.
        """
        return self._with_propagation(self.H[index])

    def mic_rows(self, index):
        """Return ``select_mics(index).rows``, the original rows of a microphone block."""
        return self.select_mics(index).rows


class CMFDictionary(MatrixMult):
    """Explicit CMF dictionary with columns ``hvec(h_j h_j^H)``.

    Microphone and source selection preserve the CMF representation. Arbitrary
    row selection returns a ``MatrixMult`` with the original row indices.

    Parameters
    ----------
    H : Tensor
        Complex propagation matrix, shared ``(c, n)`` or per problem ``(B, c, n)``.
    r_diag : bool, optional
        If True, omit the CSM diagonal (default False).

    Attributes
    ----------
    matrix : Tensor
        The dense dictionary of the active rows.
    hvec : HermitianVectorization
        Vectorization of the active microphones.
    mics : Tensor or None
        Indices of the active microphones in the full array; None for all.
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
        self._num_measurements_total = propagation._num_measurements_total
        self._num_mics_total = propagation._num_mics_total

    def _derived(self, propagation, matrix=None):
        # Skip __init__: the propagation operator already carries the selected rows and microphones.
        op = CMFDictionary.__new__(CMFDictionary)
        op._init_from_propagation(propagation, matrix)
        return op

    def detached(self):
        """Detach both ``H`` and the cached dictionary without rebuilding it."""
        return self._derived(self._propagation.detached(), self.matrix.detach())

    def select_columns(self, index):
        """Select sources from both ``H`` and the stored dictionary.

        Parameters
        ----------
        index : Tensor
            Source indices, shared ``(k,)`` or per problem ``(B, k)``.

        Returns
        -------
        CMFDictionary
            Dictionary of the selected columns.
        """
        propagation = self._propagation.select_columns(index)
        # MatrixMult selects the same columns from the cached dictionary. Reuse its
        # matrix so the new CMFDictionary need not rebuild it from the selected H.
        return self._derived(propagation, super().select_columns(index).matrix)

    def scale_columns(self, scale):
        """Scale ``H`` and the cached dictionary, see :meth:`CMFPropagation.scale_columns`."""
        return self._derived(self._propagation.scale_columns(scale), super().scale_columns(scale).matrix)

    def select_mics(self, index):
        """Build the dictionary of the selected microphone block.

        See :meth:`CMFPropagation.select_mics`.

        Parameters
        ----------
        index : Tensor
            Local microphone indices, shared ``(k,)`` or per problem ``(B, k)``.

        Returns
        -------
        CMFDictionary
            Dictionary of the subarray.
        """
        return self._derived(self._propagation.select_mics(index))

    def select_problems(self, index):
        """Select problems from ``H`` and the cached dictionary without rebuilding it.

        See :meth:`MatrixMult.select_problems`.
        """
        return self._derived(self._propagation.select_problems(index), self.matrix[index])

    def mic_rows(self, index):
        """Return the rows of a microphone block without building its dictionary.

        See :meth:`CMFPropagation.mic_rows`.
        """
        return self._propagation.mic_rows(index)
