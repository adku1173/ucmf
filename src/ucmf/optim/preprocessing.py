"""Explicit preprocessing helpers for sparse inverse problems."""

from __future__ import annotations

import torch


def normalize_sensing_matrix_columns(A):
    """Return a column-normalized sensing matrix and its column norms.

    Args:
        A: sensing matrix with shape ``(m, n)`` or ``(batch, m, n)``.

    Returns
    -------
        ``(A_norm, column_norms)`` where ``column_norms`` has shape ``(n,)`` for
        a shared matrix and ``(batch, n)`` for a batched matrix.

    The caller is responsible for transforming source coordinates explicitly,
    e.g. ``z = x * column_norms`` before solving and ``x = z / column_norms``
    after solving.
    """
    A_tensor = A if torch.is_tensor(A) else torch.as_tensor(A)
    if A_tensor.ndim not in (2, 3):
        msg = f'A must be 2D or 3D, got shape {tuple(A_tensor.shape)}'
        raise ValueError(msg)
    if A_tensor.is_complex():
        msg = 'A must be real-valued; vectorize complex CSMs before normalizing.'
        raise ValueError(msg)
    if not torch.is_floating_point(A_tensor):
        A_tensor = A_tensor.to(torch.float32)

    column_norms = torch.linalg.norm(A_tensor, dim=-2)
    if torch.any(column_norms == 0):
        msg = 'A contains zero-norm columns; cannot normalize.'
        raise ValueError(msg)
    return A_tensor / column_norms.unsqueeze(-2), column_norms
