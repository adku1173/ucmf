"""Per-solve gradient weight matrix helpers for unfolded sparse solvers."""

from __future__ import annotations

from .utils import validate_tensor

import torch


def validate_gradient_weight_shape(W: torch.Tensor, A_tensor: torch.Tensor, y: torch.Tensor):
    """Validate a per-solve effective gradient weight matrix ``W``."""
    if W.ndim not in (2, 3):
        msg = f'W must be 2D or 3D, got shape {tuple(W.shape)}'
        raise ValueError(msg)
    if tuple(W.shape[-2:]) != tuple(A_tensor.shape[-2:]):
        msg = f'W has operator shape {tuple(W.shape[-2:])} but A has operator shape {tuple(A_tensor.shape[-2:])}.'
        raise ValueError(msg)
    if W.ndim == 3 and W.shape[0] != y.shape[0]:
        msg = f'Batched W has batch size {W.shape[0]} but y has batch size {y.shape[0]}.'
        raise ValueError(msg)


def prepare_gradient_weight(W, A_tensor: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return the effective adjoint-side operator for one solve.

    The normalized data term ``1 / (2 m) * ||y - A x||_2^2`` has gradient
    ``-A.T @ (y - A x) / m``, so the descent step applies the adjoint ``A.T``.
    A model may replace that adjoint by a learned operator ``W.T``, giving the
    weighted direction ``W.T @ (y - A x) / m`` while the forward operator stays
    ``A``. This helper returns that effective adjoint-side operator ``W``.

    ``A_tensor`` (i.e. ``W = A``, the classical adjoint) is returned when ``W``
    is omitted. Otherwise ``W`` must already be a real floating tensor with the
    same device/dtype and operator shape as ``A_tensor``. It is returned
    unchanged so its autograd graph is preserved.
    """
    if W is None:
        return A_tensor
    validate_tensor(W, name='W', dtype=A_tensor.dtype, device=A_tensor.device)
    validate_gradient_weight_shape(W, A_tensor, y)
    return W


def apply_auxiliary_weight(B, A_tensor):
    """Form the effective weight from the auxiliary weight ``B``: ``W = B @ A``.

    Handles shared or per-sample ``B`` and shared or batched ``A``.
    """
    if B.ndim == 3:
        # Per-sample B: W_b = B_b @ A_b, (b, p, m) @ (b, m, n) -> (b, p, n).
        return torch.bmm(B, A_tensor)
    if A_tensor.ndim == 2:
        # Shared B and A: (p, m) @ (m, n) -> (p, n).
        return B @ A_tensor
    # Shared B, batched A: broadcast B over the batch, (p, m) @ (b, m, n) -> (b, p, n).
    return torch.bmm(B.unsqueeze(0).expand(A_tensor.shape[0], -1, -1), A_tensor)


def validate_measurement_subset(msm_subset, A_tensor, n_measurements, device):
    """Validate and return long measurement indices ``I`` into the full auxiliary weight.

    ``I`` names the original rows of the already-subsetted ``A``; it must be an
    integer tensor on ``device`` (the auxiliary weight's device).
    """
    if not torch.is_tensor(msm_subset) or msm_subset.dtype not in (torch.int32, torch.int64):
        raise ValueError('msm_subset must be an integer tensor of measurement indices.')
    if msm_subset.device != device:
        raise ValueError(f'msm_subset must be on device {device}, got {msm_subset.device}.')
    idx = msm_subset
    if idx.ndim not in (1, 2):
        raise ValueError(f'msm_subset must be 1D or 2D, got shape {tuple(idx.shape)}.')
    if idx.shape[-1] != A_tensor.shape[-2]:
        raise ValueError(f'msm_subset length {idx.shape[-1]} must match A measurement dim {A_tensor.shape[-2]}.')
    if idx.ndim == 2:
        if A_tensor.ndim != 3:
            raise ValueError('per-sample msm_subset (2D) requires batched A of shape (B, m_sub, n).')
        if idx.shape[0] != A_tensor.shape[0]:
            raise ValueError(f'msm_subset batch {idx.shape[0]} does not match A batch {A_tensor.shape[0]}.')
    if idx.numel() and (torch.any(idx < 0) or torch.any(idx >= n_measurements)):
        raise ValueError(f'msm_subset entries must lie in [0, {n_measurements}).')
    return idx.long()


def measurement_subset_weight(B, A_tensor, msm_subset, n_measurements):
    """Effective weight on a reduced measurement set ``I``: ``W_sub = B[I, I] @ A_sub``.

    ``A_tensor`` holds the active rows ``A_sub``; ``msm_subset`` gives the original
    indices ``I`` so the matching sub-block of ``B`` is selected.
    """
    idx = validate_measurement_subset(msm_subset, A_tensor, n_measurements, B.device)
    if idx.ndim == 1:
        # Shared index set: B_sub = B[I][:, I], (m_sub, m_sub).
        B_sub = B.index_select(0, idx).index_select(1, idx)
    else:
        # Per-sample index sets: B_sub[b] = B[I_b][:, I_b], (b, m_sub, m_sub),
        # via broadcast advanced indexing B[I_b[:, None], I_b[None, :]].
        B_sub = B[idx[:, :, None], idx[:, None, :]]
    return apply_auxiliary_weight(B_sub, A_tensor)
