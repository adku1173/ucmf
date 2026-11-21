"""Per-solve gradient weight matrix helpers for unfolded sparse solvers."""

from __future__ import annotations

import torch

from .utils import validate_tensor


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

    ``A_tensor`` is returned when ``W`` is omitted. Otherwise ``W`` must already
    be a real floating tensor with the same device/dtype and operator shape as
    ``A_tensor``. It is returned unchanged so its autograd graph is preserved.
    """
    if W is None:
        return A_tensor
    validate_tensor(W, name='W', dtype=A_tensor.dtype, device=A_tensor.device)
    validate_gradient_weight_shape(W, A_tensor, y)
    return W
