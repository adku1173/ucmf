"""Stateless tensor preparation and operator helpers for sparse solvers."""

from numbers import Real

import torch
import torch.nn.functional as F


def validate_tensor(value, *, name, dtype=None, device=None):
    """Require a real floating tensor and optionally an exact dtype/device.

    Validation never converts, copies, or detaches the input. Callers must
    prepare incompatible data explicitly before entering the solver.
    """
    if not torch.is_tensor(value):
        msg = f'{name} must be a real floating-point tensor.'
        raise ValueError(msg)
    if value.is_complex():
        msg = f'{name} must be real-valued; vectorize complex CSMs before solving.'
        raise ValueError(msg)
    if not torch.is_floating_point(value):
        msg = f'{name} must be a real floating-point tensor.'
        raise ValueError(msg)
    if dtype is not None and value.dtype != dtype:
        msg = f'{name} must have dtype {dtype}, got {value.dtype}. Prepare inputs before forward().'
        raise ValueError(msg)
    if device is not None and value.device != device:
        msg = f'{name} must be on device {device}, got {value.device}. Prepare inputs before forward().'
        raise ValueError(msg)


def prepare_sensing_matrix(A, *, dtype=None, device=None):
    """Validate and prepare a fixed real sensing matrix for a solver call.

    Accepts a shared matrix of shape ``(m, n)`` or batched matrices of shape
    ``(B, m, n)``. Requires a real floating-point tensor, optionally with the
    exact ``dtype``/``device``. The result is detached and contiguous, with its
    original dtype and device unchanged. No column normalization is applied.
    Raises ValueError for missing, complex, empty, mismatched, or incorrectly
    ranked input.
    """
    if A is None:
        msg = 'A must be provided in forward().'
        raise ValueError(msg)

    validate_tensor(A, name='A', dtype=dtype, device=device)
    A_tensor = A.detach()

    if A_tensor.ndim not in (2, 3):
        msg = f'A must be 2D or 3D, got shape {tuple(A_tensor.shape)}'
        raise ValueError(msg)

    if 0 in A_tensor.shape[-2:]:
        msg = 'A must have non-empty measurement and source dimensions.'
        raise ValueError(msg)
    return A_tensor.contiguous()


def to_batch_vector(tensor, name):
    """Reshape a scalar or per-sample tensor for broadcasting over sources.

    Scalars remain unchanged, a one-element vector becomes a scalar, and a
    vector of shape ``(B,)`` becomes ``(B, 1)``. Column vectors are returned
    unchanged. Other shapes raise ValueError, using ``name`` in the message.
    """
    if tensor.ndim == 0:
        return tensor
    if tensor.ndim == 1:
        if tensor.numel() == 1:
            return tensor.squeeze(0)
        return tensor.unsqueeze(1)
    if tensor.ndim == 2 and tensor.shape[1] == 1:
        return tensor
    msg = f'{name} must be scalar or have shape (B,) / (B, 1), got {tuple(tensor.shape)}'
    raise ValueError(msg)


def apply_forward_operator(A, x):
    """Map batched source vectors ``x`` of shape ``(B, n)`` to measurements.

    Applies either a shared matrix ``A`` of shape ``(m, n)`` or one matrix per
    sample of shape ``(B, m, n)``, returning a tensor of shape ``(B, m)``.
    """
    if A.ndim == 2:
        return F.linear(x, A)
    return torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)


def apply_adjoint_operator(v, Op):
    """Map real measurement vectors ``v`` back to source coordinates.

    Applies the transpose of a shared operator ``Op`` of shape ``(m, n)`` or
    batched operators of shape ``(B, m, n)`` to ``v`` of shape ``(B, m)``.
    The result has shape ``(B, n)``; no measurement-count scaling is applied.
    """
    if Op.ndim == 2:
        return v @ Op
    return torch.bmm(v.unsqueeze(1), Op).squeeze(1)


def default_gamma_from_A(A):
    """Return the step size ``m / ||A||_2**2`` for normalized least squares.

    Uses the squared spectral norm of a shared or batched sensing matrix,
    returning a scalar or a tensor of shape ``(B,)`` respectively. It is the
    largest eigenvalue of the smaller Gram matrix (``A.T @ A`` or ``A @ A.T``);
    a symmetric eigensolver is much faster than an SVD, especially on GPUs.
    The computation does not track gradients. Zero or non-finite squared
    spectral norms raise ValueError.
    """
    with torch.no_grad():
        gram = A.mT @ A if A.shape[-1] <= A.shape[-2] else A @ A.mT
        L = torch.linalg.eigvalsh(gram)[..., -1] / float(A.shape[-2])
        if torch.any(~torch.isfinite(L) | (L <= 0)):
            msg = 'A must have a finite, nonzero spectral norm when gamma is omitted.'
            raise ValueError(msg)
    return 1.0 / L


def resolve_batch_param(value, batch_size, name):
    """Reshape an already-prepared scalar or per-sample parameter.

    Returns None unchanged, numeric scalars as Python floats, and tensors as
    scalars or ``(B, 1)`` columns without dtype/device conversion or detachment.
    Non-scalar tensors must match ``batch_size``. Unsupported types and invalid
    shapes raise ValueError mentioning ``name``. Device/dtype compatibility is
    checked at the solver boundary, not by this shape-only helper.
    """
    if value is None:
        return None

    if torch.is_tensor(value):
        tensor = value
        if tensor.ndim > 2:
            msg = f'{name} must be scalar, 1D, or 2D with shape (B, 1), got {tuple(tensor.shape)}'
            raise ValueError(msg)
        tensor = to_batch_vector(tensor, name=name)
        if tensor.ndim == 0:
            return tensor
        if tensor.shape[0] != batch_size:
            msg = f'{name} has shape {tuple(tensor.shape)} but expected batch size {batch_size}.'
            raise ValueError(msg)
        return tensor

    if isinstance(value, Real):
        return float(value)
    msg = f'{name} must be a real scalar or floating-point tensor.'
    raise ValueError(msg)


def validate_shapes(y, x0, A):
    """Check that prepared measurements, warm start, and sensing matrix agree.

    Expects ``y`` of shape ``(B, m)`` and a shared ``(m, n)`` or batched
    ``(B, m, n)`` matrix ``A``. If given, ``x0`` must have shape ``(B, n)``.
    Raises ValueError on incompatible dimensions; returns None on success.
    """
    if A.ndim == 2:
        m, n = A.shape
        if y.shape[1] != m:
            msg = f'y has shape {tuple(y.shape)} but A has shape {tuple(A.shape)}'
            raise ValueError(msg)
        batch_size = y.shape[0]
    else:
        batch_size, m, n = A.shape
        if y.shape[0] != batch_size or y.shape[1] != m:
            msg = f'y has shape {tuple(y.shape)} but A has shape {tuple(A.shape)}'
            raise ValueError(msg)

    if x0 is not None and tuple(x0.shape) != (batch_size, n):
        msg = f'x0 has shape {tuple(x0.shape)} but expected {(batch_size, n)}'
        raise ValueError(msg)
