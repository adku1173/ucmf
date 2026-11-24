"""Stateless tensor preparation and operator helpers for sparse solvers."""

import math
from numbers import Real

import torch


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


@torch.no_grad()
def default_gamma_from_operator(operator):
    """Return the default step ``1 / L`` without materializing the operator.

    ``L = 1.05 * ||A||_2**2 / m`` is the Lipschitz constant of the normalized
    data term's gradient, with ``||A||_2**2`` estimated by 50 power iterations
    on ``A.T @ A``. The Rayleigh quotient is a lower bound, so the factor 1.05
    is a heuristic margin, not a convergence guarantee. The start vector comes
    from a dedicated seeded CPU generator: deterministic, without consuming the
    global RNG. A shared operator returns a scalar, a batched operator one step
    size per problem. No gradients are tracked; zero or non-finite estimates
    raise ValueError.
    """
    m, n = operator.shape
    batch_size = operator.batch_size
    generator = torch.Generator(device='cpu').manual_seed(0)
    v = torch.randn(
        1 if batch_size is None else batch_size, n, generator=generator, dtype=operator.dtype, device='cpu',
    ).to(operator.device)
    v = v / torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    for _ in range(50):
        # Apply A.T @ A without forming A or its Gram matrix.
        v = operator.adjoint(operator.forward(v))
        v = v / torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    # For unit v, ||A v||**2 is the Rayleigh quotient of A.T @ A.
    L = 1.05 * torch.linalg.vector_norm(operator.forward(v), dim=-1).square() / m
    if torch.any(~torch.isfinite(L) | (L <= 0)):
        raise ValueError('The operator must have a finite, nonzero spectral norm when gamma is omitted.')
    gamma = 1.0 / L
    return gamma.squeeze(0) if batch_size is None else gamma


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


def validate_unfolded_iterations(t, num_iterations):
    """Validate the requested depth for a fixed-depth unfolded solver.

    Unfolded models own a fixed stack of trained layers, so ``t`` must be
    non-negative and may not exceed the available ``num_iterations``.
    """
    if t < 0:
        raise ValueError('t must be non-negative.')
    if t > num_iterations:
        raise ValueError(f't ({t}) cannot exceed num_iterations ({num_iterations}).')


def softplus_inverse(value, eps=1e-12):
    """Return x with ``softplus(x) ~= max(value, eps)`` for parameter init.

    Uses double-precision Python math so the init does not silently downcast to
    float32; the result matches the requested applied value to rounding.
    """
    value = max(float(value), eps)
    return value + math.log(-math.expm1(-value))
