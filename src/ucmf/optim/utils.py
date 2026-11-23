"""Stateless tensor preparation and operator helpers for sparse solvers."""

import math
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


def apply_adjoint_operator(v, matrix):
    """Map real measurement vectors ``v`` back to source coordinates.

    Applies the transpose of a shared ``(m, n)`` or batched ``(B, m, n)``
    matrix to ``v`` of shape ``(B, m)``. The result has shape ``(B, n)``;
    no measurement-count scaling is applied.
    """
    if matrix.ndim == 2:
        return v @ matrix
    return torch.bmm(v.unsqueeze(1), matrix).squeeze(1)


DEFAULT_GAMMA_SAFETY = 1.05


def default_gamma_from_A(A):
    """Return the step size ``1 / L`` for normalized least squares.

    ``L = 1.05 * ||A||_2**2 / m`` is the Lipschitz constant of the normalized
    data term's gradient, with ``||A||_2**2`` estimated by 50 power iterations
    (:func:`estimate_squared_spectral_norm`). The estimate is a lower bound;
    the safety factor 1.05 keeps ``gamma <= m / ||A||_2**2`` in typical cases
    (power iteration converges slowly when the top singular values are close).
    Shared or batched ``A`` give a scalar or ``(B,)`` tensor. No gradients are
    tracked. Zero or non-finite estimates raise ValueError.
    """
    L = DEFAULT_GAMMA_SAFETY * estimate_squared_spectral_norm(A) / float(A.shape[-2])
    if torch.any(~torch.isfinite(L) | (L <= 0)):
        msg = 'A must have a finite, nonzero spectral norm when gamma is omitted.'
        raise ValueError(msg)
    return 1.0 / L


@torch.no_grad()
def default_gamma_from_operator(operator):
    """Estimate ``m / (1.05 * ||A||_2**2)`` without materializing the operator.

    A shared operator returns a scalar; a batched operator returns one step
    size per problem. Power iteration gives a lower bound on the squared norm,
    so the factor 1.05 is a heuristic margin, not a convergence guarantee.
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
    L = DEFAULT_GAMMA_SAFETY * torch.linalg.vector_norm(operator.forward(v), dim=-1).square() / m
    if torch.any(~torch.isfinite(L) | (L <= 0)):
        raise ValueError('The operator must have a finite, nonzero spectral norm when gamma is omitted.')
    gamma = 1.0 / L
    return gamma.squeeze(0) if batch_size is None else gamma


@torch.no_grad()
def estimate_squared_spectral_norm(A, n_iter=50, seed=0):
    """Estimate ``||A||_2**2`` by power iteration on ``A.T @ A``.

    Works for a shared ``(m, n)`` or batched ``(B, m, n)`` matrix and returns a
    scalar or a ``(B,)`` tensor. The result is the Rayleigh quotient
    ``||A v||**2`` of the final unit vector ``v``, i.e. a lower bound of the
    exact value; callers needing a safe Lipschitz bound apply a safety factor.

    The start vector is drawn from a dedicated CPU generator seeded with
    ``seed``: results are deterministic and the global RNG is not consumed.
    A fixed ``n_iter`` (no tolerance check) avoids host-device syncs.
    """
    generator = torch.Generator().manual_seed(seed)
    v = torch.randn(*A.shape[:-2], A.shape[-1], 1, generator=generator, dtype=A.dtype).to(A.device)
    v = v / torch.linalg.vector_norm(v, dim=-2, keepdim=True)
    for _ in range(n_iter):
        # v <- A.T A v / ||A.T A v||, (..., n, 1)
        v = A.mT @ (A @ v)
        v = v / torch.linalg.vector_norm(v, dim=-2, keepdim=True)
    # Rayleigh quotient v.T A.T A v = ||A v||**2 for unit v.
    return torch.linalg.vector_norm(A @ v, dim=(-2, -1)) ** 2


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
