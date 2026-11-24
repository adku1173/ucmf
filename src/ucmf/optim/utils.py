"""Stateless tensor preparation and operator helpers for sparse solvers."""

import math
from numbers import Real

import torch


def validate_tensor(value, *, name, dtype=None, device=None):
    """Require a real floating-point tensor, optionally with an exact dtype and device.

    Validation never converts, copies, or detaches the input. Callers must
    prepare incompatible data explicitly before entering the solver.

    Parameters
    ----------
    value : object
        Value to check.
    name : str
        Name used in error messages.
    dtype : torch.dtype, optional
        Required dtype.
    device : torch.device, optional
        Required device.

    Raises
    ------
    ValueError
        If ``value`` is not a real floating-point tensor or has another dtype or device.
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

    Parameters
    ----------
    tensor : Tensor
        Scalar, one-element, ``(B,)``, or ``(B, 1)`` tensor.
    name : str
        Name used in error messages.

    Returns
    -------
    Tensor
        A scalar for scalar and one-element input, otherwise a ``(B, 1)`` column.

    Raises
    ------
    ValueError
        For any other shape.
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
    global RNG. No gradients are tracked.

    Parameters
    ----------
    operator : LinearOperator
        Shared or batched measurement operator of shape ``(m, n)``.

    Returns
    -------
    Tensor
        Scalar step size for a shared operator, shape ``(B,)`` for a batched one.

    Raises
    ------
    ValueError
        If the estimated squared norm is zero or not finite.
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

    Tensors are neither converted nor detached. Dtype and device compatibility
    is checked at the solver boundary, not by this shape-only helper.

    Parameters
    ----------
    value : float, Tensor, or None
        Python scalar, scalar tensor, or per-sample tensor of shape ``(B,)`` or ``(B, 1)``.
    batch_size : int
        Expected batch size ``B`` of per-sample tensors.
    name : str
        Name used in error messages.

    Returns
    -------
    float, Tensor, or None
        None unchanged, numbers as Python floats, tensors as scalars or ``(B, 1)`` columns.

    Raises
    ------
    ValueError
        For unsupported types, invalid shapes, or a mismatching batch size.
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

    Parameters
    ----------
    t : int
        Requested number of iterations.
    num_iterations : int
        Number of trained layers.

    Raises
    ------
    ValueError
        If ``t`` is negative or exceeds ``num_iterations``.
    """
    if t < 0:
        raise ValueError('t must be non-negative.')
    if t > num_iterations:
        raise ValueError(f't ({t}) cannot exceed num_iterations ({num_iterations}).')


def softplus_inverse(value, eps=1e-12):
    """Return the unconstrained parameter whose softplus is ``max(value, eps)``.

    Uses double-precision Python math so the init does not silently downcast to
    float32; the result matches the requested applied value to rounding.

    Parameters
    ----------
    value : float
        Requested applied (positive) value.
    eps : float, optional
        Lower bound of ``value`` (default 1e-12).

    Returns
    -------
    float
        ``x`` with ``softplus(x) == max(value, eps)`` up to rounding.
    """
    value = max(float(value), eps)
    return value + math.log(-math.expm1(-value))
