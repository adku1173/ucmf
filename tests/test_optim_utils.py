"""Standalone solver utilities preserve batching and autograd behavior."""

from ucmf.optim.operator import MatrixMult
from ucmf.optim.utils import (
    default_gamma_from_operator,
    resolve_batch_param,
    to_batch_vector,
    validate_tensor,
)

import numpy as np
import pytest
import torch


@pytest.mark.parametrize('shape, expected', [((), ()), ((1,), ()), ((3,), (3, 1)), ((3, 1), (3, 1))])
def test_to_batch_vector_shapes(shape, expected):
    assert to_batch_vector(torch.ones(shape), 'lam').shape == expected


@pytest.mark.parametrize('shape', [(2, 2), (2, 1, 1)])
def test_to_batch_vector_rejects_invalid_shapes(shape):
    with pytest.raises(ValueError, match='lam must be scalar'):
        to_batch_vector(torch.ones(shape), 'lam')


def test_batch_parameters_are_reshaped_without_conversion():
    value = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    result = resolve_batch_param(value, 3, 'lam')
    assert result.dtype == value.dtype
    assert result.device == value.device
    assert result.data_ptr() == value.data_ptr()
    torch.testing.assert_close(result, value[:, None])


@pytest.mark.parametrize('factory', [np.array, list, tuple])
def test_batch_parameters_require_preprocessed_tensors(factory):
    with pytest.raises(ValueError, match='lam must be a real scalar or floating-point tensor'):
        resolve_batch_param(factory([0.1, 0.2, 0.3]), 3, 'lam')


def test_batch_parameter_scalar_missing_invalid_and_autograd_cases():
    assert resolve_batch_param(None, 3, 'lam') is None
    assert resolve_batch_param(0.1, 3, 'lam') == 0.1
    assert resolve_batch_param(torch.tensor([0.1]), 3, 'lam').shape == ()
    for value, message in [(torch.tensor([0.1, 0.2]), 'expected batch size'), (torch.ones(3, 1, 1), 'must be scalar')]:
        with pytest.raises(ValueError, match=message):
            resolve_batch_param(value, 3, 'lam')
    value = torch.ones(3, requires_grad=True)
    resolve_batch_param(value, 3, 'lam').sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value))


@pytest.mark.parametrize('batched', [False, True])
def test_default_step_size_uses_spectral_norm_without_autograd(batched):
    # ||A||_2**2 = 4 (and 16), m = 2, safety 1.05: gamma = m / (1.05 * ||A||_2**2).
    A = torch.diag(torch.tensor([2.0, 1.0]))
    if batched:
        A = torch.stack([A, 2 * A])
    A.requires_grad_()
    expected = torch.tensor([2 / 4.2, 2 / 16.8]) if batched else torch.tensor(2 / 4.2)
    result = default_gamma_from_operator(MatrixMult(A))
    torch.testing.assert_close(result, expected)
    assert not result.requires_grad


def squared_norm_estimate(A):
    """Power-iteration estimate of ``||A||_2**2`` implied by the default step."""
    return A.shape[-2] / (1.05 * default_gamma_from_operator(MatrixMult(A)))


@pytest.mark.parametrize('batched', [False, True])
def test_power_iteration_estimates_squared_spectral_norm_from_below(batched):
    # Accuracy depends on the gap sigma_2 / sigma_1 (here 0.8 by construction);
    # generic random matrices may converge much more slowly.
    rng = torch.Generator().manual_seed(1)
    batch = (3,) if batched else ()
    U = torch.linalg.qr(torch.randn(*batch, 40, 25, dtype=torch.float64, generator=rng)).Q
    V = torch.linalg.qr(torch.randn(*batch, 25, 25, dtype=torch.float64, generator=rng)).Q
    sigma = torch.linspace(0.8, 0.1, 25, dtype=torch.float64)
    sigma[0] = 1.0
    A = (U * sigma) @ V.mT
    A.requires_grad_()
    estimate = squared_norm_estimate(A)
    assert estimate.shape == batch
    assert not estimate.requires_grad
    torch.testing.assert_close(estimate, torch.ones(batch, dtype=torch.float64), rtol=1e-12, atol=0)


def test_power_iteration_is_a_lower_bound_for_random_matrices():
    A = torch.randn(20, 40, 25, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    exact = torch.linalg.matrix_norm(A, ord=2) ** 2
    assert torch.all(squared_norm_estimate(A) <= exact * (1 + 1e-12))  # Rayleigh quotient


def test_power_iteration_is_exact_for_diagonal_matrix():
    A = torch.diag(torch.tensor([3.0, 1.0, 0.5], dtype=torch.float64))
    torch.testing.assert_close(squared_norm_estimate(A), torch.tensor(9.0, dtype=torch.float64))


def test_power_iteration_is_deterministic_and_leaves_global_rng_untouched():
    A = torch.randn(30, 20, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    state = torch.get_rng_state()
    first = default_gamma_from_operator(MatrixMult(A))
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(default_gamma_from_operator(MatrixMult(A)), first)


def test_tensor_validation_does_not_modify_input_or_autograd():
    value = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
    version = value._version
    assert validate_tensor(value, name='x0', dtype=value.dtype, device=value.device) is None
    assert value._version == version
    assert value.is_leaf and value.requires_grad
