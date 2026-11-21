"""Standalone solver utilities preserve batching and autograd behavior."""

import numpy as np
import pytest
import torch

from ucmf.optim.utils import (
    apply_adjoint_operator,
    apply_forward_operator,
    default_gamma_from_A,
    prepare_sensing_matrix,
    resolve_batch_param,
    to_batch_vector,
    validate_shapes,
    validate_tensor,
)


@pytest.mark.parametrize('batched', [False, True])
def test_operators_match_per_sample_matrix_products(batched):
    torch.manual_seed(42)
    A = torch.randn((3, 4, 2) if batched else (4, 2), dtype=torch.float64, requires_grad=True)
    x = torch.randn(3, 2, dtype=A.dtype, requires_grad=True)
    v = torch.randn(3, 4, dtype=A.dtype, requires_grad=True)
    matrices = A if batched else A.expand(3, -1, -1)
    forward = apply_forward_operator(A, x)
    adjoint = apply_adjoint_operator(v, A)
    torch.testing.assert_close(forward, torch.stack([a @ row for a, row in zip(matrices, x, strict=True)]))
    torch.testing.assert_close(adjoint, torch.stack([a.T @ row for a, row in zip(matrices, v, strict=True)]))
    torch.testing.assert_close((forward * v).sum(), (x * adjoint).sum())
    (forward.sum() + adjoint.sum()).backward()
    assert all(tensor.grad is not None for tensor in (A, x, v))


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


def test_sensing_matrix_is_detached_contiguous_without_conversion():
    A = torch.ones(2, 4, dtype=torch.float64, requires_grad=True).T
    prepared, norms = prepare_sensing_matrix(A)
    assert prepared.is_contiguous()
    assert not prepared.requires_grad
    assert prepared.dtype == A.dtype
    assert prepared.device == A.device
    assert norms is None
    torch.testing.assert_close(prepared, A)
    with pytest.raises(ValueError, match='A must be a real floating-point tensor'):
        prepare_sensing_matrix([[1, 2]])


@pytest.mark.parametrize('batched', [False, True])
def test_default_step_size_uses_spectral_norm_without_autograd(batched):
    A = torch.diag(torch.tensor([2.0, 1.0]))
    if batched:
        A = torch.stack([A, 2 * A])
    A.requires_grad_()
    expected = torch.tensor([0.5, 0.125]) if batched else torch.tensor(0.5)
    result = default_gamma_from_A(A)
    torch.testing.assert_close(result, expected)
    assert not result.requires_grad


def test_tensor_validation_does_not_modify_input_or_autograd():
    value = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
    version = value._version
    assert validate_tensor(value, name='x0', dtype=value.dtype, device=value.device) is None
    assert value._version == version
    assert value.is_leaf and value.requires_grad


@pytest.mark.parametrize('batched', [False, True])
def test_validate_shapes_accepts_matching_inputs_and_rejects_mismatches(batched):
    A = torch.ones((2, 4, 3) if batched else (4, 3))
    y = torch.ones(2, 4)
    assert validate_shapes(y, None, A) is None
    assert validate_shapes(y, torch.zeros(2, 3), A) is None
    with pytest.raises(ValueError, match='y has shape'):
        validate_shapes(torch.ones(2, 5), None, A)
    with pytest.raises(ValueError, match='x0 has shape'):
        validate_shapes(y, torch.zeros(2, 4), A)
