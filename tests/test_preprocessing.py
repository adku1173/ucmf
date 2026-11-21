"""Explicit column normalization and source-coordinate transformations."""

import numpy as np
import pytest
import torch

from ucmf.optim import normalize_sensing_matrix_columns


@pytest.mark.parametrize('batched', [True, False])
def test_column_normalization_preserves_forward_model_with_explicit_coordinates(batched):
    A = torch.tensor([[3.0, 0.0], [4.0, 2.0]], dtype=torch.float64)
    if batched:
        A = torch.stack([A, 2 * A])
    original = A.clone()
    x = torch.tensor([0.2, 0.3], dtype=A.dtype)

    normalized, norms = normalize_sensing_matrix_columns(A)
    z = x * norms

    torch.testing.assert_close(torch.linalg.vector_norm(normalized, dim=-2), torch.ones_like(norms))
    torch.testing.assert_close(normalized @ z.unsqueeze(-1), A @ x.reshape(2, 1))
    torch.testing.assert_close(z / norms, x.expand_as(norms))
    torch.testing.assert_close(A, original)
    assert norms.shape == ((2, 2) if batched else (2,))
    assert normalized.dtype == A.dtype


@pytest.mark.parametrize('A', [[[3, 0], [4, 2]], np.array([[3, 0], [4, 2]]), torch.tensor([[3, 0], [4, 2]])])
def test_array_like_and_integer_inputs(A):
    normalized, norms = normalize_sensing_matrix_columns(A)

    torch.testing.assert_close(norms, torch.tensor([5.0, 2.0]))
    torch.testing.assert_close(normalized, torch.tensor([[0.6, 0.0], [0.8, 1.0]]))


@pytest.mark.parametrize(
    ('A', 'message'),
    [
        (torch.ones(3), 'A must be 2D or 3D'),
        (torch.zeros(2, 2), 'zero-norm columns'),
        (torch.tensor([[1.0, 0.0], [0.0, 0.0]]), 'zero-norm columns'),
        (torch.ones(2, 2, dtype=torch.complex64), 'A must be real-valued'),
    ],
)
def test_invalid_normalization_inputs(A, message):
    with pytest.raises(ValueError, match=message):
        normalize_sensing_matrix_columns(A)
