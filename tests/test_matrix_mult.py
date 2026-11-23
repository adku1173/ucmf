"""MatrixMult specifics; selection is tested for all operators in test_operator."""

from ucmf.optim.operator import MatrixMult

import pytest
import torch


@pytest.mark.parametrize('shape', [(6, 5), (3, 6, 5)], ids=['shared', 'batched'])
def test_keeps_input_matrix(shape):
    A = torch.randn(shape, dtype=torch.float64)
    assert MatrixMult(A).matrix is A


@pytest.mark.parametrize(
    ('matrix', 'message'),
    [
        (torch.ones(2), '2D or 3D'),
        (torch.ones(2, 2, dtype=torch.complex64), 'real-valued'),
        (torch.ones(2, 2, dtype=torch.long), 'floating'),
    ],
)
def test_matrix_validation(matrix, message):
    with pytest.raises(ValueError, match=message):
        MatrixMult(matrix)
