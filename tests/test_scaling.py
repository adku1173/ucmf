"""Column normalization of the operators and back-transformation to source strengths."""

from ucmf.scaling import normalize_operator, to_source

from cases.operators import BATCH_SIZE, OperatorCases

import torch
from pytest_cases import parametrize_with_cases


@parametrize_with_cases('operator, matrix', cases=OperatorCases)
def test_normalized_operator_has_unit_column_norms(operator, matrix):
    D, u = normalize_operator(operator)
    assert type(D) is type(operator)
    torch.testing.assert_close(u, torch.linalg.vector_norm(matrix, dim=-2))
    torch.testing.assert_close(D.column_norms(), torch.ones_like(u))


@parametrize_with_cases('operator, matrix', cases=OperatorCases)
def test_to_source_inverts_the_normalization(operator, matrix):
    D, u = normalize_operator(operator)
    q = torch.rand(BATCH_SIZE, matrix.shape[-1], dtype=matrix.dtype, generator=torch.Generator().manual_seed(5))
    x = u * q
    torch.testing.assert_close(D.forward(x), operator.forward(q))
    torch.testing.assert_close(to_source(x, u), q)
