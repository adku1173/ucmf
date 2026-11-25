"""Column normalization of the operators and back-transformation to source strengths."""

from ucmf import FISTA
from ucmf.optim.operator import CMFPropagation, MatrixMult
from ucmf.scaling import lambda_max, normalize_operator, to_source

from cases.operators import BATCH_SIZE, OperatorCases

import pytest
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


@parametrize_with_cases('operator, matrix', cases=OperatorCases)
def test_lambda_max_matches_dense_reference(operator, matrix):
    D, u = normalize_operator(operator)
    y = torch.randn(BATCH_SIZE, matrix.shape[-2], dtype=matrix.dtype, generator=torch.Generator().manual_seed(7))
    correlations = (y.unsqueeze(-2) @ (matrix / u.unsqueeze(-2))).squeeze(-2)
    torch.testing.assert_close(lambda_max(D, y), correlations.amax(-1) / matrix.shape[-2])


@pytest.mark.parametrize(('lam', 'is_zero'), [(1.0, True), (0.9, False)])
def test_equiregularized_lambda_one_is_the_zero_threshold(lam, is_zero):
    rng = torch.Generator().manual_seed(8)
    H = torch.randn(5, 12, dtype=torch.complex128, generator=rng)
    D, u = normalize_operator(CMFPropagation(H))
    y = D.forward(torch.rand(BATCH_SIZE, 12, dtype=torch.float64, generator=rng))
    x = FISTA()(y / lambda_max(D, y).unsqueeze(-1), operator=D, lam=lam, t=50)
    assert bool((x == 0).all(dim=-1).all()) is is_zero


def test_cv_scaling_after_preprocessing_matches_direct_scaling():
    """Rescaling preprocessed data on CV subsets gives ``A_s U_t^-1`` and ``y_s / lambda_t``."""
    rng = torch.Generator().manual_seed(9)
    A = MatrixMult(torch.randn(10, 6, dtype=torch.float64, generator=rng))
    y = torch.randn(BATCH_SIZE, 10, dtype=torch.float64, generator=rng) + 3.0
    train, val = torch.tensor([0, 2, 3, 5, 7, 8, 9]), torch.tensor([1, 4, 6])

    D_t, u_t = normalize_operator(A.select_rows(train))  # paper: directly from the raw data
    lam_t = lambda_max(D_t, y[:, train]).unsqueeze(-1)

    D, u = normalize_operator(A)  # preprocessing with the full operator
    y_pre = y / lambda_max(D, y).unsqueeze(-1)
    D_t2, u_t2 = normalize_operator(D.select_rows(train))
    lam_t2 = lambda_max(D_t2, y_pre[:, train]).unsqueeze(-1)

    torch.testing.assert_close(u * u_t2, u_t)
    torch.testing.assert_close(D_t2.matrix, D_t.matrix)
    torch.testing.assert_close(D.select_rows(val).scale_columns(1 / u_t2).matrix, A.select_rows(val).matrix / u_t)
    torch.testing.assert_close(y_pre[:, train] / lam_t2, y[:, train] / lam_t)
    torch.testing.assert_close(y_pre[:, val] / lam_t2, y[:, val] / lam_t)
