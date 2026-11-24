"""Problem cases ``(operator, y, x0)`` for the properties shared by all models.

The operator source (``H`` or the matrix) requires gradients, so tests can check
that models detach it.
Tags: ``cmf``, ``dictionary`` (``CMFDictionary``), ``rows`` (``MatrixMult`` row subsets).
"""

from ucmf.optim.operator import CMFDictionary, CMFPropagation, MatrixMult

import torch
from pytest_cases import case, parametrize

NUM_MICS = 4
NUM_SOURCES = 3
BATCH = 2
def cmf_problem(dtype=torch.float64):
    """Small problem every model accepts; ``H`` requires gradients."""
    rng = torch.Generator().manual_seed(3)
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    H = torch.randn(NUM_MICS, NUM_SOURCES, dtype=complex_dtype, generator=rng).requires_grad_()
    operator = CMFPropagation(H)
    y = torch.randn(BATCH, operator.shape[0], dtype=dtype, generator=rng)
    x0 = torch.randn(BATCH, NUM_SOURCES, dtype=dtype, generator=rng)
    return operator, y, x0


PER_PROBLEM_MICS = [[0, 2, 3], [0, 1, 2]]
CMF_LAYOUTS = parametrize(
    'batched_H, mics',
    [(False, None), (False, [0, 2, 3]), (False, PER_PROBLEM_MICS), (True, None), (True, PER_PROBLEM_MICS)],
    ids=['shared-H', 'shared-H-shared-mics', 'shared-H-per-problem-mics', 'batched-H', 'batched-H-per-problem-mics'],
)
R_DIAG = parametrize('r_diag', [False, True], ids=['with-diagonal', 'without-diagonal'])


def _with_data(operator, rng):
    m, n = operator.shape
    y = torch.randn(BATCH, m, dtype=operator.dtype, generator=rng)
    x0 = torch.randn(BATCH, n, dtype=operator.dtype, generator=rng)
    return operator, y, x0


def _cmf_problem(operator_type, batched_H, mics, r_diag):
    rng = torch.Generator().manual_seed(11)
    shape = (BATCH, NUM_MICS, NUM_SOURCES) if batched_H else (NUM_MICS, NUM_SOURCES)
    H = torch.randn(shape, dtype=torch.complex128, generator=rng).requires_grad_()
    operator = operator_type(H, r_diag=r_diag)
    if mics is not None:
        operator = operator.select_mics(torch.tensor(mics))
    return _with_data(operator, rng)


class ProblemCases:
    @case(tags=('cmf', 'dictionary'))
    @CMF_LAYOUTS
    @R_DIAG
    def case_dictionary(self, batched_H, mics, r_diag):
        return _cmf_problem(CMFDictionary, batched_H, mics, r_diag)

    @case(tags='cmf')
    @CMF_LAYOUTS
    @R_DIAG
    def case_propagation(self, batched_H, mics, r_diag):
        return _cmf_problem(CMFPropagation, batched_H, mics, r_diag)

    @case(tags='rows')
    @parametrize(
        'rows',
        [None, [0, 2, 3, 5, 8, 13], [[0, 2, 3, 5, 8, 13], [1, 2, 4, 5, 9, 15]]],
        ids=['full', 'shared-rows', 'per-problem-rows'],
    )
    def case_matrix(self, rows):
        rng = torch.Generator().manual_seed(12)
        A = torch.randn(NUM_MICS * NUM_MICS, NUM_SOURCES, dtype=torch.float64, generator=rng).requires_grad_()
        operator = MatrixMult(A)
        if rows is not None:
            operator = operator.select_rows(torch.tensor(rows))
        return _with_data(operator, rng)
