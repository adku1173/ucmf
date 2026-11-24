"""Operator cases ``(operator, matrix)``: each operator with its explicit reference matrix."""

from ucmf.optim.operator import CMFDictionary, CMFPropagation, HermitianVectorization, MatrixMult

import torch
from pytest_cases import case, parametrize

BATCH_SIZE = 3


def cmf_matrix(H, r_diag):
    """Dense CMF dictionary of ``H``: column ``j`` is ``hvec(h_j h_j^H)``."""
    columns = H.mT.unsqueeze(-1) * H.mH.unsqueeze(-2)  # One outer product per source.
    hvec = HermitianVectorization(H.shape[-2], r_diag=r_diag)
    return hvec(columns.real, columns.imag).mT


class OperatorCases:
    """Construct each operator alongside its explicit reference matrix.

    Tags: ``cmf`` (microphone selection and transformation), ``rows`` (row selection).
    """

    @case(tags='rows')
    @parametrize('batched', [False, True], ids=['shared', 'batched'])
    def case_matrix_mult(self, batched):
        rng = torch.Generator().manual_seed(0)
        shape = (BATCH_SIZE, 6, 5) if batched else (6, 5)
        matrix = torch.randn(shape, dtype=torch.float64, generator=rng)
        return MatrixMult(matrix), matrix

    @case(tags='cmf')
    @parametrize('batched', [False, True], ids=['shared', 'batched'])
    @parametrize('r_diag', [False, True], ids=['full', 'off-diagonal'])
    def case_cmf_propagation(self, batched, r_diag):
        return self._cmf_case(CMFPropagation, batched, r_diag)

    @case(tags=('cmf', 'rows'))
    @parametrize('batched', [False, True], ids=['shared', 'batched'])
    @parametrize('r_diag', [False, True], ids=['full', 'off-diagonal'])
    def case_cmf_dictionary(self, batched, r_diag):
        return self._cmf_case(CMFDictionary, batched, r_diag)

    @staticmethod
    def _cmf_case(operator_type, batched, r_diag):
        rng = torch.Generator().manual_seed(17)
        shape = (BATCH_SIZE, 5, 4) if batched else (5, 4)
        H = torch.randn(shape, dtype=torch.complex128, generator=rng)
        return operator_type(H, r_diag=r_diag), cmf_matrix(H, r_diag)
