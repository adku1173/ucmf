"""Tests for the CV splits of ucmf.train.objectives; skipped without the ``train`` extra."""

import pytest

pytest.importorskip('ray')

from ucmf.optim.operator import CMFDictionary, MatrixMult  # noqa: E402
from ucmf.train.objectives import split_complex_rows, split_rows  # noqa: E402

import torch  # noqa: E402

B, C = 3, 5


def cmf_dictionary(r_diag):
    H = torch.randn(C, 4, dtype=torch.complex128, generator=torch.Generator().manual_seed(0))
    return CMFDictionary(H, r_diag=r_diag)


def check_partition(operator, op_t, train, val, num_train):
    """Disjoint, complete per problem; ``op_t`` holds the train rows of ``operator``."""
    m = operator.shape[0]
    assert train.shape == (B, num_train) and val.shape == (B, m - num_train)
    for t, v in zip(train, val, strict=True):
        assert sorted(torch.cat((t, v)).tolist()) == list(range(m))
    torch.testing.assert_close(op_t.matrix, operator.matrix[train])


@pytest.mark.parametrize('operator', [MatrixMult(torch.randn(10, 4, dtype=torch.float64)), cmf_dictionary(False)])
def test_split_rows_partitions_the_rows(operator):
    y = torch.zeros(B, operator.shape[0], dtype=torch.float64)
    op_t, train, val = split_rows(operator, y, 0.8, torch.Generator().manual_seed(1))
    check_partition(operator, op_t, train, val, round(0.8 * operator.shape[0]))


@pytest.mark.parametrize('r_diag', [False, True])
def test_split_complex_rows_keeps_entries_together_and_stratifies_the_diagonal(r_diag):
    operator = cmf_dictionary(r_diag)
    y = torch.zeros(B, operator.shape[0], dtype=torch.float64)
    op_t, train, val = split_complex_rows(operator, y, 0.8, torch.Generator().manual_seed(2))
    hvec = operator.hvec
    num_diagonal, num_off = (0 if r_diag else C), C * (C - 1) // 2
    check_partition(operator, op_t, train, val, round(0.8 * num_diagonal) + 2 * round(0.8 * num_off))
    # Complex entry (i, j) of every row: real rows first, then the off-diagonal imaginary rows.
    entries = torch.stack((torch.cat((hvec.real_i, hvec.imag_i)), torch.cat((hvec.real_j, hvec.imag_j))), dim=-1)
    for t, v in zip(train, val, strict=True):
        train_entries, val_entries = ({tuple(e) for e in entries[rows].tolist()} for rows in (t, v))
        assert not train_entries & val_entries
        assert sum(i == j for i, j in train_entries) == round(0.8 * num_diagonal)


def test_splits_are_reproducible_and_differ_between_problems():
    operator = cmf_dictionary(False)
    y = torch.zeros(B, operator.shape[0], dtype=torch.float64)
    first = split_complex_rows(operator, y, 0.8, torch.Generator().manual_seed(3))[1]
    torch.testing.assert_close(split_complex_rows(operator, y, 0.8, torch.Generator().manual_seed(3))[1], first)
    assert not torch.equal(first[0], first[1])
