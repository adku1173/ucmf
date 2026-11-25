"""Tests for the CV splits of ucmf.train.objectives; skipped without the ``train`` extra."""

import pytest

pytest.importorskip('ray')

from ucmf.optim.operator import CMFDictionary, CMFPropagation, MatrixMult  # noqa: E402
from ucmf.train import train_model  # noqa: E402
from ucmf.train.objectives import cv_loss, split_complex_rows, split_mics, split_mics_cross, split_rows  # noqa: E402

from configs import training_config  # noqa: E402

import torch  # noqa: E402

B, C = 3, 5


def cmf_dictionary(r_diag):
    H = torch.randn(C, 4, dtype=torch.complex128, generator=torch.Generator().manual_seed(0))
    return CMFDictionary(H, r_diag=r_diag)


def row_entries(hvec):
    """Complex entry ``(i, j)`` of every row: real rows first, then the off-diagonal imaginary rows."""
    return torch.stack((torch.cat((hvec.real_i, hvec.imag_i)), torch.cat((hvec.real_j, hvec.imag_j))), dim=-1)


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
    entries = row_entries(hvec)
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


@pytest.mark.parametrize('operator_type', [CMFDictionary, CMFPropagation])
@pytest.mark.parametrize('r_diag', [False, True])
@pytest.mark.parametrize('split', [split_mics, split_mics_cross])
def test_mic_splits_train_on_a_block_and_validate_on_the_val_block_or_all_other_rows(split, r_diag, operator_type):
    H = torch.randn(C, 4, dtype=torch.complex128, generator=torch.Generator().manual_seed(0))
    operator = operator_type(H, r_diag=r_diag)
    y = torch.zeros(B, operator.shape[0], dtype=torch.float64)
    op_t, train, val = split(operator, y, 0.6, torch.Generator().manual_seed(4))  # 3 train, 2 val mics
    entries = row_entries(operator.hvec)
    for mics, t, v in zip(op_t.mics, train, val, strict=True):
        train_mics, val_mics = set(mics.tolist()), set(range(C)) - set(mics.tolist())
        # Exactly the train block: only train microphones and the full block size.
        assert {e for row in entries[t].tolist() for e in row} <= train_mics and len(t) == 3 * (3 - r_diag)
        val_entries = entries[v].tolist()
        if split is split_mics:  # the val block only; the cross terms stay unused
            assert all(set(e) <= val_mics for e in val_entries) and len(v) == 2 * (2 - r_diag)
        else:  # all other rows, including the cross terms
            assert sorted(torch.cat((t, v)).tolist()) == list(range(operator.shape[0]))


def test_cv_loss_scales_both_subsets_with_the_train_subset():
    rng = torch.Generator().manual_seed(5)
    matrix = torch.rand(10, 4, dtype=torch.float64, generator=rng)
    x_hat = torch.rand(B, 4, dtype=torch.float64, generator=rng)
    y = torch.rand(B, 10, dtype=torch.float64, generator=rng)
    received = {}

    def solve(y_t, D_t, t):  # records the train problem and returns fixed iterates
        received.update(y_t=y_t, D_t=D_t.matrix)
        return x_hat.expand(t, -1, -1)

    loss = cv_loss(solve, y, None, 2, MatrixMult(matrix), None, w=0.5, generator=torch.Generator().manual_seed(6))
    _, train, val = split_rows(MatrixMult(matrix), y, 0.8, torch.Generator().manual_seed(6))
    D = matrix / torch.linalg.vector_norm(matrix[train], dim=-2).unsqueeze(-2)  # A U_t^-1, (B, m, n)
    D_t, y_t = D.gather(1, train.unsqueeze(-1).expand(-1, -1, 4)), y.gather(-1, train)
    lam = ((D_t.mT @ y_t.unsqueeze(-1)).squeeze(-1).amax(-1) / 8).unsqueeze(-1)
    error = (D @ x_hat.unsqueeze(-1)).squeeze(-1).gather(-1, val) - y.gather(-1, val) / lam
    torch.testing.assert_close(received['D_t'], D_t)
    torch.testing.assert_close(received['y_t'], y_t / lam)
    torch.testing.assert_close(loss, (error.square().sum(-1) / 4).mean())  # equal iterates: weights cancel


CV_MODEL = {'num_iterations': 2, 'gamma_init': 0.3, 'theta_init': 0.01, 'num_measurements': C * C}


def cv_config(**overrides):
    H = torch.randn(C, 4, dtype=torch.complex128, generator=torch.Generator().manual_seed(7))
    settings = {'operator_name': 'CMFDictionary', 'operator_kwargs': {'H': H}, 'model_kwargs': CV_MODEL}
    return training_config(**{**settings, 'loss': 'cv', 'batch_size': B, **overrides})


@pytest.mark.parametrize('split', ['rows', 'complex_rows', 'mics', 'mics_cross'])
def test_cv_trains_with_every_split(split):
    config = cv_config(loss_kwargs={'split': split})
    y = CMFDictionary(config.operator_kwargs['H']).forward(torch.rand(2 * B, 4, dtype=torch.float64))
    result = train_model(config, y_train=y, y_val=y[:B])
    assert len(result.history) == 4 and torch.isfinite(torch.tensor(result.best_val_loss))


def test_cv_config_rejects_invalid_split_and_signed_models():
    with pytest.raises(ValueError, match='split must be one of'):
        cv_config(loss_kwargs={'split': 'nope'})
    with pytest.raises(ValueError, match="loss='cv' requires a nonnegative model"):
        cv_config(model_kwargs={**CV_MODEL, 'nonnegative': False}, data_scaling='none')
