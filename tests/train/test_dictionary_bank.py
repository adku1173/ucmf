"""Tests for training data with a dictionary bank; skipped without the ``train`` extra."""

import pytest

pytest.importorskip('ray')

from ucmf.optim.operator import MatrixMult  # noqa: E402
from ucmf.scaling import lambda_max  # noqa: E402
from ucmf.train.utils import batch_operator, iterate_batches, prepare_data  # noqa: E402

from configs import training_config  # noqa: E402

import torch  # noqa: E402

K, M, N_DIM = 3, 6, 10
INDEX = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2])  # entry 2 has fewer samples than one batch


def bank_config(**overrides):
    return training_config(**{'operator_name': 'MatrixMult', 'model_kwargs': {}, 'batch_size': 4, **overrides})


def take(batches, count):
    return [next(batches) for _ in range(count)]


def test_random_batches_select_the_entry_of_every_sample():
    for samples, problems in take(iterate_batches(bank_config(), len(INDEX), INDEX), 6):
        assert torch.equal(problems, INDEX[samples])


def test_per_dictionary_batches_share_one_entry_and_drop_incomplete_groups():
    batches = take(iterate_batches(bank_config(batch_sampling='per_dictionary'), len(INDEX), INDEX), 4)
    assert all(samples.shape == (4,) and problems.ndim == 0 for samples, problems in batches)
    assert all(torch.all(INDEX[samples] == problems) for samples, problems in batches)
    assert sorted(int(problems) for _, problems in batches[:2]) == [0, 1]  # one epoch: entry 2 is dropped


def test_batch_operator_selects_bank_entries_and_their_column_norms():
    bank, norms = MatrixMult(torch.randn(K, M, N_DIM, dtype=torch.float64)), torch.rand(K, N_DIM)
    operator, u = batch_operator(bank, norms, torch.tensor([2, 0]))
    torch.testing.assert_close(operator.matrix, bank.matrix[[2, 0]])
    torch.testing.assert_close(u, norms[[2, 0]])
    assert batch_operator(bank, norms, None) == (bank, norms)


def test_equireg_uses_the_dictionary_of_every_sample():
    bank = MatrixMult(torch.randn(K, M, N_DIM, dtype=torch.float64).abs())
    y, q = torch.rand(len(INDEX), M, dtype=torch.float64), torch.rand(len(INDEX), N_DIM, dtype=torch.float64)
    y_train, _, _, _, _, _ = prepare_data(bank_config(), bank, y, y, q, q, INDEX, INDEX)
    expected = torch.stack(
        [lambda_max(MatrixMult(bank.matrix[k]), y_i[None])[0] for k, y_i in zip(INDEX, y, strict=True)]
    )
    torch.testing.assert_close(y_train, y / expected.unsqueeze(-1))


@pytest.mark.parametrize(
    'overrides, index, message',
    [
        ({}, None, 'required exactly for a dictionary bank'),
        ({}, INDEX[:-1], 'one bank entry per sample'),
        ({'batch_sampling': 'per_dictionary', 'batch_size': 6}, INDEX, 'at least batch_size samples'),
    ],
)
def test_bank_indices_are_validated(overrides, index, message):
    bank = MatrixMult(torch.rand(K, M, N_DIM, dtype=torch.float64))
    y, q = torch.rand(len(INDEX), M, dtype=torch.float64), torch.rand(len(INDEX), N_DIM, dtype=torch.float64)
    with pytest.raises(ValueError, match=message):
        prepare_data(bank_config(**overrides), bank, y, y, q, q, index, index)
