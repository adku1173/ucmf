"""Tests for training data with a dictionary bank; skipped without the ``train`` extra."""

import pytest

pytest.importorskip('ray')

from ucmf.optim.operator import MatrixMult  # noqa: E402
from ucmf.scaling import lambda_max  # noqa: E402
from ucmf.train import train_model  # noqa: E402
from ucmf.train.utils import batch_operator, iterate_batches, prepare_data, validation_batches  # noqa: E402

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
    y_train, _, _ = prepare_data(bank_config(), bank, y, q, INDEX)
    expected = torch.stack(
        [lambda_max(MatrixMult(bank.matrix[k]), y_i[None])[0] for k, y_i in zip(INDEX, y, strict=True)]
    )
    torch.testing.assert_close(y_train, y / expected.unsqueeze(-1))


@pytest.mark.parametrize(
    'index, message', [(None, 'required exactly for a dictionary bank'), (INDEX[:-1], 'one bank entry per sample')]
)
def test_bank_indices_are_validated(index, message):
    bank = MatrixMult(torch.rand(K, M, N_DIM, dtype=torch.float64))
    y, q = torch.rand(len(INDEX), M, dtype=torch.float64), torch.rand(len(INDEX), N_DIM, dtype=torch.float64)
    with pytest.raises(ValueError, match=message):
        prepare_data(bank_config(), bank, y, q, index)


@pytest.mark.parametrize('index', [None, INDEX])
def test_per_dictionary_sampling_needs_a_full_batch_of_one_entry(index):
    with pytest.raises(ValueError, match='at least batch_size samples'):
        iterate_batches(bank_config(batch_sampling='per_dictionary', batch_size=6), len(INDEX), index)


def test_validation_batches_are_full_and_follow_a_fixed_permutation():
    y = torch.arange(len(INDEX), dtype=torch.float64).unsqueeze(-1)
    batches = validation_batches(bank_config(), y, None, INDEX)
    again = validation_batches(bank_config(), y, None, INDEX)
    assert len(batches) == 2 and all(torch.equal(a[0], b[0]) for a, b in zip(batches, again, strict=True))
    for y_batch, q_batch, problems in batches:
        assert q_batch is None and torch.equal(problems, INDEX[y_batch[:, 0].long()])
    assert not torch.equal(batches[0][0][:, 0], torch.arange(4.0, dtype=torch.float64))  # permuted


def bank_problem():
    rng = torch.Generator().manual_seed(4)
    A = torch.randn(M, N_DIM, dtype=torch.float64, generator=rng)
    q = torch.rand(16, N_DIM, dtype=torch.float64, generator=rng)
    index = torch.randint(K, (16,), generator=rng)
    return A, q @ A.T, q, index


def train(A, y, q, index, **overrides):
    model_kwargs = {'num_iterations': 2, 'gamma_init': 0.3, 'theta_init': 0.01, 'num_measurements': M}
    config = bank_config(operator_kwargs={'matrix': A}, model_kwargs=model_kwargs, **overrides)
    index_val = None if index is None else index[:8]
    return train_model(config, y_train=y, y_val=y[:8], q_train=q, q_val=q[:8], index_train=index, index_val=index_val)


def test_a_bank_of_identical_dictionaries_trains_like_the_shared_operator():
    A, y, q, index = bank_problem()
    bank, shared = train(A.expand(K, -1, -1), y, q, index).history, train(A, y, q, None).history
    for entry, expected in zip(bank, shared, strict=True):
        assert entry['train_loss'] == pytest.approx(expected['train_loss'], rel=1e-9)
        assert entry['val_loss'] == pytest.approx(expected['val_loss'], rel=1e-9)


@pytest.mark.parametrize('batch_sampling', ['random', 'per_dictionary'])
def test_training_with_a_bank_of_distinct_dictionaries(batch_sampling):
    A, y, q, index = bank_problem()
    index[:4] = 0  # one full batch of a single entry for per_dictionary
    matrices = A * torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64).view(K, 1, 1) + 0.1
    history = train(matrices, y, q, index, batch_sampling=batch_sampling).history
    assert all(torch.isfinite(torch.tensor(entry['val_loss'])) for entry in history)
