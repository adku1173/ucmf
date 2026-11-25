"""Tests for ucmf.train.objectives; skipped without the ``train`` extra."""

from types import SimpleNamespace

import pytest

pytest.importorskip('ray')

import math  # noqa: E402

from ucmf import AdaLFISTASW  # noqa: E402
from ucmf.optim.compile import all_iterates  # noqa: E402
from ucmf.optim.operator import MatrixMult  # noqa: E402
from ucmf.train import train_model  # noqa: E402
from ucmf.train.objectives import bic, build_loss, lasso, nmse, weighted_mean  # noqa: E402

from configs import training_config  # noqa: E402

import torch  # noqa: E402

M, N_DIM, BATCH, T = 6, 10, 4, 3


def test_nmse_compares_source_strengths():
    column_norms = torch.rand(N_DIM, dtype=torch.float64) + 0.5
    q = torch.rand(BATCH, N_DIM, dtype=torch.float64)
    assert nmse(column_norms * q, q, column_norms).item() == pytest.approx(0, abs=1e-12)
    assert nmse(column_norms * 2 * q, q, column_norms).item() == pytest.approx(1)


def test_weighted_mean_normalizes_weights_and_skips_earlier_outputs_for_zero_weight():
    outputs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    assert weighted_mean(outputs, lambda x: x, 0.5).item() == pytest.approx((0.25 * 1 + 0.5 * 2 + 3) / 1.75)
    assert weighted_mean(outputs, lambda x: x, 1).item() == pytest.approx(2)
    assert weighted_mean(torch.tensor([float('nan'), 3.0]), lambda x: x, 0).item() == 3


@pytest.mark.parametrize('w', [0, 0.5])
def test_build_loss_binds_weighted_nmse_of_the_model_iterates(w):
    rng = torch.Generator().manual_seed(0)
    operator = MatrixMult(torch.randn(M, N_DIM, dtype=torch.float64, generator=rng))
    y = torch.randn(BATCH, M, dtype=torch.float64, generator=rng)
    model = AdaLFISTASW(num_iterations=T, gamma_init=0.3, theta_init=0.01, num_measurements=M).double()
    column_norms = torch.rand(N_DIM, dtype=torch.float64) + 0.5
    q = torch.rand(BATCH, N_DIM, dtype=torch.float64)
    solve = lambda y, operator, t: all_iterates(model, y, t, operator=operator)  # noqa: E731
    loss = build_loss(SimpleNamespace(loss='nmse', w=w, loss_kwargs={}))(solve, y, q, T, operator, column_norms)
    weights = [w ** (T - k) for k in range(1, T + 1)]
    losses = [nmse(model(y, t=k, operator=operator), q, column_norms) for k in range(1, T + 1)]
    torch.testing.assert_close(loss, sum(a * b for a, b in zip(weights, losses, strict=True)) / sum(weights))


def test_lasso_matches_dense_reference():
    rng = torch.Generator().manual_seed(1)
    matrix = torch.randn(M, N_DIM, dtype=torch.float64, generator=rng)
    x_hat = torch.rand(BATCH, N_DIM, dtype=torch.float64, generator=rng)
    y = torch.randn(BATCH, M, dtype=torch.float64, generator=rng)
    expected = ((x_hat @ matrix.T - y).square().sum(-1) / (2 * M) + 0.1 * x_hat.sum(-1)).mean()
    torch.testing.assert_close(lasso(x_hat, y, MatrixMult(matrix), 0.1), expected)


def make_config(loss, **loss_kwargs):
    matrix = torch.randn(M, N_DIM, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    model_kwargs = {'num_iterations': 2, 'gamma_init': 0.3, 'theta_init': 0.01, 'num_measurements': M}
    operator = {'operator_name': 'MatrixMult', 'operator_kwargs': {'matrix': matrix}}
    return training_config(**operator, model_kwargs=model_kwargs, loss=loss, loss_kwargs=loss_kwargs, batch_size=BATCH)


def test_bic_matches_formula_with_constant_eps():
    rng = torch.Generator().manual_seed(3)
    matrix = torch.randn(M, N_DIM, dtype=torch.float64, generator=rng)
    x_hat = torch.rand(BATCH, N_DIM, dtype=torch.float64, generator=rng).requires_grad_()
    y = torch.randn(BATCH, M, dtype=torch.float64, generator=rng)
    eps = 0.01 * x_hat.detach().amax(-1, keepdim=True)  # a constant: no gradient through the maximum
    count = (1 - torch.exp(-x_hat.square() / (2 * eps.square()))).sum(-1)
    expected = (M * torch.log((x_hat @ matrix.T - y).square().sum(-1) / M) + math.log(M) * count).mean()
    actual = bic(x_hat, y, MatrixMult(matrix), 0.01)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(*(torch.autograd.grad(value, x_hat)[0] for value in (actual, expected)))


def test_bic_of_the_zero_solution_is_finite():
    y = torch.randn(BATCH, M, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    x_hat = torch.zeros(BATCH, N_DIM, dtype=torch.float64, requires_grad=True)
    value = bic(x_hat, y, MatrixMult(torch.ones(M, N_DIM, dtype=torch.float64)), 0.01)
    expected = (M * torch.log(y.square().sum(-1) / M)).mean()  # no active coefficients
    torch.testing.assert_close(value, expected)
    assert torch.isfinite(torch.autograd.grad(value, x_hat)[0]).all()


@pytest.mark.parametrize(('loss', 'loss_kwargs'), [('lasso', {'lam': 0.1}), ('bic', {})])
def test_unsupervised_objectives_train_without_targets(loss, loss_kwargs):
    config = make_config(loss, **loss_kwargs)
    y = config.operator_kwargs['matrix'] @ torch.rand(N_DIM, 2 * BATCH, dtype=torch.float64)
    result = train_model(config, y_train=y.T, y_val=y.T[:BATCH])
    assert len(result.history) == 4 and math.isfinite(result.best_val_loss)


@pytest.mark.parametrize(
    ('loss', 'loss_kwargs'), [('lasso', {}), ('lasso', {'lam': 0.1, 'mu': 1}), ('nmse', {'lam': 0.1})]
)
def test_loss_kwargs_must_match_the_objective(loss, loss_kwargs):
    with pytest.raises(ValueError, match=f'loss {loss!r} needs loss_kwargs'):
        make_config(loss, **loss_kwargs)
