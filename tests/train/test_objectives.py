"""Tests for ucmf.train.objectives; skipped without the ``train`` extra."""

from types import SimpleNamespace

import pytest

pytest.importorskip('ray')

from ucmf import AdaLFISTASW  # noqa: E402
from ucmf.optim.compile import all_iterates  # noqa: E402
from ucmf.optim.operator import MatrixMult  # noqa: E402
from ucmf.train.objectives import build_loss, nmse, weighted_mean  # noqa: E402

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
    loss = build_loss(SimpleNamespace(loss='nmse', w=w))(solve, y, q, T, operator, column_norms)
    weights = [w ** (T - k) for k in range(1, T + 1)]
    losses = [nmse(model(y, t=k, operator=operator), q, column_norms) for k in range(1, T + 1)]
    torch.testing.assert_close(loss, sum(a * b for a, b in zip(weights, losses, strict=True)) / sum(weights))
