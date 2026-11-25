"""Tests for ucmf.train.objectives; skipped without the ``train`` extra."""

from types import SimpleNamespace

import pytest

pytest.importorskip('ray')

from ucmf import AdaLFISTASW  # noqa: E402
from ucmf.optim.operator import MatrixMult  # noqa: E402
from ucmf.train.objectives import build_loss, nmse, weighted_sum  # noqa: E402

import torch  # noqa: E402

M, N_DIM, BATCH, T = 6, 10, 4, 3


def test_nmse_compares_source_strengths():
    column_norms = torch.rand(N_DIM, dtype=torch.float64) + 0.5
    q = torch.rand(BATCH, N_DIM, dtype=torch.float64)
    assert nmse(column_norms * q, q, column_norms).item() == pytest.approx(0, abs=1e-12)
    assert nmse(column_norms * 2 * q, q, column_norms).item() == pytest.approx(1)


def test_weighted_sum_weights_earlier_outputs_and_skips_them_for_zero_weight():
    outputs = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
    assert weighted_sum(outputs, lambda x: x, 0.5).item() == 0.25 * 1 + 0.5 * 2 + 3
    assert weighted_sum([torch.tensor(float('nan')), torch.tensor(3.0)], lambda x: x, 0).item() == 3


@pytest.mark.parametrize('w', [0, 0.5])
def test_build_loss_binds_weighted_nmse_of_the_model_iterates(w):
    rng = torch.Generator().manual_seed(0)
    operator = MatrixMult(torch.randn(M, N_DIM, dtype=torch.float64, generator=rng))
    y = torch.randn(BATCH, M, dtype=torch.float64, generator=rng)
    model = AdaLFISTASW(num_iterations=T, gamma_init=0.3, theta_init=0.01, num_measurements=M).double()
    callbacks = model.callbacks
    column_norms = torch.rand(N_DIM, dtype=torch.float64) + 0.5
    q = torch.rand(BATCH, N_DIM, dtype=torch.float64)
    loss = build_loss(SimpleNamespace(loss='nmse', w=w))(model, y, q, T, operator, column_norms)
    iterates = [model(y, t=k, operator=operator) for k in range(1, T + 1)]
    expected = sum(w ** (T - k) * nmse(x, q, column_norms) for k, x in enumerate(iterates, start=1))
    torch.testing.assert_close(loss, expected)
    assert model.callbacks is callbacks
