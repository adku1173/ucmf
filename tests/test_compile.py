"""compile_unrolled: one compiled graph reproduces eager solves for every t."""

from ucmf.optim import IterationCallback, compile_unrolled

import pytest
import torch

T_MAX = 5


@pytest.fixture(scope='module')
def case(unfolded_model_type):
    torch.manual_seed(0)
    model = unfolded_model_type(
        n_iterations=T_MAX, n_measurements=6, gamma_init=0.3, theta_init=0.01, positive=False,
    ).to(torch.float64)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 2:  # auxiliary weights
                parameter.add_(0.1 * torch.randn_like(parameter))
    A = torch.randn(6, 4, dtype=torch.float64)
    y = torch.randn(3, 6, dtype=torch.float64)
    return model, A, y, compile_unrolled(model, T_MAX, mode='default')


def _grads(model):
    return [p.grad.clone() for p in model.parameters()]


@pytest.mark.parametrize('t', range(1, T_MAX + 1))
def test_intermediate_t_matches_eager_solve_and_gradients(case, t):
    model, A, y, compiled = case
    model.zero_grad(set_to_none=True)
    actual = compiled(y, A=A)[t - 1]
    actual.square().sum().backward()
    actual_grads = _grads(model)

    model.zero_grad(set_to_none=True)
    expected = model(y, A=A, t=t)
    expected.square().sum().backward()

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    for got, want in zip(actual_grads, _grads(model), strict=True):
        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-12)


def test_returns_all_iterates_and_restores_callbacks(case):
    model, A, y, compiled = case
    seen = []

    class Counter(IterationCallback):
        def on_step_end(self, solver, x, **kwargs):
            seen.append(1)

    own = [Counter()]
    model.callbacks = own
    try:
        out = compiled(y, A=A)
        restored = model.callbacks
    finally:
        model.callbacks = []
    assert out.shape == (T_MAX, 3, 4)
    assert len(seen) == T_MAX  # the model's own callbacks stay active
    assert restored == own  # the collector is removed again
