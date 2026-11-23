"""compile_unrolled: one compiled graph reproduces eager solves for every t."""

from ucmf import AdaLFISTASW
from ucmf.optim import IterationCallback, compile_unrolled
from ucmf.optim.operator import CMFPropagation, MatrixMult

import pytest
import torch
from pytest_cases import parametrize_with_cases
from unfolded_model_cases import N_MICS, N_SOURCES, T, UnfoldedModelCases, build_model, perturb_weights


@parametrize_with_cases('make_model, dense_weights', cases=UnfoldedModelCases)
def test_compiled_model_matches_eager_with_changing_mic_subsets(make_model, dense_weights):
    """Training scenario: per-problem microphone subsets and propagation matrices change per batch."""
    rng = torch.Generator().manual_seed(8)
    model = perturb_weights(build_model(make_model, r_diag=True))
    compiled = compile_unrolled(model, T, mode='default')
    params = tuple(model.parameters())
    y_full = torch.randn(2, N_MICS * (N_MICS - 1), dtype=torch.float64, generator=rng)

    for mics in ([[0, 1, 2], [0, 2, 3]], [[1, 2, 3], [0, 1, 3]]):
        H = torch.randn(2, N_MICS, N_SOURCES, dtype=torch.complex128, generator=rng)
        operator = CMFPropagation(H, r_diag=True).select_mics(torch.tensor(mics))
        y = y_full.gather(1, operator.rows)
        actual = compiled(y, operator=operator)
        expected = torch.stack([model(y, operator=operator, t=t) for t in range(1, T + 1)])
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

        got = torch.autograd.grad(actual.square().sum(), params)
        want = torch.autograd.grad(expected.square().sum(), params)
        for grad, reference in zip(got, want, strict=True):
            torch.testing.assert_close(grad, reference, rtol=1e-10, atol=1e-12)


def test_returns_all_iterates_and_restores_callbacks():
    model = AdaLFISTASW(T, 6, 0.3, 0.01).double()
    compiled = compile_unrolled(model, T, mode='default')
    seen = []

    class Counter(IterationCallback):
        def on_step_end(self, solver, x, **kwargs):
            seen.append(1)

    own = [Counter()]
    model.callbacks = own
    out = compiled(torch.randn(3, 6, dtype=torch.float64), operator=MatrixMult(torch.randn(6, 4, dtype=torch.float64)))
    assert out.shape == (T, 3, 4)
    assert len(seen) == T  # the model's own callbacks stay active
    assert model.callbacks == own  # the collector is removed again


def test_restores_callbacks_when_solve_raises():
    model = AdaLFISTASW(T, 6, 0.3, 0.01)
    model.callbacks = own = [IterationCallback()]
    compiled = compile_unrolled(model, T, mode='default')
    operator = MatrixMult(torch.ones(6, 4))

    with pytest.raises(ValueError, match='y has shape .* but operator has shape'):
        compiled(torch.ones(3, 7), operator=operator)
    assert model.callbacks is own

    out = compiled(torch.ones(3, 6), operator=operator)
    assert out.shape == (T, 3, 4)
    assert model.callbacks is own
