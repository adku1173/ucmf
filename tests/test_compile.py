"""compile_unrolled: one compiled graph reproduces eager solves for every t."""

from ucmf import AdaLFISTASW
from ucmf.optim import IterationCallback, compile_unrolled
from ucmf.optim.compile import all_iterates
from ucmf.optim.operator import CMFPropagation, MatrixMult

from cases.models import ModelCases, T, build_model, perturb_weights
from cases.problems import NUM_MICS, NUM_SOURCES

import pytest
import torch
from pytest_cases import parametrize_with_cases


@pytest.fixture(autouse=True)
def fresh_compile_cache():
    """Each test compiles anew; otherwise the recompile limit (8) silently falls back to eager."""
    torch._dynamo.reset()


SETTINGS = [
    # device, dtype, torch.compile mode, (rtol, atol) of iterates, (rtol, atol) of gradients
    pytest.param('cpu', torch.float64, 'default', (1e-12, 1e-12), (1e-10, 1e-12), id='cpu-default'),
    pytest.param(
        'cuda', torch.float32, 'reduce-overhead', (1e-4, 1e-5), (1e-4, 1e-5), id='cuda-graphs',
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is not available'),
    ),
]


@pytest.mark.parametrize(('device', 'dtype', 'mode', 'tol', 'grad_tol'), SETTINGS)
@parametrize_with_cases('spec', cases=ModelCases, has_tag='unfolded')
def test_compiled_model_matches_eager_with_changing_mic_subsets(spec, device, dtype, mode, tol, grad_tol):
    """Training scenario: per-problem microphone subsets and propagation matrices change per batch."""
    rng = torch.Generator().manual_seed(8)
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    model = perturb_weights(build_model(spec, r_diag=True)).to(device=device, dtype=dtype)
    compiled = compile_unrolled(model, T, mode=mode)
    params = tuple(model.parameters())
    y_full = torch.randn(2, NUM_MICS * (NUM_MICS - 1), dtype=dtype, generator=rng).to(device)

    for mics in ([[0, 1, 2], [0, 2, 3]], [[1, 2, 3], [0, 1, 3]]):
        H = torch.randn(2, NUM_MICS, NUM_SOURCES, dtype=complex_dtype, generator=rng).to(device)
        operator = CMFPropagation(H, r_diag=True).select_mics(torch.tensor(mics, device=device))
        y = y_full.gather(1, operator.rows)
        for t in range(1, T + 1):
            actual = compiled(y, t, operator=operator)
            expected = torch.stack([model(y, operator=operator, t=s) for s in range(1, t + 1)])
            torch.testing.assert_close(actual, expected, rtol=tol[0], atol=tol[1])

            got = torch.autograd.grad(actual.square().sum(), params)
            want = torch.autograd.grad(expected.square().sum(), params)
            for grad, reference in zip(got, want, strict=True):
                torch.testing.assert_close(grad, reference, rtol=grad_tol[0], atol=grad_tol[1])


@parametrize_with_cases('spec', cases=ModelCases, has_tag='unfolded')
def test_inactive_iterations_keep_the_iterate(spec):
    """Eager: with ``active_iterations = s``, iterations after ``s`` keep ``x_s`` and its gradient."""
    s = 2
    rng = torch.Generator().manual_seed(3)
    model = perturb_weights(build_model(spec, r_diag=True))
    H = torch.randn(2, NUM_MICS, NUM_SOURCES, dtype=torch.complex128, generator=rng)
    operator = CMFPropagation(H, r_diag=True)
    y = torch.randn(2, NUM_MICS * (NUM_MICS - 1), dtype=torch.float64, generator=rng)
    params = tuple(model.parameters())

    actual = all_iterates(model, y, T, operator=operator, active_iterations=torch.tensor(s))
    expected = all_iterates(model, y, s, operator=operator)
    torch.testing.assert_close(actual[:s], expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[s:], expected[-1].expand(T - s, -1, -1), rtol=0, atol=0)

    got = torch.autograd.grad(actual[s - 1].square().sum(), params, allow_unused=True)
    want = torch.autograd.grad(expected[-1].square().sum(), params, allow_unused=True)
    for grad, reference in zip(got, want, strict=True):
        if reference is None:  # parameters of later iterations
            assert grad is None or not grad.any()
        else:
            torch.testing.assert_close(grad, reference, rtol=1e-12, atol=1e-14)


def test_gradient_stays_finite_when_later_iterations_overflow():
    """Untrained later iterations may diverge; they must not turn the gradient of ``x_t`` into NaN."""
    model = AdaLFISTASW(num_iterations=T, num_measurements=6, gamma_init=0.3, theta_init=0.01).double()
    with torch.no_grad():
        model._gamma[1:] = 1e200  # applied step sizes of iterations 2, ..., T overflow
    rng = torch.Generator().manual_seed(1)
    y = torch.randn(3, 6, dtype=torch.float64, generator=rng)
    operator = MatrixMult(torch.randn(6, 4, dtype=torch.float64, generator=rng))
    assert not torch.isfinite(model(y, operator=operator, t=T)).all()  # precondition: overflow
    compiled = compile_unrolled(model, T, mode='default')
    params = tuple(model.parameters())

    got = torch.autograd.grad(compiled(y, 1, operator=operator)[-1].square().sum(), params)
    want = torch.autograd.grad(model(y, operator=operator, t=1).square().sum(), params, allow_unused=True)
    for grad, reference in zip(got, want, strict=True):
        torch.testing.assert_close(grad, torch.zeros_like(grad) if reference is None else reference)


def test_compiles_once_for_all_depths():
    model = AdaLFISTASW(num_iterations=T, num_measurements=6, gamma_init=0.3, theta_init=0.01).double()
    compiled = compile_unrolled(model, T, mode='default')
    rng = torch.Generator().manual_seed(2)
    y = torch.randn(3, 6, dtype=torch.float64, generator=rng)
    operator = MatrixMult(torch.randn(6, 4, dtype=torch.float64, generator=rng))
    compiled(y, 1, operator=operator)
    with torch.compiler.set_stance('fail_on_recompile'):
        for t in range(1, T + 1):
            assert compiled(y, t, operator=operator).shape == (t, 3, 4)


def test_returns_all_iterates_and_restores_callbacks(recorder):
    model = AdaLFISTASW(num_iterations=T, num_measurements=6, gamma_init=0.3, theta_init=0.01).double()
    compiled = compile_unrolled(model, T, mode='default')
    own = [recorder]
    model.callbacks = own
    rng = torch.Generator().manual_seed(0)
    y = torch.randn(3, 6, dtype=torch.float64, generator=rng)
    out = compiled(y, T, operator=MatrixMult(torch.randn(6, 4, dtype=torch.float64, generator=rng)))
    assert out.shape == (T, 3, 4)
    assert len(recorder.end_kwargs) == T  # the model's own callbacks stay active
    assert model.callbacks == own  # the collector is removed again


def test_restores_callbacks_when_solve_raises():
    model = AdaLFISTASW(num_iterations=T, num_measurements=6, gamma_init=0.3, theta_init=0.01)
    model.callbacks = own = [IterationCallback()]
    compiled = compile_unrolled(model, T, mode='default')
    operator = MatrixMult(torch.ones(6, 4))

    with pytest.raises(ValueError, match='y has shape .* but operator has shape'):
        compiled(torch.ones(3, 7), T, operator=operator)
    assert model.callbacks is own

    out = compiled(torch.ones(3, 6), T, operator=operator)
    assert out.shape == (T, 3, 4)
    assert model.callbacks is own
