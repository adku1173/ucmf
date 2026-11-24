"""Numerical oracles and public interface tests for classical solvers."""

import inspect
import math

from ucmf import FISTA, ISTA
from ucmf.optim.operator import CMFDictionary, CMFPropagation, MatrixMult
from ucmf.optim.utils import default_gamma_from_A

import numpy as np
import pytest
import torch

SOLVERS = [ISTA, FISTA]


def numpy_reference(A, y, x0, lam, gamma, *, steps, nonnegative, accelerated):
    """Independent, sample-wise matrix-vector implementation in NumPy.

    The default step is taken from ucmf (power-iteration estimate for the whole,
    possibly batched A); its formula is tested separately in test_optim_utils.
    """
    default_steps = np.broadcast_to(default_gamma_from_A(torch.from_numpy(A)).numpy(), (len(y),))
    solutions = []
    for sample, observation in enumerate(y):
        matrix = A if A.ndim == 2 else A[sample]
        step = default_steps[sample] if gamma is None else gamma[sample]
        x = x0[sample].copy()
        extrapolated = x.copy()
        momentum = 1.0
        for _ in range(steps):
            current = extrapolated if accelerated else x
            residual = observation - matrix @ current
            proposal = current + (step / observation.size) * (matrix.T @ residual)
            threshold = step * lam[sample]
            next_x = (
                np.maximum(proposal - threshold, 0)
                if nonnegative
                else np.sign(proposal) * np.maximum(np.abs(proposal) - threshold, 0)
            )
            if accelerated:
                next_momentum = (1 + math.sqrt(1 + 4 * momentum**2)) / 2
                extrapolated = next_x + ((momentum - 1) / next_momentum) * (next_x - x)
                momentum = next_momentum
            x = next_x
        solutions.append(x)
    return np.stack(solutions)


@pytest.mark.parametrize('solver_type', SOLVERS)
@pytest.mark.parametrize('nonnegative', [True, False])
@pytest.mark.parametrize('batched_A', [True, False])
@pytest.mark.parametrize('default_gamma', [True, False])
def test_matches_independent_numpy_iterations(solver_type, nonnegative, batched_A, default_gamma):
    rng = np.random.default_rng(17)
    A = rng.normal(size=(3, 5, 4) if batched_A else (5, 4))
    y = rng.normal(size=(3, 5))
    x0 = rng.normal(size=(3, 4))
    lam = np.array([0.0, 0.015, 0.03])
    gamma = None if default_gamma else np.array([0.07, 0.11, 0.15])

    actual = solver_type(nonnegative=nonnegative)(
        torch.from_numpy(y),
        x0=torch.from_numpy(x0),
        operator=MatrixMult(torch.from_numpy(A)),
        lam=torch.from_numpy(lam),
        gamma=None if gamma is None else torch.from_numpy(gamma),
        t=9,
    )
    expected = numpy_reference(A, y, x0, lam, gamma, steps=9, nonnegative=nonnegative, accelerated=solver_type is FISTA)

    torch.testing.assert_close(actual, torch.from_numpy(expected), rtol=1e-11, atol=1e-12)
    assert actual.dtype == torch.float64
    assert actual.shape == (3, 4)


@pytest.mark.parametrize('solver_type', SOLVERS)
@pytest.mark.parametrize('batched', [False, True])
def test_cmf_representations_agree_with_default_step(solver_type, batched):
    rng = torch.Generator().manual_seed(7)
    H = torch.randn((2, 3, 4) if batched else (3, 4), dtype=torch.complex128, generator=rng)
    y = torch.randn(2, 6, dtype=torch.float64, generator=rng)
    solver = solver_type(nonnegative=False)
    expected = solver(y, operator=CMFDictionary(H, r_diag=True), lam=0.01, t=8)
    actual = solver(y, operator=CMFPropagation(H, r_diag=True), lam=0.01, t=8)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize('solver_type', SOLVERS)
@pytest.mark.parametrize('nonnegative', [True, False])
def test_orthogonal_problem_has_analytic_lasso_solution(solver_type, nonnegative):
    y = torch.tensor([1.0, -2.0, 0.01, -0.01], dtype=torch.float64)
    A = torch.eye(4, dtype=y.dtype)
    lam = 0.1
    expected = (y - 4 * lam).clamp_min(0) if nonnegative else y.sign() * (y.abs() - 4 * lam).clamp_min(0)

    actual = solver_type(nonnegative=nonnegative)(y, operator=MatrixMult(A), lam=lam, t=8)

    torch.testing.assert_close(actual, expected.unsqueeze(0))


@pytest.mark.parametrize('solver_type', SOLVERS)
@pytest.mark.parametrize('nonnegative', [True, False])
def test_solution_satisfies_normalized_lasso_kkt(solver_type, nonnegative):
    rng = torch.Generator().manual_seed(5)
    A = torch.randn(9, 4, generator=rng, dtype=torch.float64)
    y = torch.randn(2, 9, generator=rng, dtype=torch.float64)
    lam = 0.015
    x = solver_type(nonnegative=nonnegative)(y, operator=MatrixMult(A), lam=lam, t=1000)
    grad = ((x @ A.T - y) @ A) / A.shape[0]
    active = x.abs() > 1e-9

    torch.testing.assert_close(grad[active] + lam * x[active].sign(), torch.zeros_like(x[active]), atol=1e-9, rtol=0)
    if nonnegative:
        assert torch.all(x >= 0)
        assert torch.all(grad[~active] + lam >= -1e-9)
    else:
        assert torch.all(grad[~active].abs() <= lam + 1e-9)


@pytest.mark.parametrize('solver_type', SOLVERS)
def test_gradient_normalization_and_explicit_column_scaling(solver_type):
    A = torch.tensor([[3.0, 0.0], [4.0, 2.0]])
    y = torch.ones(1, 2)

    actual = solver_type(nonnegative=False)(y, operator=MatrixMult(A), lam=0.0, gamma=1.0, t=1)

    torch.testing.assert_close(actual, torch.tensor([[3.5, 1.0]]))


@pytest.mark.parametrize('solver_type', SOLVERS)
@pytest.mark.parametrize('parameter_form', ['float', 'scalar_tensor', 'vector', 'column'])
def test_scalar_and_per_sample_parameters(solver_type, parameter_form):
    def parameter(value):
        if parameter_form == 'float':
            return value
        if parameter_form == 'scalar_tensor':
            return torch.tensor(value, dtype=torch.float64)
        if parameter_form == 'vector':
            return torch.full((2,), value, dtype=torch.float64)
        return torch.full((2, 1), value, dtype=torch.float64)

    y = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float64)
    actual = solver_type(nonnegative=False)(
        y, operator=MatrixMult(torch.eye(2, dtype=y.dtype)), lam=parameter(0.1), gamma=parameter(1.0), t=1
    )
    expected = y.sign() * (y.abs() / 2 - 0.1).clamp_min(0)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('solver_type', SOLVERS)
def test_sensing_matrix_is_fixed_but_measurements_remain_differentiable(solver_type):
    A = torch.eye(2, requires_grad=True)
    y = torch.ones(1, 2, requires_grad=True)
    solver = solver_type()
    # gamma = m / ||A||_2**2 = 2: each step solves A = I exactly.
    solver(y, operator=MatrixMult(A), lam=0.0, gamma=2.0, t=2).sum().backward()

    assert A.grad is None
    torch.testing.assert_close(y.grad, torch.ones_like(y))
    assert list(solver.parameters()) == []


@pytest.mark.parametrize('solver_type', SOLVERS)
def test_autograd_through_measurements_warm_start_and_step_parameters(solver_type):
    A = torch.tensor([[1.0, 0.2], [0.1, 1.1], [0.3, -0.1]], dtype=torch.float64)
    y = torch.tensor([[1.2, 0.8, 0.2]], dtype=torch.float64, requires_grad=True)
    x0 = torch.tensor([[0.3, 0.2]], dtype=torch.float64, requires_grad=True)
    lam = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    gamma = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    solver = solver_type(nonnegative=False)

    assert torch.autograd.gradcheck(
        lambda y, x0, lam, gamma: solver(y, x0=x0, operator=MatrixMult(A), lam=lam, gamma=gamma, t=4),
        (y, x0, lam, gamma),
    )


@pytest.mark.parametrize('solver_type', SOLVERS)
def test_preprocessed_inputs_and_noncontiguous_sensing_matrix(solver_type):
    y = torch.tensor([1.0, 2.0], dtype=torch.float64)
    solver = solver_type()
    expected = solver(y, operator=MatrixMult(torch.eye(2, dtype=y.dtype)), lam=0.0, t=2)
    for A in ([[1, 0], [0, 1]], np.eye(2), torch.eye(2, dtype=torch.int64), torch.eye(2).T):
        prepared_A = torch.as_tensor(A, dtype=y.dtype, device=y.device)
        torch.testing.assert_close(solver(y, operator=MatrixMult(prepared_A), lam=0.0, t=2), expected)


@pytest.mark.parametrize('solver_type', SOLVERS)
@pytest.mark.parametrize(
    ('overrides', 'message'),
    [
        ({'operator': None}, 'operator must be a LinearOperator'),
        ({'operator': torch.eye(2)}, 'operator must be a LinearOperator'),
        ({'lam': None}, 'lam must be provided'),
        ({'operator': MatrixMult(torch.ones(3, 2))}, 'y has shape'),
        ({'operator': MatrixMult(torch.ones(3, 2, 2))}, 'y has shape'),
        ({'operator': MatrixMult(torch.ones(2, 0))}, 'non-empty'),
        ({'operator': MatrixMult(torch.zeros(2, 2))}, 'nonzero spectral norm'),
        ({'y': torch.tensor(1.0)}, 'y must be a real floating-point tensor'),
        ({'y': torch.ones(1, 1, 2)}, 'y must be a real floating-point tensor'),
        ({'y': torch.ones(2, 2, dtype=torch.int64)}, 'y must be a real floating-point tensor'),
        ({'y': torch.ones(2, 2, dtype=torch.complex64)}, 'y must be a real floating-point tensor'),
        ({'x0': torch.ones(2, 3)}, 'x0 has shape'),
        ({'lam': torch.tensor([0.1, 0.2, 0.3])}, 'expected batch size'),
        ({'gamma': torch.ones(2, 2)}, 'gamma must be scalar'),
        ({'lam': torch.ones(2, 1, 1)}, 'lam must be scalar'),
    ],
)
def test_invalid_inputs_raise_clear_errors(solver_type, overrides, message):
    inputs = {'y': torch.ones(2, 2), 'operator': MatrixMult(torch.eye(2)), 'lam': 0.1, 't': 1}
    inputs.update(overrides)
    with pytest.raises(ValueError, match=message):
        solver_type()(**inputs)


@pytest.mark.parametrize('solver_type', SOLVERS)
def test_forward_exposes_only_classical_solver_inputs(solver_type):
    assert list(inspect.signature(solver_type.forward).parameters) == [
        'self', 'y', 'x0', 't', 'operator', 'lam', 'gamma',
    ]


@pytest.mark.parametrize('solver_type', SOLVERS)
@pytest.mark.parametrize('weight', [None, torch.eye(2)])
def test_forward_rejects_independent_gradient_weights(solver_type, weight):
    with pytest.raises(TypeError, match="unexpected keyword argument 'W'"):
        solver_type()(torch.ones(2), operator=MatrixMult(torch.eye(2)), lam=0.0, W=weight)


@pytest.mark.parametrize('solver_type', SOLVERS)
def test_constructor_has_no_gradient_mode_selector(solver_type):
    assert list(inspect.signature(solver_type.__init__).parameters) == ['self', 'nonnegative', 'callbacks']
    assert not hasattr(solver_type(), 'ls_mode')
    assert not hasattr(solver_type(), 'tls_grad')


@pytest.mark.parametrize('solver_type', SOLVERS)
@pytest.mark.parametrize('mode', ['ls', 'tls', 'unknown'])
def test_removed_gradient_mode_is_not_silently_accepted(solver_type, mode):
    with pytest.raises(TypeError, match="unexpected keyword argument 'ls_mode'"):
        solver_type(ls_mode=mode)


@pytest.mark.parametrize('solver_type', SOLVERS)
@pytest.mark.parametrize('batched_A', [True, False])
def test_one_step_matches_normalized_adjoint_residual(solver_type, batched_A):
    A = torch.tensor([[1.0, 0.1], [0.3, 1.5]], dtype=torch.float64)
    if batched_A:
        A = torch.stack([A, 2 * A])
    y = torch.tensor([[1.0, 2.0], [0.5, 0.8]], dtype=A.dtype)
    x = torch.tensor([[0.2, 0.3], [0.1, 0.2]], dtype=A.dtype)
    residual = y - (A @ x.unsqueeze(-1)).squeeze(-1)
    expected = x + (residual.unsqueeze(-2) @ A).squeeze(-2) / y.shape[1]
    actual = solver_type(nonnegative=False)(y, x0=x, operator=MatrixMult(A), lam=0.0, gamma=1.0, t=1)
    torch.testing.assert_close(actual, expected)


