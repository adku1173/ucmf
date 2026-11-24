"""Acoular solver adapters solve the same problem as Acoular's LassoLars solver."""

from acoular.solvers import L1RegularizedLeastSquaresProblem, LassoLarsSolver

from ucmf.acoular import FISTATorchSolver, ISTATorchSolver

import numpy as np
import pytest
import torch


def _problem(solver):
    return L1RegularizedLeastSquaresProblem(
        solver=solver, alpha=1e-3, nonnegative=True, dictionary_scaling='unit_l2', unit_mult=1.0
    )


@pytest.fixture
def system():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((40, 12))
    x_true = np.zeros(12)
    x_true[[1, 5, 9]] = [1.0, 0.6, 0.8]
    return A, A @ x_true


@pytest.mark.parametrize('solver_type', [ISTATorchSolver, FISTATorchSolver])
def test_torch_solver_matches_lasso_lars(system, solver_type):
    A, y = system
    reference = _problem(LassoLarsSolver(backend_kwargs={'fit_intercept': False})).solve(A, y).solution
    result = _problem(solver_type(n_iter=20000, tol=1e-12)).solve(A, y)

    assert isinstance(result.solution, np.ndarray)
    assert result.info['converged']
    assert 0 < result.info['num_steps'] < 20000
    np.testing.assert_allclose(result.solution, reference, atol=1e-8)


def test_without_tol_runs_all_iterations_without_stopper(system):
    A, y = system
    result = _problem(FISTATorchSolver(n_iter=50)).solve(A, y)
    reference = _problem(FISTATorchSolver(n_iter=50, tol=0.0)).solve(A, y)
    assert result.info == {}
    assert reference.info == {'num_steps': 50, 'converged': False}
    np.testing.assert_array_equal(result.solution, reference.solution)


def test_start_value_is_used_as_warm_start(system):
    A, y = system
    problem = _problem(FISTATorchSolver(n_iter=20000, tol=1e-10, check_every=1))
    cold = problem.solve(A, y)
    warm = problem.solve(A, y, start_value=cold.solution * np.linalg.norm(A, axis=0))
    assert warm.info['num_steps'] < cold.info['num_steps']


def test_digest_depends_on_algorithm_and_configuration():
    solver = FISTATorchSolver()
    digests = {ISTATorchSolver().digest, solver.digest}
    for name, value in [
        ('n_iter', 10),
        ('tol', 1e-3),
        ('check_every', 7),
        ('device', 'meta'),
        ('precision', 'float32'),
    ]:
        setattr(solver, name, value)
        digests.add(solver.digest)
    assert len(digests) == 7


def test_float32_precision_matches_float64(system):
    A, y = system
    reference = _problem(FISTATorchSolver(n_iter=500)).solve(A, y).solution
    solution = _problem(FISTATorchSolver(n_iter=500, precision='float32')).solve(A, y).solution
    np.testing.assert_allclose(solution, reference, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_cuda_matches_cpu(system):
    A, y = system
    cpu = _problem(FISTATorchSolver(n_iter=500)).solve(A, y).solution
    cuda = _problem(FISTATorchSolver(n_iter=500, device='cuda')).solve(A, y).solution
    np.testing.assert_allclose(cuda, cpu, atol=1e-12)
