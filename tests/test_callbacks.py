"""Callback lifecycle, scalar handling, stopping, and graph retention."""

from types import SimpleNamespace

from ucmf import FISTA, ISTA
from ucmf.optim import (
    IntermediateOutputCollector,
    SolutionChangeStopper,
    StepParamCollector,
)
from ucmf.optim.operator import MatrixMult

import pytest
import torch


@pytest.mark.parametrize('solver_type', [ISTA, FISTA])
def test_default_step_and_threshold_use_normalized_objective(solver_type):
    collector = StepParamCollector()
    solver = solver_type(positive=False, callbacks=[collector])
    A = torch.tensor([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])

    solver(torch.zeros(1, 3), operator=MatrixMult(A), lam=0.2, t=1)

    # gamma = m / (1.05 * ||A||_2**2) = 3 / (1.05 * 4); theta = lam * gamma.
    torch.testing.assert_close(collector.gammas_tensor, torch.tensor([0.75 / 1.05]))
    torch.testing.assert_close(collector.thetas_tensor, torch.tensor([0.15 / 1.05]))


@pytest.mark.parametrize('gamma', [0.25, torch.tensor(0.25), [0.25, 0.5]])
def test_step_collector_handles_scalars_and_per_sample_values(gamma):
    collector = StepParamCollector()
    y = torch.ones(2, 2, dtype=torch.float64)
    prepared_gamma = torch.as_tensor(gamma, dtype=y.dtype) if not isinstance(gamma, float) else gamma
    operator = MatrixMult(torch.eye(2, dtype=y.dtype))
    FISTA(callbacks=[collector])(y, operator=operator, lam=0.2, gamma=prepared_gamma, t=3)
    expected_gamma = torch.as_tensor(gamma, dtype=y.dtype)
    if expected_gamma.ndim:
        expected_gamma = expected_gamma.unsqueeze(-1)
    expected_gamma = torch.stack([expected_gamma] * 3)

    torch.testing.assert_close(collector.gammas_tensor, expected_gamma)
    torch.testing.assert_close(collector.thetas_tensor, expected_gamma * 0.2)
    assert not collector.gammas_tensor.requires_grad

    collector.reset()
    with pytest.raises(RuntimeError, match='No gamma values'):
        _ = collector.gammas_tensor
    with pytest.raises(RuntimeError, match='No theta values'):
        _ = collector.thetas_tensor


def test_step_collector_detaches_and_copies_parameters():
    collector = StepParamCollector()
    gamma = torch.tensor(0.5, requires_grad=True)
    FISTA(callbacks=[collector])(torch.ones(2), operator=MatrixMult(torch.eye(2)), lam=0.1, gamma=gamma, t=1)
    with torch.no_grad():
        gamma.fill_(2.0)

    torch.testing.assert_close(collector.gammas_tensor, torch.tensor([0.5]))
    assert not collector.gammas_tensor.requires_grad
    assert not collector.thetas_tensor.requires_grad


@pytest.mark.parametrize('solver_type', [ISTA, FISTA])
def test_classical_callbacks_receive_operator_as_weight_and_fixed_step_parameters(solver_type, recorder):
    """Order, detached operator, and y are tested for all models in test_models."""
    A = torch.eye(2, requires_grad=True)
    solver_type(callbacks=[recorder])(torch.ones(1, 2), operator=MatrixMult(A), lam=0.1, gamma=0.2, t=3)
    for kwargs in recorder.end_kwargs:
        operator = kwargs['A']
        assert isinstance(operator, MatrixMult) and kwargs['W'] is operator
        torch.testing.assert_close(operator.matrix, A)
        assert kwargs['gamma'] == 0.2
        assert kwargs['theta'] == pytest.approx(0.02)


@pytest.mark.parametrize('solver_type', [ISTA, FISTA])
def test_stopper_converges_and_resets_between_solves(solver_type):
    stopper = SolutionChangeStopper(tol=1e-9, check_every=1)
    outputs = IntermediateOutputCollector()
    solver = solver_type(callbacks=[stopper, outputs])
    # gamma = m / ||A||_2**2 = 3: one step solves A = I exactly.
    x = solver(torch.ones(2, 3), operator=MatrixMult(torch.eye(3)), lam=0.0, gamma=3.0, t=100)

    torch.testing.assert_close(x, torch.ones(2, 3))
    assert stopper.converged
    assert stopper.stop_iter == 1
    assert stopper.num_steps == 2
    assert len(outputs.intermediates) == 2

    solver(torch.zeros(2, 3), operator=MatrixMult(torch.eye(3)), lam=0.0, gamma=3.0, t=100)
    assert stopper.converged
    assert stopper.stop_iter == 0
    assert stopper.num_steps == 1
    assert len(outputs.intermediates) == 1


@pytest.mark.parametrize('solver_type', [ISTA, FISTA])
def test_stopper_checks_only_every_check_every_steps(solver_type):
    stopper = SolutionChangeStopper(tol=1e-9, check_every=5)
    solver = solver_type(callbacks=[stopper])
    solver(torch.ones(2, 3), operator=MatrixMult(torch.eye(3)), lam=0.0, gamma=3.0, t=100)
    assert stopper.converged
    assert stopper.stop_iter == 4
    assert stopper.num_steps == 5

    stopper = SolutionChangeStopper(tol=1e-9, check_every=5)
    solver = solver_type(callbacks=[stopper])
    solver(torch.ones(2, 3), operator=MatrixMult(torch.eye(3)), lam=0.0, gamma=3.0, t=3)
    assert not stopper.converged
    assert stopper.last_change is None
    assert stopper.num_steps == 3


def test_relative_stopping_requires_every_sample_to_converge():
    stopper = SolutionChangeStopper(tol=0.01, check_every=1)
    solver = SimpleNamespace(iter=0, stop=False)
    before = torch.tensor([[1000.0], [0.001]], dtype=torch.float64)
    after = before + torch.tensor([[0.001], [0.0001]], dtype=torch.float64)
    stopper.on_step_begin(solver, before)
    stopper.on_step_end(solver, after)

    assert stopper.last_change[0] < stopper.tol
    assert stopper.last_change[1] > stopper.tol
    assert not solver.stop
    assert not stopper.converged

    solver.iter = 1
    stopper.on_step_begin(solver, after)
    stopper.on_step_end(solver, after.clone())
    assert solver.stop
    assert stopper.converged
    assert stopper.num_steps == 2


def test_absolute_stopping_and_zero_solution_are_finite():
    stopper = SolutionChangeStopper(tol=0.1, relative=False, check_every=1)
    solver = SimpleNamespace(iter=0, stop=False)
    stopper.on_step_begin(solver, torch.tensor([[0.0], [1000.0]]))
    stopper.on_step_end(solver, torch.tensor([[0.01], [1000.01]]))
    assert solver.stop
    assert torch.isfinite(stopper.last_change).all()

    relative = SolutionChangeStopper(check_every=1)
    solver = SimpleNamespace(iter=0, stop=False)
    relative.on_step_begin(solver, torch.zeros(2, 1))
    relative.on_step_end(solver, torch.zeros(2, 1))
    assert solver.stop
    torch.testing.assert_close(relative.last_change, torch.zeros(2))


