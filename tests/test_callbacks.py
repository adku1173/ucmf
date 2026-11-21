"""Callback lifecycle, scalar handling, stopping, and graph retention."""

from types import SimpleNamespace

from ucmf import FISTA, ISTA
from ucmf.optim import (
    IntermediateOutputCollector,
    IterationCallback,
    SolutionChangeStopper,
    StepParamCollector,
)

import pytest
import torch


@pytest.mark.parametrize('solver_type', [ISTA, FISTA])
def test_default_step_and_threshold_use_normalized_objective(solver_type):
    collector = StepParamCollector()
    solver = solver_type(positive=False, callbacks=[collector])
    A = torch.tensor([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])

    solver(torch.zeros(1, 3), A=A, lam=0.2, t=1)

    torch.testing.assert_close(collector.gammas_tensor, torch.tensor([0.75]))
    torch.testing.assert_close(collector.thetas_tensor, torch.tensor([0.15]))


@pytest.mark.parametrize('gamma', [0.25, torch.tensor(0.25), [0.25, 0.5]])
def test_step_collector_handles_scalars_and_per_sample_values(gamma):
    collector = StepParamCollector()
    y = torch.ones(2, 2, dtype=torch.float64)
    prepared_gamma = torch.as_tensor(gamma, dtype=y.dtype) if not isinstance(gamma, float) else gamma
    FISTA(callbacks=[collector])(y, A=torch.eye(2, dtype=y.dtype), lam=0.2, gamma=prepared_gamma, t=3)
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
    FISTA(callbacks=[collector])(torch.ones(2), A=torch.eye(2), lam=0.1, gamma=gamma, t=1)
    with torch.no_grad():
        gamma.fill_(2.0)

    torch.testing.assert_close(collector.gammas_tensor, torch.tensor([0.5]))
    assert not collector.gammas_tensor.requires_grad
    assert not collector.thetas_tensor.requires_grad


@pytest.mark.parametrize('solver_type', [ISTA, FISTA])
def test_callbacks_receive_ordered_iterations_and_classical_adjoint(solver_type):
    events = []
    A = torch.eye(2)
    y = torch.ones(1, 2)

    class Recorder(IterationCallback):
        def on_step_begin(self, solver, x, **kwargs):
            events.append(('begin', solver.iter))

        def on_step_end(self, solver, x, **kwargs):
            events.append(('end', solver.iter))
            torch.testing.assert_close(kwargs['A'], A)
            torch.testing.assert_close(kwargs['W'], A)
            torch.testing.assert_close(kwargs['y'], y)
            torch.testing.assert_close(solver.y, y)
            assert kwargs['gamma'] == 0.2
            assert kwargs['theta'] == pytest.approx(0.02)

    solver_type(callbacks=[Recorder()])(y, A=A, lam=0.1, gamma=0.2, t=3)
    assert events == [(event, k) for k in range(3) for event in ('begin', 'end')]


@pytest.mark.parametrize('solver_type', [ISTA, FISTA])
@pytest.mark.parametrize('stop_at', ['begin', 'end'])
def test_custom_callbacks_can_stop_at_either_boundary(solver_type, stop_at):
    outputs = IntermediateOutputCollector()

    class Stop(IterationCallback):
        def on_step_begin(self, solver, x, **kwargs):
            if stop_at == 'begin':
                solver.stop = True

        def on_step_end(self, solver, x, **kwargs):
            if stop_at == 'end':
                solver.stop = True

    actual = solver_type(callbacks=[Stop(), outputs])(torch.ones(2), A=torch.eye(2), lam=0.0, t=8)

    completed = 0 if stop_at == 'begin' else 1
    assert len(outputs.intermediates) == completed
    torch.testing.assert_close(actual, torch.full((1, 2), float(completed)))


@pytest.mark.parametrize('solver_type', [ISTA, FISTA])
def test_stopper_converges_and_resets_between_solves(solver_type):
    stopper = SolutionChangeStopper(tol=1e-9, check_every=1)
    outputs = IntermediateOutputCollector()
    solver = solver_type(callbacks=[stopper, outputs])
    x = solver(torch.ones(2, 3), A=torch.eye(3), lam=0.0, t=100)

    torch.testing.assert_close(x, torch.ones(2, 3))
    assert stopper.converged
    assert stopper.stop_iter == 1
    assert stopper.n_steps == 2
    assert len(outputs.intermediates) == 2

    solver(torch.zeros(2, 3), A=torch.eye(3), lam=0.0, t=100)
    assert stopper.converged
    assert stopper.stop_iter == 0
    assert stopper.n_steps == 1
    assert len(outputs.intermediates) == 1


@pytest.mark.parametrize('solver_type', [ISTA, FISTA])
def test_stopper_checks_only_every_check_every_steps(solver_type):
    stopper = SolutionChangeStopper(tol=1e-9, check_every=5)
    solver = solver_type(callbacks=[stopper])
    solver(torch.ones(2, 3), A=torch.eye(3), lam=0.0, t=100)
    assert stopper.converged
    assert stopper.stop_iter == 4
    assert stopper.n_steps == 5

    stopper = SolutionChangeStopper(tol=1e-9, check_every=5)
    solver = solver_type(callbacks=[stopper])
    solver(torch.ones(2, 3), A=torch.eye(3), lam=0.0, t=3)
    assert not stopper.converged
    assert stopper.last_change is None
    assert stopper.n_steps == 3


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
    assert stopper.n_steps == 2


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


@pytest.mark.parametrize('solver_type', [ISTA, FISTA])
def test_intermediate_outputs_keep_autograd_graph_and_reset(solver_type):
    collector = IntermediateOutputCollector()
    solver = solver_type(callbacks=[collector])
    y = torch.ones(1, 2, requires_grad=True)
    solver(y, A=torch.eye(2), gamma=0.5, lam=0.0, t=4)
    assert len(collector.intermediates) == 4
    assert all(x.requires_grad for x in collector.intermediates)
    sum(x.square().sum() for x in collector.intermediates).backward()
    assert y.grad is not None and torch.all(y.grad > 0)

    solver(y.detach(), A=torch.eye(2), lam=0.0, t=2)
    assert len(collector.intermediates) == 2
    collector.reset()
    assert collector.intermediates == []
