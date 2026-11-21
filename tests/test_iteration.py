"""Explicit loops retain scheduling, recurrent autograd, and callback behavior."""

from ucmf import FISTA, ISTA
from ucmf.optim import IntermediateOutputCollector, IterationCallback
from ucmf.optim.weights import prepare_gradient_weight

import pytest
import torch


def scheduled_reference(A, y, x0, gammas, thetas, *, accelerated, x_shift=0.0):
    """Independent textbook loop, including an optional begin-callback change."""
    x = x0.clone()
    z = x0.clone()
    tk = 1.0
    for gamma, theta in zip(gammas, thetas, strict=True):
        previous = x + x_shift
        point = z if accelerated else previous
        proposal = point + gamma * ((y - point @ A.T) @ A) / y.shape[1]
        next_x = proposal.sign() * (proposal.abs() - theta).clamp_min(0)
        if accelerated:
            next_tk = (1 + (1 + 4 * tk**2) ** 0.5) / 2
            z = next_x + (tk - 1) / next_tk * (next_x - previous)
            tk = next_tk
        x = next_x
    return x


@pytest.mark.parametrize('solver_type', [ISTA, FISTA])
def test_loop_keeps_warm_start_unchanged_and_resets_between_solves(solver_type):
    solver = solver_type(positive=False)
    A = torch.tensor([[1.0, 0.2], [0.1, 1.3]], dtype=torch.float64)
    y = torch.tensor([[1.0, 0.5]], dtype=A.dtype)
    x0 = torch.tensor([[0.3, 0.1]], dtype=A.dtype, requires_grad=True)
    saved = x0.detach().clone()
    expected = scheduled_reference(A, y, x0, [0.2] * 9, [0.002] * 9, accelerated=solver_type is FISTA)
    first = solver(y, x0=x0, A=A, gamma=0.2, lam=0.01, t=9)
    second = solver(y, x0=x0, A=A, gamma=0.2, lam=0.01, t=9)

    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(x0, saved)
    assert first.data_ptr() != x0.data_ptr()
    first.sum().backward()
    first_grad = x0.grad.clone()
    x0.grad = None
    second.sum().backward()
    torch.testing.assert_close(x0.grad, first_grad)


class ScheduledFISTA(FISTA):
    """Test-only ALFISTA-style subclass, not a migrated production model."""

    def __init__(self, callbacks=None):
        super().__init__(positive=False, callbacks=callbacks)
        self.gammas = torch.nn.Parameter(torch.tensor([0.1, 0.3, 0.2], dtype=torch.float64))
        self.thetas = torch.nn.Parameter(torch.tensor([0.01, 0.02, 0.015], dtype=torch.float64))

    def forward(self, y, x0=None, t=1, A=None, lam=None, gamma=None, W=None):
        y, x0, A, gamma, theta, operator = self._setup_forward(y, x0, A, lam, gamma, W=W)
        return self._iterate(y, x0, A, gamma, theta, t, Op_tensor=operator)

    def _setup_forward(self, y, x0, A, lam, gamma, W=None):
        y, x0, A, _ = self._prepare_forward_inputs(y, x0, A)
        return y, x0, A, None, None, prepare_gradient_weight(W, A, y)

    def _step_params(self, k):
        return self.gammas[k], self.thetas[k]

    def _iterate(self, y, x0, A_tensor, gamma_value, theta_value, t, Op_tensor=None):
        return self._iterate_accelerated(y, x0, A_tensor, t, self._step_params, Op_tensor=Op_tensor)


def test_scheduled_steps_match_reference_values_and_gradients():
    outputs = IntermediateOutputCollector()
    solver = ScheduledFISTA(callbacks=[outputs])
    A = torch.tensor([[1.0, 0.2], [0.1, 1.3]], dtype=torch.float64)
    y = torch.tensor([[1.0, 0.5]], dtype=A.dtype, requires_grad=True)
    actual = solver(y, A=A, t=3)
    expected = scheduled_reference(A, y, torch.zeros_like(y), solver.gammas, solver.thetas, accelerated=True)
    torch.testing.assert_close(actual, expected)

    variables = (solver.gammas, solver.thetas, y)
    actual_grads = torch.autograd.grad(actual.square().sum(), variables, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().sum(), variables)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)
    sum(x.square().sum() for x in outputs.intermediates).backward()
    assert solver.gammas.grad is not None and torch.all(solver.gammas.grad.abs() > 0)
    assert solver.thetas.grad is not None and torch.all(solver.thetas.grad.abs() > 0)
    assert len(outputs.intermediates) == 3
    assert list(solver.state_dict()) == ['gammas', 'thetas']


@pytest.mark.parametrize('solver_type', [ISTA, FISTA])
def test_own_recurrent_loop_keeps_graphs_and_resets_local_hidden_state(solver_type):
    visited = []

    class RecurrentSolver(solver_type):
        def __init__(self):
            super().__init__(positive=False)
            self.gain = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))

        def _iterate(self, y, x0, A_tensor, gamma_value, theta_value, t, Op_tensor=None):
            if Op_tensor is None:
                Op_tensor = A_tensor
            grad_func, shrink_func = self._iteration_functions()
            x = self._init_solver_vector(y, x0, A_tensor)
            z = self._init_solver_vector(y, x0, A_tensor) if solver_type is FISTA else x
            tk = 1.0
            hidden = y.new_zeros(y.shape[0], 1)
            cell = y.new_zeros(y.shape[0], 1)
            lam = theta_value / gamma_value

            self._start_iterations(y)
            for k in range(t):
                if not self._on_iteration_begin(k, x):
                    break
                visited.append((k, hidden, cell))
                point = z if solver_type is FISTA else x
                grad = grad_func(y, point, A_tensor, Op=Op_tensor)
                cell = torch.tanh(cell + self.gain * grad.mean(dim=1, keepdim=True))
                hidden = 0.5 * hidden + cell
                gamma = 0.1 + hidden.sigmoid() * 0.05
                theta = lam * gamma
                x_next = shrink_func(point + gamma * grad, theta)
                if solver_type is FISTA:
                    tk, z = self._nesterov_momentum(tk, x, x_next)
                x = x_next
                if not self._on_iteration_end(x, gamma, theta, A_tensor, y, W_tensor=Op_tensor):
                    break
            return x

    solver = RecurrentSolver()
    A = torch.tensor([[1.0, 0.2], [0.1, 1.3]], dtype=torch.float64)
    y = torch.tensor([[1.0, 0.5]], dtype=A.dtype)
    first = solver(y, A=A, gamma=0.2, lam=0.01, t=4)
    first.square().sum().backward()
    first_grad = solver.gain.grad.clone()
    solver.zero_grad()
    second = solver(y, A=A, gamma=0.2, lam=0.01, t=4)
    second.square().sum().backward()  # No reference to the first solve's freed graph.

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(solver.gain.grad, first_grad)
    assert first_grad.abs() > 0
    assert visited[1][1].grad_fn is not None
    torch.testing.assert_close(visited[0][1], torch.zeros(1, 1, dtype=A.dtype))
    torch.testing.assert_close(visited[4][1], torch.zeros(1, 1, dtype=A.dtype))
    assert visited[0][1] is not visited[4][1]
    assert list(solver.state_dict()) == ['gain']
    assert not any(name in vars(solver) for name in ('x', 'z', 'tk', 'hidden', 'cell'))


def test_schedule_only_runs_for_executed_steps():
    calls = []

    class RecordingSchedule(ScheduledFISTA):
        def _step_params(self, k):
            calls.append(k)
            return super()._step_params(k)

    class Stop(IterationCallback):
        def on_step_begin(self, solver, x, **kwargs):
            solver.stop = solver.iter == 2

    solver = RecordingSchedule(callbacks=[Stop()])
    y = torch.ones(2, dtype=torch.float64)
    for steps in (5, 0, 5):
        solver(y, A=torch.eye(2, dtype=y.dtype), t=steps)
    assert calls == [0, 1, 0, 1]


def test_existing_momentum_hook_can_disable_acceleration():
    class UnacceleratedFISTA(FISTA):
        def _nesterov_momentum(self, tk, x, x_next):
            return 1.0, x_next

    A = torch.tensor([[1.0, 0.2], [0.3, 1.3]])
    y = torch.tensor([[1.0, 0.5]])
    expected = ISTA(positive=False)(y, A=A, lam=0.01, gamma=0.2, t=9)
    actual = UnacceleratedFISTA(positive=False)(y, A=A, lam=0.01, gamma=0.2, t=9)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('solver_type', [ISTA, FISTA])
def test_gradient_and_callback_hooks_are_dispatched(solver_type):
    seen = []

    class CustomSolver(solver_type):
        def ls_grad(self, y, x, A, Op=None):
            seen.append('gradient')
            return 2 * super().ls_grad(y, x, A, Op=Op)

        def _start_iterations(self, y):
            seen.append('start')
            super()._start_iterations(y)

        def _on_iteration_begin(self, k, x):
            seen.append('begin')
            return super()._on_iteration_begin(k, x)

        def _on_iteration_end(self, x, gamma, theta, A_tensor, y, W_tensor=None):
            seen.append('end')
            return super()._on_iteration_end(x, gamma, theta, A_tensor, y, W_tensor=W_tensor)

    A = torch.eye(2)
    y = torch.ones(1, 2)
    actual = CustomSolver()(y, A=A, lam=0.0, gamma=0.2, t=3)
    expected = solver_type()(y, A=A, lam=0.0, gamma=0.4, t=3)
    torch.testing.assert_close(actual, expected)
    assert seen == ['start', 'begin', 'gradient', 'end', 'begin', 'gradient', 'end', 'begin', 'gradient', 'end']


@pytest.mark.parametrize('solver_type', [ISTA, FISTA])
def test_begin_callbacks_can_adjust_iterates(solver_type):
    class Shift(IterationCallback):
        def on_step_begin(self, solver, x, **kwargs):
            x.add_(0.05)

    A = torch.tensor([[1.0, 0.2], [0.3, 1.3]], dtype=torch.float64)
    y = torch.tensor([[1.0, 0.5]], dtype=A.dtype)
    with torch.no_grad():
        actual = solver_type(positive=False, callbacks=[Shift()])(y, A=A, lam=0.01, gamma=0.2, t=5)
    expected = scheduled_reference(
        A, y, torch.zeros_like(y), [0.2] * 5, [0.002] * 5, accelerated=solver_type is FISTA, x_shift=0.05
    )
    torch.testing.assert_close(actual, expected)
