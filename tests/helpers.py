"""Plain test helpers: callbacks, operator inspection, and dense reference iterations."""

import math

from ucmf.optim import IterationCallback
from ucmf.optim.neural_augmentation import NeuralAugmentation
from ucmf.optim.operator import CMFDictionary, CMFPropagation

import torch
import torch.nn.functional as F


class Recorder(IterationCallback):
    """Record ``(boundary, iteration)`` events and the keywords of every end callback."""

    def __init__(self):
        self.events = []
        self.end_kwargs = []

    def on_step_begin(self, solver, x, **kwargs):
        self.events.append(('begin', solver.iter))

    def on_step_end(self, solver, x, **kwargs):
        self.events.append(('end', solver.iter))
        self.end_kwargs.append(kwargs)


class StopAt(IterationCallback):
    """Request a stop at the ``'begin'`` or ``'end'`` boundary of one iteration."""

    def __init__(self, iteration, boundary):
        self.iteration = iteration
        self.boundary = boundary

    def on_step_begin(self, solver, x, **kwargs):
        if self.boundary == 'begin' and solver.iter == self.iteration:
            solver.stop = True

    def on_step_end(self, solver, x, **kwargs):
        if self.boundary == 'end' and solver.iter == self.iteration:
            solver.stop = True


class MustNotRun(IterationCallback):
    """Fail if an iteration begins, e.g. for inputs that must be rejected beforehand."""

    def on_step_begin(self, solver, x, **kwargs):
        raise AssertionError('Invalid inputs must be rejected before iteration begins.')


def assert_operator_matches(operator, matrix, x, r):
    """``forward`` and ``adjoint`` act like the dense ``matrix`` (shared or per problem) on ``x`` / ``r``."""
    torch.testing.assert_close(operator.forward(x), (matrix @ x.unsqueeze(-1)).squeeze(-1))
    torch.testing.assert_close(operator.adjoint(r), (matrix.mT @ r.unsqueeze(-1)).squeeze(-1))


def r_diag_of(operator):
    """Diagonal setting a model needs for ``operator``; dense problems use the full CSM."""
    return isinstance(operator, (CMFDictionary, CMFPropagation)) and operator.r_diag


def operator_source(operator):
    """The tensor the operator is built from."""
    return operator.H if isinstance(operator, (CMFDictionary, CMFPropagation)) else operator.matrix


def scheduled_step_params(gammas, thetas):
    """Step parameters ``gammas[k]``, ``thetas[k]`` for ``reference_iterates``."""
    return 0, lambda residual, direction, k: (gammas[k], thetas[k], k + 1)


def model_step_params(model, batch_size):
    """Step parameters of a learned or NA model for ``reference_iterates``.

    NA: the controller is fed independently computed features of the residual
    and the direction.
    """
    if not isinstance(model, NeuralAugmentation):
        return scheduled_step_params(model.gamma, model.theta)
    controller = model.controller

    def step(residual, direction, state):
        features = torch.stack((residual.abs().sum(1) / residual.shape[1], direction.abs().sum(1)), dim=1)
        features = (features - controller.feature_mean) / controller.feature_std
        hidden, cell = controller.lstm(features, state)
        gamma, theta = F.softplus(controller.linear(torch.relu(controller.lll(cell)))).split(1, dim=1)
        return gamma, theta, (hidden, cell)

    return (controller.hidden_init.expand(batch_size, -1), controller.cell_init.expand(batch_size, -1)), step


def reference_iterates(M, P, Q, A, y, x0, t, step_params, *, positive=False, accelerated=True, x_shift=0.0):
    """Independent dense (F)ISTA iteration; returns all ``t`` iterates ``(t, B, n)``.

    The direction at the current point ``p`` is ``(M.T @ y - P.T @ (Q @ p)) / m``.
    ``step_params = (state, step)`` with ``step(residual, direction, state) ->
    (gamma, theta, state)`` gets the physical residual ``y - A p``. ``x_shift``
    emulates a begin callback that adds it to ``x`` in place before each step.
    """
    measurement = (M.mT @ y.unsqueeze(-1)).squeeze(-1)
    state, step = step_params
    x, z, tk, iterates = x0, x0, 1.0, []
    for _ in range(t):
        x = x + x_shift
        point = z if accelerated else x
        feedback = (P.mT @ (Q @ point.unsqueeze(-1))).squeeze(-1)
        direction = (measurement - feedback) / y.shape[-1]
        residual = y - (A @ point.unsqueeze(-1)).squeeze(-1)
        gamma, theta, state = step(residual, direction, state)
        proposal = point + gamma * direction
        if positive:
            x_next = torch.relu(proposal - theta)
        else:
            x_next = proposal.sign() * torch.relu(proposal.abs() - theta)
        if accelerated:
            t_next = (1 + math.sqrt(1 + 4 * tk**2)) / 2
            z = x_next + (tk - 1) / t_next * (x_next - x)
            tk = t_next
        x = x_next
        iterates.append(x)
    return torch.stack(iterates)
