"""Weighted steps belong to subclasses that explicitly override forward."""

import math

from ucmf import FISTA, ISTA
from ucmf.optim import IterationCallback
from ucmf.optim.utils import default_gamma_from_A

import numpy as np
import pytest
import torch


def weighted_numpy_reference(A, W, y, x0, lam, gamma, *, positive, accelerated):
    """Independent weighted-update oracle, not a classical LASSO solver.

    The default step is taken from ucmf (power-iteration estimate for the whole A).
    """
    default_steps = np.broadcast_to(default_gamma_from_A(torch.from_numpy(A)).numpy(), (len(y),))
    solutions = []
    for sample, observation in enumerate(y):
        matrix = A if A.ndim == 2 else A[sample]
        weight = W if W.ndim == 2 else W[sample]
        step = default_steps[sample] if gamma is None else gamma[sample]
        x = x0[sample].copy()
        z = x.copy()
        tk = 1.0
        for _ in range(9):
            point = z if accelerated else x
            proposal = point + (step / observation.size) * (weight.T @ (observation - matrix @ point))
            threshold = step * lam[sample]
            next_x = (
                np.maximum(proposal - threshold, 0)
                if positive
                else np.sign(proposal) * np.maximum(np.abs(proposal) - threshold, 0)
            )
            if accelerated:
                next_tk = (1 + math.sqrt(1 + 4 * tk**2)) / 2
                z = next_x + (tk - 1) / next_tk * (next_x - x)
                tk = next_tk
            x = next_x
        solutions.append(x)
    return np.stack(solutions)


@pytest.mark.parametrize('positive', [True, False])
@pytest.mark.parametrize('batched_A', [True, False])
@pytest.mark.parametrize('batched_W', [True, False])
@pytest.mark.parametrize('default_gamma', [True, False])
def test_weighted_subclass_matches_numpy_reference(weighted_solver_type, positive, batched_A, batched_W, default_gamma):
    rng = np.random.default_rng(17)
    A = rng.normal(size=(3, 5, 4) if batched_A else (5, 4))
    W = rng.normal(size=(3, 5, 4) if batched_W else (5, 4))
    y = rng.normal(size=(3, 5))
    x0 = rng.normal(size=(3, 4))
    lam = np.array([0.0, 0.015, 0.03])
    gamma = None if default_gamma else np.array([0.07, 0.11, 0.15])

    actual = weighted_solver_type(positive=positive)(
        torch.from_numpy(y),
        x0=torch.from_numpy(x0),
        A=torch.from_numpy(A),
        W=torch.from_numpy(W),
        lam=torch.from_numpy(lam),
        gamma=None if gamma is None else torch.from_numpy(gamma),
        t=9,
    )
    expected = weighted_numpy_reference(
        A, W, y, x0, lam, gamma, positive=positive, accelerated=issubclass(weighted_solver_type, FISTA)
    )
    torch.testing.assert_close(actual, torch.from_numpy(expected), rtol=1e-11, atol=1e-12)


def test_weighted_forward_can_default_to_the_classical_adjoint(weighted_solver_type):
    classical_type = FISTA if issubclass(weighted_solver_type, FISTA) else ISTA
    A = torch.tensor([[1.0, 0.2], [0.3, 1.3]])
    y = torch.tensor([[1.0, 0.5]])
    actual = weighted_solver_type()(y, A=A, lam=0.01, t=9)
    expected = classical_type()(y, A=A, lam=0.01, t=9)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_weighted_subclass_preserves_autograd(weighted_solver_type):
    A = torch.tensor([[1.0, 0.2], [0.1, 1.1], [0.3, -0.1]], dtype=torch.float64)
    y = torch.tensor([[1.2, 0.8, 0.2]], dtype=A.dtype, requires_grad=True)
    x0 = torch.tensor([[0.3, 0.2]], dtype=A.dtype, requires_grad=True)
    W = A.clone().requires_grad_()
    lam = torch.tensor(0.01, dtype=A.dtype, requires_grad=True)
    gamma = torch.tensor(0.2, dtype=A.dtype, requires_grad=True)
    solver = weighted_solver_type(positive=False)
    assert torch.autograd.gradcheck(
        lambda y, x0, W, lam, gamma: solver(y, x0=x0, A=A, W=W, lam=lam, gamma=gamma, t=4),
        (y, x0, W, lam, gamma),
    )


def test_subclass_does_not_register_external_weights_as_its_parameters(weighted_solver_type):
    A = torch.eye(2, requires_grad=True)
    W = torch.nn.Parameter(2 * torch.eye(2))
    solver = weighted_solver_type()
    solver(torch.ones(2), A=A, W=W, gamma=0.2, lam=0.0, t=3).sum().backward()
    assert W.grad is not None and torch.isfinite(W.grad).all()
    assert A.grad is None
    assert list(solver.parameters()) == []


@pytest.mark.parametrize(
    ('W', 'message'),
    [
        (torch.ones(2), 'W must be 2D or 3D'),
        (torch.ones(3, 2), 'operator shape'),
        (torch.ones(3, 2, 2), 'Batched W has batch size'),
        (torch.ones(2, 2, dtype=torch.complex64), 'W must be real-valued'),
    ],
)
def test_subclass_forward_validates_its_weights(weighted_solver_type, W, message):
    with pytest.raises(ValueError, match=message):
        weighted_solver_type()(torch.ones(2, 2), A=torch.eye(2), W=W, lam=0.1)


def test_weighted_forward_reports_its_effective_adjoint_to_callbacks(weighted_solver_type):
    seen = []
    W = 2 * torch.eye(2)

    class Recorder(IterationCallback):
        def on_step_end(self, solver, x, **kwargs):
            seen.append(kwargs['W'])

    weighted_solver_type(callbacks=[Recorder()])(torch.ones(2), A=torch.eye(2), W=W, gamma=0.2, lam=0.01, t=3)
    assert len(seen) == 3
    for operator in seen:
        torch.testing.assert_close(operator, W)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is not available')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_weighted_subclass_preserves_cuda_and_batched_gradients(weighted_solver_type, dtype):
    rng = torch.Generator().manual_seed(42)
    A = torch.randn(2, 5, 3, generator=rng, dtype=dtype)
    y = torch.randn(2, 5, generator=rng, dtype=dtype)
    lam = torch.tensor([0.01, 0.03], dtype=dtype)
    solver = weighted_solver_type(positive=False)
    expected = solver(y, A=A, W=A, lam=lam, t=15)
    y_cuda = y.cuda().requires_grad_()
    W_cuda = A.cuda().requires_grad_()
    actual = solver(y_cuda, A=A.cuda(), W=W_cuda, lam=lam.cuda(), t=15)
    actual.square().sum().backward()
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-6)
    assert y_cuda.grad is not None and torch.isfinite(y_cuda.grad).all()
    assert W_cuda.grad is not None and torch.isfinite(W_cuda.grad).all()
