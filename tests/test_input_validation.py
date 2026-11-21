"""Forward validates caller-prepared tensors before any solver iteration."""

import numpy as np
import pytest
import torch

from ucmf import FISTA, ISTA
from ucmf.optim import IterationCallback


@pytest.mark.parametrize('solver_type', [ISTA, FISTA])
@pytest.mark.parametrize('name', ['A', 'x0', 'lam', 'gamma'])
@pytest.mark.parametrize('mismatch', ['dtype', 'device', 'integer', 'complex', 'array'])
def test_forward_rejects_incompatible_inputs_before_callbacks(solver_type, name, mismatch):
    class MustNotRun(IterationCallback):
        def on_step_begin(self, solver, x, **kwargs):
            pytest.fail('Invalid inputs must be rejected before iteration begins.')

    inputs = {
        'y': torch.ones(2, 2),
        'A': torch.eye(2),
        'x0': torch.zeros(2, 2),
        'lam': torch.full((2,), 0.1),
        'gamma': torch.full((2,), 0.2),
    }
    value = inputs[name]
    saved = value.clone()
    if mismatch == 'dtype':
        inputs[name] = value.double()
        message = f'{name} must have dtype'
    elif mismatch == 'device':
        inputs[name] = value.to('meta')
        message = f'{name} must be on device'
    elif mismatch == 'integer':
        inputs[name] = value.long()
        message = f'{name} must be a real floating-point tensor'
    elif mismatch == 'complex':
        inputs[name] = value.to(torch.complex64)
        message = f'{name} must be real-valued'
    else:
        inputs[name] = value.numpy()
        message = f'{name} must be a real'

    with pytest.raises(ValueError, match=message):
        solver_type(callbacks=[MustNotRun()])(**inputs, t=1)
    torch.testing.assert_close(value, saved)
    assert value.dtype == torch.float32 and value.device.type == 'cpu'


@pytest.mark.parametrize('solver_type', [ISTA, FISTA])
@pytest.mark.parametrize('name', ['A', 'x0', 'lam', 'gamma'])
def test_forward_requires_explicit_array_preprocessing(solver_type, name):
    inputs = {'y': torch.ones(2, 2), 'A': torch.eye(2), 'lam': 0.1, 'gamma': 0.2}
    array = np.ones((2, 2) if name in ('A', 'x0') else (2,))
    inputs[name] = torch.as_tensor(array, dtype=inputs['y'].dtype, device=inputs['y'].device)
    assert solver_type()(**inputs, t=1).shape == (2, 2)


@pytest.mark.parametrize('name', ['A', 'x0', 'lam', 'gamma'])
def test_mismatches_are_rejected_even_for_zero_iterations(name):
    inputs = {'y': torch.ones(2, 2), 'A': torch.eye(2), 'lam': 0.1, 'gamma': 0.2}
    inputs[name] = torch.ones((2, 2) if name in ('A', 'x0') else (2,), dtype=torch.float64)
    with pytest.raises(ValueError, match=f'{name} must have dtype'):
        ISTA()(**inputs, t=0)


@pytest.mark.parametrize('mismatch', ['dtype', 'device', 'integer', 'array'])
def test_weighted_forward_rejects_incompatible_weights(weighted_solver_type, mismatch):
    W = torch.eye(2)
    if mismatch == 'dtype':
        W = W.double()
        message = 'W must have dtype'
    elif mismatch == 'device':
        W = W.to('meta')
        message = 'W must be on device'
    elif mismatch == 'integer':
        W = W.long()
        message = 'W must be a real floating-point tensor'
    else:
        W = W.numpy()
        message = 'W must be a real floating-point tensor'
    with pytest.raises(ValueError, match=message):
        weighted_solver_type()(torch.ones(2), A=torch.eye(2), W=W, lam=0.1, t=0)
