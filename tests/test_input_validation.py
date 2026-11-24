"""Forward validates caller-prepared operator, lam, and gamma; x0 is tested for all models in test_models."""

from ucmf import FISTA, ISTA
from ucmf.optim.operator import MatrixMult

from helpers import MustNotRun

import numpy as np
import pytest
import torch


@pytest.mark.parametrize('solver_type', [ISTA, FISTA])
@pytest.mark.parametrize('name', ['operator', 'lam', 'gamma'])
@pytest.mark.parametrize('mismatch', ['dtype', 'device', 'integer', 'complex', 'array'])
def test_forward_rejects_incompatible_inputs_before_callbacks(solver_type, name, mismatch):
    inputs = {
        'y': torch.ones(2, 2),
        'operator': MatrixMult(torch.eye(2)),
        'lam': torch.full((2,), 0.1),
        'gamma': torch.full((2,), 0.2),
    }
    value = inputs[name].matrix if name == 'operator' else inputs[name]
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

    if name == 'operator':
        if mismatch in ('dtype', 'device'):
            inputs[name] = MatrixMult(inputs[name])
        else:
            message = 'operator must be a LinearOperator'
    with pytest.raises(ValueError, match=message):
        solver_type(callbacks=[MustNotRun()])(**inputs, t=1)
    torch.testing.assert_close(value, saved)
    assert value.dtype == torch.float32 and value.device.type == 'cpu'


@pytest.mark.parametrize('solver_type', [ISTA, FISTA])
@pytest.mark.parametrize('name', ['operator', 'x0', 'lam', 'gamma'])
def test_forward_requires_explicit_array_preprocessing(solver_type, name):
    inputs = {'y': torch.ones(2, 2), 'operator': MatrixMult(torch.eye(2)), 'lam': 0.1, 'gamma': 0.2}
    array = np.ones((2, 2) if name in ('operator', 'x0') else (2,))
    prepared = torch.as_tensor(array, dtype=inputs['y'].dtype, device=inputs['y'].device)
    inputs[name] = MatrixMult(prepared) if name == 'operator' else prepared
    assert solver_type()(**inputs, t=1).shape == (2, 2)


@pytest.mark.parametrize('name', ['operator', 'lam', 'gamma'])
def test_mismatches_are_rejected_even_for_zero_iterations(name):
    inputs = {'y': torch.ones(2, 2), 'operator': MatrixMult(torch.eye(2)), 'lam': 0.1, 'gamma': 0.2}
    value = torch.ones((2, 2) if name == 'operator' else (2,), dtype=torch.float64)
    inputs[name] = MatrixMult(value) if name == 'operator' else value
    with pytest.raises(ValueError, match=f'{name} must have dtype'):
        ISTA()(**inputs, t=0)
