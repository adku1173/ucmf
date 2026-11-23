"""Single-weight Ada-LFISTA specifics; shared properties are in test_unfolded_models."""

from ucmf import AdaLFISTASW
from ucmf.optim.operator import CMFPropagation

import pytest
import torch


def test_parameter_names():
    model = AdaLFISTASW(2, 3, 0.1, 0.01)
    assert [name for name, _ in model.named_parameters()] == ['B', '_gamma', '_theta']


def test_precomputed_rejects_matrix_free_operator():
    operator = CMFPropagation(torch.ones(2, 3, dtype=torch.complex64))
    model = AdaLFISTASW(1, 4, 0.2, 0.01, weight_mode='precomputed')
    with pytest.raises(ValueError, match="'precomputed' requires a dense"):
        model(torch.ones(4), operator=operator)


def test_rejects_unknown_weight_mode():
    with pytest.raises(ValueError, match='weight_mode'):
        AdaLFISTASW(1, 4, 0.2, 0.01, weight_mode='auto')


def test_rejects_auxiliary_weight_of_wrong_shape():
    with pytest.raises(ValueError, match='B must have shape'):
        AdaLFISTASW(n_iterations=2, n_measurements=3, gamma_init=0.1, theta_init=0.01, B=torch.eye(4))
