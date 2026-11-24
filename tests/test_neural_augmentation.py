"""Neural-augmentation specifics for every NA model (ModelCases with tag ``na``)."""

from ucmf.optim import StepParamCollector
from ucmf.optim.neural_augmentation import _StepParamLSTM

from cases.models import ModelCases, T, build_model, dense_matrix, perturb_weights
from cases.problems import BATCH, NUM_MICS, cmf_problem

import pytest
import torch
from pytest_cases import parametrize_with_cases

SPEC = 'spec'


@parametrize_with_cases(SPEC, cases=ModelCases, has_tag='na')
def test_outputs_are_nan_until_normalizers_are_initialized(spec):
    operator, y, x0 = cmf_problem()
    model = spec.make_model(NUM_MICS, False, **spec.model_kwargs).double()
    assert torch.isnan(spec.solve(model, y, operator, x0)).all()


@pytest.mark.parametrize('warm_start', [False, True], ids=['zeros', 'warm-start'])
@parametrize_with_cases(SPEC, cases=ModelCases, has_tag='na')
def test_init_normalizers_uses_first_iteration_features(spec, warm_start):
    operator, y, x0 = cmf_problem()
    model = perturb_weights(build_model(spec))
    model.init_normalizers(y, operator, x0=x0 if warm_start else None)
    start = x0 if warm_start else torch.zeros_like(x0)
    with torch.no_grad():
        M, P, Q = spec.dense_weights(model, operator)
        residual = y - (dense_matrix(operator) @ start.unsqueeze(-1)).squeeze(-1)
        direction = ((M.mT @ y.unsqueeze(-1)) - P.mT @ (Q @ start.unsqueeze(-1))).squeeze(-1) / y.shape[1]
    features = torch.stack((residual.abs().sum(1) / y.shape[1], direction.abs().sum(1)), dim=1)
    torch.testing.assert_close(model.controller.feature_mean, features.mean(0))
    torch.testing.assert_close(model.controller.feature_std, features.std(0, unbiased=False))


def test_features_are_mean_absolute_residual_and_direction_l1_norm():
    residual = torch.tensor([[1.0, -2.0], [0.0, 4.0]])
    direction = torch.tensor([[3.0, -4.0, 0.5], [0.0, 0.0, -1.0]])
    expected = torch.tensor([[1.5, 7.5], [2.0, 1.0]])
    torch.testing.assert_close(_StepParamLSTM.features(residual, direction), expected)


@parametrize_with_cases(SPEC, cases=ModelCases, has_tag='na')
def test_depth_is_not_limited(spec):
    operator, y, x0 = cmf_problem()
    assert spec.solve(build_model(spec), y, operator, x0, t=T + 20).shape == x0.shape


@parametrize_with_cases(SPEC, cases=ModelCases, has_tag='na')
def test_initial_controller_predicts_constant_per_sample_step_parameters(spec):
    operator, y, x0 = cmf_problem()
    collector = StepParamCollector()
    model = build_model(spec)
    model.callbacks = [collector]
    spec.solve(model, y, operator, x0)
    # The readout weight starts at zero, so every prediction equals the init.
    torch.testing.assert_close(collector.gammas_tensor, torch.full((T, BATCH, 1), 0.3, dtype=y.dtype))
    torch.testing.assert_close(collector.thetas_tensor, torch.full((T, BATCH, 1), 0.01, dtype=y.dtype))
