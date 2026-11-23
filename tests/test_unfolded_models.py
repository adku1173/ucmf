"""Properties shared by all unfolded models (UnfoldedModelCases x ProblemCases)."""

from ucmf import FISTA
from ucmf.optim import IntermediateOutputCollector, IterationCallback
from ucmf.optim.operator import CMFPropagation, MatrixMult
from ucmf.optim.utils import softplus_inverse

import pytest
import torch
import torch.nn.functional as F
from pytest_cases import parametrize_with_cases
from unfolded_model_cases import (
    N_MICS,
    N_SOURCES,
    ProblemCases,
    T,
    UnfoldedModelCases,
    build_model,
    dense_matrix,
    operator_source,
    perturb_weights,
    r_diag_of,
    reference_iterates,
)

MODEL = 'make_model, dense_weights'
PROBLEM = 'operator, y, x0'


class Recorder(IterationCallback):
    def __init__(self):
        self.seen = {}

    def on_step_end(self, solver, x, **kwargs):
        self.seen = kwargs


def assert_iterates_match_reference(make_model, dense_weights, operator, y, x0, positive):
    model = perturb_weights(build_model(make_model, r_diag_of(operator), positive=positive))
    collector = IntermediateOutputCollector()
    model.callbacks = [collector]
    with torch.no_grad():
        model(y, x0=x0, operator=operator, t=T)
        expected = reference_iterates(*dense_weights(model, operator), y, x0, model.gamma, model.theta, positive)
    torch.testing.assert_close(torch.stack(collector.intermediates), expected)


def assert_gradients_match_reference(make_model, dense_weights, operator, y, x0):
    model = perturb_weights(build_model(make_model, r_diag_of(operator)))
    params = tuple(model.parameters())
    actual = model(y, x0=x0, operator=operator, t=T)
    *actual_grads, source_grad = torch.autograd.grad(
        actual.square().sum(), (*params, operator_source(operator)), allow_unused=True
    )
    assert source_grad is None  # the operator is fixed
    expected = reference_iterates(*dense_weights(model, operator), y, x0, model.gamma, model.theta, False)[-1]
    expected_grads = torch.autograd.grad(expected.square().sum(), params)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)
        assert torch.count_nonzero(expected_grad)


def assert_operator_matches(operator, matrix, x, r):
    torch.testing.assert_close(operator.forward(x), (matrix @ x.unsqueeze(-1)).squeeze(-1))
    torch.testing.assert_close(operator.adjoint(r), (matrix.mT @ r.unsqueeze(-1)).squeeze(-1))


@pytest.mark.parametrize('positive', [False, True], ids=['signed', 'positive'])
@parametrize_with_cases(PROBLEM, cases=ProblemCases, has_tag='cmf')
@parametrize_with_cases(MODEL, cases=UnfoldedModelCases)
def test_iterates_match_reference(make_model, dense_weights, operator, y, x0, positive):
    assert_iterates_match_reference(make_model, dense_weights, operator, y, x0, positive)


@parametrize_with_cases(PROBLEM, cases=ProblemCases, has_tag='cmf')
@parametrize_with_cases(MODEL, cases=UnfoldedModelCases)
def test_gradients_match_reference(make_model, dense_weights, operator, y, x0):
    assert_gradients_match_reference(make_model, dense_weights, operator, y, x0)


@parametrize_with_cases(PROBLEM, cases=ProblemCases, has_tag='rows')
@parametrize_with_cases(MODEL, cases=UnfoldedModelCases, has_tag='dense')
def test_row_subsets_match_reference(make_model, dense_weights, operator, y, x0):
    assert_iterates_match_reference(make_model, dense_weights, operator, y, x0, positive=False)
    assert_gradients_match_reference(make_model, dense_weights, operator, y, x0)


@pytest.mark.parametrize('positive', [False, True], ids=['signed', 'positive'])
@parametrize_with_cases(PROBLEM, cases=ProblemCases, has_tag='cmf')
@parametrize_with_cases(MODEL, cases=UnfoldedModelCases)
def test_identity_weights_reproduce_fista(make_model, dense_weights, operator, y, x0, positive):
    """Identity auxiliary weights and constant gamma/theta = lam * gamma give FISTA."""
    lam, gamma = 0.03, 0.12
    model = build_model(make_model, r_diag_of(operator), positive=positive, gamma_init=gamma, theta_init=lam * gamma)
    with torch.no_grad():  # set in float64 so the softplus round-trip is exact
        model._gamma.fill_(softplus_inverse(gamma))
        model._theta.fill_(softplus_inverse(lam * gamma))
    expected = FISTA(positive=positive)(y, x0=x0, operator=operator, lam=lam, gamma=gamma, t=T)
    actual = model(y, x0=x0, operator=operator, t=T)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@parametrize_with_cases(PROBLEM, cases=ProblemCases, has_tag='cmf')
@parametrize_with_cases(MODEL, cases=UnfoldedModelCases)
def test_callbacks_receive_physical_and_effective_operator(make_model, dense_weights, operator, y, x0):
    model = perturb_weights(build_model(make_model, r_diag_of(operator)))
    recorder = Recorder()
    model.callbacks = [recorder]
    with torch.no_grad():
        model(y, x0=x0, operator=operator, t=T)
        M, _, _ = dense_weights(model, operator)
        assert_operator_matches(recorder.seen['A'], dense_matrix(operator), x0, y)
        assert_operator_matches(recorder.seen['W'], M, x0, y)


@parametrize_with_cases(PROBLEM, cases=ProblemCases, has_tag='cmf')
@parametrize_with_cases(MODEL, cases=UnfoldedModelCases, has_tag='two_weights')
def test_callbacks_receive_recurrent_operator(make_model, dense_weights, operator, y, x0):
    model = perturb_weights(build_model(make_model, r_diag_of(operator)))
    recorder = Recorder()
    model.callbacks = [recorder]
    with torch.no_grad():
        model(y, x0=x0, operator=operator, t=T)
        _, P, _ = dense_weights(model, operator)
        assert_operator_matches(recorder.seen['W_recurrent'], P, x0, y)


@parametrize_with_cases(MODEL, cases=UnfoldedModelCases)
def test_applied_step_parameters_are_softplus_of_raw_parameters(make_model, dense_weights):
    model = build_model(make_model, gamma_init=0.3, theta_init=0.01)
    torch.testing.assert_close(model.gamma, F.softplus(model._gamma))
    torch.testing.assert_close(model.theta, F.softplus(model._theta))
    torch.testing.assert_close(model.gamma, torch.full((T,), 0.3, dtype=torch.float64), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(model.theta, torch.full((T,), 0.01, dtype=torch.float64), rtol=1e-6, atol=1e-6)


@parametrize_with_cases(PROBLEM, cases=ProblemCases, has_tag='dictionary')
@parametrize_with_cases(MODEL, cases=UnfoldedModelCases, has_tag='weight_mode')
def test_precomputed_weights_match_factored_weights(make_model, dense_weights, operator, y, x0):
    factored = perturb_weights(build_model(make_model, r_diag_of(operator)))
    precomputed = build_model(make_model, r_diag_of(operator), weight_mode='precomputed')
    precomputed.load_state_dict(factored.state_dict())
    expected = factored(y, x0=x0, operator=operator, t=T)
    actual = precomputed(y, x0=x0, operator=operator, t=T)
    torch.testing.assert_close(actual, expected)
    expected_grads = torch.autograd.grad(expected.square().sum(), tuple(factored.parameters()))
    actual_grads = torch.autograd.grad(actual.square().sum(), tuple(precomputed.parameters()))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


@parametrize_with_cases(MODEL, cases=UnfoldedModelCases)
def test_constructor_validation(make_model, dense_weights):
    with pytest.raises(ValueError, match='n_iterations must be positive'):
        make_model(N_MICS, False, n_iterations=0, gamma_init=0.1, theta_init=0.01)
    with pytest.raises(ValueError, match='n_measurements must be positive|n_mics must exceed'):
        make_model(0, False, n_iterations=2, gamma_init=0.1, theta_init=0.01)


@pytest.mark.parametrize(('t', 'message'), [(-1, 'non-negative'), (T + 1, 'cannot exceed n_iterations')])
@parametrize_with_cases(MODEL, cases=UnfoldedModelCases)
def test_forward_rejects_invalid_depth(make_model, dense_weights, t, message):
    model = build_model(make_model)
    operator = CMFPropagation(torch.ones(N_MICS, N_SOURCES, dtype=torch.complex128))
    with pytest.raises(ValueError, match=message):
        model(torch.ones(1, operator.shape[0], dtype=torch.float64), operator=operator, t=t)


@parametrize_with_cases(MODEL, cases=UnfoldedModelCases)
def test_forward_rejects_operator_of_another_array_size(make_model, dense_weights):
    model = build_model(make_model)
    operator = CMFPropagation(torch.ones(N_MICS + 1, N_SOURCES, dtype=torch.complex128))
    with pytest.raises(ValueError, match='but Br? expects'):
        model(torch.ones(1, operator.shape[0], dtype=torch.float64), operator=operator)


@pytest.mark.parametrize(
    ('idx', 'batch', 'message'),
    [
        (torch.tensor([0.0, 1.0, 2.0, 3.0]), None, 'integer tensor'),
        (torch.tensor([0, 1, 2]), None, 'must match the measurement dimension'),
        (torch.zeros(2, 2, 2, dtype=torch.long), None, '1D .*or 2D'),
        (torch.tensor([[0, 1, 2, 3]]), None, 'requires a batched operator'),
        (torch.zeros(3, 4, dtype=torch.long), 2, 'batch size 3 but the operator has 2'),
    ],
)
@parametrize_with_cases(MODEL, cases=UnfoldedModelCases, has_tag='dense')
def test_row_subset_validation(make_model, dense_weights, idx, batch, message):
    model = build_model(make_model)
    A = torch.randn(4, 4, dtype=torch.float64) if batch is None else torch.randn(batch, 4, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match=message):
        model(torch.randn(batch or 1, 4, dtype=torch.float64), operator=MatrixMult(A, rows=idx), t=T)


@parametrize_with_cases(MODEL, cases=UnfoldedModelCases, has_tag='cmf_only')
def test_rejects_operator_without_cmf_coordinates(make_model, dense_weights):
    model = build_model(make_model)
    m = N_MICS * N_MICS
    with pytest.raises(ValueError, match='CMFDictionary or CMFPropagation'):
        model(torch.ones(1, m, dtype=torch.float64), operator=MatrixMult(torch.eye(m, dtype=torch.float64)))


@parametrize_with_cases(MODEL, cases=UnfoldedModelCases, has_tag='cmf_only')
def test_rejects_mismatched_diagonal_setting(make_model, dense_weights):
    model = build_model(make_model, r_diag=False)
    operator = CMFPropagation(torch.ones(N_MICS, N_SOURCES, dtype=torch.complex128), r_diag=True)
    with pytest.raises(ValueError, match='r_diag'):
        model(torch.ones(1, operator.shape[0], dtype=torch.float64), operator=operator)
