"""Properties shared by all models (ModelCases x ProblemCases)."""

from ucmf import FISTA
from ucmf.optim import IntermediateOutputCollector, IterationCallback
from ucmf.optim.operator import CMFPropagation, MatrixMult

from cases.models import ModelCases, T, build_model, dense_matrix, perturb_weights, set_constant_step_params
from cases.problems import NUM_MICS, NUM_SOURCES, ProblemCases, cmf_problem
from helpers import MustNotRun, StopAt, assert_operator_matches, operator_source, r_diag_of, reference_iterates

import pytest
import torch
import torch.nn.functional as F
from pytest_cases import parametrize_with_cases

SPEC = 'spec'
PROBLEM = 'operator, y, x0'


def reference(spec, model, operator, y, x0, nonnegative=False, x_shift=0.0):
    """All ``T`` iterates of the independent dense reference for ``model``."""
    return reference_iterates(
        *spec.dense_weights(model, operator), dense_matrix(operator), y, x0, T,
        spec.step_params(model, y.shape[0]), nonnegative=nonnegative, accelerated=spec.accelerated, x_shift=x_shift,
    )


def assert_iterates_match_reference(spec, operator, y, x0, nonnegative):
    model = perturb_weights(build_model(spec, r_diag_of(operator), nonnegative=nonnegative))
    collector = IntermediateOutputCollector()
    model.callbacks = [collector]
    with torch.no_grad():
        spec.solve(model, y, operator, x0)
        expected = reference(spec, model, operator, y, x0, nonnegative)
    torch.testing.assert_close(torch.stack(collector.intermediates), expected)


def assert_gradients_match_reference(spec, operator, y, x0):
    """Gradients in the model parameters and ``y`` match; the operator stays fixed."""
    model = perturb_weights(build_model(spec, r_diag_of(operator)))
    y = y.clone().requires_grad_()
    variables = (*model.parameters(), y)
    actual = spec.solve(model, y, operator, x0)
    *actual_grads, source_grad = torch.autograd.grad(
        actual.square().sum(), (*variables, operator_source(operator)), allow_unused=True
    )
    assert source_grad is None
    expected = reference(spec, model, operator, y, x0)[-1]
    expected_grads = torch.autograd.grad(expected.square().sum(), variables)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)
        assert torch.count_nonzero(expected_grad)


@pytest.mark.parametrize('nonnegative', [False, True], ids=['signed', 'nonnegative'])
@parametrize_with_cases(PROBLEM, cases=ProblemCases, has_tag='cmf')
@parametrize_with_cases(SPEC, cases=ModelCases)
def test_iterates_match_reference(spec, operator, y, x0, nonnegative):
    assert_iterates_match_reference(spec, operator, y, x0, nonnegative)


@parametrize_with_cases(PROBLEM, cases=ProblemCases, has_tag='cmf')
@parametrize_with_cases(SPEC, cases=ModelCases)
def test_gradients_match_reference(spec, operator, y, x0):
    assert_gradients_match_reference(spec, operator, y, x0)


@parametrize_with_cases(PROBLEM, cases=ProblemCases, has_tag='rows')
@parametrize_with_cases(SPEC, cases=ModelCases, has_tag='dense')
def test_row_subsets_match_reference(spec, operator, y, x0):
    assert_iterates_match_reference(spec, operator, y, x0, nonnegative=False)
    assert_gradients_match_reference(spec, operator, y, x0)


@pytest.mark.parametrize('nonnegative', [False, True], ids=['signed', 'nonnegative'])
@parametrize_with_cases(PROBLEM, cases=ProblemCases, has_tag='cmf')
@parametrize_with_cases(SPEC, cases=ModelCases, has_tag='unfolded')
def test_identity_weights_reproduce_fista(spec, operator, y, x0, nonnegative):
    """Identity auxiliary weights and constant gamma/theta = lam * gamma give FISTA."""
    lam, gamma = 0.03, 0.12
    model = build_model(spec, r_diag_of(operator), nonnegative=nonnegative)
    set_constant_step_params(model, gamma, lam * gamma)  # in float64 so the softplus round-trip is exact
    expected = FISTA(nonnegative=nonnegative)(y, x0=x0, operator=operator, lam=lam, gamma=gamma, t=T)
    actual = spec.solve(model, y, operator, x0)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@parametrize_with_cases(PROBLEM, cases=ProblemCases, has_tag='cmf')
@parametrize_with_cases(SPEC, cases=ModelCases)
def test_callbacks_receive_physical_and_effective_operator(spec, operator, y, x0, recorder):
    model = perturb_weights(build_model(spec, r_diag_of(operator)))
    model.callbacks = [recorder]
    with torch.no_grad():
        spec.solve(model, y, operator, x0)
        M, _, _ = spec.dense_weights(model, operator)
        assert_operator_matches(recorder.end_kwargs[-1]['A'], dense_matrix(operator), x0, y)
        assert_operator_matches(recorder.end_kwargs[-1]['W'], M, x0, y)


@parametrize_with_cases(PROBLEM, cases=ProblemCases, has_tag='cmf')
@parametrize_with_cases(SPEC, cases=ModelCases, has_tag='two_weights')
def test_callbacks_receive_recurrent_operator(spec, operator, y, x0, recorder):
    model = perturb_weights(build_model(spec, r_diag_of(operator)))
    model.callbacks = [recorder]
    with torch.no_grad():
        spec.solve(model, y, operator, x0)
        _, P, _ = spec.dense_weights(model, operator)
        assert_operator_matches(recorder.end_kwargs[-1]['W_recurrent'], P, x0, y)


@parametrize_with_cases(SPEC, cases=ModelCases, has_tag='learned')
def test_applied_step_parameters_are_softplus_of_raw_parameters(spec):
    model = build_model(spec, gamma_init=0.3, theta_init=0.01)
    torch.testing.assert_close(model.gamma, F.softplus(model._gamma))
    torch.testing.assert_close(model.theta, F.softplus(model._theta))
    torch.testing.assert_close(model.gamma, torch.full((T,), 0.3, dtype=torch.float64), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(model.theta, torch.full((T,), 0.01, dtype=torch.float64), rtol=1e-6, atol=1e-6)


def assert_weight_mode_matches_default(spec, operator, y, x0, weight_mode):
    """Iterates and parameter gradients of ``weight_mode`` match the model's default weight mode."""
    default = perturb_weights(build_model(spec, r_diag_of(operator)))
    model = build_model(spec, r_diag_of(operator), weight_mode=weight_mode)
    model.load_state_dict(default.state_dict())
    expected = spec.solve(default, y, operator, x0)
    actual = spec.solve(model, y, operator, x0)
    torch.testing.assert_close(actual, expected)
    expected_grads = torch.autograd.grad(expected.square().sum(), tuple(default.parameters()))
    actual_grads = torch.autograd.grad(actual.square().sum(), tuple(model.parameters()))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


@parametrize_with_cases(PROBLEM, cases=ProblemCases, has_tag='dictionary')
@parametrize_with_cases(SPEC, cases=ModelCases, has_tag='weight_mode')
def test_precomputed_weights_match_default_weights(spec, operator, y, x0):
    assert_weight_mode_matches_default(spec, operator, y, x0, 'precomputed')


@parametrize_with_cases(PROBLEM, cases=ProblemCases, has_tag='cmf')
@parametrize_with_cases(SPEC, cases=ModelCases, has_tag='matrix_free')
def test_factored_reduced_weights_match_matrix_free_weights(spec, operator, y, x0):
    """The dense ``B`` formed from ``Br`` reproduces the congruence, also on subsets and per problem."""
    assert_weight_mode_matches_default(spec, operator, y, x0, 'factored')


@parametrize_with_cases(SPEC, cases=ModelCases, has_tag='learned')
def test_constructor_rejects_non_positive_depth(spec):
    with pytest.raises(ValueError, match='num_iterations must be positive'):
        spec.make_model(NUM_MICS, False, **{**spec.model_kwargs, 'num_iterations': 0})


@parametrize_with_cases(SPEC, cases=ModelCases, has_tag='unfolded')
def test_constructor_rejects_empty_array(spec):
    with pytest.raises(ValueError, match='num_measurements must be positive|num_mics must exceed'):
        spec.make_model(0, False, **spec.model_kwargs)


@parametrize_with_cases(SPEC, cases=ModelCases)
def test_forward_rejects_negative_depth(spec):
    operator, y, _ = cmf_problem()
    with pytest.raises(ValueError, match='non-negative'):
        spec.solve(build_model(spec), y, operator, t=-1)


@parametrize_with_cases(SPEC, cases=ModelCases, has_tag='learned')
def test_forward_rejects_depth_beyond_num_iterations(spec):
    operator, y, _ = cmf_problem()
    with pytest.raises(ValueError, match='cannot exceed num_iterations'):
        spec.solve(build_model(spec), y, operator, t=T + 1)


@parametrize_with_cases(SPEC, cases=ModelCases, has_tag='unfolded')
def test_forward_rejects_operator_of_another_array_size(spec):
    model = build_model(spec)
    operator = CMFPropagation(torch.ones(NUM_MICS + 1, NUM_SOURCES, dtype=torch.complex128))
    with pytest.raises(ValueError, match='but Br? expects'):
        spec.solve(model, torch.ones(1, operator.shape[0], dtype=torch.float64), operator)


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
@parametrize_with_cases(SPEC, cases=ModelCases, has_tag=('unfolded', 'dense'))
def test_row_subset_validation(spec, idx, batch, message):
    model = build_model(spec)
    rng = torch.Generator().manual_seed(0)
    A = torch.randn((4, 4) if batch is None else (batch, 4, 4), dtype=torch.float64, generator=rng)
    with pytest.raises(ValueError, match=message):
        spec.solve(model, torch.randn(batch or 1, 4, dtype=torch.float64, generator=rng), MatrixMult(A, rows=idx))


@parametrize_with_cases(SPEC, cases=ModelCases, has_tag='cmf_only')
def test_rejects_operator_without_cmf_coordinates(spec):
    model = build_model(spec)
    m = NUM_MICS * NUM_MICS
    with pytest.raises(ValueError, match='CMFDictionary or CMFPropagation'):
        spec.solve(model, torch.ones(1, m, dtype=torch.float64), MatrixMult(torch.eye(m, dtype=torch.float64)))


@parametrize_with_cases(SPEC, cases=ModelCases, has_tag='cmf_only')
def test_rejects_mismatched_diagonal_setting(spec):
    model = build_model(spec, r_diag=False)
    operator = CMFPropagation(torch.ones(NUM_MICS, NUM_SOURCES, dtype=torch.complex128), r_diag=True)
    with pytest.raises(ValueError, match='r_diag'):
        spec.solve(model, torch.ones(1, operator.shape[0], dtype=torch.float64), operator)


@parametrize_with_cases(SPEC, cases=ModelCases)
def test_callbacks_run_in_order_with_detached_operator(spec, recorder):
    operator, y, x0 = cmf_problem()
    model = build_model(spec)
    model.callbacks = [recorder]
    spec.solve(model, y, operator, x0, t=3)
    assert recorder.events == [(event, k) for k in range(3) for event in ('begin', 'end')]
    assert not hasattr(model, 'y')  # y reaches callbacks only through the end-callback metadata
    for kwargs in recorder.end_kwargs:
        assert not kwargs['A'].H.requires_grad
        torch.testing.assert_close(kwargs['y'], y)


@pytest.mark.parametrize('boundary', ['begin', 'end'])
@parametrize_with_cases(SPEC, cases=ModelCases)
def test_callbacks_can_stop_at_either_boundary(spec, boundary):
    operator, y, x0 = cmf_problem()
    outputs = IntermediateOutputCollector()
    model = build_model(spec)
    model.callbacks = [StopAt(0, boundary), outputs]
    actual = spec.solve(model, y, operator, x0)
    if boundary == 'begin':
        assert outputs.intermediates == []
        torch.testing.assert_close(actual, x0)
    else:
        assert len(outputs.intermediates) == 1
        torch.testing.assert_close(actual, outputs.intermediates[0])


@parametrize_with_cases(SPEC, cases=ModelCases)
def test_intermediate_outputs_keep_autograd_graph_and_reset(spec):
    operator, y, x0 = cmf_problem()
    collector = IntermediateOutputCollector()
    model = build_model(spec)
    model.callbacks = [collector]
    y = y.clone().requires_grad_()
    spec.solve(model, y, operator, x0)
    assert len(collector.intermediates) == T
    assert all(x.requires_grad for x in collector.intermediates)
    sum(x.square().sum() for x in collector.intermediates).backward()
    assert torch.count_nonzero(y.grad)

    spec.solve(model, y.detach(), operator, x0, t=2)
    assert len(collector.intermediates) == 2
    collector.reset()
    assert collector.intermediates == []


@parametrize_with_cases(SPEC, cases=ModelCases)
def test_warm_start_stays_unchanged_and_solves_are_independent(spec):
    operator, y, x0 = cmf_problem()
    model = perturb_weights(build_model(spec))
    x0 = x0.clone().requires_grad_()
    saved = x0.detach().clone()
    first = spec.solve(model, y, operator, x0)
    second = spec.solve(model, y, operator, x0)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(x0, saved)
    assert first.data_ptr() != x0.data_ptr()
    first.sum().backward()
    first_grad = x0.grad.clone()
    x0.grad = None
    second.sum().backward()
    torch.testing.assert_close(x0.grad, first_grad)


@parametrize_with_cases(SPEC, cases=ModelCases)
def test_zero_steps_return_a_copy_of_the_warm_start_or_zeros(spec):
    operator, y, x0 = cmf_problem()
    model = build_model(spec)
    actual = spec.solve(model, y[0], operator, x0[0], t=0)
    torch.testing.assert_close(actual, x0[:1], rtol=0, atol=0)
    assert actual.data_ptr() != x0.data_ptr()
    torch.testing.assert_close(spec.solve(model, y, operator, t=0), torch.zeros_like(x0), rtol=0, atol=0)


@parametrize_with_cases(SPEC, cases=ModelCases)
def test_begin_callbacks_can_adjust_iterates(spec):
    class Shift(IterationCallback):
        def on_step_begin(self, solver, x, **kwargs):
            x.add_(0.05)

    operator, y, x0 = cmf_problem()
    model = perturb_weights(build_model(spec))
    model.callbacks = [Shift()]
    with torch.no_grad():
        actual = spec.solve(model, y, operator, x0)
        expected = reference(spec, model, operator, y, x0, x_shift=0.05)[-1]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('t', [0, 1])
@pytest.mark.parametrize('mismatch', ['dtype', 'device', 'integer', 'complex', 'array'])
@parametrize_with_cases(SPEC, cases=ModelCases)
def test_invalid_warm_start_is_rejected_before_callbacks(spec, mismatch, t):
    operator, y, x0 = cmf_problem()
    invalid, message = {
        'dtype': (x0.float(), 'x0 must have dtype'),
        'device': (x0.to('meta'), 'x0 must be on device'),
        'integer': (x0.long(), 'x0 must be a real floating-point tensor'),
        'complex': (x0.to(torch.complex128), 'x0 must be real-valued'),
        'array': (x0.numpy(), 'x0 must be a real'),
    }[mismatch]
    model = build_model(spec)
    model.callbacks = [MustNotRun()]
    with pytest.raises(ValueError, match=message):
        spec.solve(model, y, operator, invalid, t=t)


@pytest.mark.parametrize('t', [0, 1])
@parametrize_with_cases(SPEC, cases=ModelCases)
def test_invalid_operator_is_rejected_before_callbacks(spec, t):
    operator, y, x0 = cmf_problem()
    model = build_model(spec)
    model.callbacks = [MustNotRun()]
    with pytest.raises(ValueError, match='operator must have dtype'):
        spec.solve(model, y, CMFPropagation(operator.H.detach().to(torch.complex64)), x0, t=t)
    with pytest.raises(ValueError, match='LinearOperator|CMFDictionary or CMFPropagation'):
        spec.solve(model, y, dense_matrix(operator), x0, t=t)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is not available')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@parametrize_with_cases(SPEC, cases=ModelCases)
def test_cuda_matches_cpu_with_autograd(spec, dtype):
    operator, y, x0 = cmf_problem(dtype)
    operator = CMFPropagation(operator.H.detach())
    model = perturb_weights(build_model(spec)).to(dtype)
    expected = spec.solve(model, y, operator, x0)
    model = model.cuda()
    y_cuda = y.cuda().requires_grad_()
    actual = spec.solve(model, y_cuda, CMFPropagation(operator.H.cuda()), x0.cuda())
    actual.square().sum().backward()
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-6)
    assert torch.isfinite(y_cuda.grad).all()


@parametrize_with_cases(SPEC, cases=ModelCases)
def test_state_dict_holds_the_expected_parameters_and_buffers(spec):
    assert list(build_model(spec).state_dict()) == spec.state_names


@parametrize_with_cases(SPEC, cases=ModelCases, has_tag='weight_mode')
def test_rejects_unknown_weight_mode(spec):
    with pytest.raises(ValueError, match='weight_mode'):
        spec.make_model(NUM_MICS, False, **spec.model_kwargs, weight_mode='auto')


@parametrize_with_cases(SPEC, cases=ModelCases, has_tag='weight_mode')
def test_precomputed_weights_require_a_dense_operator(spec):
    operator, y, x0 = cmf_problem()
    with pytest.raises(ValueError, match="'precomputed' requires a dense"):
        spec.solve(build_model(spec, weight_mode='precomputed'), y, operator, x0)


@parametrize_with_cases(SPEC, cases=ModelCases, has_tag=('unfolded', 'dense'))
def test_rejects_initial_auxiliary_weights_of_wrong_shape(spec):
    m = NUM_MICS * NUM_MICS
    for name in spec.init_weights:
        with pytest.raises(ValueError, match=f'{name} must have shape'):
            spec.make_model(NUM_MICS, False, **spec.model_kwargs, **{name: torch.eye(m + 1)})
