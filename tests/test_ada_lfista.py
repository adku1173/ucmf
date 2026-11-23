"""Two-matrix Ada-LFISTA specifics; shared properties are in test_unfolded_models."""

from ucmf import AdaLFISTA
from ucmf.optim import IterationCallback
from ucmf.optim.operator import CMFPropagation, MatrixMult

import pytest
import torch

M_FULL, N, T = 6, 4, 4


def make_model(*, num_measurements=M_FULL, weight_mode='factored'):
    torch.manual_seed(0)
    model = AdaLFISTA(
        num_iterations=T, num_measurements=num_measurements, gamma_init=0.3, theta_init=0.01,
        positive=False, weight_mode=weight_mode,
    )
    model = model.double()
    with torch.no_grad():
        model.B1.add_(0.2 * torch.randn_like(model.B1))
        model.B2.add_(0.2 * torch.randn_like(model.B2))
    return model


def test_parameter_names():
    assert [name for name, _ in make_model().named_parameters()] == ['B1', 'B2', '_gamma', '_theta']


@pytest.mark.parametrize('weight_mode', ['factored', 'precomputed'])
def test_paper_numbering_b1_recurrence_b2_measurement(weight_mode):
    """B1 defines the recurrence (W_recurrent), B2 the measurement branch (W); FA2026 swapped them."""
    model = make_model(weight_mode=weight_mode)
    A = torch.randn(M_FULL, N, dtype=torch.float64)
    seen = {}

    class Recorder(IterationCallback):
        def on_step_end(self, solver, x, **kwargs):
            seen.update(kwargs)

    model.callbacks = [Recorder()]
    y = torch.randn(2, M_FULL, dtype=torch.float64)
    with torch.no_grad():
        model(y, operator=MatrixMult(A), t=T)
    torch.testing.assert_close(seen['A'].matrix, A)
    x = A.new_ones(2, N)
    for name, B in [('W', model.B2), ('W_recurrent', model.B1)]:
        reference = MatrixMult(B.detach() @ A)
        torch.testing.assert_close(seen[name].forward(x), reference.forward(x))
        torch.testing.assert_close(seen[name].adjoint(y), reference.adjoint(y))


def test_precomputed_rejects_matrix_free_operator():
    operator = CMFPropagation(torch.ones(2, N, dtype=torch.complex128), r_diag=True)
    model = make_model(num_measurements=operator.shape[0], weight_mode='precomputed')
    with pytest.raises(ValueError, match="'precomputed' requires a dense"):
        model(torch.ones(operator.shape[0], dtype=operator.dtype), operator=operator)


def test_rejects_unknown_weight_mode():
    with pytest.raises(ValueError, match='weight_mode'):
        make_model(weight_mode='auto')


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'B1': torch.eye(M_FULL + 1)}, 'B1 must have shape'),
        ({'B2': torch.eye(M_FULL + 1)}, 'B2 must have shape'),
    ],
)
def test_rejects_auxiliary_weights_of_wrong_shape(kwargs, message):
    arguments = {'num_iterations': T, 'num_measurements': M_FULL, 'gamma_init': 0.1, 'theta_init': 0.01, **kwargs}
    with pytest.raises(ValueError, match=message):
        AdaLFISTA(**arguments)
