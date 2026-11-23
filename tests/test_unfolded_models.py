"""Properties shared by all unfolded models (see the unfolded_model_type fixture)."""

from ucmf import FISTA, AdaLFISTA, AdaLFISTASW
from ucmf.optim.utils import softplus_inverse

import pytest
import torch
import torch.nn.functional as F

PARAMETER_NAMES = {AdaLFISTASW: ['B', '_gamma', '_theta'], AdaLFISTA: ['B1', 'B2', '_gamma', '_theta']}


def make_model(model_type, n_measurements=6, t=5, **kwargs):
    """Float64 model with dense, non-identity auxiliary weights (all 2D parameters)."""
    torch.manual_seed(0)
    model = model_type(n_iterations=t, n_measurements=n_measurements, gamma_init=0.3, theta_init=0.01, **kwargs)
    model = model.double()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 2:
                parameter.add_(0.2 * torch.randn_like(parameter))
    return model


def subblock_model(model, idx):
    """Full-set model whose auxiliary weights are the sub-blocks ``B[I, I]`` of ``model``."""
    sub = make_model(type(model), n_measurements=idx.shape[-1], t=model.n_iterations)
    with torch.no_grad():
        for (name, parameter), source in zip(sub.named_parameters(), model.parameters(), strict=True):
            parameter.copy_(source[idx][:, idx] if name.startswith('B') else source)
    return sub


@pytest.mark.parametrize('positive', [True, False])
@pytest.mark.parametrize('batched_A', [True, False])
def test_identity_weights_reproduce_fista(unfolded_model_type, positive, batched_A):
    """Identity auxiliary weights and constant gamma/theta = lam * gamma give FISTA."""
    torch.manual_seed(0)
    A = torch.randn(3, 5, 4, dtype=torch.float64) if batched_A else torch.randn(5, 4, dtype=torch.float64)
    y = torch.randn(3, 5, dtype=torch.float64)
    lam, gamma, t = 0.03, 0.12, 7
    model = unfolded_model_type(
        n_iterations=t, n_measurements=5, gamma_init=gamma, theta_init=lam * gamma, positive=positive,
    ).double()
    with torch.no_grad():  # set in float64 so the softplus round-trip is exact
        model._gamma.fill_(softplus_inverse(gamma))
        model._theta.fill_(softplus_inverse(lam * gamma))
    expected = FISTA(positive=positive)(y, A=A, lam=lam, gamma=gamma, t=t)
    torch.testing.assert_close(model(y, A=A, t=t), expected, rtol=1e-12, atol=1e-12)


def test_parameters_applied_values_and_gradients(unfolded_model_type):
    model = make_model(unfolded_model_type, positive=False)
    assert [name for name, _ in model.named_parameters()] == PARAMETER_NAMES[unfolded_model_type]
    # The public gamma/theta are the actually applied values, softplus(_gamma/_theta).
    torch.testing.assert_close(model.gamma, F.softplus(model._gamma))
    torch.testing.assert_close(model.theta, F.softplus(model._theta))
    torch.testing.assert_close(model.gamma, torch.full((5,), 0.3, dtype=torch.float64), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(model.theta, torch.full((5,), 0.01, dtype=torch.float64), rtol=1e-6, atol=1e-6)
    A, y = torch.randn(6, 4, dtype=torch.float64), torch.randn(2, 6, dtype=torch.float64)
    model(y, A=A, t=5).square().sum().backward()
    for parameter in model.parameters():
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


def test_measurement_subset_matches_subblock_model(unfolded_model_type):
    torch.manual_seed(1)
    A, y = torch.randn(6, 4, dtype=torch.float64), torch.randn(3, 6, dtype=torch.float64)
    model = make_model(unfolded_model_type)
    idx = torch.tensor([0, 2, 3, 5])
    expected = subblock_model(model, idx)(y[:, idx], A=A[idx], t=5)
    torch.testing.assert_close(model(y[:, idx], A=A[idx], msm_subset=idx, t=5), expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(model(y, A=A, msm_subset=torch.arange(6), t=5), model(y, A=A, t=5), rtol=0, atol=0)


def test_per_sample_measurement_subset_matches_subblock_models(unfolded_model_type):
    torch.manual_seed(2)
    A, y = torch.randn(3, 6, 4, dtype=torch.float64), torch.randn(3, 6, dtype=torch.float64)
    model = make_model(unfolded_model_type)
    idx = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]])
    A_sub = torch.stack([A[i][idx[i]] for i in range(3)])
    y_sub = torch.stack([y[i][idx[i]] for i in range(3)])
    rows = [subblock_model(model, idx[i])(y_sub[i : i + 1], A=A_sub[i], t=5) for i in range(3)]
    actual = model(y_sub, A=A_sub, msm_subset=idx, t=5)
    torch.testing.assert_close(actual, torch.cat(rows), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ('idx', 'batch', 'message'),
    [
        (torch.tensor([0.0, 1.0, 2.0, 3.0]), None, 'integer tensor'),
        (torch.tensor([0, 1, 2]), None, 'must match A measurement dim'),
        (torch.zeros(2, 2, 2, dtype=torch.long), None, '1D or 2D'),
        (torch.tensor([[0, 1, 2, 3]]), None, 'requires batched A'),
        (torch.zeros(3, 4, dtype=torch.long), 2, 'does not match A batch'),
    ],
)
def test_measurement_subset_validation(unfolded_model_type, idx, batch, message):
    model = unfolded_model_type(n_iterations=3, n_measurements=6, gamma_init=0.3, theta_init=0.01)
    A = torch.randn(4, 4) if batch is None else torch.randn(batch, 4, 4)
    with pytest.raises(ValueError, match=message):
        model(torch.randn(batch or 1, 4), A=A, msm_subset=idx, t=3)


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'n_iterations': 0}, 'n_iterations must be positive'),
        ({'n_measurements': 0}, 'n_measurements must be positive'),
    ],
)
def test_constructor_validation(unfolded_model_type, kwargs, message):
    with pytest.raises(ValueError, match=message):
        unfolded_model_type(**{'n_iterations': 2, 'n_measurements': 3, 'gamma_init': 0.1, 'theta_init': 0.01, **kwargs})


@pytest.mark.parametrize(
    ('n_measurements_y', 't', 'message'),
    [(5, -1, 'non-negative'), (5, 99, 'cannot exceed n_iterations'), (4, 3, 'but B expects 5')],
)
def test_forward_validation(unfolded_model_type, n_measurements_y, t, message):
    model = unfolded_model_type(n_iterations=3, n_measurements=5, gamma_init=0.2, theta_init=0.01)
    with pytest.raises(ValueError, match=message):
        model(torch.randn(1, n_measurements_y), A=torch.randn(n_measurements_y, 4), t=t)
