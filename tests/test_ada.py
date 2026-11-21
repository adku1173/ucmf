"""Equivalence and training-surface tests for single-weight Ada-LFISTA."""

from ucmf import FISTA, AdaLFISTASW
from ucmf.optim.utils import softplus_inverse

import pytest
import torch
import torch.nn.functional as F


def _ada_matching_fista(A, lam, gamma, t, positive):
    """Ada-LFISTA-SW configured to reproduce FISTA: B = I, constant gamma/theta.

    Parameters are set in A's dtype so the softplus round-trip stays exact and
    the comparison is not limited by the default float32 parameter storage.
    """
    model = AdaLFISTASW(
        n_iterations=t, n_measurements=A.shape[-2], gamma_init=gamma, theta_init=lam * gamma, positive=positive,
    ).to(A.dtype)
    with torch.no_grad():
        model._gamma.fill_(softplus_inverse(gamma))
        model._theta.fill_(softplus_inverse(lam * gamma))
    return model


@pytest.mark.parametrize('positive', [True, False])
@pytest.mark.parametrize('batched_A', [True, False])
def test_identity_auxiliary_weight_reproduces_fista(positive, batched_A):
    torch.manual_seed(0)
    A = torch.randn(3, 5, 4, dtype=torch.float64) if batched_A else torch.randn(5, 4, dtype=torch.float64)
    y = torch.randn(3, 5, dtype=torch.float64)
    lam, gamma, t = 0.03, 0.12, 7
    expected = FISTA(positive=positive)(y, A=A, lam=lam, gamma=gamma, t=t)
    actual = _ada_matching_fista(A, lam, gamma, t, positive)(y, A=A, t=t)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_trainable_parameters_and_applied_values():
    model = AdaLFISTASW(n_iterations=4, n_measurements=3, gamma_init=0.2, theta_init=0.05)
    assert [name for name, _ in model.named_parameters()] == ['B', '_gamma', '_theta']
    # The public gamma/theta are the actually applied values, softplus(_gamma/_theta).
    torch.testing.assert_close(model.gamma, F.softplus(model._gamma))
    torch.testing.assert_close(model.theta, F.softplus(model._theta))
    torch.testing.assert_close(model.gamma, torch.full((4,), 0.2), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(model.theta, torch.full((4,), 0.05), rtol=1e-6, atol=1e-6)


def test_auxiliary_weight_and_step_params_receive_gradients():
    torch.manual_seed(0)
    A = torch.randn(5, 4, dtype=torch.float64)
    y = torch.randn(2, 5, dtype=torch.float64)
    model = AdaLFISTASW(
        n_iterations=6, n_measurements=5, gamma_init=0.2, theta_init=0.01, positive=False,
    ).to(torch.float64)
    with torch.no_grad():
        model.B.add_(0.01 * torch.randn_like(model.B))
    model(y, A=A, t=6).square().sum().backward()
    grads = {name: param.grad for name, param in model.named_parameters()}
    for name in ('B', '_gamma', '_theta'):
        assert grads[name] is not None and torch.isfinite(grads[name]).all()
    # B actually influences the output, so its gradient is non-trivial.
    assert grads['B'].abs().max() > 0


def _randomized_model(n_measurements, t=5):
    """Ada-LFISTA-SW with a dense, non-identity auxiliary weight for subset tests."""
    model = AdaLFISTASW(
        n_iterations=t, n_measurements=n_measurements, gamma_init=0.3, theta_init=0.01,
    ).to(torch.float64)
    with torch.no_grad():
        model.B.copy_(torch.randn(n_measurements, n_measurements, dtype=torch.float64))
    return model


def _subblock_model(model, idx):
    """Full-set solver whose B is the selected sub-block ``B[I, I]`` of ``model``."""
    sub = AdaLFISTASW(
        n_iterations=model.n_iterations, n_measurements=idx.shape[-1], gamma_init=0.3, theta_init=0.01,
    ).to(torch.float64)
    with torch.no_grad():
        sub.B.copy_(model.B[idx][:, idx])
    return sub


def test_measurement_subset_matches_subblock_solver():
    torch.manual_seed(0)
    A = torch.randn(6, 4, dtype=torch.float64)
    y = torch.randn(3, 6, dtype=torch.float64)
    model = _randomized_model(6)
    idx = torch.tensor([0, 2, 3, 5])
    actual = model(y[:, idx], A=A[idx], msm_subset=idx, t=5)
    expected = _subblock_model(model, idx)(y[:, idx], A=A[idx], t=5)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_full_set_equals_identity_measurement_subset():
    torch.manual_seed(1)
    A = torch.randn(6, 4, dtype=torch.float64)
    y = torch.randn(3, 6, dtype=torch.float64)
    model = _randomized_model(6)
    subset = model(y, A=A, msm_subset=torch.arange(6), t=5)
    torch.testing.assert_close(subset, model(y, A=A, t=5), rtol=0, atol=0)


def test_per_sample_measurement_subset_matches_subblock_solvers():
    torch.manual_seed(2)
    b = 3
    A = torch.randn(b, 6, 4, dtype=torch.float64)
    y = torch.randn(b, 6, dtype=torch.float64)
    model = _randomized_model(6)
    idx = torch.stack([torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 4]), torch.tensor([2, 3, 4, 5])])
    A_sub = torch.stack([A[i][idx[i]] for i in range(b)])
    y_sub = torch.stack([y[i][idx[i]] for i in range(b)])
    actual = model(y_sub, A=A_sub, msm_subset=idx, t=5)
    rows = [_subblock_model(model, idx[i])(y_sub[i : i + 1], A=A_sub[i], t=5) for i in range(b)]
    torch.testing.assert_close(actual, torch.cat(rows), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ('idx', 'message'),
    [
        (torch.tensor([0.0, 1.0, 2.0, 3.0]), 'integer tensor'),
        (torch.tensor([0, 1, 2]), 'must match A measurement dim'),
        (torch.zeros(2, 2, 2, dtype=torch.long), '1D or 2D'),
        (torch.tensor([[0, 1, 2, 3]]), 'requires batched A'),
    ],
)
def test_measurement_subset_validation(idx, message):
    model = AdaLFISTASW(n_iterations=3, n_measurements=6, gamma_init=0.3, theta_init=0.01)
    with pytest.raises(ValueError, match=message):
        model(torch.randn(1, 4), A=torch.randn(4, 4), msm_subset=idx, t=3)


def test_per_sample_measurement_subset_rejects_batch_mismatch():
    model = AdaLFISTASW(n_iterations=3, n_measurements=6, gamma_init=0.3, theta_init=0.01)
    idx = torch.zeros(3, 4, dtype=torch.long)  # index batch 3 != A/y batch 2
    with pytest.raises(ValueError, match='does not match A batch'):
        model(torch.randn(2, 4), A=torch.randn(2, 4, 4), msm_subset=idx, t=3)


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'n_iterations': 0, 'n_measurements': 3}, 'n_iterations must be positive'),
        ({'n_iterations': 2, 'n_measurements': 0}, 'n_measurements must be positive'),
        ({'n_iterations': 2, 'n_measurements': 3, 'B': torch.eye(4)}, 'B must have shape'),
    ],
)
def test_constructor_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AdaLFISTASW(gamma_init=0.1, theta_init=0.01, **kwargs)


@pytest.mark.parametrize(('t', 'message'), [(-1, 'non-negative'), (99, 'cannot exceed n_iterations')])
def test_forward_validates_iteration_count(t, message):
    model = AdaLFISTASW(n_iterations=3, n_measurements=5, gamma_init=0.2, theta_init=0.01)
    with pytest.raises(ValueError, match=message):
        model(torch.randn(1, 5), A=torch.randn(5, 4), t=t)


def test_forward_rejects_A_measurement_mismatch():
    model = AdaLFISTASW(n_iterations=3, n_measurements=5, gamma_init=0.2, theta_init=0.01)
    with pytest.raises(ValueError, match='but B expects 5'):
        model(torch.randn(1, 4), A=torch.randn(4, 4), t=3)
