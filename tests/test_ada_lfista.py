"""Two-matrix Ada-LFISTA specifics; shared properties are in test_unfolded_models."""

import math

from ucmf import AdaLFISTA
from ucmf.optim import IntermediateOutputCollector, IterationCallback

import pytest
import torch

M_FULL, N, T = 6, 4, 4


def oracle(A, y, B1, B2, gammas, thetas, positive):
    """Algorithm 1 per sample in column-vector form: R = B1 D (W1), M = B2 D (W2)."""
    history = []
    for D, signal, W1, W2 in zip(A, y, B1, B2, strict=True):
        R, M = W1 @ D, W2 @ D
        x = torch.zeros(D.shape[1], dtype=D.dtype)
        z, tk, iterates = x.clone(), 1.0, []
        for gamma, theta in zip(gammas, thetas, strict=True):
            value = z + gamma * (M.T @ signal - R.T @ (R @ z)) / len(signal)
            x_next = torch.relu(value - theta) if positive else value.sign() * torch.relu(value.abs() - theta)
            t_next = (1 + math.sqrt(1 + 4 * tk**2)) / 2
            z = x_next + (tk - 1) / t_next * (x_next - x)
            x, tk = x_next, t_next
            iterates.append(x)
        history.append(torch.stack(iterates))
    return torch.stack(history, dim=1)  # (T, b, n)


def make_model(positive=False):
    torch.manual_seed(0)
    model = AdaLFISTA(n_iterations=T, n_measurements=M_FULL, gamma_init=0.3, theta_init=0.01, positive=positive)
    model = model.double()
    with torch.no_grad():
        model.B1.add_(0.2 * torch.randn_like(model.B1))
        model.B2.add_(0.2 * torch.randn_like(model.B2))
    return model


@pytest.mark.parametrize('positive', [False, True])
@pytest.mark.parametrize('mode', ['shared', 'batched', 'subset', 'per_sample_subset'])
def test_matches_algorithm_1_at_every_iteration(mode, positive):
    model = make_model(positive)
    torch.manual_seed(1)
    A_full = torch.randn(2, M_FULL, N, dtype=torch.float64)
    if mode in ('shared', 'subset'):
        A_full[1] = A_full[0]
    y_full = torch.randn(2, M_FULL, dtype=torch.float64)
    idx = torch.stack([torch.arange(M_FULL)] * 2)
    if 'subset' in mode:
        idx = torch.tensor([[0, 2, 3, 5], [0, 2, 3, 5] if mode == 'subset' else [1, 2, 4, 5]])
    A = torch.stack([A_full[i][idx[i]] for i in range(2)])
    y = torch.stack([y_full[i][idx[i]] for i in range(2)])
    B1 = [model.B1.detach()[idx[i]][:, idx[i]] for i in range(2)]
    B2 = [model.B2.detach()[idx[i]][:, idx[i]] for i in range(2)]
    msm_subset = {'subset': idx[0], 'per_sample_subset': idx}.get(mode)

    collector = IntermediateOutputCollector()
    model.callbacks = [collector]
    out = model(y, A=A[0] if mode in ('shared', 'subset') else A, t=T, msm_subset=msm_subset)
    expected = oracle(A, y, B1, B2, model.gamma.detach(), model.theta.detach(), positive)
    torch.testing.assert_close(torch.stack(collector.intermediates).detach(), expected)
    torch.testing.assert_close(out.detach(), expected[-1])


def test_callbacks_receive_both_effective_operators():
    model = make_model()
    A = torch.randn(M_FULL, N, dtype=torch.float64)
    seen = {}

    class Recorder(IterationCallback):
        def on_step_end(self, solver, x, **kwargs):
            seen.update(kwargs)

    model.callbacks = [Recorder()]
    with torch.no_grad():
        model(torch.randn(2, M_FULL, dtype=torch.float64), A=A, t=T)
    torch.testing.assert_close(seen['A'], A)
    torch.testing.assert_close(seen['W'], model.B2.detach() @ A)  # measurement branch M
    torch.testing.assert_close(seen['W_recurrent'], model.B1.detach() @ A)  # recurrence R


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'B1': torch.eye(M_FULL + 1)}, 'B1 must have shape'),
        ({'B2': torch.eye(M_FULL + 1)}, 'B2 must have shape'),
    ],
)
def test_rejects_auxiliary_weights_of_wrong_shape(kwargs, message):
    arguments = {'n_iterations': T, 'n_measurements': M_FULL, 'gamma_init': 0.1, 'theta_init': 0.01, **kwargs}
    with pytest.raises(ValueError, match=message):
        AdaLFISTA(**arguments)
