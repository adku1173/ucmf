"""Single-weight Ada-LFISTA specifics; shared properties are in test_unfolded_models."""

from ucmf import AdaLFISTASW
from ucmf.optim import IterationCallback

import pytest
import torch


def test_callbacks_receive_effective_weight():
    torch.manual_seed(0)
    model = AdaLFISTASW(n_iterations=3, n_measurements=5, gamma_init=0.2, theta_init=0.01).double()
    with torch.no_grad():
        model.B.add_(0.2 * torch.randn_like(model.B))
    A = torch.randn(5, 4, dtype=torch.float64)
    seen = {}

    class Recorder(IterationCallback):
        def on_step_end(self, solver, x, **kwargs):
            seen.update(kwargs)

    model.callbacks = [Recorder()]
    with torch.no_grad():
        model(torch.randn(2, 5, dtype=torch.float64), A=A, t=3)
    torch.testing.assert_close(seen['A'], A)
    torch.testing.assert_close(seen['W'], model.B.detach() @ A)  # effective adjoint W = B @ A


def test_rejects_auxiliary_weight_of_wrong_shape():
    with pytest.raises(ValueError, match='B must have shape'):
        AdaLFISTASW(n_iterations=2, n_measurements=3, gamma_init=0.1, theta_init=0.01, B=torch.eye(4))
