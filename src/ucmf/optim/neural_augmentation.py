"""Neural augmentation: an LSTM predicts per-sample step sizes and thresholds.

Adapted from NA-ALISTA (Behrens et al.) as in FA2026. In every iteration, the
controller sees the normalized L1 norms of the residual ``y - A z`` (divided
by ``m``) and of the update direction, and predicts ``gamma_k`` and the
applied threshold ``theta_k`` for each sample.
"""

from .utils import softplus_inverse

import torch
import torch.nn.functional as F
from torch import nn


class _StepParamLSTM(nn.Module):
    """LSTM controller with a softplus readout initialized to fixed ``gamma`` / ``theta``.

    The feature normalization buffers start as NaN and must be set explicitly
    (``NeuralAugmentation.init_normalizers``) before the first solve.
    """

    def __init__(self, dim, gamma_init, theta_init):
        super().__init__()
        # Creation order matches FA2026 (readout, LSTM, initial states) for equal seeded inits.
        self.lll = nn.Linear(dim, dim)
        self.linear = nn.Linear(dim, 2)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.copy_(torch.tensor([softplus_inverse(gamma_init), softplus_inverse(theta_init)]))
        self.lstm = nn.LSTMCell(2, dim)
        self.hidden_init = nn.Parameter(torch.randn(dim))
        self.cell_init = nn.Parameter(torch.randn(dim))
        self.register_buffer('feature_mean', torch.full((2,), float('nan')))
        self.register_buffer('feature_std', torch.full((2,), float('nan')))

    @staticmethod
    def features(residual, direction):
        """Per-sample ``(||residual||_1 / m, ||direction||_1)`` with shape ``(B, 2)``."""
        residual_norm = torch.linalg.vector_norm(residual, ord=1, dim=1) / residual.shape[1]
        return torch.stack((residual_norm, torch.linalg.vector_norm(direction, ord=1, dim=1)), dim=1)

    def initial_state(self, batch_size):
        """Return the learned initial ``(hidden, cell)`` for ``batch_size`` samples."""
        return self.hidden_init.expand(batch_size, -1), self.cell_init.expand(batch_size, -1)

    def forward(self, residual, direction, hidden, cell):
        """Advance the LSTM by one iteration.

        Returns ``gamma`` and ``theta`` of shape ``(B, 1)`` and the new ``hidden`` and ``cell``.
        """
        features = (self.features(residual, direction) - self.feature_mean) / self.feature_std.clamp_min(1e-6)
        hidden, cell = self.lstm(features, (hidden, cell))
        params = F.softplus(self.linear(torch.relu(self.lll(cell))))
        return params[:, :1], params[:, 1:], hidden, cell


class NeuralAugmentation:
    """Mixin predicting the step size and threshold of every iteration with an LSTM.

    Combine with a weight base whose ``_prepare_update`` provides
    ``residual_gradient(z) -> (y - A z, direction)``. The controller is not tied
    to a fixed depth, so any ``t >= 0`` is allowed. Call :meth:`init_normalizers`
    once before solving; until then the outputs are NaN. All arguments are
    keyword-only. The operator must have unit column norms (unchecked
    precondition), see :func:`ucmf.scaling.normalize_operator`.

    Parameters
    ----------
    gamma_init : float
        Initial applied step size of the readout.
    theta_init : float
        Initial applied soft-threshold of the readout.
    lstm_dim : int, optional
        Hidden size of the LSTM (default 128).
    **kwargs
        Arguments of the weight base.

    Attributes
    ----------
    controller : torch.nn.Module
        LSTM, readout, learned initial states, and the feature normalization buffers.
    """

    def __init__(self, *, gamma_init, theta_init, lstm_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.controller = _StepParamLSTM(lstm_dim, gamma_init, theta_init)

    @torch.no_grad()
    def init_normalizers(self, y, operator, x0=None):
        """Set the feature normalization from the first-iteration features of a batch.

        Uses the batch mean and population standard deviation of the features
        at ``z = x0``.

        Parameters
        ----------
        y : Tensor
            Measurements of shape ``(B, m)`` or ``(m,)``.
        operator : LinearOperator
            Measurement operator.
        x0 : Tensor, optional
            Point of the first iteration, shape ``(B, n)`` or ``(n,)``; zeros if omitted.
        """
        y, x0, _, residual_gradient, _ = self._prepare_update(y, x0, operator)
        features = self.controller.features(*residual_gradient(x0))
        self.controller.feature_mean.copy_(features.mean(dim=0))
        self.controller.feature_std.copy_(features.std(dim=0, unbiased=False))

    def forward(self, y, x0=None, t=1, operator=None, active_iterations=None):
        """Run ``t`` accelerated iterations with controller-predicted step parameters.

        Parameters
        ----------
        y : Tensor
            Measurements of shape ``(B, m)`` or ``(m,)``.
        x0 : Tensor, optional
            Warm start of shape ``(B, n)`` or ``(n,)``; zeros if omitted.
        t : int, optional
            Number of iterations, ``t >= 0`` (default 1).
        operator : LinearOperator
            Measurement operator; treated as fixed.
        active_iterations : Tensor, optional
            0-d integer tensor ``s <= t``: iterations ``s + 1, ..., t`` keep the iterate, so the
            result is ``x_s``. One compiled graph of depth ``t`` thus serves every depth ``s``,
            see :func:`~ucmf.optim.compile.compile_unrolled`. Default: all ``t`` iterations.

        Returns
        -------
        Tensor
            Solution of shape ``(B, n)``.
        """
        if t < 0:
            raise ValueError('t must be non-negative.')
        y, x0, _, residual_gradient, callback_metadata = self._prepare_update(y, x0, operator)
        shrink_func = self.get_shrink_func()
        x = x0.clone()
        z = x0.clone()
        tk = 1.0
        hidden, cell = self.controller.initial_state(y.shape[0])

        self._start_iterations()
        for k in range(t):
            if not self._on_iteration_begin(k, x):
                break
            residual, direction = residual_gradient(z)
            gamma_k, theta_k, hidden, cell = self.controller(residual, direction, hidden, cell)
            x_next = shrink_func(z + gamma_k * direction, theta_k)
            tk, z_next = self._nesterov_momentum(tk, x, x_next)
            x = self._keep_inactive(k, active_iterations, x_next, x)
            z = self._keep_inactive(k, active_iterations, z_next, z)
            if not self._on_iteration_end(x, gamma_k, theta_k, **callback_metadata):
                break
        return x
