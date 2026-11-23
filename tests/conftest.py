"""Test-only subclasses demonstrating subclass ownership of a weighted forward."""

# Acoular must be loaded before NumPy (imported via ucmf/torch).
import acoular  # noqa: F401
from ucmf import FISTA, ISTA, AdaLFISTA, AdaLFISTASW

import pytest


class _WeightedForward:
    """Only the subclass exposes W; no unfolded model is migrated here."""

    def forward(self, y, x0=None, t=1, A=None, lam=None, gamma=None, W=None):
        y, x0, A, gamma, theta, operator = self._setup_forward(y, x0, A, lam, gamma, W=W)
        return self._iterate(y, x0, A, gamma, theta, t, Op_tensor=operator)


class WeightedISTA(_WeightedForward, ISTA):
    pass


class WeightedFISTA(_WeightedForward, FISTA):
    pass


@pytest.fixture(params=[WeightedISTA, WeightedFISTA], ids=['weighted-ISTA', 'weighted-FISTA'])
def weighted_solver_type(request):
    return request.param


@pytest.fixture(scope='module', params=[AdaLFISTASW, AdaLFISTA], ids=['AdaLFISTASW', 'AdaLFISTA'])
def unfolded_model_type(request):
    """Every unfolded model; shared properties are tested once for all of them."""
    return request.param
