"""Training objectives: one function ``f(solve, y, q, t, operator, column_norms, *, w)`` each.

``solve(y, operator, t)`` returns the iterates ``x_1, ..., x_t`` as ``(t, B, n)``, eager or
compiled. Each objective returns the loss; :func:`weighted_mean` explains ``w``.
"""

import math
from functools import partial

from ..scaling import to_source

import torch


def weighted_mean(outputs, criterion, w):
    """Mean of ``criterion(x_k)`` over ``[x_1, ..., x_t]`` with weights ``w ** (t - k)``.

    The weights are normalized to sum one, so the loss scale is independent of ``t`` and ``w``.
    ``w == 0`` evaluates only ``x_t``, so non-finite earlier losses cannot enter as ``0 * inf``.
    """
    if w == 0:
        return criterion(outputs[-1])
    t = len(outputs)
    weights = [w ** (t - k) for k in range(1, t + 1)]
    return sum(weight * criterion(x) for weight, x in zip(weights, outputs, strict=True)) / sum(weights)


def nmse(x_hat, q, u):
    """Batch-mean ``||q_hat - q||^2 / ||q||^2``, ``q_hat = to_source(x_hat, u)``, ``q != 0``."""
    error = torch.linalg.vector_norm(to_source(x_hat, u) - q, dim=-1).square()
    return (error / torch.linalg.vector_norm(q, dim=-1).square()).mean()


def lasso(x_hat, y, operator, lam):
    """Batch-mean ``||D x_hat - y||^2 / (2 m) + lam ||x_hat||_1`` with ``D = operator``."""
    data_fit = torch.linalg.vector_norm(operator.forward(x_hat) - y, dim=-1).square() / (2 * y.shape[-1])
    return (data_fit + lam * x_hat.abs().sum(dim=-1)).mean()


def bic(x_hat, y, operator, alpha):
    """Batch-mean smoothed BIC ``m log(||D x_hat - y||^2 / m) + log(m) ||x_hat||_{0,eps}``.

    The smoothed count is ``sum_j (1 - exp(-x_j^2 / (2 eps^2)))`` (Gaussian family).
    ``eps = alpha ||x_hat||_inf`` per sample is detached, so it acts as a constant. Its lower
    bound keeps the frequent zero solution finite, where ``eps`` would vanish (``0 / 0``).
    """
    m = y.shape[-1]
    variance = torch.linalg.vector_norm(operator.forward(x_hat) - y, dim=-1).square() / m
    eps = alpha * x_hat.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    count = (1 - torch.exp(-x_hat.square() / (2 * eps.square()))).sum(dim=-1)
    return (m * torch.log(variance) + math.log(m) * count).mean()


def nmse_loss(solve, y, q, t, operator, column_norms, *, w):
    """Weighted NMSE of the iterates against the source-strength targets ``q``."""
    return weighted_mean(solve(y, operator, t), lambda x_hat: nmse(x_hat, q, column_norms), w)


def lasso_loss(solve, y, q, t, operator, column_norms, *, w, lam):
    """Weighted LASSO criterion of the iterates; equireg makes ``lam`` relative to lambda_max."""
    return weighted_mean(solve(y, operator, t), lambda x_hat: lasso(x_hat, y, operator, lam), w)


def bic_loss(solve, y, q, t, operator, column_norms, *, w, alpha=1e-2):
    """Weighted smoothed BIC of the iterates; ``alpha = 1e-2`` puts ``eps`` 20 dB below the max."""
    return weighted_mean(solve(y, operator, t), lambda x_hat: bic(x_hat, y, operator, alpha), w)


OBJECTIVES = {'nmse': nmse_loss, 'lasso': lasso_loss, 'bic': bic_loss}


def build_loss(config):
    """Bind the objective ``config.loss`` to ``config.w`` and ``config.loss_kwargs``."""
    return partial(OBJECTIVES[config.loss], w=config.w, **config.loss_kwargs)
