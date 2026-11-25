"""Training objectives: one function ``f(solve, y, q, t, operator, column_norms, *, w)`` each.

``solve(y, operator, t)`` returns the iterates ``x_1, ..., x_t`` as ``(t, B, n)``, eager or
compiled. Each objective returns the loss; :func:`weighted_mean` explains ``w``.
"""

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


def nmse_loss(solve, y, q, t, operator, column_norms, *, w):
    """Weighted NMSE of the iterates against the source-strength targets ``q``."""
    return weighted_mean(solve(y, operator, t), lambda x_hat: nmse(x_hat, q, column_norms), w)


OBJECTIVES = {'nmse': nmse_loss}


def build_loss(config):
    """Bind the objective ``config.loss`` to the iterate weight ``config.w``."""
    return partial(OBJECTIVES[config.loss], w=config.w)
