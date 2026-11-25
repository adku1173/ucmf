"""Training objectives: one function ``f(model, y, q, t, operator, column_norms, *, w)`` each.

Each solves the batch at depth ``t`` and returns the loss; :func:`weighted_sum` explains ``w``.
"""

from functools import partial

from ..optim.callbacks import IntermediateOutputCollector
from ..scaling import to_source

import torch


def solve_outputs(model, y, operator, t, w):
    """Solve; return ``[x_t]`` for ``w == 0``, otherwise ``[x_1, ..., x_t]`` with their graphs."""
    if w == 0:
        return [model(y, t=t, operator=operator)]
    collector = IntermediateOutputCollector()
    previous_callbacks = model.callbacks
    model.callbacks = [*previous_callbacks, collector]
    try:
        model(y, t=t, operator=operator)
    finally:
        model.callbacks = previous_callbacks
    return collector.intermediates


def weighted_sum(outputs, criterion, w):
    """Sum ``w ** (t - k) * criterion(x_k)`` over ``[x_1, ..., x_t]``; final weight one.

    ``w == 0`` evaluates only ``x_t``, so non-finite earlier losses cannot enter as ``0 * inf``.
    """
    if w == 0:
        return criterion(outputs[-1])
    t = len(outputs)
    return sum(w ** (t - k) * criterion(x) for k, x in enumerate(outputs, start=1))


def nmse(x_hat, q, u):
    """Batch-mean ``||q_hat - q||^2 / ||q||^2``, ``q_hat = to_source(x_hat, u)``, ``q != 0``."""
    error = torch.linalg.vector_norm(to_source(x_hat, u) - q, dim=-1).square()
    return (error / torch.linalg.vector_norm(q, dim=-1).square()).mean()


def nmse_loss(model, y, q, t, operator, column_norms, *, w):
    """Weighted NMSE of the iterates against the source-strength targets ``q``."""
    outputs = solve_outputs(model, y, operator, t, w)
    return weighted_sum(outputs, lambda x_hat: nmse(x_hat, q, column_norms), w)


OBJECTIVES = {'nmse': nmse_loss}


def build_loss(config):
    """Bind the objective ``config.loss`` to the iterate weight ``config.w``."""
    return partial(OBJECTIVES[config.loss], w=config.w)
