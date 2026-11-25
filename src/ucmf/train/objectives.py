"""Training objectives: one function ``f(solve, y, q, t, operator, column_norms, *, w)`` each.

``solve(y, operator, t)`` returns the iterates ``x_1, ..., x_t`` as ``(t, B, n)``, eager or
compiled. Each objective returns the loss; :func:`weighted_mean` explains ``w``.
"""

import math
from functools import partial

from ..scaling import to_source

import torch


def _partition(group, batch_size, train_ratio, generator):
    """Split the indices ``group`` per problem at random into ``(B, k)`` train and the val rest."""
    num_train = round(train_ratio * group.numel())
    order = torch.rand(batch_size, group.numel(), generator=generator, device=group.device).argsort(dim=-1)
    return group[order[:, :num_train]], group[order[:, num_train:]]


def split_rows(operator, y, train_ratio, generator):
    """CV split of the scalar rows per problem; real and imaginary parts are independent.

    Returns ``(op_t, train_rows, val_rows)`` with local row indices of shape ``(B, k)``;
    ``op_t`` needs ``select_rows`` (a dense operator).
    """
    train, val = _partition(torch.arange(y.shape[-1], device=y.device), y.shape[0], train_ratio, generator)
    train, val = train.sort(dim=-1).values, val.sort(dim=-1).values
    return operator.select_rows(train), train, val


def split_complex_rows(operator, y, train_ratio, generator):
    """CV split of the complex CSM entries per problem; Re and Im of an entry stay together.

    Diagonal (real-only) and off-diagonal entries are split separately with the same ratio, so
    all problems get the same number of rows. Requires a ``CMFDictionary``; see :func:`split_rows`.
    """
    hvec = operator.hvec
    off_diagonal = hvec.real_i != hvec.real_j
    real_rows = torch.arange(hvec.num_real, device=y.device)
    imag_rows = hvec.num_real + off_diagonal.cumsum(0) - 1  # imaginary row of each off-diagonal real row
    diagonal_t, diagonal_v = _partition(real_rows[~off_diagonal], y.shape[0], train_ratio, generator)
    off_t, off_v = _partition(real_rows[off_diagonal], y.shape[0], train_ratio, generator)
    train = torch.cat((diagonal_t, off_t, imag_rows[off_t]), dim=-1).sort(dim=-1).values
    val = torch.cat((diagonal_v, off_v, imag_rows[off_v]), dim=-1).sort(dim=-1).values
    return operator.select_rows(train), train, val


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
