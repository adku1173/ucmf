"""Training objectives: one function ``f(solve, y, q, t, operator, column_norms, *, w)`` each.

``solve(y, operator, t)`` returns the iterates ``x_1, ..., x_t`` as ``(t, B, n)``, eager or
compiled. Each objective returns the loss; :func:`weighted_mean` explains ``w``.
"""

import inspect
import math
from functools import partial

from ..scaling import lambda_max, normalize_operator, to_source

import torch


def _partition(group, batch_size, train_ratio, generator):
    """Split the indices ``group`` per problem at random into ``(B, k)`` train and the val rest."""
    num_train = round(train_ratio * group.numel())
    order = torch.rand(batch_size, group.numel(), generator=generator, device=group.device).argsort(dim=-1)
    return group[order[:, :num_train]], group[order[:, num_train:]]


def split_rows(operator, y, train_ratio, generator):
    """CV split of the scalar rows per problem; real and imaginary parts are independent.

    Returns ``(op_t, train_rows, val_rows)`` with local row indices of shape ``(B, k)``;
    ``op_t`` needs ``select_rows`` (a dense operator). A ``CMFDictionary`` keeps its CMF
    coordinates, so the reduced models act on the principal submatrix of their weight.
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


def _partition_mics(operator, y, train_ratio, generator):
    """Split the microphones of a CMF operator per problem into train and val microphones."""
    return _partition(torch.arange(operator.hvec.num_mics, device=y.device), y.shape[0], train_ratio, generator)


def split_mics(operator, y, train_ratio, generator):
    """CV split of the microphones per problem into the CSM blocks of disjoint microphone sets.

    The cross terms between both sets stay unused; ``train_ratio`` refers to microphones.
    Requires a CMF operator over all rows, so that its ``rows`` index ``y``; see :func:`split_rows`.
    """
    train_mics, val_mics = _partition_mics(operator, y, train_ratio, generator)
    op_t = operator.select_mics(train_mics)
    return op_t, op_t.rows, operator.mic_rows(val_mics)


def split_mics_cross(operator, y, train_ratio, generator):
    """Like :func:`split_mics`, but validate on all other rows: val block and cross terms."""
    train_mics, _ = _partition_mics(operator, y, train_ratio, generator)
    op_t = operator.select_mics(train_mics)
    outside = torch.ones_like(y, dtype=torch.bool).scatter(-1, op_t.rows, False)
    val = torch.arange(y.shape[-1], device=y.device).expand_as(y)[outside].view(y.shape[0], -1)
    return op_t, op_t.rows, val


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


SPLITS = {'rows': split_rows, 'complex_rows': split_complex_rows, 'mics': split_mics, 'mics_cross': split_mics_cross}


def cv_loss(solve, y, q, t, operator, column_norms, *, w, generator, split='rows', train_ratio=0.8):
    """Weighted CV residual ``||D_v x - y_v||^2 / (2 m_v)`` of iterates solved on the train rows.

    Column norms and ``lambda_max`` of the train subset also scale the val rows (ADR 0009);
    ``D_v x`` is the full forward of ``x / u_t``, so no val operator is formed.
    """
    op_t, train, val = SPLITS[split](operator, y, train_ratio, generator)
    D_t, u_t = normalize_operator(op_t)
    lam = lambda_max(D_t, y.gather(-1, train)).unsqueeze(-1)
    y_t, y_v = y.gather(-1, train) / lam, y.gather(-1, val) / lam

    def residual(x_hat):
        error = operator.forward(x_hat / u_t).gather(-1, val) - y_v
        return (torch.linalg.vector_norm(error, dim=-1).square() / (2 * val.shape[-1])).mean()

    return weighted_mean(solve(y_t, D_t, t), residual, w)


OBJECTIVES = {'nmse': nmse_loss, 'lasso': lasso_loss, 'bic': bic_loss, 'cv': cv_loss}
SUPPLIED = {  # keywords bound by build_loss from the config, not by loss_kwargs
    'w': lambda config: config.w,
    'generator': lambda config: torch.Generator(config.torch_device).manual_seed(config.seed),
}


def build_loss(config):
    """Bind ``config.loss`` to ``config.loss_kwargs`` and the :data:`SUPPLIED` keywords it takes."""
    objective = OBJECTIVES[config.loss]
    parameters = inspect.signature(objective).parameters
    supplied = {name: make(config) for name, make in SUPPLIED.items() if name in parameters}
    return partial(objective, **supplied, **config.loss_kwargs)
