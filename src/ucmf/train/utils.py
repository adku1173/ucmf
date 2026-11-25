"""Losses, model/data helpers, and the Ray Tune report hook for training."""

import warnings

import ucmf

from ..optim import operator as operator_module
from ..scaling import to_source

import ray.tune
import torch
import torch.nn.functional as F


def batch_loss(config, x_hat, y, q_true, operator, column_norms):
    """Batch-mean ``config.loss`` of the normalized solution ``x_hat`` for ``operator = A U^-1``.

    'mse'/'nmse' compare the source strengths ``x_hat / column_norms`` with ``q_true``;
    'relative_residual' compares ``operator.forward(x_hat)`` with ``y``.
    """
    if config.loss == 'relative_residual':
        residual = operator.forward(x_hat) - y
        numerator = torch.linalg.vector_norm(residual, dim=1) ** 2
        denominator = torch.linalg.vector_norm(y, dim=1).clamp_min(1e-12) ** 2
        return (0.5 * numerator / denominator).mean()
    q_hat = to_source(x_hat, column_norms)
    if config.loss == 'mse':
        return F.mse_loss(q_hat, q_true)
    numerator = torch.linalg.vector_norm(q_hat - q_true, dim=1) ** 2
    denominator = torch.linalg.vector_norm(q_true, dim=1).clamp_min(1e-12) ** 2
    return (numerator / denominator).mean()


def ray_report(metrics):
    """Report to Ray Tune; outside a Tune session Ray ignores the call (its warning is silenced)."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='`report` is meant to only be called', category=UserWarning)
        ray.tune.report(metrics)


def clone_state(model):
    """Detached on-device copy of the ``state_dict``; allocate once, then use ``copy_state``."""
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def copy_state(state, model):
    """Overwrite the preallocated ``state`` in place with the model's current ``state_dict``."""
    for key, value in model.state_dict().items():
        state[key].copy_(value)


def iterate_batches(config, num_samples):
    """Endless minibatch indices on the configured device, without replacement per epoch.

    Every epoch draws a fresh permutation (seeded by ``config.seed``) and splits it into
    batches of ``config.batch_size``; the last batch of an epoch may be smaller.
    """
    generator = torch.Generator(device=config.torch_device).manual_seed(config.seed)
    while True:
        permutation = torch.randperm(num_samples, generator=generator, device=config.torch_device)
        yield from permutation.split(config.batch_size)


def build_operator(config):
    """Build the operator named by ``config.operator_name`` from ``config.operator_kwargs``."""
    return getattr(operator_module, config.operator_name)(**config.operator_kwargs)


def build_model(config):
    """Build the solver named by ``config.model_name`` on the configured device and precision."""
    model = getattr(ucmf, config.model_name)(**config.model_kwargs)
    return model.to(dtype=config.torch_dtype, device=config.torch_device)


def build_optimizer(config, model):
    """Build a fresh Adam optimizer for the model parameters at ``config.learning_rate``."""
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate)


def evaluate(config, model, y, q, operator, column_norms, stage):
    """No-grad loss of the model at solver depth ``stage`` on ``(y, q)``."""
    with torch.no_grad():
        x_hat = model(y, t=stage, operator=operator)
        return batch_loss(config, x_hat, y, q, operator, column_norms).item()


def prepare_data(config, y_train, y_val, q_train, q_val):
    """Validate the training data and move it to the config's device and precision.

    Returns
    -------
    tuple
        ``(y_train, y_val, q_train, q_val)``; missing targets stay None.

    Raises
    ------
    ValueError
        If ``y_train``/``y_val`` are not 2D, or a supervised loss misses targets.
    """
    if y_train.ndim != 2 or y_val.ndim != 2:
        raise ValueError(f'y_train and y_val must be 2D, got {tuple(y_train.shape)} and {tuple(y_val.shape)}.')
    if config.supervised and (q_train is None or q_val is None):
        raise ValueError(f'loss {config.loss!r} requires q_train and q_val.')
    return tuple(
        None if tensor is None else tensor.to(dtype=config.torch_dtype, device=config.torch_device)
        for tensor in (y_train, y_val, q_train, q_val)
    )
