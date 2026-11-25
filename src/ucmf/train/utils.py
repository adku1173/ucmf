"""Model/data helpers and the Ray Tune report hook for training."""

import warnings

import ucmf

from ..optim import operator as operator_module
from ..optim.compile import all_iterates, compile_unrolled
from ..scaling import lambda_max

import ray.tune
import torch


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
    full batches of ``config.batch_size``; the remaining ``num_samples % batch_size``
    indices are dropped, so all batches share one shape (one compiled graph).
    """
    generator = torch.Generator(device=config.torch_device).manual_seed(config.seed)
    num_batches = num_samples // config.batch_size
    while True:
        permutation = torch.randperm(num_samples, generator=generator, device=config.torch_device)
        yield from permutation[: num_batches * config.batch_size].view(num_batches, config.batch_size)


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


def build_solve(config, model):
    """Return ``solve(y, operator, t)`` giving the iterates ``(t, B, n)`` of ``model``.

    With ``config.compile``, one graph of depth ``config.stages`` serves all stages;
    stage ``t`` takes its first ``t`` iterates.
    """
    if config.compile:
        compiled = compile_unrolled(model, config.stages)
        return lambda y, operator, t: compiled(y, operator=operator)[:t]
    return lambda y, operator, t: all_iterates(model, y, t, operator=operator)


def evaluate(objective, solve, y, q, operator, column_norms, stage):
    """No-grad ``objective`` at solver depth ``stage`` on ``(y, q)``."""
    with torch.no_grad():
        return objective(solve, y, q, stage, operator, column_norms).item()


def prepare_data(config, operator, y_train, y_val, q_train, q_val):
    """Validate the training data, move it to the config's device and precision, and scale it.

    With ``config.data_scaling == 'equireg'``, measurements and targets of every sample
    are divided by ``lambda_max`` of the normalized ``operator``.

    Returns
    -------
    tuple
        ``(y_train, y_val, q_train, q_val)``; missing targets stay None.

    Raises
    ------
    ValueError
        If ``y_train``/``y_val`` are not 2D, ``y_train`` has fewer samples than one batch,
        a supervised loss misses targets, or the operator does not match the configured
        dtype and device.
    """
    if y_train.ndim != 2 or y_val.ndim != 2:
        raise ValueError(f'y_train and y_val must be 2D, got {tuple(y_train.shape)} and {tuple(y_val.shape)}.')
    if y_train.shape[0] < config.batch_size:
        raise ValueError(f'y_train has {y_train.shape[0]} samples, fewer than batch_size={config.batch_size}.')
    if config.supervised and (q_train is None or q_val is None):
        raise ValueError(f'loss {config.loss!r} requires q_train and q_val.')
    y_train, y_val, q_train, q_val = (
        None if tensor is None else tensor.to(dtype=config.torch_dtype, device=config.torch_device)
        for tensor in (y_train, y_val, q_train, q_val)
    )
    if operator.dtype != y_train.dtype:
        raise ValueError(f'operator must have dtype {y_train.dtype}, got {operator.dtype}.')
    if operator.device != y_train.device:
        raise ValueError(f'operator must be on device {y_train.device}, got {operator.device}.')
    if config.data_scaling == 'equireg':
        y_train, q_train = _equiregularize(operator, y_train, q_train)
        y_val, q_val = _equiregularize(operator, y_val, q_val)
    return y_train, y_val, q_train, q_val


def _equiregularize(operator, y, q):
    """Divide measurements and optional targets per sample by ``lambda_max``."""
    lam = lambda_max(operator, y).unsqueeze(-1)
    return y / lam, None if q is None else q / lam
