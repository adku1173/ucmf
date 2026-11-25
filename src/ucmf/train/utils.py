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


def iterate_batches(config, num_samples, index=None):
    """Endless minibatches ``(samples, problems)`` on the device, without replacement per epoch.

    Every epoch draws a fresh permutation (seeded by ``config.seed``) and splits it into
    full batches of ``config.batch_size``; incomplete batches are dropped, so all batches
    share one shape (one compiled graph). ``index`` assigns each sample its entry of a
    dictionary bank; ``problems`` selects the entries of a batch (see :func:`batch_operator`):
    None without a bank, ``index[samples]`` for 'random' sampling, and for 'per_dictionary'
    the single 0-dim entry shared by all samples of the batch.
    """
    generator = torch.Generator(device=config.torch_device).manual_seed(config.seed)
    B, device = config.batch_size, config.torch_device
    while True:
        permutation = torch.randperm(num_samples, generator=generator, device=device)
        if config.batch_sampling == 'random':
            for samples in permutation[: num_samples // B * B].view(-1, B):
                yield samples, None if index is None else index[samples]
            continue
        grouped = permutation[index[permutation].argsort(stable=True)]  # random order within each entry
        counts = torch.bincount(index).tolist()
        batches = [chunk for group in grouped.split(counts) for chunk in group.split(B) if len(chunk) == B]
        for i in torch.randperm(len(batches), generator=generator, device=device).tolist():
            yield batches[i], index[batches[i][0]]


def batch_operator(operator, column_norms, problems):
    """Operator and column norms of a batch: the bank entries ``problems``, or all for None."""
    if problems is None:
        return operator, column_norms
    return operator.select_problems(problems), column_norms[problems]


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


def prepare_data(config, operator, y_train, y_val, q_train, q_val, index_train=None, index_val=None):
    """Validate the training data, move it to the config's device and precision, and scale it.

    With ``config.data_scaling == 'equireg'``, measurements and targets of every sample
    are divided by ``lambda_max`` of the normalized ``operator``. For a dictionary bank
    (a per-problem ``operator``), ``index_train`` ``(N,)`` and ``index_val`` ``(M,)``
    give the bank entry of every sample; integer values in ``[0, K)`` are a precondition.

    Returns
    -------
    tuple
        ``(y_train, y_val, q_train, q_val, index_train, index_val)``; missing ones stay None.

    Raises
    ------
    ValueError
        If ``y_train``/``y_val`` are not 2D or have fewer samples than one batch, a supervised
        loss misses targets, the operator does not match the configured dtype and device,
        the indices do not match a bank, or 'per_dictionary' sampling has no full batch.
    """
    if y_train.ndim != 2 or y_val.ndim != 2:
        raise ValueError(f'y_train and y_val must be 2D, got {tuple(y_train.shape)} and {tuple(y_val.shape)}.')
    for name, y in (('y_train', y_train), ('y_val', y_val)):
        if y.shape[0] < config.batch_size:
            raise ValueError(f'{name} has {y.shape[0]} samples, fewer than batch_size={config.batch_size}.')
    if (index_train is None) != (index_val is None) or (index_train is None) != (operator.batch_size is None):
        raise ValueError('index_train and index_val are required exactly for a dictionary bank (per-problem operator).')
    if index_train is not None and (index_train.shape != y_train.shape[:1] or index_val.shape != y_val.shape[:1]):
        raise ValueError('index_train and index_val must hold one bank entry per sample.')
    if config.batch_sampling == 'per_dictionary' and (
        index_train is None or torch.bincount(index_train).max() < config.batch_size
    ):
        raise ValueError("batch_sampling='per_dictionary' needs a bank entry with at least batch_size samples.")
    if config.supervised and (q_train is None or q_val is None):
        raise ValueError(f'loss {config.loss!r} requires q_train and q_val.')
    y_train, y_val, q_train, q_val = (
        None if tensor is None else tensor.to(dtype=config.torch_dtype, device=config.torch_device)
        for tensor in (y_train, y_val, q_train, q_val)
    )
    index_train, index_val = (None if i is None else i.to(config.torch_device) for i in (index_train, index_val))
    if operator.dtype != y_train.dtype:
        raise ValueError(f'operator must have dtype {y_train.dtype}, got {operator.dtype}.')
    if operator.device != y_train.device:
        raise ValueError(f'operator must be on device {y_train.device}, got {operator.device}.')
    if config.data_scaling == 'equireg':
        y_train, q_train = _equiregularize(operator, y_train, q_train, index_train)
        y_val, q_val = _equiregularize(operator, y_val, q_val, index_val)
    return y_train, y_val, q_train, q_val, index_train, index_val


def _equiregularize(operator, y, q, index):
    """Divide measurements and optional targets per sample by ``lambda_max`` of its dictionary.

    A bank is processed per entry, so no per-sample copy of the dictionaries is formed.
    """
    if index is None:
        lam = lambda_max(operator, y)
    else:
        lam = y.new_empty(y.shape[0])
        for entry in index.unique().tolist():
            mask = index == entry
            lam[mask] = lambda_max(operator.select_problems(entry), y[mask])
    lam = lam.unsqueeze(-1)
    return y / lam, None if q is None else q / lam
