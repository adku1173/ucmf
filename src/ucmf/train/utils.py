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
    the single 0-dim entry shared by all samples of the batch. 'per_dictionary' drops the
    incomplete batch of every entry, which favors frequent entries in small datasets.

    Raises
    ------
    ValueError
        If 'per_dictionary' sampling has no bank or no entry with a full batch.
    """
    if config.batch_sampling == 'per_dictionary':
        if index is None or torch.bincount(index).max() < config.batch_size:
            raise ValueError("batch_sampling='per_dictionary' needs a bank entry with at least batch_size samples.")
    return _batches(config, num_samples, index)


def _batches(config, num_samples, index):
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


def validation_batches(config, y, q, index):
    """Split a validation set into fixed batches ``(y, q, problems)`` of ``config.batch_size``.

    A permutation seeded by ``config.seed`` precedes the split, so the dropped incomplete
    batch holds random samples, not the end of ordered data (e.g. the highest frequencies).
    ``problems`` selects one bank entry per sample, or is None without a bank.
    """
    generator = torch.Generator(device=config.torch_device).manual_seed(config.seed)
    B, num_samples = config.batch_size, y.shape[0]
    permutation = torch.randperm(num_samples, generator=generator, device=config.torch_device)
    # Keep a multiple of B (drop_last) and split it into rows of B sample indices.
    blocks = permutation[: num_samples // B * B].view(-1, B)
    batches = []
    for samples in blocks:
        q_batch = None if q is None else q[samples]  # targets only for supervised losses
        problems = None if index is None else index[samples]  # bank entries only with a bank
        batches.append((y[samples], q_batch, problems))
    return batches


def batch_operator(operator, column_norms, problems):
    """Operator and column norms of a batch: the bank entries ``problems``, or all for None."""
    if problems is None:
        return operator, column_norms
    return operator.select_problems(problems), column_norms[problems]


def build_operator(config):
    """Build the operator named by ``config.operator_name`` from ``config.operator_kwargs``."""
    return getattr(operator_module, config.operator_name)(**config.operator_kwargs)


def build_model(config):
    """Build the solver named by ``config.model_name`` on the configured device and precision.

    The random initialization (e.g. the LSTM of NA models) is seeded by ``config.seed``. It runs
    on the CPU in a forked RNG, so the caller's global RNG stays untouched.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = getattr(ucmf, config.model_name)(**config.model_kwargs)
    return model.to(dtype=config.torch_dtype, device=config.torch_device)


def build_optimizer(config, model):
    """Build a fresh Adam optimizer for the model parameters at ``config.learning_rate``."""
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate)


def build_solve(config, model):
    """Return ``solve(y, operator, t)`` giving the iterates ``(t, B, n)`` of ``model``.

    With ``config.compile``, one graph of depth ``config.stages`` serves all stages;
    in stage ``t`` the later iterations keep ``x_t``.
    """
    if config.compile:
        compiled = compile_unrolled(model, config.stages)
        return lambda y, operator, t: compiled(y, t, operator=operator)
    return lambda y, operator, t: all_iterates(model, y, t, operator=operator)


def evaluate(objective, solve, batches, operator, column_norms, stage):
    """No-grad mean ``objective`` at solver depth ``stage`` over the :func:`validation_batches`."""
    with torch.no_grad():
        losses = [
            objective(solve, y, q, stage, *batch_operator(operator, column_norms, problems))
            for y, q, problems in batches
        ]
        return torch.stack(losses).mean().item()


def prepare_data(config, operator, y, q=None, index=None):
    """Move one dataset to the config's device and precision, validate it, and scale it.

    With ``config.data_scaling == 'equireg'``, measurements and targets of every sample
    are divided by ``lambda_max`` of the normalized ``operator``. For a dictionary bank
    (a per-problem ``operator`` of ``K`` entries), ``index`` ``(N,)`` gives the bank entry
    of every sample; integer values in ``[0, K)`` are a precondition.

    Returns
    -------
    tuple
        ``(y, q, index)``; missing ones stay None.

    Raises
    ------
    ValueError
        See :func:`_validate_data`.
    """
    y, q = (None if t is None else t.to(dtype=config.torch_dtype, device=config.torch_device) for t in (y, q))
    index = None if index is None else index.to(config.torch_device)
    _validate_data(config, operator, y, q, index)
    if config.data_scaling == 'equireg':
        y, q = _equiregularize(operator, y, q, index, config.batch_size)
    return y, q, index


def _validate_data(config, operator, y, q, index):
    """Check one moved dataset against the config and the operator.

    Raises
    ------
    ValueError
        If ``y`` is not 2D or has fewer samples than one batch, a supervised loss misses
        targets, the operator does not match the configured dtype and device, or ``index``
        is not given exactly for a bank with one entry per sample.
    """
    if y.ndim != 2:
        raise ValueError(f'y must be 2D, got shape {tuple(y.shape)}.')
    if y.shape[0] < config.batch_size:
        raise ValueError(f'y has {y.shape[0]} samples, fewer than batch_size={config.batch_size}.')
    if config.supervised and q is None:
        raise ValueError(f'loss {config.loss!r} requires source-strength targets q.')

    if operator.dtype != y.dtype:
        raise ValueError(f'operator must have dtype {y.dtype}, got {operator.dtype}.')
    if operator.device != y.device:
        raise ValueError(f'operator must be on device {y.device}, got {operator.device}.')

    if (index is not None) != (operator.batch_size is not None):
        raise ValueError('index is required exactly for a dictionary bank (per-problem operator).')
    if index is not None and index.shape != y.shape[:1]:
        raise ValueError(f'index must hold one bank entry per sample, got shape {tuple(index.shape)}.')


def _equiregularize(operator, y, q, index, batch_size):
    """Divide measurements and optional targets per sample by ``lambda_max`` of its dictionary.

    A bank is processed per entry, so no per-sample copy of the dictionaries is formed.
    ``lambda_max`` runs in chunks of ``batch_size`` samples: a matrix-free adjoint of the
    whole dataset at once would not fit (e.g. 30 GiB for 1M samples, 16 mics, 256 points).
    """
    if index is None:
        lam = _chunked_lambda_max(operator, y, batch_size)
    else:
        lam = y.new_empty(y.shape[0])
        for entry in index.unique().tolist():
            mask = index == entry
            lam[mask] = _chunked_lambda_max(operator.select_problems(entry), y[mask], batch_size)
    lam = lam.unsqueeze(-1)
    return y / lam, None if q is None else q / lam


def _chunked_lambda_max(operator, y, batch_size):
    """``lambda_max`` of a shared operator per sample, computed in chunks of ``batch_size``."""
    return torch.cat([lambda_max(operator, chunk) for chunk in y.split(batch_size)])
