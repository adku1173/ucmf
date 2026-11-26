"""Compile an unfolded solver once and expose all intermediate iterates."""

from .callbacks import IntermediateOutputCollector

import torch


def all_iterates(model, y, t, **forward_kwargs):
    """Run ``t >= 1`` iterations and return all iterates ``x_1, ..., x_t``.

    Parameters
    ----------
    model : torch.nn.Module
        Solver; its callbacks are restored afterwards, also if the solve raises.
    y : Tensor
        Measurements of shape ``(B, m)``.
    t : int
        Number of iterations.
    **forward_kwargs
        Further arguments of ``model.forward``, e.g. ``operator``.

    Returns
    -------
    Tensor
        Iterates of shape ``(t, B, n)`` with their computational graphs.
    """
    collector = IntermediateOutputCollector()
    previous_callbacks = model.callbacks
    model.callbacks = [*previous_callbacks, collector]
    try:
        model(y, t=t, **forward_kwargs)
    finally:
        model.callbacks = previous_callbacks
    return torch.stack(collector.intermediates)


def compile_unrolled(model, t_max, mode='reduce-overhead'):
    """Return a compiled function giving the iterates ``x_1, ..., x_t`` for any ``t <= t_max``.

    ``torch.compile`` unrolls the iteration loop, so every distinct ``t`` would
    otherwise be a separate graph and trigger a recompilation (e.g. in staged
    training with t = 1, 2, ..., T). Here the solver is compiled once for
    ``t_max``, and ``t`` enters the graph as a tensor (``active_iterations`` of the
    model's ``forward``): iterations after ``t`` are still computed but keep
    ``x_t``, so they contribute exactly zero gradient. Iterating on with the untrained
    parameters of later iterations could overflow and turn the gradient into NaN
    (``0 * inf``).

    The model is an input of the compiled graph, not a captured constant, so new
    instances of the same model type (e.g. in a hyperparameter search) reuse the
    graph. Each further model type, ``t_max``, or ``mode`` compiles anew; after
    ``torch._dynamo.config.recompile_limit`` (8) such variants, torch falls back
    to eager execution.

    Parameters
    ----------
    model : torch.nn.Module
        Unfolded solver, e.g. ``AdaLFISTASW``. Its own callbacks must be compilable (no
        host syncs such as ``.item()``); they stay active.
    t_max : int
        Number of iterations of the compiled graph.
    mode : str, optional
        ``torch.compile`` mode; the default ``'reduce-overhead'`` uses CUDA graphs on GPU.

    Returns
    -------
    callable
        ``f(y, t, **forward_kwargs)`` returning the iterates of shape ``(t, B, n)``;
        ``forward_kwargs`` are passed to ``model.forward``.
    """

    def unrolled(solver, y, active_iterations, **forward_kwargs):
        # t_max stays a graph constant, active_iterations is a graph input
        return all_iterates(solver, y, t_max, active_iterations=active_iterations, **forward_kwargs)

    compiled = torch.compile(unrolled, mode=mode)

    def solve(y, t, **forward_kwargs):
        active_iterations = torch.full((), t, device=y.device)  # a fill kernel, no host-to-device copy
        return compiled(model, y, active_iterations, **forward_kwargs)[:t]

    return solve
