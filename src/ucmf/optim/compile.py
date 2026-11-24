"""Compile an unfolded solver once and expose all intermediate iterates."""

from .callbacks import IntermediateOutputCollector

import torch


def compile_unrolled(model, t_max, mode='reduce-overhead'):
    """Return a compiled function giving all iterates ``x_1, ..., x_{t_max}``.

    ``torch.compile`` unrolls the iteration loop, so every distinct ``t`` would
    otherwise be a separate graph and trigger a recompilation (e.g. in staged
    training with t = 1, 2, ..., T). Here the solver is compiled once for
    ``t_max``; the result of ``t`` iterations is ``out[t - 1]``. Iterations
    after ``t`` are still computed; they do not change ``x_t`` and contribute
    zero gradient to it.

    The model is an input of the compiled graph, not a captured constant, so new
    instances of the same model type (e.g. in a hyperparameter search) reuse the
    graph. Each further model type, ``t_max``, or ``mode`` compiles anew; after
    ``torch._dynamo.config.recompile_limit`` (8) such variants, torch falls back
    to eager execution.

    Parameters
    ----------
    model : torch.nn.Module
        Solver, e.g. ``AdaLFISTASW``. Its own callbacks must be compilable (no
        host syncs such as ``.item()``); they stay active.
    t_max : int
        Number of iterations of the compiled graph.
    mode : str, optional
        ``torch.compile`` mode; the default ``'reduce-overhead'`` uses CUDA graphs on GPU.

    Returns
    -------
    callable
        ``f(y, **forward_kwargs)`` returning a tensor of shape ``(t_max, B, n)``;
        ``forward_kwargs`` are passed to ``model.forward`` with ``t = t_max``.
    """

    def all_intermediates(solver, y, **forward_kwargs):
        collector = IntermediateOutputCollector()
        previous_callbacks = solver.callbacks
        solver.callbacks = [*previous_callbacks, collector]
        try:
            solver(y, t=t_max, **forward_kwargs)
        finally:
            solver.callbacks = previous_callbacks
        return torch.stack(collector.intermediates)

    compiled = torch.compile(all_intermediates, mode=mode)
    return lambda y, **forward_kwargs: compiled(model, y, **forward_kwargs)
