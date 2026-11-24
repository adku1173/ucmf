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
    collector = IntermediateOutputCollector()

    def all_intermediates(y, **forward_kwargs):
        previous_callbacks = model.callbacks
        model.callbacks = [*previous_callbacks, collector]
        try:
            model(y, t=t_max, **forward_kwargs)
        finally:
            model.callbacks = previous_callbacks
        return torch.stack(collector.intermediates)

    return torch.compile(all_intermediates, mode=mode)
