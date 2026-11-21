"""Problem-independent callbacks for iterative solvers (no acoustic dependencies)."""

import torch


class IterationCallback:
    """Base class for callbacks receiving the solver and its current iterate.

    Override ``on_step_begin`` or ``on_step_end``. Set ``solver.stop = True``
    in either method to stop the current solve. Iteration indices are zero-based.
    """

    def on_step_begin(self, solver, x, **kwargs):
        """Called before the step, with ``solver.iter`` set to its index."""

    def on_step_end(self, solver, x, **kwargs):
        """Receive gamma, theta, A, W, y; classical solvers report W=A."""


class StepParamCollector(IterationCallback):
    """Record detached step sizes and thresholds at each iteration.

    ``gammas_tensor`` and ``thetas_tensor`` stack the collected values along
    their first (iteration) dimension. Remaining dimensions follow the supplied
    parameters: scalars give ``(T,)``, per-sample values give ``(T, B, 1)``.
    Call ``reset()`` before reusing the collector for a new solve.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear accumulated gamma and theta lists."""
        self.gammas = []
        self.thetas = []

    def on_step_end(self, solver, x, gamma=None, theta=None, **kwargs):
        """Store tensors and Python scalars on the iterate's device and dtype."""
        if gamma is not None:
            self.gammas.append(torch.as_tensor(gamma, device=x.device, dtype=x.dtype).detach().clone())
        if theta is not None:
            self.thetas.append(torch.as_tensor(theta, device=x.device, dtype=x.dtype).detach().clone())

    @property
    def gammas_tensor(self):
        """Stack collected step sizes along the iteration dimension."""
        if not self.gammas:
            msg = 'No gamma values collected. Run a forward pass first.'
            raise RuntimeError(msg)
        return torch.stack(self.gammas)

    @property
    def thetas_tensor(self):
        """Stack collected thresholds along the iteration dimension."""
        if not self.thetas:
            msg = 'No theta values collected. Run a forward pass first.'
            raise RuntimeError(msg)
        return torch.stack(self.thetas)


class IntermediateOutputCollector(IterationCallback):
    """Record iterates WITHOUT detaching their computational graphs.

    ``intermediates`` contains one ``(B, n)`` tensor per completed step. The
    buffer resets automatically at the beginning of iteration zero. For a
    zero-iteration solve, no callbacks run; call ``reset()`` explicitly.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear collected iterates."""
        self.intermediates = []

    def on_step_begin(self, solver, x, **kwargs):
        """Clear the buffer at the first iteration of a new solve."""
        if getattr(solver, 'iter', None) == 0:
            self.reset()

    def on_step_end(self, solver, x, **kwargs):
        """Append the current iterate with its computational graph attached."""
        self.intermediates.append(x)


class SolutionChangeStopper(IterationCallback):
    """Stop once consecutive iterates change by less than ``tol``.

    The default relative criterion is
    ``||x_k - x_previous|| / max(||x_k||, ||x_previous||, eps) < tol``.
    Set ``relative=False`` to use absolute changes. All samples in a batch
    must satisfy the criterion. State resets at iteration zero of each solve.
    ``stop_iter`` is zero-based; ``n_steps`` counts completed steps.
    """

    def __init__(self, tol=1e-6, relative=True, eps=1e-12):
        self.tol = float(tol)
        self.relative = bool(relative)
        self.eps = float(eps)
        self.reset()

    def reset(self):
        """Clear state from a previous solver run."""
        self.previous = None
        self.last_change = None
        self.converged = False
        self.stop_iter = None
        self.n_steps = 0

    @staticmethod
    def _batch_norm(x):
        if x.ndim == 1:
            return torch.linalg.vector_norm(x).reshape(1)
        return torch.linalg.vector_norm(x.reshape(x.shape[0], -1), dim=1)

    def on_step_begin(self, solver, x, **kwargs):
        """Store the pre-step iterate for comparison at step end."""
        if getattr(solver, 'iter', None) == 0:
            self.reset()
        self.previous = x.detach().clone()

    def on_step_end(self, solver, x, **kwargs):
        """Request stopping when all iterate changes are below tolerance."""
        if self.previous is None:
            self.previous = x.detach().clone()
            return

        x_det = x.detach()
        diff = self._batch_norm(x_det - self.previous)
        if self.relative:
            scale = torch.maximum(self._batch_norm(x_det), self._batch_norm(self.previous))
            diff = diff / scale.clamp_min(self.eps)

        self.last_change = diff.detach().clone()
        self.n_steps = int(getattr(solver, 'iter', 0)) + 1
        if torch.all(diff < self.tol):
            self.converged = True
            self.stop_iter = int(getattr(solver, 'iter', self.n_steps - 1))
            solver.stop = True
