"""Problem-independent callbacks for iterative solvers (no acoustic dependencies)."""

import torch


class IterationCallback:
    """Base class for callbacks receiving the solver and its current iterate.

    Override :meth:`on_step_begin` or :meth:`on_step_end`. Set ``solver.stop = True``
    in either method to stop the current solve. Iteration indices are zero-based.
    """

    def on_step_begin(self, solver, x, **kwargs):
        """Handle the beginning of an iteration.

        Parameters
        ----------
        solver : ISTA
            The running solver; ``solver.iter`` is the index of the coming step.
        x : Tensor
            Current iterate of shape ``(B, n)``; may be modified in place.
        **kwargs
            No keywords are passed at the beginning of a step.
        """

    def on_step_end(self, solver, x, **kwargs):
        """Handle the end of an iteration.

        Parameters
        ----------
        solver : ISTA
            The running solver; ``solver.iter`` is the index of the completed step.
        x : Tensor
            Iterate after the step, shape ``(B, n)``, with its autograd graph.
        **kwargs
            ``gamma`` and ``theta`` of the step, the detached measurement operator
            ``A``, the effective weight operator ``W`` (``A`` for classical solvers),
            ``y``, and model-specific entries such as ``W_recurrent``.
        """


class StepParamCollector(IterationCallback):
    """Record detached step sizes and thresholds at each iteration.

    Call :meth:`reset` before reusing the collector for a new solve.

    Attributes
    ----------
    gammas, thetas : list of Tensor
        Collected values on the iterate's device and dtype, one per completed step.
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
        """Collected step sizes stacked along a leading iteration axis.

        Scalar values give shape ``(T,)``, per-sample values ``(T, B, 1)``.

        Raises
        ------
        RuntimeError
            If no values were collected.
        """
        if not self.gammas:
            msg = 'No gamma values collected. Run a forward pass first.'
            raise RuntimeError(msg)
        return torch.stack(self.gammas)

    @property
    def thetas_tensor(self):
        """Collected thresholds stacked along a leading iteration axis.

        Scalar values give shape ``(T,)``, per-sample values ``(T, B, 1)``.

        Raises
        ------
        RuntimeError
            If no values were collected.
        """
        if not self.thetas:
            msg = 'No theta values collected. Run a forward pass first.'
            raise RuntimeError(msg)
        return torch.stack(self.thetas)


class IntermediateOutputCollector(IterationCallback):
    """Record iterates without detaching their computational graphs.

    The buffer resets automatically at the beginning of iteration zero. A
    zero-iteration solve runs no callbacks; call :meth:`reset` explicitly.

    Attributes
    ----------
    intermediates : list of Tensor
        One ``(B, n)`` iterate per completed step.
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

    The relative criterion is
    ``||x_k - x_previous|| / max(||x_k||, ||x_previous||, eps) < tol``; all samples
    of a batch must satisfy it. State resets at iteration zero of each solve.

    The criterion is evaluated only for every ``check_every``-th step (after
    ``check_every``, ``2 * check_every``, ... completed steps), comparing that
    step's consecutive iterates. Other steps cost no tensor operations and,
    on GPUs, no host-device synchronization. Stopping can therefore happen up
    to ``check_every - 1`` steps after the change first fell below ``tol``.

    Parameters
    ----------
    tol : float, optional
        Tolerance of the change (default 1e-6).
    relative : bool, optional
        If True (default), use the relative change, otherwise ``||x_k - x_previous||``.
    eps : float, optional
        Lower bound of the relative scale (default 1e-12).
    check_every : int, optional
        Evaluate the criterion every ``check_every`` steps (default 100).

    Attributes
    ----------
    converged : bool
        Whether the last solve stopped by this criterion.
    stop_iter : int or None
        Zero-based index of the step at which the solve stopped.
    num_steps : int
        Number of completed steps of the last solve.
    last_change : Tensor or None
        Per-sample change at the last checked step.
    """

    def __init__(self, tol=1e-6, relative=True, eps=1e-12, check_every=100):
        self.tol = float(tol)
        self.relative = bool(relative)
        self.eps = float(eps)
        self.check_every = int(check_every)
        self.reset()

    def reset(self):
        """Clear state from a previous solver run."""
        self.previous = None
        self.last_change = None
        self.converged = False
        self.stop_iter = None
        self.num_steps = 0

    @staticmethod
    def _batch_norm(x):
        if x.ndim == 1:
            return torch.linalg.vector_norm(x).reshape(1)
        return torch.linalg.vector_norm(x.reshape(x.shape[0], -1), dim=1)

    def _is_check_step(self, solver):
        return (solver.iter + 1) % self.check_every == 0

    def on_step_begin(self, solver, x, **kwargs):
        """Store the pre-step iterate if this step is checked."""
        if solver.iter == 0:
            self.reset()
        if self._is_check_step(solver):
            self.previous = x.detach().clone()

    def on_step_end(self, solver, x, **kwargs):
        """On checked steps, request stopping when all iterate changes are below tolerance."""
        self.num_steps = solver.iter + 1
        if not self._is_check_step(solver):
            return

        x_det = x.detach()
        diff = self._batch_norm(x_det - self.previous)
        if self.relative:
            scale = torch.maximum(self._batch_norm(x_det), self._batch_norm(self.previous))
            diff = diff / scale.clamp_min(self.eps)

        self.last_change = diff.detach().clone()
        if torch.all(diff < self.tol):
            self.converged = True
            self.stop_iter = solver.iter
            solver.stop = True
