"""Acoular solver adapters for the PyTorch ISTA/FISTA solvers.

The solvers plug into Acoular's inverse-problem interface (Acoular branch
``cmf-problem-assembly``) and solve an
:class:`acoular.solvers.L1RegularizedLeastSquaresProblem`, e.g. inside
:class:`acoular.BeamformerCMF`::

    problem = L1RegularizedLeastSquaresProblem(
        solver=FISTATorchSolver(device='cuda'),
        alpha=1e-9,
        nonnegative=True,
        dictionary_scaling='unit_l2',
    )
    bf = acoular.BeamformerCMF(freq_data=ps, steer=st, problem=problem)

The regularization follows Acoular's LassoLars convention: the problem
receives data multiplied by ``problem.unit_mult``, and the solver uses
``lam = problem.alpha * problem.unit_mult``. Since the ucmf objective
``||y - A x||² / (2 m) + lam * ||x||₁`` equals scikit-learn's Lasso
objective, results are directly comparable to ``LassoLarsSolver`` with
``fit_intercept=False``.

Import Acoular before ``ucmf`` (``import acoular`` first): ``ucmf`` loads
NumPy via PyTorch, and Acoular must be loaded before NumPy.
"""

from abc import abstractmethod

from acoular.internal import digest
from acoular.solvers import LeastSquaresSolver, SolverOutput

from .optim import FISTA, ISTA, SolutionChangeStopper
from .optim.operator import MatrixMult

import torch
from traits.api import Enum, Float, Int, Property, Str, Union, cached_property


class TorchSolver(LeastSquaresSolver):
    """Base class for Acoular solvers backed by a ucmf PyTorch solver.

    Requires a problem providing ``alpha``, ``unit_mult`` and ``nonnegative``
    (:class:`acoular.solvers.L1RegularizedLeastSquaresProblem`). The
    dictionary, data and optional start value are copied to :attr:`device`
    with :attr:`precision`; the solution is returned as a NumPy array.
    """

    #: Torch device used for solving, e.g. ``'cpu'`` or ``'cuda'``.
    device = Str('cpu')

    #: Floating-point precision of the solve.
    precision = Enum('float64', 'float32')

    #: Number of iterations per frequency bin (maximum if :attr:`tol` is set).
    n_iter = Int(5000)

    #: Optional relative solution-change tolerance for early stopping via
    #: :class:`ucmf.optim.SolutionChangeStopper`. ``None`` (default) runs exactly
    #: :attr:`n_iter` iterations without callbacks.
    tol = Union(None, Float)

    #: Number of steps between stopping checks if :attr:`tol` is set.
    check_every = Int(100)

    #: Unique identifier for this solver configuration. (read-only)
    digest = Property(depends_on=['device', 'precision', 'n_iter', 'tol', 'check_every'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @abstractmethod
    def _create_solver(self, positive, callbacks):
        """Return the ucmf solver module used for one solve."""

    def solve(self, problem, dictionary_matrix, data, start_value=None):
        """Solve one frequency bin.

        With :attr:`tol` set, ``info`` reports ``num_steps`` and ``converged``;
        otherwise it is empty.
        """
        dtype = getattr(torch, self.precision)
        operator = MatrixMult(torch.as_tensor(dictionary_matrix, dtype=dtype, device=self.device))
        y = torch.as_tensor(data, dtype=dtype, device=self.device)
        x0 = None if start_value is None else torch.as_tensor(start_value, dtype=dtype, device=self.device)
        stopper = None if self.tol is None else SolutionChangeStopper(tol=self.tol, check_every=self.check_every)
        solver = self._create_solver(positive=problem.nonnegative, callbacks=[] if stopper is None else [stopper])
        with torch.no_grad():
            x = solver(y, x0=x0, t=self.n_iter, operator=operator, lam=problem.alpha * problem.unit_mult)
        info = {} if stopper is None else {'num_steps': stopper.num_steps, 'converged': stopper.converged}
        return SolverOutput(solution=x[0].cpu().numpy(), info=info)


class ISTATorchSolver(TorchSolver):
    """L1-regularized least squares via :class:`ucmf.ISTA`."""

    def _create_solver(self, positive, callbacks):
        return ISTA(positive=positive, callbacks=callbacks)


class FISTATorchSolver(TorchSolver):
    """L1-regularized least squares via :class:`ucmf.FISTA`."""

    def _create_solver(self, positive, callbacks):
        return FISTA(positive=positive, callbacks=callbacks)
