"""Runtime of ucmf's torch ISTA/FISTA versus PyLops ISTA/FISTA for CMF.

Solves the three-source CMF problem of ``sparse_recovery.py`` (third-octave
band at 8 kHz) with a fixed number of iterations and without early stopping.
PyLops has no non-negativity constraint, so all solvers solve the same
unconstrained problem (``nonnegative=False``). Only the time spent inside the
solvers is measured (including step-size computation and data transfer);
CSM computation and dictionary assembly are excluded. CPU solvers run single-
and multi-threaded, ucmf additionally on CUDA if available.

Run from the repository root: uv run python examples/benchmark_solvers.py
"""

import time
from pathlib import Path

import acoular as ac
from acoular.solvers import L1RegularizedLeastSquaresProblem, LeastSquaresSolver, SolverBase, SolverOutput

from ucmf.acoular import FISTATorchSolver, ISTATorchSolver

import numpy as np
import torch
from pylops import MatrixMult
from pylops.optimization.sparsity import fista, ista
from threadpoolctl import threadpool_limits
from traits.api import Callable, Float, Instance, Int

N_ITER = 1000
ac.config.global_caching = 'none'


class PylopsSolver(LeastSquaresSolver):
    """PyLops ISTA/FISTA for the ucmf objective without early stopping.

    PyLops minimizes ``||y - A x||² + eps ||x||₁``. With ``eps = 2 m lam`` this
    matches ``||y - A x||² / (2 m) + lam ||x||₁`` for ``lam = alpha * unit_mult``.
    """

    algorithm = Callable()
    n_iter = Int()

    def solve(self, problem, dictionary_matrix, data, start_value=None):
        eps = 2 * dictionary_matrix.shape[0] * problem.alpha * problem.unit_mult
        x, _, _ = self.algorithm(Op=MatrixMult(dictionary_matrix), y=data, niter=self.n_iter, eps=eps, tol=0.0)
        return SolverOutput(solution=x)


class TimedSolver(LeastSquaresSolver):
    """Accumulate the time spent in the wrapped solver."""

    solver = Instance(SolverBase)
    elapsed = Float(0.0)

    def solve(self, problem, dictionary_matrix, data, start_value=None):
        start = time.perf_counter()
        output = self.solver.solve(problem, dictionary_matrix, data, start_value)
        self.elapsed += time.perf_counter() - start
        return output


# Three sources in front of the 64-microphone array; the CSM is computed once.
sfreq = 51200
mg = ac.MicGeom(file=Path(ac.__file__).parent / 'xml' / 'array_64.xml')
n1 = ac.WNoiseGenerator(sample_freq=sfreq, num_samples=sfreq, seed=1)
n2 = ac.WNoiseGenerator(sample_freq=sfreq, num_samples=sfreq, seed=2, rms=0.7)
n3 = ac.WNoiseGenerator(sample_freq=sfreq, num_samples=sfreq, seed=3, rms=0.5)
p1 = ac.PointSource(signal=n1, mics=mg, loc=(-0.1, -0.1, -0.3))
p2 = ac.PointSource(signal=n2, mics=mg, loc=(0.15, 0, -0.3))
p3 = ac.PointSource(signal=n3, mics=mg, loc=(0, 0.1, -0.3))
spectra = ac.PowerSpectra(source=ac.Mixer(source=p1, sources=[p2, p3]), block_size=128, window='Hanning')
ps = ac.PowerSpectraImport(csm=spectra.csm, frequencies=spectra.fftfreq())
rg = ac.RectGrid(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=-0.3, increment=0.01)
st = ac.SteeringVector(grid=rg, mics=mg)


def run(solver):
    """Return the solver time and the CMF map of the third-octave band at 8 kHz."""
    timed = TimedSolver(solver=solver)
    problem = L1RegularizedLeastSquaresProblem(
        solver=timed, alpha=1e-9, nonnegative=False, dictionary_scaling='unit_l2'
    )
    bf = ac.BeamformerCMF(freq_data=ps, steer=st, problem=problem)
    result = bf.synthetic(8000, 3)
    return timed.elapsed, result


n_threads = torch.get_num_threads()
cpu_solvers = {
    'PyLops': {
        'ISTA': PylopsSolver(algorithm=ista, n_iter=N_ITER),
        'FISTA': PylopsSolver(algorithm=fista, n_iter=N_ITER),
    },
    'ucmf CPU float64': {'ISTA': ISTATorchSolver(n_iter=N_ITER), 'FISTA': FISTATorchSolver(n_iter=N_ITER)},
}
results = {}
for threads in (1, n_threads):
    torch.set_num_threads(threads)
    with threadpool_limits(limits=threads, user_api='blas'):
        for name, solvers in cpu_solvers.items():
            for algorithm, solver in solvers.items():
                results[algorithm, name, threads] = run(solver)

if torch.cuda.is_available():
    for precision in ('float64', 'float32'):
        run(FISTATorchSolver(n_iter=1, device='cuda', precision=precision))  # CUDA/cuSOLVER initialization
        for algorithm, solver_type in (('ISTA', ISTATorchSolver), ('FISTA', FISTATorchSolver)):
            solver = solver_type(n_iter=N_ITER, device='cuda', precision=precision)
            results[algorithm, f'ucmf CUDA {precision}', '-'] = run(solver)

print(f'{N_ITER} iterations per frequency bin')
print('| Algorithm | Implementation | Threads | Time [s] | Speed-up vs. PyLops (1 thread) | Rel. diff. to PyLops |')
print('|---|---|---|---|---|---|')
for (algorithm, name, threads), (elapsed, result) in results.items():
    reference_time, reference = results[algorithm, 'PyLops', 1]
    difference = np.linalg.norm(result - reference) / np.linalg.norm(reference)
    print(f'| {algorithm} | {name} | {threads} | {elapsed:.2f} | {reference_time / elapsed:.1f}x | {difference:.1e} |')
