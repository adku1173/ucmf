"""CMF of three synthetic sources with ucmf's ISTA and FISTA as Acoular solvers.

Uses the three-source setup from Acoular's introductory examples and compares
the custom torch solvers with Acoular's LassoLars solver on the same problem.

Run from the repository root: uv run python examples/sparse_recovery.py
"""

from pathlib import Path

import acoular as ac
import matplotlib.pyplot as plt
from acoular.solvers import L1RegularizedLeastSquaresProblem, LassoLarsSolver

from ucmf.acoular import FISTATorchSolver, ISTATorchSolver

ac.config.global_caching = 'none'

# Simulate three sources in front of the 64-microphone array.
sfreq = 51200
num_samples = sfreq
mg = ac.MicGeom(file=Path(ac.__file__).parent / 'xml' / 'array_64.xml')
n1 = ac.WNoiseGenerator(sample_freq=sfreq, num_samples=num_samples, seed=1)
n2 = ac.WNoiseGenerator(sample_freq=sfreq, num_samples=num_samples, seed=2, rms=0.7)
n3 = ac.WNoiseGenerator(sample_freq=sfreq, num_samples=num_samples, seed=3, rms=0.5)
p1 = ac.PointSource(signal=n1, mics=mg, loc=(-0.1, -0.1, -0.3))
p2 = ac.PointSource(signal=n2, mics=mg, loc=(0.15, 0, -0.3))
p3 = ac.PointSource(signal=n3, mics=mg, loc=(0, 0.1, -0.3))
ts = ac.Mixer(source=p1, sources=[p2, p3])

# Define spectral processing, grid and steering vector.
ps = ac.PowerSpectra(source=ts, block_size=128, window='Hanning')
rg = ac.RectGrid(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=-0.3, increment=0.01)
st = ac.SteeringVector(grid=rg, mics=mg)

# Solve the same L1-regularized CMF problem with three solvers.
solvers = {
    'LassoLars': LassoLarsSolver(backend_kwargs={'fit_intercept': False}),
    'ISTA': ISTATorchSolver(),
    'FISTA': FISTATorchSolver(),
}
plt.figure(figsize=(12, 3.5))
for i, (name, solver) in enumerate(solvers.items(), start=1):
    problem = L1RegularizedLeastSquaresProblem(
        solver=solver, alpha=1e-9, nonnegative=True, dictionary_scaling='unit_l2'
    )
    bf = ac.BeamformerCMF(freq_data=ps, steer=st, problem=problem)
    Lm = ac.L_p(bf.synthetic(8000, 3))

    plt.subplot(1, 3, i)
    plt.imshow(Lm.T, origin='lower', vmin=Lm.max() - 20, extent=rg.extent, interpolation='nearest')
    plt.colorbar()
    plt.title(name)
plt.tight_layout()
plt.show()
