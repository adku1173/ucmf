# ucmf

## Installation and checks

Python 3.13 or newer is required.

```bash
uv sync
uv run pytest
uv run ruff check .
uv run python examples/sparse_recovery.py
uv run python examples/benchmark_solvers.py
```

Runtime dependencies are PyTorch and NumPy. The `examples` dependency group
(included in `dev`, installed by `uv sync`) adds Acoular, Matplotlib, PyLops,
and threadpoolctl.
Acoular is taken from the `cmf-problem-assembly` branch via `[tool.uv.sources]`;
the `acoular.solvers` interface used by `ucmf.acoular` is not yet released on PyPI,
so with `pip install -e .` install Acoular separately from that branch:
`pip install "acoular @ git+https://github.com/acoular/acoular.git@cmf-problem-assembly"`.
`import ucmf` never imports Acoular; only `ucmf.acoular` does.
Import Acoular before `ucmf` and NumPy (numba/OpenBLAS threading, see Acoular's
configuration); ruff's isort section order in `pyproject.toml` enforces this, as in Acoular.
TensorFlow, datasets, training code, and the original `fa2026` package are not required.

## Usage

```python
import torch
from ucmf import ISTA, FISTA
from ucmf.optim import SolutionChangeStopper

A = torch.eye(3)                     # real sensing matrix (m, n)
y = torch.tensor([[1.0, 0.0, 0.5]])  # measurements (B, m)
stopper = SolutionChangeStopper(tol=1e-6)
solver = FISTA(positive=True, callbacks=[stopper])
with torch.no_grad():
    x_hat = solver(y, A=A, lam=0.01, t=100)
# x_hat has shape (B, n); ISTA has the same interface.
```

Existing imports can change from `from fa2026.optim import FISTA, ISTA` to
`from ucmf.optim import FISTA, ISTA`. Top-level imports as above also work.

### Objective and conventions

The solvers minimize

```text
min_x  ||y - A x||² / (2 m) + lam * ||x||₁
```

`positive=True` additionally constrains `x >= 0` (the default).
The negative LS gradient is `A.T @ (y - A @ x) / m`, the threshold is
`gamma * lam`, and the default step size is `gamma = m / ||A||₂²`.
Thus `lam` is normalized by the measurement count: an equivalent unnormalized
LASSO uses `lambda_unnormalized = m * lam`.

- `y`: real floating-point tensor, `(m,)` or `(B, m)`.
- `A`: real floating-point tensor, `(m, n)` or `(B, m, n)`.
- `x0`: optional warm-start tensor, `(n,)` for a single sample or `(B, n)`.
- `lam`: required, Python scalar or floating tensor, scalar or per-sample `(B,)` / `(B, 1)`.
- `gamma`: optional, with the same scalar/per-sample forms.
- `t`: non-negative iteration count. Output is always `(B, n)`.

All tensor inputs must already have the same dtype and device as `y`.
Forward validates this and raises `ValueError` for mismatches; it never moves or
casts caller inputs. Convert arrays and prepare tensors explicitly beforehand:

```python
A = torch.as_tensor(A, dtype=y.dtype, device=y.device)
# Likewise prepare x0 and tensor-valued lam/gamma, if supplied.
```

There is **no implicit column normalization**. If desired, use
`normalize_sensing_matrix_columns(A)` and explicitly transform coordinates
`z = x * column_norms`, then recover `x = z / column_norms` after solving.
Regularization then acts on `z`, not on the unscaled physical source powers.
For CMF inputs, `A` and `y` must use the same vectorization, row ordering,
diagonal-removal setting, and off-diagonal weights (the source repo uses
`sqrt(2)` weights). The core solvers do not construct or vectorize CSMs; with
[`ucmf.acoular`](#acoular-integration), Acoular's `BeamformerCMF` does this.

### Callbacks and differentiability

`ucmf.optim` also exports:

- `IterationCallback`: override `on_step_begin` / `on_step_end`; set
  `solver.stop = True` to stop. End callbacks receive `gamma`, `theta`, `A`, `W`, `y`;
  the compatibility keyword `W` always equals `A` for classical ISTA/FISTA.
- `SolutionChangeStopper`: relative or absolute iterate-change stopping; every
  sample in the batch must meet the tolerance. Resets on iteration zero.
  The criterion is evaluated only every `check_every` steps (default 100),
  comparing that step's consecutive iterates; checking every step costs about
  +30 % (CPU) to +150 % (CUDA) per iteration for a 4032 × 1681 matrix,
  checking every 100 steps about +1–4 %. Stopping may thus occur up to
  `check_every - 1` steps after the change first fell below `tol`.
- `StepParamCollector`: detached parameter histories, manually reset between
  solves. Scalar parameters produce `(T,)`, per-sample parameters `(T, B, 1)`.
- `IntermediateOutputCollector`: intermediate iterates with their autograd
  graphs, automatically cleared on iteration zero. With `t=0`, no callbacks run.

The solvers have no trainable parameters. Gradients can flow through `y`, `x0`,
`lam`, and `gamma`; `A` is treated as fixed and detached. Use `torch.no_grad()`
for inference if no graph is needed.

The classical call interface is
`forward(y, x0=None, t=1, A=None, lam=None, gamma=None)`.
It does not accept an independent gradient weight `W`, including `W=None`.
Later unfolded models can override `forward` to expose or construct their own
weights and reuse the protected preparation/iteration hooks. Weighted updates
are not classical ISTA/FISTA; their step-size guarantees need separate treatment.

Only normalized least squares is supported. The old `ls_mode` argument and
TLS implementation have been removed; do not pass `ls_mode`, even `'ls'`.

## Acoular integration

`ucmf.acoular` provides ISTA and FISTA as custom solvers for Acoular's
inverse-problem interface, e.g. for `acoular.BeamformerCMF`:

```python
import acoular as ac
from acoular.solvers import L1RegularizedLeastSquaresProblem
from ucmf.acoular import FISTATorchSolver

problem = L1RegularizedLeastSquaresProblem(
    solver=FISTATorchSolver(device='cuda', precision='float32', n_iter=5000),
    alpha=1e-9,
    nonnegative=True,
    dictionary_scaling='unit_l2',
)
bf = ac.BeamformerCMF(freq_data=ps, steer=st, problem=problem)
```

- `TorchSolver(LeastSquaresSolver)` is the shared base class; `ISTATorchSolver`
  and `FISTATorchSolver` select the ucmf solver.
- Traits: `device` (`'cpu'`, `'cuda'`, ...), `precision` (`'float64'` or
  `'float32'`), `n_iter` (iterations per frequency bin), the optional `tol`,
  and `check_every` (default 100). By default (`tol=None`) exactly `n_iter`
  iterations run without callbacks. Setting `tol` enables
  `SolutionChangeStopper(tol=tol, check_every=check_every)`. All traits enter
  the solver digest, so Acoular's caching distinguishes configurations.
- The problem must be an `L1RegularizedLeastSquaresProblem`: positivity comes
  from `problem.nonnegative`, and the regularization is
  `lam = problem.alpha * problem.unit_mult`, matching Acoular's LassoLars
  convention. The ucmf objective equals scikit-learn's Lasso objective, so
  results are directly comparable to `LassoLarsSolver(backend_kwargs={'fit_intercept': False})`.
  Scaling is done by the problem, not the solver.
- An optional `start_value` is used as warm start `x0`. With `tol` set,
  `SolverOutput.info` reports `n_steps` and `converged`; otherwise it is empty.

### Example: three sources

[`examples/sparse_recovery.py`](examples/sparse_recovery.py) simulates the
three white-noise sources of Acoular's introductory example in front of the
64-microphone array and solves the same CMF problem (third-octave band at
8 kHz, `alpha=1e-9`) with LassoLars, ISTA, and FISTA (5000 iterations each,
CPU, float64). The titles show the total computation time, including the
per-bin CSM and dictionary assembly shared by all solvers:

![CMF maps of three sources with LassoLars, ISTA, and FISTA](docs/images/three_sources_cmf.png)

FISTA matches the LassoLars map up to a relative difference of about `2e-4`.
ISTA has not yet converged after 5000 iterations and still yields
spatially smeared sources. For this small, very sparse problem, LassoLars
is faster than 5000 first-order iterations on the CPU.

### Benchmark: ucmf vs. PyLops

[`examples/benchmark_solvers.py`](examples/benchmark_solvers.py) compares the
solver time (including step-size computation and data transfer, excluding CSM
and dictionary assembly) of ucmf and PyLops ISTA/FISTA on the same problem:
1000 iterations per frequency bin without early stopping, 5 frequency bins,
dictionary of 4032 × 1681. PyLops has no non-negativity constraint, so all
solvers use `nonnegative=False`; PyLops' objective `||y - A x||² + eps ||x||₁`
matches with `eps = 2 m alpha unit_mult`. Thread counts are set with
`torch.set_num_threads` and threadpoolctl (Acoular sets
`OPENBLAS_NUM_THREADS=1` on import). Results on 2× Intel Xeon Silver 4214R (24 cores) and an RTX 3090:

| Algorithm | Implementation | Threads | Time [s] | Speed-up vs. PyLops (1 thread) | Rel. diff. to PyLops |
|---|---|---|---|---|---|
| ISTA | PyLops | 1 | 38.69 | 1.0x | 0 |
| ISTA | ucmf CPU float64 | 1 | 40.30 | 1.0x | 5.9e-16 |
| ISTA | PyLops | 24 | 9.95 | 3.9x | 3.6e-16 |
| ISTA | ucmf CPU float64 | 24 | 5.20 | 7.4x | 4.7e-16 |
| ISTA | ucmf CUDA float64 | - | 1.40 | 27.7x | 4.7e-16 |
| ISTA | ucmf CUDA float32 | - | 0.68 | 56.5x | 8.0e-06 |
| FISTA | PyLops | 1 | 56.08 | 1.0x | 0 |
| FISTA | ucmf CPU float64 | 1 | 40.76 | 1.4x | 1.4e-13 |
| FISTA | PyLops | 24 | 12.29 | 4.6x | 1.8e-13 |
| FISTA | ucmf CPU float64 | 24 | 4.65 | 12.1x | 1.3e-13 |
| FISTA | ucmf CUDA float64 | - | 1.55 | 36.1x | 9.7e-14 |
| FISTA | ucmf CUDA float32 | - | 0.81 | 69.0x | 8.4e-04 |

Both implementations produce the same iterates up to rounding. Single-threaded,
ISTA runs equally fast in both; PyLops FISTA is slower than PyLops ISTA,
ucmf FISTA is not. Multi-threaded, ucmf is 1.9–2.6× faster than PyLops, and on
the GPU another 3–8× faster than its own multi-threaded CPU run.

## Extending the iteration

ISTA and FISTA use explicit loops with local variables. There are no state or
context dataclasses, iteration runner, or `nonlocal` momentum updates.
Subclasses reuse the gradient/proximal and callback helpers. Learned FISTA
steps can use `_iterate_accelerated(..., step_params_fn=...)`; recurrent models
can implement their own loop with local hidden/cell tensors.

See the [architecture guide](docs/architecture.md) for extension examples and
[domain vocabulary](GLOSSARY.md) for the iteration variables. The earlier
state-framework hooks are removed. Stateless tensor/operator helpers live in
`src/ucmf/optim/utils.py` and are imported directly, not exposed as class aliases.

See [migration notes](docs/migration.md) for scope, provenance, and the later
CMF-model migration boundary.
