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
from ucmf.optim.operator import MatrixMult

A = torch.eye(3)                     # real sensing matrix (m, n)
operator = MatrixMult(A)
y = torch.tensor([[1.0, 0.0, 0.5]])  # measurements (B, m)
stopper = SolutionChangeStopper(tol=1e-6)
solver = FISTA(nonnegative=True, callbacks=[stopper])
with torch.no_grad():
    x_hat = solver(y, operator=operator, lam=0.01, t=100)
# x_hat has shape (B, n); ISTA has the same interface.
```

Existing imports can change from `from fa2026.optim import FISTA, ISTA` to
`from ucmf.optim import FISTA, ISTA`. Top-level imports as above also work.

### Objective and conventions

The solvers minimize

```text
min_x  ||y - A x||² / (2 m) + lam * ||x||₁
```

`nonnegative=True` additionally constrains `x >= 0` (the default).
The negative LS gradient is `A.T @ (y - A @ x) / m`, the threshold is
`gamma * lam`, and the default step size is `gamma = m / (1.05 * ||A||₂²)`,
with the squared norm estimated by 50 power iterations.
Thus `lam` is normalized by the measurement count: an equivalent unnormalized
LASSO uses `lambda_unnormalized = m * lam`.

- `y`: real floating-point tensor, `(m,)` or `(B, m)`.
- `operator`: shared or per-problem `LinearOperator`. Wrap a real `(m, n)` or
  `(B, m, n)` tensor in `MatrixMult`; for CMF, use the dense `CMFDictionary` or
  `CMFPropagation`, which works directly with the propagation matrix `H`.
- `x0`: optional warm-start tensor, `(n,)` for a single sample or `(B, n)`.
- `lam`: required, Python scalar or floating tensor, scalar or per-sample `(B,)` / `(B, 1)`.
- `gamma`: optional, with the same scalar/per-sample forms.
- `t`: non-negative iteration count. Output is always `(B, n)`.

All tensor inputs must already have the same dtype and device as `y`.
Forward validates this and raises `ValueError` for mismatches; it never moves or
casts caller inputs. Convert arrays and prepare tensors explicitly beforehand:

```python
operator = MatrixMult(torch.as_tensor(A, dtype=y.dtype, device=y.device))
# Likewise prepare x0 and tensor-valued lam/gamma, if supplied.
```

There is **no column normalization**; regularization acts on the unscaled
source powers. Normalization will be revisited later.
For CMF inputs, `A` and `y` must use the same vectorization, row ordering,
diagonal-removal setting, and off-diagonal weights (the source repo uses
`sqrt(2)` weights). The core solvers do not construct or vectorize CSMs; with
[`ucmf.acoular`](#acoular-integration), Acoular's `BeamformerCMF` does this.

### Callbacks and differentiability

`ucmf.optim` also exports:

- `IterationCallback`: override `on_step_begin` / `on_step_end`; set
  `solver.stop = True` to stop. End callbacks receive `gamma`, `theta`, `A`, `W`, `y`;
  `A` and `W` both refer to the same detached operator for classical ISTA/FISTA.
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

ISTA and FISTA have no trainable parameters. Gradients flow through `y`, `x0`, `lam`,
and `gamma`; the operator is detached. Use `torch.no_grad()` for inference.

## Unfolded models

| Model | Learned weights | Operator |
| --- | --- | --- |
| `AdaLFISTASW` / `NA_AdaLFISTASW` | `B` (`m × m`) | any |
| `AdaLFISTA` / `NA_AdaLFISTA` | `B1`, `B2` (`m × m`) | any |
| `AdaLFISTASWR` / `NA_AdaLFISTASWR` | complex `Br` (`c × c`) | CMF |
| `AdaLFISTAR` / `NA_AdaLFISTAR` | complex `Br1`, `Br2` (`c × c`) | CMF |

`AdaLFISTA*` learn one `gamma`/`theta` per iteration; `NA_AdaLFISTA*` predict them per
sample with an LSTM. Weights start at the identity, which reproduces FISTA.
The models require an operator with unit column norms; `ucmf.scaling` normalizes
it and maps the solution back to source strengths. Constructors are keyword-only:

```python
from ucmf import NA_AdaLFISTASW
from ucmf.scaling import normalize_operator, to_source

D, u = normalize_operator(operator)  # D = A U^-1
model = NA_AdaLFISTASW(num_measurements=m, gamma_init=0.3, theta_init=0.01)
model.init_normalizers(y, D)  # once, before solving with NA models
x = model(y, operator=D, t=10)
q = to_source(x, u)  # source strengths of the original operator
```

Equiregularization divides each measurement by `lam_max = lambda_max(D, y)` (from
`ucmf.scaling`, nonnegative LASSO) before solving, so that `lam = 1` gives the zero
solution for every problem; physical source strengths are then
`lam_max.unsqueeze(-1) * to_source(x, u)`. Training applies it by default
(`data_scaling='equireg'`).

For subsets, use `D.select_rows(...)` or `select_mics(...)` and align `y` with
`D.rows`; the column norms of the full operator stay valid for `to_source`. `compile_unrolled(model, t_max)` compiles once for all depths. See the
[architecture guide](docs/architecture.md).

## Training

`ucmf.train` (extra `train`, Ray Tune) trains an unfolded model stage by stage
(`t = 1..stages`). It normalizes the operator, equiregularizes the data
(`data_scaling='equireg'`), and compiles one graph for all stages (`compile=True`):

```python
from ucmf.train import TrainingConfig, train_model

config = TrainingConfig(
    model_name='NA_AdaLFISTASW', model_kwargs={'num_measurements': m, 'gamma_init': 0.3, 'theta_init': 0.01},
    operator_name='CMFPropagation', operator_kwargs={'H': H},
    loss='cv', loss_kwargs={'split': 'mics_cross'}, w=0.5,
)
result = train_model(config, y_train=y_train, y_val=y_val)  # q_train/q_val for 'nmse'
```

Objectives: `nmse` (supervised), `lasso` (`lam`), `bic` (`alpha=1e-2`), and `cv`
(`split` in `rows`, `complex_rows`, `mics`, `mics_cross`; `train_ratio=0.8`). With
`w > 0`, earlier iterates enter with normalized weights `w^(t-k)`. See
[ADR 0010](docs/adr/0010-training-objectives.md).

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
- The problem must be an `L1RegularizedLeastSquaresProblem`: non-negativity comes
  from `problem.nonnegative`, and the regularization is
  `lam = problem.alpha * problem.unit_mult`, matching Acoular's LassoLars
  convention. The ucmf objective equals scikit-learn's Lasso objective, so
  results are directly comparable to `LassoLarsSolver(backend_kwargs={'fit_intercept': False})`.
  Scaling is done by the problem, not the solver.
- An optional `start_value` is used as warm start `x0`. With `tol` set,
  `SolverOutput.info` reports `num_steps` and `converged`; otherwise it is empty.

### Example: three sources

[`examples/sparse_recovery.py`](examples/sparse_recovery.py) simulates the
three white-noise sources of Acoular's introductory example in front of the
64-microphone array and solves the same CMF problem (third-octave band at
8 kHz, `alpha=1e-9`) with LassoLars (CPU) and ISTA and FISTA (2000 iterations
each, CUDA GPU, float64; the example requires a CUDA device). The titles show
the total computation time, including the per-bin CSM and dictionary assembly
shared by all solvers:

![CMF maps of three sources with LassoLars, ISTA, and FISTA](docs/images/three_sources_cmf.png)

FISTA matches the LassoLars map up to a relative difference of about `8e-4`.
ISTA has not yet converged after 2000 iterations and still yields
spatially smeared sources. For this small, very sparse problem, LassoLars
on the CPU is still faster than 2000 first-order iterations on the GPU.

### Benchmark: ucmf vs. PyLops

[`examples/benchmark_solvers.py`](examples/benchmark_solvers.py) compares the
solver time (including step-size computation and data transfer, excluding CSM
and dictionary assembly) of ucmf and PyLops ISTA/FISTA on the same problem:
1000 iterations per frequency bin without early stopping, 5 frequency bins,
dictionary of 4032 × 1681. PyLops has no non-negativity constraint, so all
solvers use `nonnegative=False`; PyLops' objective `||y - A x||² + eps ||x||₁`
matches with `eps = 2 m alpha unit_mult`. PyLops gets ucmf's default step size
(its `alpha = gamma / m`), so both run identical iterations. Thread counts are set with
`torch.set_num_threads` and threadpoolctl (Acoular sets
`OPENBLAS_NUM_THREADS=1` on import). Results on 2× Intel Xeon Silver 4214R (24 cores) and an RTX 3090:

| Algorithm | Implementation | Threads | Time [s] | Speed-up vs. PyLops (1 thread) | Rel. diff. to PyLops |
|---|---|---|---|---|---|
| ISTA | PyLops | 1 | 37.05 | 1.0x | 0 |
| ISTA | ucmf CPU float64 | 1 | 36.28 | 1.0x | 5.0e-16 |
| ISTA | PyLops | 24 | 10.28 | 3.6x | 4.0e-16 |
| ISTA | ucmf CPU float64 | 24 | 5.25 | 7.1x | 4.4e-16 |
| ISTA | ucmf CUDA float64 | - | 0.95 | 38.9x | 4.2e-16 |
| ISTA | ucmf CUDA float32 | - | 0.67 | 55.1x | 2.4e-04 |
| FISTA | PyLops | 1 | 53.82 | 1.0x | 0 |
| FISTA | ucmf CPU float64 | 1 | 36.46 | 1.5x | 1.3e-13 |
| FISTA | PyLops | 24 | 13.08 | 4.1x | 1.4e-13 |
| FISTA | ucmf CPU float64 | 24 | 4.36 | 12.3x | 2.5e-13 |
| FISTA | ucmf CUDA float64 | - | 0.97 | 55.4x | 4.6e-13 |
| FISTA | ucmf CUDA float32 | - | 0.72 | 74.9x | 8.1e-04 |

Both implementations produce the same iterates up to rounding. Single-threaded,
ISTA runs equally fast in both; PyLops FISTA is slower than PyLops ISTA,
ucmf FISTA is not. Multi-threaded, ucmf is 2.0–3.0× faster than PyLops, and on
the GPU another 4.5–8× faster than its own multi-threaded CPU run.

### Benchmark: dense and matrix-free CMF operators

[`examples/benchmark_operators.py`](examples/benchmark_operators.py) measures all 280 supported configurations in both modes.
Select a CUDA device with `CUDA_VISIBLE_DEVICES`, then run `uv run python examples/benchmark_operators.py --steps 15 --repeats 7`.

RTX 3090 (GPUs 001 and 002), PyTorch 2.12.0+cu130, float32 without TF32, `r_diag=True`, 15 iterations. ISTA/FISTA run
without gradients; Ada timings include parameter gradients but no optimizer step. Solver-side weight preparation is timed;
operator construction and step-size estimation are excluded. Shared cases reuse one operator; distinct cases alternate two
independent batches, each with one dictionary per sample. Batch 1 needs no distinct case; batch 512 is measured at 16 microphones
and a 16 × 16 grid only. `MatrixMult` and `CMFDictionary` use the same dense products; only the latter retains CMF metadata.

Each configuration and execution mode runs in a fresh process. Times are medians of seven synchronized runs after warm-up.
Compiled runs use `torch.compile(mode='reduce-overhead', fullgraph=True)`. The first compiled call includes compilation or cache
loading, but not process startup. Solutions and, where applicable, gradients are checked against eager `CMFPropagation` before
warm timing. Unsupported combinations are omitted; no measured configuration ran out of memory.

Memory peaks are measured during warm timing. **Allocated** is memory occupied by tensors tracked by PyTorch, including live
inputs. **Reserved** is memory held by the PyTorch CUDA allocator, including cached blocks and any CUDA graph pools. Neither
metric reports total device memory or the peak during operator construction, compilation, or warm-up. In the tables, Eager
and Compiled memory cells show **allocated / reserved** in MiB.

For 64 microphones, 4,096 grid points and batch 32 with distinct operators, compiled `CMFPropagation` is 2.9–3.6× faster than
the fastest dense path for each model. With one shared operator at the same size, dense FISTA is faster (3.47 vs. 14.29 ms).

<details>
<summary>16 microphones, 16 × 16 grid (256 points)</summary>

| Batch | Dictionaries | Model | Operator | Weight | Eager ms | Compiled ms | First compiled s | Eager allocated / reserved MiB | Compiled allocated / reserved MiB |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | shared | ISTA | MatrixMult | — | 1.710 | 0.240 | 3.061 | 8.4 / 22.0 | 0.2 / 24.0 |
| 1 | shared | ISTA | CMFDictionary | — | 1.775 | 0.244 | 3.113 | 8.4 / 22.0 | 0.3 / 24.0 |
| 1 | shared | ISTA | CMFPropagation | — | 11.169 | 0.649 | 9.166 | 8.3 / 22.0 | 0.1 / 24.0 |
| 1 | shared | FISTA | MatrixMult | — | 2.187 | 0.245 | 4.308 | 8.4 / 22.0 | 0.2 / 24.0 |
| 1 | shared | FISTA | CMFDictionary | — | 2.230 | 0.252 | 4.366 | 8.4 / 22.0 | 0.3 / 24.0 |
| 1 | shared | FISTA | CMFPropagation | — | 11.703 | 0.656 | 9.794 | 8.3 / 22.0 | 0.1 / 24.0 |
| 1 | shared | AdaLFISTASW | MatrixMult | factored | 14.695 | 1.214 | 8.572 | 17.2 / 22.0 | 0.5 / 24.0 |
| 1 | shared | AdaLFISTASW | MatrixMult | precomputed | 12.146 | 1.315 | 8.454 | 17.5 / 22.0 | 0.5 / 24.0 |
| 1 | shared | AdaLFISTASW | CMFDictionary | factored | 15.584 | 1.391 | 8.449 | 17.2 / 22.0 | 0.5 / 24.0 |
| 1 | shared | AdaLFISTASW | CMFDictionary | precomputed | 14.795 | 1.474 | 2.099 | 17.5 / 22.0 | 0.5 / 24.0 |
| 1 | shared | AdaLFISTASW | CMFPropagation | factored | 43.263 | 2.162 | 20.182 | 17.0 / 22.0 | 0.3 / 24.0 |
| 1 | shared | AdaLFISTA | MatrixMult | factored | 17.188 | 1.258 | 9.186 | 17.4 / 22.0 | 0.7 / 24.0 |
| 1 | shared | AdaLFISTA | MatrixMult | precomputed | 15.281 | 1.553 | 8.568 | 17.7 / 22.0 | 0.7 / 24.0 |
| 1 | shared | AdaLFISTA | CMFDictionary | factored | 18.222 | 1.289 | 8.773 | 17.5 / 22.0 | 0.7 / 24.0 |
| 1 | shared | AdaLFISTA | CMFDictionary | precomputed | 15.622 | 1.236 | 8.426 | 17.7 / 22.0 | 0.7 / 24.0 |
| 1 | shared | AdaLFISTA | CMFPropagation | factored | 45.460 | 2.195 | 20.864 | 17.3 / 22.0 | 0.5 / 24.0 |
| 1 | shared | AdaLFISTASWR | CMFDictionary | factored | 43.437 | 2.270 | 20.694 | 16.7 / 22.0 | 0.3 / 24.0 |
| 1 | shared | AdaLFISTASWR | CMFPropagation | factored | 64.617 | 3.065 | 34.115 | 16.6 / 22.0 | 0.1 / 24.0 |
| 1 | shared | AdaLFISTAR | CMFDictionary | factored | 68.575 | 3.528 | 38.273 | 16.8 / 22.0 | 0.3 / 24.0 |
| 1 | shared | AdaLFISTAR | CMFPropagation | factored | 85.637 | 4.013 | 54.670 | 16.7 / 22.0 | 0.1 / 24.0 |
| 32 | shared | ISTA | MatrixMult | — | 1.714 | 0.408 | 3.310 | 8.5 / 22.0 | 0.3 / 24.0 |
| 32 | shared | ISTA | CMFDictionary | — | 1.730 | 0.411 | 3.110 | 8.6 / 22.0 | 0.3 / 24.0 |
| 32 | shared | ISTA | CMFPropagation | — | 10.840 | 0.695 | 9.200 | 10.4 / 24.0 | 0.1 / 26.0 |
| 32 | shared | FISTA | MatrixMult | — | 2.357 | 0.405 | 4.441 | 8.6 / 22.0 | 0.3 / 24.0 |
| 32 | shared | FISTA | CMFDictionary | — | 2.357 | 0.412 | 4.403 | 8.6 / 22.0 | 0.3 / 24.0 |
| 32 | shared | FISTA | CMFPropagation | — | 11.410 | 0.704 | 9.883 | 10.5 / 24.0 | 0.1 / 26.0 |
| 32 | shared | AdaLFISTASW | MatrixMult | factored | 13.347 | 1.549 | 8.732 | 18.6 / 24.0 | 0.6 / 26.0 |
| 32 | shared | AdaLFISTASW | MatrixMult | precomputed | 13.793 | 1.604 | 8.480 | 18.9 / 24.0 | 0.6 / 26.0 |
| 32 | shared | AdaLFISTASW | CMFDictionary | factored | 15.973 | 1.594 | 8.910 | 18.6 / 24.0 | 0.7 / 26.0 |
| 32 | shared | AdaLFISTASW | CMFDictionary | precomputed | 15.456 | 1.543 | 8.794 | 18.9 / 24.0 | 0.7 / 26.0 |
| 32 | shared | AdaLFISTASW | CMFPropagation | factored | 41.611 | 2.366 | 21.225 | 21.1 / 26.0 | 0.5 / 28.0 |
| 32 | shared | AdaLFISTA | MatrixMult | factored | 20.099 | 2.057 | 10.020 | 19.3 / 24.0 | 0.9 / 26.0 |
| 32 | shared | AdaLFISTA | MatrixMult | precomputed | 16.325 | 1.809 | 10.501 | 19.6 / 24.0 | 0.9 / 26.0 |
| 32 | shared | AdaLFISTA | CMFDictionary | factored | 18.020 | 2.122 | 10.247 | 19.3 / 24.0 | 0.9 / 26.0 |
| 32 | shared | AdaLFISTA | CMFDictionary | precomputed | 18.252 | 1.973 | 10.169 | 19.6 / 24.0 | 0.9 / 26.0 |
| 32 | shared | AdaLFISTA | CMFPropagation | factored | 46.000 | 2.750 | 22.427 | 21.8 / 26.0 | 0.7 / 30.0 |
| 32 | shared | AdaLFISTASWR | CMFDictionary | factored | 43.689 | 2.592 | 22.119 | 20.5 / 26.0 | 0.5 / 26.0 |
| 32 | shared | AdaLFISTASWR | CMFPropagation | factored | 66.778 | 3.204 | 35.471 | 23.2 / 28.0 | 0.3 / 30.0 |
| 32 | shared | AdaLFISTAR | CMFDictionary | factored | 80.481 | 3.725 | 39.864 | 23.5 / 28.0 | 0.5 / 30.0 |
| 32 | shared | AdaLFISTAR | CMFPropagation | factored | 82.683 | 4.263 | 54.905 | 26.2 / 32.0 | 0.3 / 32.0 |
| 32 | distinct | ISTA | MatrixMult | — | 1.855 | 0.431 | 3.643 | 23.3 / 42.0 | 15.1 / 64.0 |
| 32 | distinct | ISTA | CMFDictionary | — | 1.846 | 0.439 | 3.349 | 25.3 / 44.0 | 17.1 / 66.0 |
| 32 | distinct | ISTA | CMFPropagation | — | 11.149 | 0.950 | 9.709 | 13.4 / 26.0 | 2.1 / 44.0 |
| 32 | distinct | FISTA | MatrixMult | — | 2.378 | 0.445 | 4.540 | 23.3 / 42.0 | 15.1 / 64.0 |
| 32 | distinct | FISTA | CMFDictionary | — | 2.401 | 0.441 | 4.565 | 25.3 / 44.0 | 17.1 / 66.0 |
| 32 | distinct | FISTA | CMFPropagation | — | 11.925 | 0.978 | 10.527 | 13.4 / 26.0 | 2.1 / 44.0 |
| 32 | distinct | AdaLFISTASW | MatrixMult | factored | 17.225 | 1.673 | 9.887 | 33.4 / 64.0 | 15.4 / 66.0 |
| 32 | distinct | AdaLFISTASW | MatrixMult | precomputed | 13.367 | 1.613 | 9.789 | 55.4 / 82.0 | 15.4 / 84.0 |
| 32 | distinct | AdaLFISTASW | CMFDictionary | factored | 17.296 | 1.678 | 10.138 | 35.4 / 66.0 | 17.4 / 68.0 |
| 32 | distinct | AdaLFISTASW | CMFDictionary | precomputed | 16.309 | 2.074 | 9.996 | 57.4 / 84.0 | 17.4 / 86.0 |
| 32 | distinct | AdaLFISTASW | CMFPropagation | factored | 42.458 | 3.372 | 21.215 | 24.0 / 30.0 | 2.4 / 32.0 |
| 32 | distinct | AdaLFISTA | MatrixMult | factored | 22.187 | 1.951 | 10.162 | 34.1 / 64.0 | 15.6 / 66.0 |
| 32 | distinct | AdaLFISTA | MatrixMult | precomputed | 19.301 | 1.603 | 9.171 | 56.2 / 84.0 | 15.6 / 86.0 |
| 32 | distinct | AdaLFISTA | CMFDictionary | factored | 22.300 | 2.034 | 10.075 | 36.1 / 66.0 | 17.6 / 68.0 |
| 32 | distinct | AdaLFISTA | CMFDictionary | precomputed | 16.474 | 1.642 | 9.387 | 58.2 / 86.0 | 17.6 / 88.0 |
| 32 | distinct | AdaLFISTA | CMFPropagation | factored | 47.345 | 2.999 | 21.978 | 24.7 / 30.0 | 2.7 / 32.0 |
| 32 | distinct | AdaLFISTASWR | CMFDictionary | factored | 46.333 | 2.685 | 22.682 | 37.2 / 68.0 | 17.2 / 68.0 |
| 32 | distinct | AdaLFISTASWR | CMFPropagation | factored | 64.046 | 3.134 | 36.428 | 26.1 / 30.0 | 2.3 / 32.0 |
| 32 | distinct | AdaLFISTAR | CMFDictionary | factored | 70.368 | 4.090 | 39.984 | 40.2 / 70.0 | 17.3 / 72.0 |
| 32 | distinct | AdaLFISTAR | CMFPropagation | factored | 85.672 | 4.717 | 55.674 | 29.2 / 34.0 | 2.3 / 34.0 |
| 512 | shared | ISTA | MatrixMult | — | 1.815 | 0.489 | 3.358 | 11.3 / 24.0 | 1.2 / 28.0 |
| 512 | shared | ISTA | CMFDictionary | — | 1.834 | 0.482 | 3.260 | 11.3 / 24.0 | 1.2 / 28.0 |
| 512 | shared | ISTA | CMFPropagation | — | 11.269 | 3.003 | 9.705 | 44.1 / 58.0 | 1.0 / 62.0 |
| 512 | shared | FISTA | MatrixMult | — | 2.288 | 0.509 | 4.595 | 11.8 / 26.0 | 1.2 / 28.0 |
| 512 | shared | FISTA | CMFDictionary | — | 2.310 | 0.496 | 4.422 | 11.8 / 26.0 | 1.2 / 28.0 |
| 512 | shared | FISTA | CMFPropagation | — | 11.831 | 3.019 | 10.085 | 44.6 / 58.0 | 1.0 / 62.0 |
| 512 | shared | AdaLFISTASW | MatrixMult | factored | 16.518 | 1.878 | 10.630 | 42.9 / 48.0 | 3.4 / 50.0 |
| 512 | shared | AdaLFISTASW | MatrixMult | precomputed | 14.396 | 1.785 | 10.396 | 43.2 / 48.0 | 3.4 / 48.0 |
| 512 | shared | AdaLFISTASW | CMFDictionary | factored | 16.540 | 2.117 | 10.531 | 43.0 / 48.0 | 3.4 / 50.0 |
| 512 | shared | AdaLFISTASW | CMFDictionary | precomputed | 14.610 | 1.880 | 10.255 | 43.2 / 48.0 | 3.4 / 48.0 |
| 512 | shared | AdaLFISTASW | CMFPropagation | factored | 45.227 | 7.354 | 22.206 | 88.0 / 96.0 | 3.2 / 102.0 |
| 512 | shared | AdaLFISTA | MatrixMult | factored | 17.530 | 2.486 | 11.449 | 50.2 / 56.0 | 3.6 / 58.0 |
| 512 | shared | AdaLFISTA | MatrixMult | precomputed | 19.045 | 2.422 | 11.414 | 50.9 / 56.0 | 3.6 / 58.0 |
| 512 | shared | AdaLFISTA | CMFDictionary | factored | 17.161 | 2.633 | 11.682 | 50.2 / 56.0 | 3.7 / 58.0 |
| 512 | shared | AdaLFISTA | CMFDictionary | precomputed | 19.520 | 2.275 | 11.237 | 50.9 / 56.0 | 3.7 / 58.0 |
| 512 | shared | AdaLFISTA | CMFPropagation | factored | 47.496 | 8.271 | 23.649 | 96.7 / 106.0 | 3.4 / 110.0 |
| 512 | shared | AdaLFISTASWR | CMFDictionary | factored | 45.903 | 3.328 | 24.660 | 79.9 / 88.0 | 3.2 / 76.0 |
| 512 | shared | AdaLFISTASWR | CMFPropagation | factored | 65.811 | 8.391 | 37.556 | 125.2 / 134.0 | 3.0 / 122.0 |
| 512 | shared | AdaLFISTAR | CMFDictionary | factored | 68.649 | 5.033 | 43.284 | 127.4 / 134.0 | 3.3 / 108.0 |
| 512 | shared | AdaLFISTAR | CMFPropagation | factored | 86.990 | 10.454 | 58.016 | 173.8 / 182.0 | 3.1 / 154.0 |
| 512 | distinct | ISTA | MatrixMult | — | 4.792 | 4.983 | 3.723 | 251.1 / 264.0 | 240.9 / 388.0 |
| 512 | distinct | ISTA | CMFDictionary | — | 4.814 | 4.987 | 3.288 | 283.1 / 296.0 | 272.9 / 420.0 |
| 512 | distinct | ISTA | CMFPropagation | — | 11.659 | 4.780 | 9.781 | 92.0 / 106.0 | 33.0 / 348.0 |
| 512 | distinct | FISTA | MatrixMult | — | 4.920 | 5.012 | 4.844 | 251.6 / 264.0 | 240.9 / 388.0 |
| 512 | distinct | FISTA | CMFDictionary | — | 4.944 | 5.000 | 4.594 | 283.6 / 296.0 | 272.9 / 420.0 |
| 512 | distinct | FISTA | CMFPropagation | — | 12.064 | 4.813 | 10.291 | 92.5 / 106.0 | 33.0 / 350.0 |
| 512 | distinct | AdaLFISTASW | MatrixMult | factored | 18.211 | 10.218 | 11.281 | 282.7 / 288.0 | 243.2 / 410.0 |
| 512 | distinct | AdaLFISTASW | MatrixMult | precomputed | 20.705 | 11.497 | 11.100 | 639.4 / 648.0 | 243.2 / 644.0 |
| 512 | distinct | AdaLFISTASW | CMFDictionary | factored | 17.818 | 10.344 | 11.479 | 314.7 / 320.0 | 275.2 / 442.0 |
| 512 | distinct | AdaLFISTASW | CMFDictionary | precomputed | 21.087 | 11.687 | 11.532 | 671.5 / 680.0 | 275.2 / 676.0 |
| 512 | distinct | AdaLFISTASW | CMFPropagation | factored | 43.107 | 9.711 | 21.314 | 135.9 / 144.0 | 35.2 / 164.0 |
| 512 | distinct | AdaLFISTA | MatrixMult | factored | 19.352 | 11.145 | 12.046 | 290.0 / 296.0 | 243.4 / 418.0 |
| 512 | distinct | AdaLFISTA | MatrixMult | precomputed | 30.077 | 13.525 | 10.936 | 648.2 / 656.0 | 243.4 / 658.0 |
| 512 | distinct | AdaLFISTA | CMFDictionary | factored | 19.149 | 10.989 | 11.857 | 322.0 / 328.0 | 275.4 / 450.0 |
| 512 | distinct | AdaLFISTA | CMFDictionary | precomputed | 30.180 | 13.583 | 11.065 | 680.2 / 688.0 | 275.4 / 690.0 |
| 512 | distinct | AdaLFISTA | CMFPropagation | factored | 43.112 | 10.589 | 23.612 | 144.7 / 154.0 | 35.4 / 172.0 |
| 512 | distinct | AdaLFISTASWR | CMFDictionary | factored | 46.872 | 11.595 | 25.449 | 351.6 / 358.0 | 275.0 / 466.0 |
| 512 | distinct | AdaLFISTASWR | CMFPropagation | factored | 66.430 | 11.056 | 38.132 | 173.2 / 182.0 | 35.0 / 186.0 |
| 512 | distinct | AdaLFISTAR | CMFDictionary | factored | 69.917 | 13.672 | 41.798 | 399.1 / 406.0 | 275.0 / 526.0 |
| 512 | distinct | AdaLFISTAR | CMFPropagation | factored | 85.028 | 13.032 | 56.517 | 221.7 / 230.0 | 35.0 / 218.0 |

</details>

<details>
<summary>16 microphones, 64 × 64 grid (4,096 points)</summary>

| Batch | Dictionaries | Model | Operator | Weight | Eager ms | Compiled ms | First compiled s | Eager allocated / reserved MiB | Compiled allocated / reserved MiB |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | shared | ISTA | MatrixMult | — | 1.893 | 0.339 | 3.283 | 11.9 / 22.0 | 3.8 / 44.0 |
| 1 | shared | ISTA | CMFDictionary | — | 2.024 | 0.341 | 3.104 | 12.4 / 22.0 | 4.3 / 44.0 |
| 1 | shared | ISTA | CMFPropagation | — | 11.124 | 0.735 | 9.430 | 10.2 / 24.0 | 0.5 / 32.0 |
| 1 | shared | FISTA | MatrixMult | — | 2.305 | 0.351 | 4.676 | 12.0 / 22.0 | 3.8 / 44.0 |
| 1 | shared | FISTA | CMFDictionary | — | 2.403 | 0.346 | 4.347 | 12.5 / 22.0 | 4.3 / 44.0 |
| 1 | shared | FISTA | CMFPropagation | — | 11.912 | 0.725 | 10.924 | 10.2 / 24.0 | 0.5 / 32.0 |
| 1 | shared | AdaLFISTASW | MatrixMult | factored | 15.878 | 1.283 | 8.379 | 21.1 / 42.0 | 4.0 / 44.0 |
| 1 | shared | AdaLFISTASW | MatrixMult | precomputed | 18.295 | 1.500 | 8.189 | 32.0 / 42.0 | 4.0 / 44.0 |
| 1 | shared | AdaLFISTASW | CMFDictionary | factored | 16.297 | 1.274 | 8.707 | 21.6 / 42.0 | 4.5 / 44.0 |
| 1 | shared | AdaLFISTASW | CMFDictionary | precomputed | 15.282 | 1.446 | 8.638 | 32.5 / 42.0 | 4.5 / 44.0 |
| 1 | shared | AdaLFISTASW | CMFPropagation | factored | 43.054 | 2.175 | 20.639 | 19.2 / 24.0 | 0.8 / 26.0 |
| 1 | shared | AdaLFISTA | MatrixMult | factored | 19.602 | 1.532 | 9.638 | 21.4 / 42.0 | 4.3 / 44.0 |
| 1 | shared | AdaLFISTA | MatrixMult | precomputed | 15.381 | 1.743 | 9.259 | 32.5 / 42.0 | 4.3 / 44.0 |
| 1 | shared | AdaLFISTA | CMFDictionary | factored | 18.706 | 1.520 | 9.593 | 21.9 / 42.0 | 4.8 / 44.0 |
| 1 | shared | AdaLFISTA | CMFDictionary | precomputed | 17.872 | 1.567 | 8.871 | 33.0 / 42.0 | 4.8 / 44.0 |
| 1 | shared | AdaLFISTA | CMFPropagation | factored | 46.449 | 2.640 | 21.889 | 19.5 / 24.0 | 1.0 / 26.0 |
| 1 | shared | AdaLFISTASWR | CMFDictionary | factored | 43.780 | 2.318 | 22.793 | 21.2 / 42.0 | 4.4 / 44.0 |
| 1 | shared | AdaLFISTASWR | CMFPropagation | factored | 73.556 | 2.577 | 34.665 | 18.9 / 24.0 | 0.6 / 26.0 |
| 1 | shared | AdaLFISTAR | CMFDictionary | factored | 68.940 | 3.496 | 37.877 | 21.3 / 42.0 | 4.4 / 44.0 |
| 1 | shared | AdaLFISTAR | CMFPropagation | factored | 110.164 | 4.124 | 55.182 | 19.0 / 24.0 | 0.7 / 26.0 |
| 32 | shared | ISTA | MatrixMult | — | 1.845 | 0.552 | 3.639 | 14.9 / 40.0 | 4.8 / 40.0 |
| 32 | shared | ISTA | CMFDictionary | — | 1.818 | 0.558 | 3.268 | 15.4 / 40.0 | 5.3 / 40.0 |
| 32 | shared | ISTA | CMFPropagation | — | 11.420 | 2.827 | 9.665 | 41.8 / 56.0 | 0.6 / 64.0 |
| 32 | shared | FISTA | MatrixMult | — | 3.272 | 0.682 | 4.480 | 15.4 / 40.0 | 4.8 / 42.0 |
| 32 | shared | FISTA | CMFDictionary | — | 2.558 | 0.574 | 4.909 | 15.9 / 40.0 | 5.3 / 42.0 |
| 32 | shared | FISTA | CMFPropagation | — | 11.438 | 2.853 | 10.357 | 42.3 / 56.0 | 0.6 / 64.0 |
| 32 | shared | AdaLFISTASW | MatrixMult | factored | 15.749 | 2.684 | 10.635 | 39.0 / 56.0 | 6.0 / 58.0 |
| 32 | shared | AdaLFISTASW | MatrixMult | precomputed | 17.189 | 2.129 | 11.086 | 47.0 / 56.0 | 6.0 / 60.0 |
| 32 | shared | AdaLFISTASW | CMFDictionary | factored | 16.821 | 2.240 | 10.727 | 39.5 / 56.0 | 6.5 / 58.0 |
| 32 | shared | AdaLFISTASW | CMFDictionary | precomputed | 13.050 | 2.329 | 10.449 | 47.5 / 56.0 | 6.5 / 60.0 |
| 32 | shared | AdaLFISTASW | CMFPropagation | factored | 42.366 | 6.665 | 21.977 | 66.1 / 74.0 | 2.8 / 74.0 |
| 32 | shared | AdaLFISTA | MatrixMult | factored | 18.794 | 2.230 | 11.907 | 39.8 / 56.0 | 6.3 / 60.0 |
| 32 | shared | AdaLFISTA | MatrixMult | precomputed | 15.612 | 2.584 | 11.949 | 55.7 / 64.0 | 6.3 / 70.0 |
| 32 | shared | AdaLFISTA | CMFDictionary | factored | 20.298 | 2.421 | 12.089 | 40.3 / 58.0 | 6.8 / 60.0 |
| 32 | shared | AdaLFISTA | CMFDictionary | precomputed | 12.966 | 2.486 | 11.425 | 56.2 / 64.0 | 6.8 / 70.0 |
| 32 | shared | AdaLFISTA | CMFPropagation | factored | 44.839 | 7.234 | 23.286 | 67.3 / 74.0 | 3.0 / 92.0 |
| 32 | shared | AdaLFISTASWR | CMFDictionary | factored | 46.224 | 2.997 | 23.070 | 41.6 / 58.0 | 6.3 / 58.0 |
| 32 | shared | AdaLFISTASWR | CMFPropagation | factored | 66.572 | 7.590 | 37.460 | 68.2 / 76.0 | 2.6 / 74.0 |
| 32 | shared | AdaLFISTAR | CMFDictionary | factored | 80.340 | 3.980 | 42.010 | 44.5 / 62.0 | 6.4 / 62.0 |
| 32 | shared | AdaLFISTAR | CMFPropagation | factored | 84.375 | 9.173 | 56.813 | 71.7 / 78.0 | 2.7 / 94.0 |
| 32 | distinct | ISTA | MatrixMult | — | 4.764 | 4.971 | 3.747 | 250.2 / 264.0 | 240.1 / 384.0 |
| 32 | distinct | ISTA | CMFDictionary | — | 4.768 | 4.964 | 3.317 | 282.2 / 296.0 | 272.1 / 416.0 |
| 32 | distinct | ISTA | CMFPropagation | — | 15.450 | 8.617 | 9.812 | 88.8 / 104.0 | 32.1 / 344.0 |
| 32 | distinct | FISTA | MatrixMult | — | 4.894 | 5.021 | 4.566 | 250.7 / 264.0 | 240.1 / 386.0 |
| 32 | distinct | FISTA | CMFDictionary | — | 4.902 | 5.035 | 4.620 | 282.7 / 296.0 | 272.1 / 418.0 |
| 32 | distinct | FISTA | CMFPropagation | — | 19.328 | 8.648 | 10.442 | 89.3 / 104.0 | 32.1 / 346.0 |
| 32 | distinct | AdaLFISTASW | MatrixMult | factored | 17.933 | 9.905 | 11.589 | 275.2 / 280.0 | 242.3 / 402.0 |
| 32 | distinct | AdaLFISTASW | MatrixMult | precomputed | 20.905 | 11.965 | 11.661 | 632.0 / 640.0 | 242.3 / 522.0 |
| 32 | distinct | AdaLFISTASW | CMFDictionary | factored | 17.783 | 10.011 | 11.632 | 307.2 / 312.0 | 274.3 / 434.0 |
| 32 | distinct | AdaLFISTASW | CMFDictionary | precomputed | 20.877 | 11.944 | 11.679 | 664.0 / 672.0 | 274.3 / 554.0 |
| 32 | distinct | AdaLFISTASW | CMFPropagation | factored | 44.996 | 17.117 | 21.373 | 113.1 / 122.0 | 34.3 / 152.0 |
| 32 | distinct | AdaLFISTA | MatrixMult | factored | 19.557 | 10.548 | 11.814 | 276.0 / 280.0 | 242.5 / 404.0 |
| 32 | distinct | AdaLFISTA | MatrixMult | precomputed | 30.138 | 12.715 | 10.767 | 640.7 / 648.0 | 242.5 / 530.0 |
| 32 | distinct | AdaLFISTA | CMFDictionary | factored | 21.863 | 10.587 | 11.172 | 308.1 / 312.0 | 274.5 / 436.0 |
| 32 | distinct | AdaLFISTA | CMFDictionary | precomputed | 30.036 | 12.792 | 10.468 | 672.7 / 680.0 | 274.5 / 562.0 |
| 32 | distinct | AdaLFISTA | CMFPropagation | factored | 48.413 | 18.119 | 22.336 | 114.3 / 122.0 | 34.5 / 154.0 |
| 32 | distinct | AdaLFISTASWR | CMFDictionary | factored | 48.923 | 11.008 | 24.193 | 309.3 / 314.0 | 274.1 / 434.0 |
| 32 | distinct | AdaLFISTASWR | CMFPropagation | factored | 67.619 | 18.063 | 37.765 | 115.2 / 124.0 | 34.1 / 154.0 |
| 32 | distinct | AdaLFISTAR | CMFDictionary | factored | 69.330 | 12.659 | 43.450 | 312.3 / 318.0 | 274.1 / 436.0 |
| 32 | distinct | AdaLFISTAR | CMFPropagation | factored | 87.407 | 18.276 | 57.732 | 118.7 / 126.0 | 34.2 / 156.0 |

</details>

<details>
<summary>64 microphones, 16 × 16 grid (256 points)</summary>

| Batch | Dictionaries | Model | Operator | Weight | Eager ms | Compiled ms | First compiled s | Eager allocated / reserved MiB | Compiled allocated / reserved MiB |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | shared | ISTA | MatrixMult | — | 1.708 | 0.591 | 3.364 | 12.1 / 22.0 | 4.0 / 44.0 |
| 1 | shared | ISTA | CMFDictionary | — | 1.740 | 0.349 | 3.270 | 12.3 / 22.0 | 4.2 / 44.0 |
| 1 | shared | ISTA | CMFPropagation | — | 10.561 | 0.938 | 9.298 | 8.8 / 22.0 | 0.2 / 26.0 |
| 1 | shared | FISTA | MatrixMult | — | 2.200 | 0.344 | 4.541 | 12.1 / 22.0 | 4.0 / 44.0 |
| 1 | shared | FISTA | CMFDictionary | — | 2.217 | 0.352 | 4.420 | 12.3 / 22.0 | 4.2 / 44.0 |
| 1 | shared | FISTA | CMFPropagation | — | 11.173 | 0.932 | 9.926 | 8.8 / 22.0 | 0.2 / 26.0 |
| 1 | shared | AdaLFISTASW | MatrixMult | factored | 17.637 | 3.281 | 8.665 | 206.6 / 278.0 | 66.0 / 216.0 |
| 1 | shared | AdaLFISTASW | MatrixMult | precomputed | 17.110 | 1.533 | 8.247 | 148.2 / 214.0 | 66.0 / 216.0 |
| 1 | shared | AdaLFISTASW | CMFDictionary | factored | 17.565 | 3.288 | 8.566 | 206.7 / 278.0 | 66.2 / 216.0 |
| 1 | shared | AdaLFISTASW | CMFDictionary | precomputed | 16.241 | 2.240 | 8.594 | 148.4 / 214.0 | 66.2 / 216.0 |
| 1 | shared | AdaLFISTASW | CMFPropagation | factored | 47.822 | 4.431 | 20.027 | 203.4 / 214.0 | 62.3 / 152.0 |
| 1 | shared | AdaLFISTA | MatrixMult | factored | 19.766 | 6.310 | 8.740 | 268.8 / 342.0 | 128.0 / 344.0 |
| 1 | shared | AdaLFISTA | MatrixMult | precomputed | 16.886 | 2.232 | 8.275 | 272.2 / 342.0 | 128.0 / 344.0 |
| 1 | shared | AdaLFISTA | CMFDictionary | factored | 21.731 | 6.280 | 8.686 | 269.0 / 342.0 | 128.2 / 344.0 |
| 1 | shared | AdaLFISTA | CMFDictionary | precomputed | 17.493 | 2.248 | 8.412 | 272.4 / 342.0 | 128.2 / 344.0 |
| 1 | shared | AdaLFISTA | CMFPropagation | factored | 47.270 | 7.475 | 20.833 | 265.8 / 278.0 | 124.3 / 280.0 |
| 1 | shared | AdaLFISTASWR | CMFDictionary | factored | 43.477 | 2.348 | 21.353 | 22.2 / 44.0 | 4.2 / 44.0 |
| 1 | shared | AdaLFISTASWR | CMFPropagation | factored | 63.906 | 3.220 | 34.829 | 19.3 / 24.0 | 0.3 / 24.0 |
| 1 | shared | AdaLFISTAR | CMFDictionary | factored | 64.645 | 3.356 | 37.822 | 24.0 / 44.0 | 4.3 / 46.0 |
| 1 | shared | AdaLFISTAR | CMFPropagation | factored | 84.279 | 4.591 | 56.816 | 21.1 / 26.0 | 0.4 / 26.0 |
| 32 | shared | ISTA | MatrixMult | — | 1.897 | 0.548 | 3.475 | 14.1 / 24.0 | 4.9 / 46.0 |
| 32 | shared | ISTA | CMFDictionary | — | 2.573 | 0.548 | 3.131 | 14.2 / 24.0 | 5.1 / 46.0 |
| 32 | shared | ISTA | CMFPropagation | — | 10.794 | 1.585 | 9.501 | 19.9 / 24.0 | 1.2 / 30.0 |
| 32 | shared | FISTA | MatrixMult | — | 2.309 | 0.551 | 4.503 | 14.1 / 24.0 | 4.9 / 46.0 |
| 32 | shared | FISTA | CMFDictionary | — | 2.440 | 0.549 | 4.552 | 14.3 / 24.0 | 5.1 / 46.0 |
| 32 | shared | FISTA | CMFPropagation | — | 12.004 | 1.591 | 10.157 | 20.0 / 24.0 | 1.2 / 30.0 |
| 32 | shared | AdaLFISTASW | MatrixMult | factored | 18.084 | 8.578 | 8.922 | 216.0 / 288.0 | 67.1 / 246.0 |
| 32 | shared | AdaLFISTASW | MatrixMult | precomputed | 17.050 | 2.844 | 8.751 | 149.2 / 224.0 | 67.1 / 182.0 |
| 32 | shared | AdaLFISTASW | CMFDictionary | factored | 17.644 | 8.758 | 8.968 | 216.2 / 288.0 | 67.3 / 246.0 |
| 32 | shared | AdaLFISTASW | CMFDictionary | precomputed | 14.376 | 2.510 | 9.009 | 149.4 / 224.0 | 67.3 / 182.0 |
| 32 | shared | AdaLFISTASW | CMFPropagation | factored | 44.274 | 10.750 | 21.127 | 225.5 / 304.0 | 63.3 / 252.0 |
| 32 | shared | AdaLFISTA | MatrixMult | factored | 20.520 | 16.366 | 10.051 | 285.5 / 358.0 | 129.1 / 318.0 |
| 32 | shared | AdaLFISTA | MatrixMult | precomputed | 18.306 | 3.238 | 9.723 | 273.3 / 352.0 | 129.1 / 318.0 |
| 32 | shared | AdaLFISTA | CMFDictionary | factored | 21.745 | 16.501 | 10.112 | 285.7 / 360.0 | 129.3 / 318.0 |
| 32 | shared | AdaLFISTA | CMFDictionary | precomputed | 17.187 | 3.287 | 9.656 | 273.5 / 352.0 | 129.3 / 318.0 |
| 32 | shared | AdaLFISTA | CMFPropagation | factored | 45.698 | 18.824 | 22.139 | 296.9 / 378.0 | 125.4 / 324.0 |
| 32 | shared | AdaLFISTASWR | CMFDictionary | factored | 43.472 | 3.110 | 21.762 | 70.1 / 92.0 | 5.4 / 82.0 |
| 32 | shared | AdaLFISTASWR | CMFPropagation | factored | 64.033 | 5.256 | 35.245 | 88.0 / 106.0 | 1.4 / 82.0 |
| 32 | shared | AdaLFISTAR | CMFDictionary | factored | 67.238 | 5.213 | 38.478 | 117.5 / 140.0 | 5.4 / 142.0 |
| 32 | shared | AdaLFISTAR | CMFPropagation | factored | 99.746 | 7.830 | 56.792 | 136.3 / 156.0 | 1.5 / 158.0 |
| 32 | distinct | ISTA | MatrixMult | — | 4.818 | 5.055 | 3.749 | 262.1 / 276.0 | 253.0 / 404.0 |
| 32 | distinct | ISTA | CMFDictionary | — | 4.821 | 5.088 | 3.284 | 270.2 / 296.0 | 261.1 / 444.0 |
| 32 | distinct | ISTA | CMFPropagation | — | 11.447 | 1.992 | 9.731 | 31.8 / 46.0 | 9.1 / 148.0 |
| 32 | distinct | FISTA | MatrixMult | — | 4.918 | 5.052 | 4.597 | 262.2 / 276.0 | 253.0 / 404.0 |
| 32 | distinct | FISTA | CMFDictionary | — | 4.940 | 5.059 | 4.588 | 270.3 / 296.0 | 261.1 / 444.0 |
| 32 | distinct | FISTA | CMFPropagation | — | 11.960 | 1.976 | 10.351 | 31.8 / 46.0 | 9.1 / 148.0 |
| 32 | distinct | AdaLFISTASW | MatrixMult | factored | 18.643 | 17.082 | 10.001 | 464.1 / 496.0 | 315.1 / 612.0 |
| 32 | distinct | AdaLFISTASW | MatrixMult | precomputed | 47.162 | 37.998 | 9.389 | 2441.8 / 2732.0 | 315.1 / 4448.0 |
| 32 | distinct | AdaLFISTASW | CMFDictionary | factored | 20.598 | 17.130 | 9.597 | 472.2 / 496.0 | 323.3 / 652.0 |
| 32 | distinct | AdaLFISTASW | CMFDictionary | precomputed | 47.120 | 38.264 | 9.626 | 2449.9 / 2732.0 | 323.3 / 4488.0 |
| 32 | distinct | AdaLFISTASW | CMFPropagation | factored | 41.352 | 11.451 | 20.483 | 233.4 / 282.0 | 71.3 / 302.0 |
| 32 | distinct | AdaLFISTA | MatrixMult | factored | 26.554 | 25.364 | 10.623 | 533.5 / 566.0 | 377.1 / 690.0 |
| 32 | distinct | AdaLFISTA | MatrixMult | precomputed | 56.753 | 39.574 | 9.901 | 2565.8 / 2796.0 | 377.1 / 4520.0 |
| 32 | distinct | AdaLFISTA | CMFDictionary | factored | 26.655 | 25.198 | 10.475 | 541.7 / 568.0 | 385.3 / 730.0 |
| 32 | distinct | AdaLFISTA | CMFDictionary | precomputed | 56.807 | 39.914 | 9.543 | 2573.9 / 2796.0 | 385.3 / 4560.0 |
| 32 | distinct | AdaLFISTA | CMFPropagation | factored | 51.666 | 19.424 | 22.670 | 304.9 / 352.0 | 133.3 / 376.0 |
| 32 | distinct | AdaLFISTASWR | CMFDictionary | factored | 50.255 | 11.749 | 23.285 | 326.1 / 344.0 | 261.3 / 480.0 |
| 32 | distinct | AdaLFISTASWR | CMFPropagation | factored | 94.024 | 5.765 | 35.111 | 99.8 / 126.0 | 9.3 / 122.0 |
| 32 | distinct | AdaLFISTAR | CMFDictionary | factored | 70.543 | 14.251 | 41.545 | 373.4 / 392.0 | 261.4 / 540.0 |
| 32 | distinct | AdaLFISTAR | CMFPropagation | factored | 100.783 | 8.208 | 55.910 | 148.1 / 176.0 | 9.4 / 198.0 |

</details>

<details>
<summary>64 microphones, 64 × 64 grid (4,096 points)</summary>

| Batch | Dictionaries | Model | Operator | Weight | Eager ms | Compiled ms | First compiled s | Eager allocated / reserved MiB | Compiled allocated / reserved MiB |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | shared | ISTA | MatrixMult | — | 2.719 | 2.739 | 3.400 | 72.2 / 86.0 | 64.0 / 152.0 |
| 1 | shared | ISTA | CMFDictionary | — | 2.728 | 2.739 | 3.133 | 74.3 / 86.0 | 66.1 / 172.0 |
| 1 | shared | ISTA | CMFPropagation | — | 10.758 | 1.068 | 9.571 | 16.3 / 22.0 | 2.1 / 84.0 |
| 1 | shared | FISTA | MatrixMult | — | 2.812 | 2.731 | 4.381 | 72.2 / 86.0 | 64.0 / 152.0 |
| 1 | shared | FISTA | CMFDictionary | — | 2.826 | 3.013 | 4.402 | 74.3 / 86.0 | 66.1 / 172.0 |
| 1 | shared | FISTA | CMFPropagation | — | 10.859 | 1.084 | 9.651 | 16.3 / 22.0 | 2.1 / 84.0 |
| 1 | shared | AdaLFISTASW | MatrixMult | factored | 19.086 | 7.785 | 8.626 | 267.0 / 342.0 | 126.1 / 216.0 |
| 1 | shared | AdaLFISTASW | MatrixMult | precomputed | 25.366 | 17.981 | 8.240 | 335.0 / 406.0 | 126.1 / 344.0 |
| 1 | shared | AdaLFISTASW | CMFDictionary | factored | 17.144 | 7.817 | 8.498 | 269.1 / 342.0 | 128.2 / 236.0 |
| 1 | shared | AdaLFISTASW | CMFDictionary | precomputed | 22.992 | 18.037 | 8.357 | 337.1 / 406.0 | 128.2 / 364.0 |
| 1 | shared | AdaLFISTASW | CMFPropagation | factored | 45.342 | 4.808 | 20.128 | 205.7 / 278.0 | 64.2 / 172.0 |
| 1 | shared | AdaLFISTA | MatrixMult | factored | 21.638 | 10.905 | 9.313 | 329.3 / 406.0 | 188.1 / 344.0 |
| 1 | shared | AdaLFISTA | MatrixMult | precomputed | 28.028 | 18.703 | 9.136 | 397.3 / 470.0 | 188.1 / 472.0 |
| 1 | shared | AdaLFISTA | CMFDictionary | factored | 24.144 | 10.909 | 9.540 | 331.4 / 406.0 | 190.2 / 364.0 |
| 1 | shared | AdaLFISTA | CMFDictionary | precomputed | 28.006 | 18.704 | 8.849 | 399.4 / 470.0 | 190.2 / 492.0 |
| 1 | shared | AdaLFISTA | CMFPropagation | factored | 50.859 | 7.988 | 21.894 | 268.1 / 342.0 | 126.2 / 300.0 |
| 1 | shared | AdaLFISTASWR | CMFDictionary | factored | 42.550 | 6.599 | 22.157 | 84.6 / 108.0 | 66.2 / 172.0 |
| 1 | shared | AdaLFISTASWR | CMFPropagation | factored | 62.458 | 3.390 | 34.880 | 27.2 / 44.0 | 2.3 / 44.0 |
| 1 | shared | AdaLFISTAR | CMFDictionary | factored | 66.134 | 8.166 | 38.551 | 86.4 / 110.0 | 66.3 / 174.0 |
| 1 | shared | AdaLFISTAR | CMFPropagation | factored | 114.898 | 4.902 | 54.930 | 29.1 / 46.0 | 2.3 / 46.0 |
| 32 | shared | ISTA | MatrixMult | — | 3.435 | 3.437 | 3.330 | 75.1 / 88.0 | 65.0 / 154.0 |
| 32 | shared | ISTA | CMFDictionary | — | 3.435 | 3.456 | 3.198 | 77.2 / 88.0 | 67.0 / 174.0 |
| 32 | shared | ISTA | CMFPropagation | — | 20.543 | 14.071 | 9.643 | 144.2 / 154.0 | 3.1 / 260.0 |
| 32 | shared | FISTA | MatrixMult | — | 3.560 | 3.467 | 4.571 | 75.6 / 90.0 | 65.0 / 156.0 |
| 32 | shared | FISTA | CMFDictionary | — | 3.569 | 3.467 | 4.523 | 77.7 / 90.0 | 67.0 / 176.0 |
| 32 | shared | FISTA | CMFPropagation | — | 20.235 | 14.286 | 10.039 | 144.7 / 154.0 | 3.1 / 262.0 |
| 32 | shared | AdaLFISTASW | MatrixMult | factored | 17.115 | 14.178 | 10.528 | 289.7 / 368.0 | 129.0 / 306.0 |
| 32 | shared | AdaLFISTASW | MatrixMult | precomputed | 25.861 | 23.993 | 10.176 | 357.6 / 432.0 | 129.0 / 372.0 |
| 32 | shared | AdaLFISTASW | CMFDictionary | factored | 20.861 | 14.197 | 10.875 | 291.7 / 368.0 | 131.1 / 326.0 |
| 32 | shared | AdaLFISTASW | CMFDictionary | precomputed | 25.584 | 23.984 | 10.648 | 359.7 / 432.0 | 131.1 / 392.0 |
| 32 | shared | AdaLFISTASW | CMFPropagation | factored | 51.616 | 38.450 | 22.173 | 309.4 / 384.0 | 67.1 / 278.0 |
| 32 | shared | AdaLFISTA | MatrixMult | factored | 24.263 | 22.167 | 11.990 | 360.1 / 440.0 | 191.0 / 384.0 |
| 32 | shared | AdaLFISTA | MatrixMult | precomputed | 30.695 | 29.046 | 11.413 | 428.2 / 504.0 | 191.0 / 514.0 |
| 32 | shared | AdaLFISTA | CMFDictionary | factored | 24.133 | 22.229 | 12.065 | 362.1 / 440.0 | 193.1 / 404.0 |
| 32 | shared | AdaLFISTA | CMFDictionary | precomputed | 30.872 | 29.146 | 11.181 | 430.2 / 504.0 | 193.1 / 534.0 |
| 32 | shared | AdaLFISTA | CMFPropagation | factored | 59.477 | 47.265 | 22.659 | 379.9 / 458.0 | 129.1 / 402.0 |
| 32 | shared | AdaLFISTASWR | CMFDictionary | factored | 44.791 | 8.633 | 23.533 | 146.1 / 216.0 | 69.1 / 224.0 |
| 32 | shared | AdaLFISTASWR | CMFPropagation | factored | 71.449 | 32.747 | 37.835 | 225.8 / 292.0 | 5.1 / 286.0 |
| 32 | shared | AdaLFISTAR | CMFDictionary | factored | 65.695 | 11.145 | 42.186 | 193.9 / 264.0 | 69.2 / 282.0 |
| 32 | shared | AdaLFISTAR | CMFPropagation | factored | 100.883 | 36.167 | 56.993 | 274.6 / 344.0 | 5.2 / 318.0 |
| 32 | distinct | ISTA | MatrixMult | — | 72.056 | 77.002 | 3.873 | 4043.1 / 4056.0 | 4033.0 / 6076.0 |
| 32 | distinct | ISTA | CMFDictionary | — | 72.056 | 77.097 | 3.528 | 4171.2 / 4184.0 | 4161.1 / 6204.0 |
| 32 | distinct | ISTA | CMFPropagation | — | 27.648 | 21.427 | 2.518 | 332.2 / 346.0 | 129.1 / 1308.0 |
| 32 | distinct | FISTA | MatrixMult | — | 72.158 | 77.021 | 4.643 | 4043.6 / 4058.0 | 4033.0 / 6076.0 |
| 32 | distinct | FISTA | CMFDictionary | — | 72.169 | 77.026 | 4.670 | 4171.7 / 4186.0 | 4161.1 / 6204.0 |
| 32 | distinct | FISTA | CMFPropagation | — | 28.088 | 21.615 | 10.445 | 332.7 / 348.0 | 129.1 / 1310.0 |
| 32 | distinct | AdaLFISTASW | MatrixMult | factored | 150.630 | 154.032 | 11.853 | 4257.7 / 4272.0 | 4097.0 / 6292.0 |
| 32 | distinct | AdaLFISTASW | MatrixMult | precomputed | 712.880 | 586.415 | 11.662 | 10181.6 / 10192.0 | 4097.0 / 10164.0 |
| 32 | distinct | AdaLFISTASW | CMFDictionary | factored | 150.637 | 153.996 | 11.201 | 4385.8 / 4400.0 | 4225.1 / 6420.0 |
| 32 | distinct | AdaLFISTASW | CMFDictionary | precomputed | 712.870 | 586.469 | 11.499 | 10309.8 / 10320.0 | 4225.1 / 10292.0 |
| 32 | distinct | AdaLFISTASW | CMFPropagation | factored | 64.900 | 48.168 | 22.585 | 497.5 / 512.0 | 193.2 / 514.0 |
| 32 | distinct | AdaLFISTA | MatrixMult | factored | 163.152 | 166.493 | 11.828 | 4328.1 / 4344.0 | 4159.0 / 6368.0 |
| 32 | distinct | AdaLFISTA | MatrixMult | precomputed | 865.560 | 604.748 | 11.973 | 10252.2 / 10264.0 | 4159.0 / 10242.0 |
| 32 | distinct | AdaLFISTA | CMFDictionary | factored | 163.203 | 166.666 | 12.539 | 4456.2 / 4472.0 | 4287.1 / 6496.0 |
| 32 | distinct | AdaLFISTA | CMFDictionary | precomputed | 865.743 | 603.841 | 12.069 | 10380.3 / 10392.0 | 4287.1 / 10370.0 |
| 32 | distinct | AdaLFISTA | CMFPropagation | factored | 74.962 | 57.505 | 23.645 | 567.9 / 584.0 | 255.2 / 658.0 |
| 32 | distinct | AdaLFISTASWR | CMFDictionary | factored | 158.954 | 148.440 | 25.149 | 4240.1 / 4248.0 | 4163.2 / 6250.0 |
| 32 | distinct | AdaLFISTASWR | CMFPropagation | factored | 77.808 | 42.046 | 38.027 | 413.9 / 420.0 | 131.2 / 522.0 |
| 32 | distinct | AdaLFISTAR | CMFDictionary | factored | 178.956 | 155.234 | 43.623 | 4287.9 / 4296.0 | 4163.2 / 6310.0 |
| 32 | distinct | AdaLFISTAR | CMFPropagation | factored | 104.021 | 45.441 | 57.464 | 462.7 / 472.0 | 131.3 / 554.0 |

</details>

## Extending and testing

Loops are explicit; subclasses reuse the protected preparation, gradient, and callback
helpers ([architecture guide](docs/architecture.md)). Shared model properties are tested
once via pytest-cases in `tests/cases/`. See the [migration notes](docs/migration.md) for FA2026.
