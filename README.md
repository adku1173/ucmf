# ucmf

## Installation and checks

Python 3.13 or newer is required.

```bash
uv sync
uv run pytest
uv run ruff check .
uv run python examples/sparse_recovery.py
```

Runtime dependencies are PyTorch and NumPy. The `examples` dependency group
(included in `dev`, installed by `uv sync`) adds Acoular and Matplotlib.
Acoular is taken from the `cmf-problem-assembly` branch via `[tool.uv.sources]`;
the `acoular.solvers` interface used by `ucmf.acoular` is not yet released on PyPI,
so with `pip install -e .` install Acoular separately from that branch:
`pip install "acoular @ git+https://github.com/acoular/acoular.git@cmf-problem-assembly"`.
`import ucmf` never imports Acoular; only `ucmf.acoular` does.
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
    solver=FISTATorchSolver(device='cuda', precision='float32', n_iter=5000, tol=1e-8),
    alpha=1e-9,
    nonnegative=True,
    dictionary_scaling='unit_l2',
)
bf = ac.BeamformerCMF(freq_data=ps, steer=st, problem=problem)
```

- `TorchSolver(LeastSquaresSolver)` is the shared base class; `ISTATorchSolver`
  and `FISTATorchSolver` select the ucmf solver.
- Traits: `device` (`'cpu'`, `'cuda'`, ...), `precision` (`'float64'` or
  `'float32'`), `n_iter` (maximum iterations per frequency bin), and `tol`
  (relative solution change for `SolutionChangeStopper`). All enter the
  solver digest, so Acoular's caching distinguishes configurations.
- The problem must be an `L1RegularizedLeastSquaresProblem`: positivity comes
  from `problem.nonnegative`, and the regularization is
  `lam = problem.alpha * problem.unit_mult`, matching Acoular's LassoLars
  convention. The ucmf objective equals scikit-learn's Lasso objective, so
  results are directly comparable to `LassoLarsSolver(backend_kwargs={'fit_intercept': False})`.
  Scaling is done by the problem, not the solver.
- An optional `start_value` is used as warm start `x0`.
  `SolverOutput.info` reports `n_steps` and `converged`.

### Example: three sources

[`examples/sparse_recovery.py`](examples/sparse_recovery.py) simulates the
three white-noise sources of Acoular's introductory example in front of the
64-microphone array and solves the same CMF problem (third-octave band at
8 kHz, `alpha=1e-9`) with LassoLars, ISTA, and FISTA (5000 iterations each):

![CMF maps of three sources with LassoLars, ISTA, and FISTA](docs/images/three_sources_cmf.png)

FISTA matches the LassoLars map up to a relative difference of about `2e-4`.
ISTA has not yet converged after 5000 iterations and still yields
spatially smeared sources.

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
