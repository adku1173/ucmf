# ucmf

## Installation and checks

Python 3.10 or newer is required.

```bash
uv sync
uv run pytest
uv run ruff check .
uv run python examples/sparse_recovery.py
```

Alternatively, use `pip install -e .` (and install `pytest` separately for tests).
Runtime dependencies are PyTorch and NumPy; Acoular, TensorFlow, datasets,
training code, and the original `fa2026` package are not required.

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
`sqrt(2)` weights). This package does not construct or vectorize CSMs yet.

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
