# Code Context

Repository analysis for matrix-free working-set IR-FISTA-B integration.

- Reference verified: `ab8721053810413928304cc219a83ec8c0a4400f` (`ab87210`).
- Read-only inspection; no files written or modified.
- Existing untracked `.pi/` and `docs/ir_fista.md` preserved.
- Report destination for routing: `/home/kujawski/Documents/Code/ucmf/.pi/research/ir-fista-b/repository_analysis.md`.

## Files Retrieved

1. `src/ucmf/optim/ista.py` (lines 1–321): objective, validation, proximal steps, callbacks, ISTA/FISTA loops.
2. `src/ucmf/optim/operator.py` (lines 1–227, 375–449; selection symbols located through line 563): operator contract, dense restriction, matrix-free CMF restriction.
3. `src/ucmf/optim/callbacks.py` (lines 1–213): callback metadata, collectors, iterate-change stopping.
4. `src/ucmf/optim/utils.py` (lines 1–212): strict tensor validation, batch parameters, matrix-free spectral estimate.
5. `src/ucmf/optim/ada_lfista.py` (lines 1–128): two-weight adaptive direction.
6. `src/ucmf/optim/ada_lfista_sw.py` (lines 1–112): single-weight adaptive direction.
7. `src/ucmf/optim/ada_lfista_r.py` (lines 1–109): reduced two-weight CMF model.
8. `src/ucmf/optim/ada_lfista_swr.py` (lines 1–101): reduced single-weight CMF model.
9. `src/ucmf/optim/learned_params.py` (lines 65–86): learned schedule dispatch.
10. `src/ucmf/optim/compile.py` (lines 1–45): fixed-depth compilation and intermediate collection.
11. `src/ucmf/optim/__init__.py` (lines 1–32), `src/ucmf/__init__.py` (lines 1–27): public exports.
12. `tests/test_ista.py` (lines 1–243; additional test symbols located through line 248): numerical oracles, KKT checks, public interface.
13. `tests/conftest.py` (lines 1–14): unconditional Acoular dependency during test collection.
14. `tests/test_matrix_mult.py`, `tests/test_iteration.py`, `tests/test_input_validation.py`, `tests/test_callbacks.py`: targeted test-symbol inspection.
15. `README.md` (lines 1–125): installation, conventions, preprocessing, differentiation.
16. `docs/architecture.md` (lines 1–57): explicit-loop architecture and model dependencies.
17. `examples/benchmark_solvers.py` (lines 1–90): solver timing and PyLops scaling.
18. `examples/benchmark_operators.py` (lines 1–48): CUDA benchmark filters and timing scope.
19. `pyproject.toml`: dependency groups and test/lint configuration.

## Key Code

### Objective and positive shrinkage

`src/ucmf/optim/ista.py:22–27,94–110,188–201` defines

```text
f(x) = ||y - A x||² / (2m) + lam ||x||₁
negative_gradient(x) = Aᵀ(y - A x) / m
theta = gamma * lam
```

The positive proximal operator is explicitly:

```text
relu(x - theta)
```

It is **not** shrinkage of `abs(x)` (`ista.py:76–92`). Signed shrinkage is `x - clamp(x, -theta, theta)` (`58–74`).

For residual correlation `c = Aᵀ(y - A x)/m`:

- Positive problem: active coordinates require `c = lam`; inactive coordinates require `c <= lam`.
- Signed problem: active coordinates require `c = lam * sign(x)`; inactive coordinates require `|c| <= lam`.

These signs match `tests/test_ista.py:104–118`. Working-set selection must use the one-sided violation for the positive problem.

### Public solver interface

`src/ucmf/optim/ista.py:234–278`:

```text
forward(self, y, x0=None, t=1, operator=None, lam=None, gamma=None)
```

Canonical usage:

```text
solver(y, operator=operator, lam=lam, t=t)
```

Contracts:

- `y`: real floating `(m,)` or `(B,m)`.
- `x0`: `(n,)` for a single problem or `(B,n)`; otherwise initialized to zeros.
- Output always `(B,n)`, including single-input solves.
- Operator must inherit `LinearOperator`; dense tensors require `MatrixMult`.
- All tensors must already match `y.dtype` and `y.device`.
- `lam` and `gamma`: Python scalar, scalar tensor, `(B,)`, or `(B,1)`.
- Operators are detached; gradients remain available through measurements, warm starts, and tensor-valued regularization/step parameters.
- `t=0` still validates and resolves parameters, including default spectral estimation, but executes no callbacks.

Exact classical signatures are tested in `tests/test_ista.py:219–235`; do not modify existing signatures just to accommodate IR-specific settings.

### FISTA recurrence and reuse points

`src/ucmf/optim/ista.py:288–321`:

```text
t_next = (1 + sqrt(1 + 4*t_k²))/2
z_next = x_next + (t_k - 1)/t_next * (x_next - x)
```

`_iterate_accelerated` accepts a prepared gradient closure and `step_params_fn(k)`. Preparation, shrinkage, momentum, and callback helpers are reusable independently.

There is no existing restart, line search, working-set controller, or optimality certificate in the classical loop.

## Architecture

### Operator layer and restricted capability

`LinearOperator` requires only:

- Metadata: `shape=(m,n)`, `batch_size`, `dtype`, `device`, `rows`.
- `forward((B,n)) -> (B,m)`.
- `adjoint((B,m)) -> (B,n)`.
- `detached()`.

Column restriction is **not** part of the base contract (`operator.py:50–119`).

Concrete restriction capabilities:

| Operator | Column restriction | Consequence |
|---|---|---|
| `MatrixMult` | `select_columns(index)`, lines 192–207 | Selects stored matrix columns; no source Gram needed, but selected dense storage is materialized. |
| `CMFPropagation` | `select_columns(index)`, lines 433–447 | Selects propagation columns; remains matrix-free with respect to the CMF dictionary. |
| `CMFDictionary` | Override located at line 545 | Dense CMF implementation has source selection; inspect override before extending its metadata behavior. |
| `_AuxiliaryOperator` | None in lines 210–227 | A composition cannot automatically be restricted through its public API. |
| Arbitrary `LinearOperator` | Not guaranteed | Requires fallback or explicit capability validation. |

Indices can be shared `(k,)` or per-problem `(B,k)`, must be integer tensors on the operator device, and are converted to `long`. Uniqueness and range are caller preconditions, intentionally not checked to avoid GPU synchronization (`operator.py:1–11,24–47`).

Important limitations:

- Per-problem sets are rectangular: all samples have the same `k`.
- Per-problem restriction of a shared operator promotes it to a batched operator.
- No automatic original-source index metadata exists; the solver must retain its own local-to-global source mapping.
- `rows` records **measurement** provenance, not source indices.
- Repeated restriction uses local indices into the current operator.
- `MatrixMult` is not inherently matrix-free: it stores the full dictionary.
- `CMFPropagation` avoids dictionary and Gram construction but still stores `H` and forms microphone covariance intermediates (`operator.py:410–419`).

A generic fallback can scatter a restricted vector into full source coordinates, call the full forward, then gather the full adjoint. This preserves correctness without dense matrix access, but forfeits restricted computational savings and allocates full-width vectors. It should be identified as a fallback, not advertised as efficient restriction.

### Scaling and step-size estimation

`utils.py:80–122` estimates

```text
L = 1.05 * ||A||₂² / m
gamma = 1/L
```

using 50 matrix-free power iterations and one final forward. The seeded CPU generator does not consume global RNG.

Risks:

- The estimate is a lower-bound approximation with heuristic inflation, not a guaranteed upper bound.
- A zero or nonfinite estimate raises `ValueError`.
- Re-estimating on every working-set update can dominate runtime.
- Empty restricted operators are incompatible with current input preparation.
- Column restriction leaves measurement count `m` unchanged.
- Row restriction changes the current solver denominator to the active measurement count, not `_num_measurements_total`.

A safe full-operator Lipschitz bound also bounds a column-restricted operator, although it may be conservative. If IR-FISTA-B requires guaranteed descent, add appropriate verification/backtracking rather than relying exclusively on the existing heuristic.

Equivalent unnormalized LASSO regularization is `m * lam` (`README.md:48–61`). The PyLops benchmark additionally documents its convention as `eps=2*m*lam`, with its step equal to `gamma/m` (`benchmark_solvers.py:35–42`).

### Adaptive models are related infrastructure, not equivalent objectives

- SW: `Aᵀ Bᵀ(y-Az)/m` (`ada_lfista_sw.py:1–16,46–52`).
- Two-weight: `(Mᵀy - RᵀRz)/m`, with `R=B1 A`, `M=B2 A` (`ada_lfista.py:1–13,48–66`).
- SWR/R replace full measurement weights with learned complex microphone congruences.
- `LearnedParams` owns the forward path and uses independent positive learned `gamma/theta`; it does not accept `lam` (`learned_params.py:75–86`).
- Neural variants own a controller loop; two-weight controller residuals require an extra physical forward (`ada_lfista.py:61–63`).

Do not apply ordinary LASSO KKT screening/certification to these learned directions without additional mathematical justification. A new classical IR-FISTA-B solver should reuse low-level helpers without changing adaptive model behavior or their mixin method resolution.

### Callbacks and stoppers

`ista.py:203–218` and `callbacks.py:6–39` establish:

- Begin callback receives full current `x`, no metadata.
- End callback receives `gamma`, `theta`, `A`, `W`, `y`; classical `W=A`.
- `solver.iter` is zero-based.
- A callback stops by setting `solver.stop=True`.
- Callback iterates retain autograd graphs.

`SolutionChangeStopper` (`callbacks.py:130–213`) compares consecutive iterates, checks every 100 completed steps by default, and requires every batch sample to pass. It is not a KKT or dual-gap certificate. It synchronizes when evaluating the tensor condition in Python.

Working-set-specific pitfalls:

- Inner restricted iterates change shape as sets change; existing collectors and stoppers expect stable global source coordinates.
- A tiny restricted change does not prove omitted coordinates are optimal.
- Preserve callback meaning by exposing reconstructed `(B,n)` iterates at a clearly documented public iteration boundary.
- Decide whether `t` counts outer working-set updates or inner proximal steps; current solvers count proximal steps.
- Restart/remap momentum when the working set changes; blindly keeping restricted momentum with changed coordinate order is incorrect.
- Solvers/callbacks are not reentrant (`docs/architecture.md:18–24`).

### Preprocessing

No tracked `preprocessing.py` exists at this reference; path discovery also found none outside the excluded environment/Git paths.

Current preprocessing responsibilities are distributed across:

- `utils.validate_tensor`: strict validation, never conversion.
- `ISTA._prepare_operator_inputs`: batch axes, shape checks, zero initialization, operator detachment.
- `HermitianVectorization`: CMF real coordinates and Frobenius-isometric weights.
- Caller code: explicit array conversion, dtype/device movement.
- Acoular integration: external problem assembly, as described by README.

There is no automatic column normalization (`README.md:72–86`). Any proposed normalization must account for changed coefficient units and coordinate-dependent penalties; it cannot silently preserve scalar `lam`.

## Concrete Integration Seams

1. **New independent solver module**, e.g. `src/ucmf/optim/ir_fista_b.py`.
   - Prefer subclassing `FISTA` for validation, shrinkage, callback helpers, and momentum.
   - Override `forward` and implement a local working-set loop.
   - Keep existing ISTA/FISTA and adaptive model interfaces unchanged.

2. **Restricted-operator capability boundary** in `operator.py`.
   - Explicitly detect native `select_columns`.
   - Either document native restriction as required or supply a scatter/gather wrapper.
   - Preserve original measurement metadata and detached semantics.
   - Keep source mapping solver-local unless a separately designed public source-index contract is needed.

3. **Full residual/global correlation versus restricted inner solve**.
   - Compute global correlation with the full operator for selection and certification.
   - Use native restricted operators for repeated inner forward/adjoint operations.
   - Scatter the restricted solution into full output coordinates.
   - Preserve nonzero warm-start coordinates when initializing sets.
   - Define removal, re-entry, support growth, and empty-set behavior explicitly.

4. **Step-size policy**.
   - Reuse a full-operator step conservatively or estimate a restricted step when worthwhile.
   - Account separately for setup cost and inner iteration cost.
   - Add line-search/restart logic if required by IR-FISTA-B theory.

5. **Diagnostics**.
   - Keep `forward` returning a tensor.
   - Expose detached diagnostic state or callback metadata for working-set sizes, global violations, inner/outer counts, and operator-call totals.
   - Document timing and stopping semantics.

6. **Exports/docs/tests**.
   - Add exports in both `src/ucmf/optim/__init__.py` and `src/ucmf/__init__.py` if top-level discoverability is intended.
   - Update README and architecture documentation.
   - Preserve the existing untracked `docs/ir_fista.md`; do not overwrite it.
   - Check `tests/test_package.py` and model-case registration before changing public exports or extending shared model coverage.

7. **Compilation remains optional**.
   - `compile_unrolled` assumes fixed-depth collection and stacks every iterate (`compile.py:8–45`).
   - Dynamic working-set widths, host decisions, and changing inner iteration counts do not fit that assumption automatically.
   - Establish eager correctness first; do not promise current single-graph behavior.

## Tests and Validation Plan

Existing high-value coverage:

- `tests/test_ista.py:17–74`: independent NumPy iterate oracle.
- `tests/test_ista.py:79–118`: dense/matrix-free equivalence, analytic solutions, normalized KKT.
- `tests/test_ista.py:122–187`: scaling, parameter broadcasting, gradients, preprocessing.
- `tests/test_ista.py:190–243`: invalid inputs and exact classical signatures.
- `tests/test_iteration.py`: schedule hooks, callback dispatch, momentum overrides, local-state resets.
- `tests/test_callbacks.py`: parameter metadata, stopper resets, all-batch stopping.
- `tests/test_operator.py`: inspect/extend operator selection and adjoint tests.
- `tests/test_input_validation.py`: strict boundary validation, including zero iterations.
- `tests/test_models.py`, `tests/cases/models.py`: shared model properties.
- `tests/test_compile.py`: existing compilation contract.

New tests should cover:

- Positive threshold rejects negative proposals rather than reflecting their magnitudes.
- Dense oracle agreement and final global KKT for both constraints if supported.
- Same solution with `MatrixMult` and `CMFPropagation`.
- Full working set reproduces the intended FISTA baseline.
- Omitted violated coordinates re-enter; zero support is handled without constructing an empty operator.
- Shared and distinct batched operators; nonidentical batch supports.
- Native restriction versus generic fallback, including adjoint identity.
- Warm-start preservation, source mapping, reordered sets, repeated solves.
- Correct `m` scaling under source restriction.
- Callback global shapes and honest stopping semantics.
- Explicit dtype/device errors and no unintended operator gradients.
- Operator-call/memory assertions preventing dictionary or source-Gram materialization.

## Dependencies and Commands

Runtime requirements from `pyproject.toml`:

- Python `>=3.13`
- NumPy `>=1.24`
- PyTorch `>=2.2`

Development adds pytest/pytest-cases, Ruff, and example dependencies. Acoular comes from the configured `cmf-problem-assembly` Git branch. Even core pytest runs require Acoular because `tests/conftest.py:3–4` imports it unconditionally.

Recommended commands, **not executed during this read-only analysis**:

```bash
uv sync
uv run pytest tests/test_ista.py tests/test_operator.py tests/test_matrix_mult.py
uv run pytest tests/test_callbacks.py tests/test_iteration.py tests/test_input_validation.py
uv run pytest tests/test_models.py tests/test_optim_utils.py tests/test_package.py
uv run pytest
uv run ruff check .
```

Existing benchmarks:

```bash
uv run python examples/benchmark_solvers.py
uv run python examples/benchmark_operators.py --help
uv run python examples/benchmark_operators.py \
  --mics 16 --grid 16 --batch 1 --layout shared \
  --model FISTA --operator CMFPropagation \
  --weight factored --mode eager --steps 15 --repeats 7
```

`benchmark_operators.py` is CUDA-oriented and excludes operator construction and spectral estimation; `benchmark_solvers.py` includes step estimation and transfer but excludes CSM/dictionary assembly. Report these differences rather than directly comparing incompatible timings.

For IR-FISTA-B benchmarking, add eager matrix-free cases measuring setup, global scans, restricted inner solves, total forward/adjoint counts, working-set sizes, peak memory, and time to a common **global** optimality tolerance. Fixed iteration counts alone are insufficient.

## Start Here

Open `src/ucmf/optim/ista.py:146–218,281–321`, followed by `src/ucmf/optim/operator.py:50–119,192–207,375–447`.

These establish the reusable preparation/callback/momentum seams and the central design constraint: efficient column restriction exists on concrete operators but is not guaranteed by `LinearOperator`.

## Residual Risks and Open Decisions

- IR-FISTA-B algorithm details were not specified here; restart, screening, inner accuracy, and `t` semantics require an explicit algorithm contract.
- Efficient ragged per-sample supports are unsupported by the current rectangular selection API.
- Global scans can dominate matrix-free runtime despite cheap restricted iterations.
- Existing default spectral estimates are heuristic.
- Current iterate-change stopping cannot certify a working-set solution.
- Dynamic working sets are not automatically compatible with existing compilation.
- Tests and benchmarks were not executed; environment readiness is unverified.
- No report file was written, honoring the explicit no-working-tree-edits constraint; this complete artifact is supplied for routing.