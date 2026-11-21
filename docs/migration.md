# Migration from FA2026_code

## Provenance and scope

The initial extraction used `FA2026_code` at commit
`8bd9e64c35dbe65bb4e605529115fb530113e144`. The source solver files were
unmodified in that working tree. The original repo remains untouched and is
not a runtime dependency.

| Source under `src/fa2026/optim/` | Destination under `src/ucmf/optim/` |
| --- | --- |
| `ista.py` | ISTA/FISTA, now with explicit local loops and LS only |
| `weights.py` | Protected weight preparation for subclass-owned forwards |
| `preprocessing.py` | Explicit column normalization |
| general classes in `callbacks.py` | Stopping, parameter histories, intermediate outputs |

The package does not import the source optimizer initializer, training stack,
or acoustic callbacks. Runtime dependencies are only PyTorch and NumPy.

## Preserved LS behavior

- Objective: `||y - A x||² / (2 m) + lam * ||x||₁`.
- Negative gradient: `A.T @ residual / m`; threshold: `gamma * lam`.
- Default step: `1 / (||A||₂² / m)`, separately per batched A.
- Positive/signed shrinkage and the same FISTA/Nesterov recurrence.
- Shared/batched matrices, scalar/per-sample parameters, and cloned warm starts.
- Always-batched outputs and fixed/detached A; output dtype/device match y.
- Gradients through y, x0, lam, gamma, and subclass-supplied W.
- Explicit column normalization and physical-coordinate rescaling.
- Callback order/keywords, zero-based iteration indices, begin/end stopping,
  relative/absolute batch convergence, and intermediate computational graphs.

`StepParamCollector` also accepts Python scalars, correcting a source failure
when gamma was an explicit float. Invalid shapes, dtypes, iteration counts,
complex operators, and zero default-step spectral norms raise clear errors.

## Current interface and intentional removals

Constructors accept `positive=True, callbacks=None`. The public forward is
`forward(y, x0=None, t=1, A=None, lam=None, gamma=None)`.

- W is not a classical input, including W=None. Unfolded models own their
  weighted forwards and can reuse `_setup_forward(..., W=...)` and the
  protected iteration hooks. Callback metadata reports W=A for classical solves.
- TLS is removed, not corrected or aliased. `ls_mode` and `tls_grad` are gone;
  even `ls_mode='ls'` is no longer accepted. Drop the obsolete keyword from
  original unfolded constructors' `super().__init__` calls when porting them.
- Silent dtype/device conversion and array-like forward inputs are removed.
  Preprocess arrays into real floating tensors and align all tensor inputs
  before calling forward. Python scalar lam/gamma remain supported. Forward
  validates dtype/device compatibility rather than transferring or casting.
- Stateless utilities are imported directly, not retained as class aliases.
  None of the source unfolded subclasses override the eight extracted helpers.
  NA_ALISTA calls `_resolve_batch_param`; when porting, import
  `resolve_batch_param` from `ucmf.optim.utils` and validate optional tensor lam
  explicitly. Place learned models with `model.to(...)` before forward.
- The intermediate state/context dataclasses, runner, and their transition
  hooks are removed. So are the generic `_run_iterations` and
  `_iterate_scheduled` drivers. Use explicit loops and callback helpers.

`_iterate_accelerated(..., step_params_fn=...)`, preparation tuple results,
gradient/shrinkage hooks, warm-start initialization, momentum, and
callback helpers remain reusable. Non-default W is an unfolded update, not a
classical LASSO gradient with the same convergence guarantee.

See [the architecture guide](architecture.md),
[the weighted-forward decision](adr/0002-classical-forward-without-weights.md),
and [the local-loop decision](adr/0003-local-loops-and-ls-only.md).
The earlier [state-record decision](adr/0001-explicit-iteration-state.md) is
retained as a superseded historical record, not the current design.

## Validation

The self-contained tests do not import fa2026. They cover independent NumPy
iterations, analytic solutions and KKT conditions, shapes/parameter forms,
CPU/CUDA, autograd, callback lifecycle, learned step schedules, local recurrent
loops, and weighted test subclasses with their own forward. Removal tests
ensure obsolete mode keywords and public W are rejected.

At the local-loop refactor, all 198 tests passed. Direct comparison with the
preceding implementation was bit-exact in 384 LS trajectory cases and eight
backward cases on CPU/CUDA. Another 192 comparisons exercised the original
scheduled/recurrent source classes through test-only constructor ports that
drop the obsolete ls_mode keyword; outputs, gradients, and model state matched
exactly. There is no compatibility mode argument in production ucmf.

## Deferred unfolded migration

No learned production model, controller, weight strategy, training objective,
checkpoint format, physical model, or generator is migrated here.

The source separates iteration logic (`sfista.py`, `alfista.py`, `na_alista.py`)
from weight strategies (`ada.py`, `congruence.py`, `learned_weight.py`,
`rank_one_weight.py`). The source's `ADA-CMF-ALFISTA` and
`ADA-CMF-NA-ALFISTA` use a compact congruence transformation through real Q_re /
Q_im matrices. Their later public LISTA names should be decided during porting,
not silently equated to accelerated FISTA architectures now.

Before porting CMF weights, preserve the corrected sqrt(2)-weighted Hermitian
CSM convention, diagonal removal, and zero-fill/transform/select behavior for
measurement subsets. These concerns do not belong in classical solver inputs.
