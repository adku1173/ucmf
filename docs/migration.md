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
| `ada.py` (`AdaptiveGradientWrapper`) + `alfista.py` (`ALFISTA`) | `ada_lfista_sw.py`: `AdaLFISTASW` (model `ADA-ALFISTA`, paper name Ada-LFISTA-SW) |
| `lfista.py` (`LFISTA`, `TwoMatrixGradientWrapper`) | `ada_lfista.py`: `AdaLFISTA` (model `ADA-LFISTA`, paper name Ada-LFISTA) |

The package does not import the source optimizer initializer, training stack,
or acoustic callbacks. Runtime dependencies are only PyTorch and NumPy.

## Preserved LS behavior

- Objective: `||y - A x||² / (2 m) + lam * ||x||₁`.
- Negative gradient: `A.T @ residual / m`; threshold: `gamma * lam`.
- Positive/signed shrinkage and the same FISTA/Nesterov recurrence.
- Shared/batched matrices, scalar/per-sample parameters, and cloned warm starts.
- Always-batched outputs and fixed/detached A; output dtype/device match y.
- Gradients through y, x0, lam, gamma, and subclass-supplied W.
- Explicit column normalization and physical-coordinate rescaling.
- Callback order/keywords, zero-based iteration indices, begin/end stopping,
  relative/absolute batch convergence, and intermediate computational graphs.

Changed: the default step is `1 / L` with `L = 1.05 * ||A||₂² / m`, where
`||A||₂²` is estimated by 50 power iterations (separately per batched A)
instead of the exact eigenvalue; default-step results therefore differ
slightly from the source. See [0005](adr/0005-power-iteration-default-step.md).

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

## Unfolded models

`ADA-ALFISTA` is migrated as `AdaLFISTASW`, a FISTA subclass with its own
forward instead of a wrapper ([0004](adr/0004-single-weight-ada-lfista.md)):

- Trainable `B`, `_gamma`, `_theta`; `gamma` / `theta` give the applied values.
  `lam` and `gamma` are not forward inputs.
- `measurement_indices` is renamed to `msm_subset`; its index range is no
  longer checked in forward (no GPU sync). `n_measurements` stays the full size.
- The wrapper's `A_init` argument is dropped (B always starts as identity).
- New, without source counterpart: `compile_unrolled` compiles once for
  staged training (the source did not compile).

`ADA-LFISTA` is migrated as `AdaLFISTA` ([0006](adr/0006-two-matrix-ada-lfista.md)).
Its weights follow the paper's numbering: `B1` is the recurrence (W1), `B2`
the measurement branch (W2), i.e. swapped relative to FA2026's `B1` / `B2`.

Still deferred: the compact CMF
congruence weights (`ADA-CMF-*`, paper suffix `-R`), NA controllers, training
objectives, checkpoint formats, physical models, and generators.

The source separates iteration logic (`sfista.py`, `alfista.py`, `na_alista.py`)
from weight strategies (`ada.py`, `congruence.py`, `learned_weight.py`,
`rank_one_weight.py`). The source's `ADA-CMF-ALFISTA` and
`ADA-CMF-NA-ALFISTA` use a compact congruence transformation through real Q_re /
Q_im matrices. Their later public LISTA names should be decided during porting,
not silently equated to accelerated FISTA architectures now.

Before porting CMF weights, preserve the corrected sqrt(2)-weighted Hermitian
CSM convention, diagonal removal, and zero-fill/transform/select behavior for
measurement subsets. These concerns do not belong in classical solver inputs.
