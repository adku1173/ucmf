# Solver architecture

The classical interface is:

```python
from ucmf import FISTA
from ucmf.optim.operator import MatrixMult

solver = FISTA(positive=True, callbacks=[])
x = solver(y, x0=None, t=100, operator=MatrixMult(A), lam=0.01, gamma=None)
```

ISTA has the same interface. There is no public W input and no gradient-mode
selector: both classes use normalized least squares. See
[the current decision](adr/0003-local-loops-and-ls-only.md).

## Read the algorithm directly

```text
forward → _prepare_operator_inputs → resolve lam/gamma/theta → _iterate
    ISTA:  x ← proximal step at x
    FISTA: _iterate_accelerated
             x_next ← proximal step at z
             tk, z, x ← Nesterov update
```

`ISTA._iterate` and `FISTA._iterate_accelerated` each contain an explicit loop.
FISTA keeps x, z, and tk local; a new call always initializes them afresh.
There are no dataclasses, context objects, runner, or state-mutating closures.
The small amount of repeated loop control is intentional: the full algorithm
is visible without following a generic iteration framework.

The shared mathematics remains in `step`, `_make_gradient`, shrinkage/operator helpers,
and `_nesterov_momentum`. Stateless tensor preparation, batch-parameter handling,
operator application, step-size calculation, and shape validation live in
`src/ucmf/optim/utils.py` and are called directly. They are not subclass hooks.
The source SFISTA, ALFISTA, NA_ALISTA, and NA_ALFISTA classes override none of
these utilities. `_make_gradient(y, operator, weight_operator)` binds the data
and operators once per solve and returns the negative gradient direction at `x`.

Input conversion belongs to preprocessing. Classical forward requires a
`LinearOperator` and checks that its shape, batch size, dtype, and device
match the inputs. It detaches the operator and never silently moves or casts
inputs.
Internally allocated solution vectors use `y.new_zeros(...)`.
Learned models must likewise be placed explicitly with `model.to(...)` before
forward; model-owned schedules/controllers remain responsible for their outputs.

Both loops call `_start_iterations`,
`_on_iteration_begin`, and `_on_iteration_end` for the same callback lifecycle:

1. Set callback-visible y and clear the stop flag.
2. Before each step, set the iteration index and run begin callbacks.
3. Stop immediately if requested; otherwise compute the step.
4. Run end callbacks with the completed solution, gamma, theta, and the
   separate `callback_metadata` (`A`, `W`, `y`, and model-specific fields).

For classical solves A and W refer to the same detached operator. Callback order, batched
stopping, intermediate autograd graphs, and zero-step behavior are unchanged.
The existing callback attributes (`y`, `iter`, `stop`) remain mutable;
one shared solver/callback instance is not made concurrent or reentrant.

## Learned step schedules

`_iterate_accelerated` takes a prepared `gradient(x)`, `callback_metadata`,
and a `step_params_fn` for the schedule.
The schedule returns `(gamma_k, theta_k)` for one iteration. It runs after
begin callbacks, exactly once per executed step, and never for t=0.

For example, an already-prepared learned subclass can implement:

```python
def _step_params(self, k):
    return self.gammas[k], self.thetas[k]

def _iterate(self, y, x0, n_sources, gamma, theta, t, *, gradient, callback_metadata):
    return self._iterate_accelerated(
        y, x0, n_sources, t, self._step_params,
        gradient=gradient, callback_metadata=callback_metadata,
    )
```

The schedule can select trainable tensors without detaching them. The model
owns parameter constraints and valid iteration counts. For a tied LASSO
threshold, return `lam * gamma_k`; directly learned theta is a different policy.

## Recurrent models own their loop

A neurally augmented model can override forward or `_iterate`, initialize
hidden/cell tensors locally, and predict gamma/theta inside a normal for-loop.
Reuse the prepared `gradient(x)`, `get_shrink_func`, `_init_solver_vector`,
and the callback helpers. An accelerated version
can reuse `_nesterov_momentum` with local x, z, tk; it returns `(tk, z)`
and the caller assigns `x = x_next`.

This is the same straightforward control-flow pattern as
[NA-ALISTA's reference implementation](https://github.com/feeds/na-alista/blob/master/utils/algorithms.py).
No state dataclass is needed for autograd. The seam is the concrete schedule
or model-owned loop, not a
universal controller framework; locality and leverage come from the shared
mathematics and callback helpers rather than another adapter.

## Subclasses own weighted forward inputs

A weighted model overrides forward instead of passing W to `super().forward`.
It prepares the fixed operator with `_prepare_operator_inputs` and builds its
effective weight as a `LinearOperator` from its own parameters, e.g. with
`make_weight_operator(operator, self.B)` from `weights.py`:

```python
y, x0, operator, _ = self._prepare_operator_inputs(y, x0, operator)
weight_operator = make_weight_operator(operator, self.B)
return self._iterate_accelerated(
    y, x0, operator.shape[1], t, step_params_fn,
    gradient=self._make_gradient(y, operator, weight_operator),
    callback_metadata={'A': operator, 'W': weight_operator, 'y': y},
)
```

The operator is detached; the weight operator keeps its autograd graph.
There is no tensor A/W path. The classical automatic LS step-size guarantee
need not hold for W != A.

## Single-weight Ada-LFISTA

`AdaLFISTASW` (`src/ucmf/optim/ada_lfista_sw.py`) uses the same operator
interface as FISTA, with a learned adjoint-side weight.

```text
forward(y, x0, t, operator)
    _prepare_operator_inputs                # validate and detach A
    B_active = B or B[I, I]                   # I = operator.rows
    W = B_active @ A                         # factored or precomputed
    _iterate_accelerated(..., step_params_fn=k -> (gamma[k], theta[k]))
        r = y - A.forward(z)
        x_next = shrink(z + gamma_k * W.adjoint(r) / m, theta_k)
```

Trainable parameters are `B` (identity init), `_gamma` and `_theta`; the
properties `gamma` / `theta` are the applied, softplus-constrained values.
The constructor's `weight_mode='factored'` (default) applies
`Aᵀ(B_activeᵀr)` without forming `W`. `weight_mode='precomputed'` forms
`W` once per solve and requires `MatrixMult` or `CMFDictionary`; it never
converts `CMFPropagation` to a dense dictionary.

For microphone CV, call `select_mics` and align `y` using `operator.rows`.
These indices must refer to the full measurement coordinates of `B`.
End callbacks receive the detached measurement operator as `A` and the
effective weight operator as `W` in both modes; gradients through `B` are preserved.

`compile_unrolled(model, t_max)` (`compile.py`) compiles one graph for `t_max`
iterations and returns all iterates `(t_max, B, n)` via an
`IntermediateOutputCollector`. Staged training uses `out[t - 1]` without a
recompilation per stage.

`AdaLFISTA` (`ada_lfista.py`, [0006](adr/0006-two-matrix-ada-lfista.md)) uses
`operator=` and restricts both `B1` and `B2` using `operator.rows`. It computes
`M.adjoint(y)` once with the factored `M = B2_active @ A`. The recurrence
uses `R.adjoint(R.forward(z))` with `R = B1_active @ A`. By default `R` is
factored; `weight_mode='precomputed'` builds only `R` and requires a dense
measurement operator. End callbacks receive operators `A`, `W = M`, and
`W_recurrent = R`.

## Intentional removals

- `ls_mode`, `tls_grad`, and TLS dispatch/warnings: LS is the only objective.
- `IterationState`, `IterationContext`, `_IterationRunner`, and `iteration.py`.
- `initialize_state`, `step_parameters`, `iteration_step`, `update_state`.
- The generic `_run_iterations` and `_iterate_scheduled` drivers. Custom loops
  now live in `_iterate` or `_iterate_accelerated` and reuse the callback helpers.

When porting original unfolded constructors, remove `ls_mode='ls'` from their
`super().__init__` call. NA_ALISTA's `_resolve_batch_param(...)` call should
become an import of `resolve_batch_param` from `ucmf.optim.utils`; validate its
optional tensor-valued lam before reshaping, without the old `reference_tensor`
argument. No compatibility alias is needed. Do not silently map TLS requests
to LS. See the [migration guide](migration.md) for the gradient and iteration
hooks used by custom subclasses.

“The interface is the test surface”: numerical oracles, KKT checks, autograd,
unfolded models against a shared dense reference, learned schedules, local recurrent loops, repeat
solves, and callback behavior are tested rather than state-container internals.
