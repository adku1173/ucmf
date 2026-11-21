# Solver architecture

The classical interface is:

```python
solver = FISTA(positive=True, callbacks=[])
x = solver(y, x0=None, t=100, A=A, lam=0.01, gamma=None)
```

ISTA has the same interface. There is no public W input and no gradient-mode
selector: both classes use normalized least squares. See
[domain vocabulary](../GLOSSARY.md) and [the current decision](adr/0003-local-loops-and-ls-only.md).

## Read the algorithm directly

```text
forward → _setup_forward → _iterate
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

The shared mathematics remains in `step`, `ls_grad`, shrinkage/operator helpers,
and `_nesterov_momentum`. Stateless tensor preparation, batch-parameter handling,
operator application, step-size calculation, and shape validation live in
`src/ucmf/optim/utils.py` and are called directly. They are not subclass hooks.
The source SFISTA, ALFISTA, NA_ALISTA, and NA_ALFISTA classes override none of
these utilities. Custom gradient behavior belongs in `ls_grad` / `get_grad_func`;
shrinkage and the iteration logic remain in `ista.py`.

Input conversion belongs to preprocessing: forward requires real floating
tensors with matching dtype/device for y, A, x0, tensor-valued lam/gamma, and
subclass-supplied W. It validates these before iteration and never silently
moves/casts them. Internally allocated solution vectors use `y.new_zeros(...)`.
Learned models must likewise be placed explicitly with `model.to(...)` before
forward; model-owned schedules/controllers remain responsible for their outputs.

Both loops call `_start_iterations`,
`_on_iteration_begin`, and `_on_iteration_end` for the same callback lifecycle:

1. Set callback-visible y and clear the stop flag.
2. Before each step, set the iteration index and run begin callbacks.
3. Stop immediately if requested; otherwise compute the step.
4. Run end callbacks with the completed solution, gamma, theta, A, W, y.

For classical solves the callback operator W equals A. Callback order, batched
stopping, intermediate autograd graphs, and zero-step behavior are unchanged.
The existing callback attributes (`y`, `iter`, `stop`, `l2_norms`) remain mutable;
one shared solver/callback instance is not made concurrent or reentrant.

## Learned step schedules

`_iterate_accelerated(y, x0, A_tensor, t, step_params_fn, Op_tensor=None)`
retains the schedule interface used by the original SFISTA/ALFISTA classes.
The schedule returns `(gamma_k, theta_k)` for one iteration. It runs after
begin callbacks, exactly once per executed step, and never for t=0.

For example, an already-prepared learned subclass can implement:

```python
def _step_params(self, k):
    return self.gammas[k], self.thetas[k]

def _iterate(self, y, x0, A_tensor, gamma_value, theta_value, t, Op_tensor=None):
    return self._iterate_accelerated(
        y, x0, A_tensor, t, self._step_params, Op_tensor=Op_tensor,
    )
```

The schedule can select trainable tensors without detaching them. The model
owns parameter constraints and valid iteration counts. For a tied LASSO
threshold, return `lam * gamma_k`; directly learned theta is a different policy.

## Recurrent models own their loop

A neurally augmented model can override forward or `_iterate`, initialize
hidden/cell tensors locally, and predict gamma/theta inside a normal for-loop.
Reuse `_prepare_forward_inputs`, `_iteration_functions`, `_init_solver_vector`,
`_validate_iteration_count`, and the callback helpers. An accelerated version
can reuse `_nesterov_momentum` with local x, z, tk.

This is the same straightforward control-flow pattern as
[NA-ALISTA's reference implementation](https://github.com/feeds/na-alista/blob/master/utils/algorithms.py).
No state dataclass is needed for autograd, and no unfolded production model is
introduced here. The seam is the concrete schedule or model-owned loop, not a
universal controller framework; locality and leverage come from the shared
mathematics and callback helpers rather than another adapter.

## Subclasses own weighted forward inputs

A weighted model overrides forward instead of passing W to `super().forward`:

```python
from ucmf import FISTA

class WeightedForwardExample(FISTA):
    def forward(self, y, x0=None, t=1, A=None, lam=None, gamma=None, W=None):
        y, x0, A, _norms, gamma, theta, operator = self._setup_forward(
            y, x0, A, lam, gamma, W=W,
        )
        return self._iterate(y, x0, A, gamma, theta, t, Op_tensor=operator)
```

This illustration is not an unfolded model implementation. A real model may
construct W from its own parameters. Protected preparation preserves shared /
batched weight validation, dtype/device compatibility checks, and W's autograd graph.
The classical automatic LS step-size guarantee need not hold for W != A.

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
to LS. Protected
preparation tuple results, `_iterate` / `_iterate_accelerated` calling
contracts, gradients, shrinkage, momentum, and callback helpers remain.

“The interface is the test surface”: numerical oracles, KKT checks, autograd,
weighted-forward subclasses, learned schedules, local recurrent loops, repeat
solves, and callback behavior are tested rather than state-container internals.
