# Solver architecture

## Classical solvers

```text
forward → _prepare_operator_inputs → resolve lam/gamma/theta → _iterate
  ISTA:  x ← shrink(x + gamma · gradient(x), theta)
  FISTA: _iterate_accelerated: x_next ← step at z; tk, z ← Nesterov update
```

- Explicit loops with local `x`, `z`, `tk`; no state objects ([0003](adr/0003-local-loops-and-ls-only.md)).
- `_prepare_operator_inputs` validates `y`, `x0`, and the operator (shape, batch size, dtype,
  device), detaches the operator, and turns a missing `x0` into zeros. It never casts or moves.
- `_make_gradient(y, operator, weight_operator)` binds the data once per solve and returns
  `gradient(x)`; subclasses override it to change the direction.
- Stateless helpers live in `utils.py` and are no subclass hooks.

## Callback lifecycle

`_start_iterations` clears `stop`. Per step, `_on_iteration_begin` sets `iter` and runs the
begin callbacks (a requested stop ends the loop), then the step runs, then
`_on_iteration_end` passes `gamma`, `theta`, and `callback_metadata` (`A`, `W`, `y`,
model-specific entries) to the end callbacks. Classical solves pass `W = A`.
Solvers and callbacks are not reentrant.

## Unfolded models: weight base × step-parameter mixin

| Weight base | `LearnedParams` | `NeuralAugmentation` |
| --- | --- | --- |
| `_SingleWeight` (`B`) | `AdaLFISTASW` | `NA_AdaLFISTASW` |
| `_TwoWeights` (`B1`, `B2`) | `AdaLFISTA` | `NA_AdaLFISTA` |
| `_ReducedSingleWeight` (`Br`) | `AdaLFISTASWR` | `NA_AdaLFISTASWR` |
| `_ReducedTwoWeights` (`Br1`, `Br2`) | `AdaLFISTAR` | `NA_AdaLFISTAR` |

- Weight bases subclass FISTA and implement `_prepare_update(y, x0, operator)` →
  `(y, x0, gradient, residual_gradient, callback_metadata)`.
- Mixins own `forward`: `LearnedParams` runs `_iterate_accelerated` with per-layer
  `softplus(_gamma)`, `softplus(_theta)`; `NeuralAugmentation` runs its own loop with local
  LSTM state ([0008](adr/0008-neural-augmentation-mixins.md)). New step-parameter kinds are
  new mixins, new weight types new bases.
- Directions: SW/SWR `W.T (y - A z) / m` with `W = B A`; LFISTA/R `(M.T y - R.T R z) / m`
  with `R = B1 A`, `M = B2 A` ([0004](adr/0004-single-weight-ada-lfista.md),
  [0006](adr/0006-two-matrix-ada-lfista.md)). SWR/R use the congruence
  `B v = hvec(Br hvec*(v) Br^H)`. Subsets restrict the weights via `operator.rows` / `operator.mics`.
- `weight_mode='precomputed'` (SW, LFISTA) forms the dense product once per solve.

## Scaling

Solvers work in normalized coordinates and never see the column norms
([0009](adr/0009-column-normalization.md)):

```text
A → normalize_operator → (D = A U^-1, u) → solver(y, operator=D) → x → to_source(x, u) → q
ȳ → y = ȳ / lambda_max(D, ȳ),  q = q̄ / lambda_max       q̄ = lambda_max · q
```

- Operators provide `column_norms()` and `scale_columns(scale)`; `ucmf.scaling` composes them.
- Ada models require unit column norms (unchecked precondition); ISTA/FISTA accept any operator.
- Training normalizes the configured operator and, with `data_scaling='equireg'`, divides
  `y` and `q` per sample by `lambda_max` once; supervised losses compare `q`.
- CV recomputes both scalings on the training subset and applies them to the validation subset.

## Compilation

`compile_unrolled(model, t_max)` compiles once and returns all iterates `(t_max, B, n)`;
stage `t` uses `out[t - 1]`. All models compile to one graph without host syncs.

## Tests

`tests/cases/` holds the pytest-cases: `ModelCases` (every model as a `ModelSpec`, tagged
`classical`/`unfolded`, `learned`/`na`, `dense`, `cmf_only`, `weight_mode`, `two_weights`),
`ProblemCases`, and `OperatorCases`. `tests/helpers.py` holds callbacks and the dense
reference iteration. Shared properties are tested once in `test_models.py`.
