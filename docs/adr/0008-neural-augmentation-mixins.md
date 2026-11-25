# 0008: Neural augmentation via step-parameter mixins

Status: Accepted — requested by the author.

## Context

FA2026 combined NA controllers (`NA_ALFISTA`, `NA_LFISTA`) with ADA wrappers. In ucmf,
unfolded models own their forward ([0006](0006-two-matrix-ada-lfista.md)). Four weight types × two step-parameter kinds
must not duplicate code.

## Decision

- Each weight type is a private FISTA base implementing
  `_prepare_update(y, x0, operator) -> (y, x0, gradient, residual_gradient, callback_metadata)`.
- Step parameters are mixins owning `forward`: `LearnedParams` (per-layer softplus
  `gamma`/`theta`) and `NeuralAugmentation` (LSTM). Models are one-line combinations,
  e.g. `NA_AdaLFISTASW(NeuralAugmentation, _SingleWeight)`. Constructors are
  keyword-only; the `state_dict` keys of the Ada models are unchanged.
- `residual_gradient(z)` returns `(y - A z, direction)`: SW/SWR reuse the residual,
  LFISTA/R pay one extra `A.forward`, only for NA.
- Controller as FA2026's `NormLSTMCB` with init readout: features `||y - A z||_1 / m`
  and `||d||_1`, `LSTMCell`, readout `softplus(linear(relu(lll(cell))))` predicting
  `gamma_k`, `theta_k` per sample. Only direct `theta` is migrated; `theta = lam * gamma`
  would be another mixin with `forward(..., lam)`.
- Explicit feature normalization: NaN buffers, set by `init_normalizers(y, operator, x0=None)`
  (batch mean and population std at `z = x0`). FA2026 set them lazily via `.item()` per step.
- Not migrated: `NA_ALISTA`, `NormLSTMB`/`NormLSTMC`, `LinearTransform`, NA without Ada weight.

## Consequences

- Bit-exact to FA2026 `NA_ALFISTA` for identity weights (checked once, not in CI).
- No host syncs: every model compiles to one graph, also with `compile_unrolled` and CUDA graphs.
- Without `init_normalizers`, outputs are NaN. NA depth is unlimited (`t >= 0`).
