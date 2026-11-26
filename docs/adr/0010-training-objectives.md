# 0010: Training objectives as functions over all iterates

Status: Accepted — requested by the author; implemented.

## Context

`batch_loss` selected `mse`, `nmse`, or `relative_residual` by `if` branches on the
evaluated output. FA2026 trains with NMSE, LASSO, smoothed BIC, and CV, and can weight
earlier unfolded outputs (`weighted_intermediate_loss`). CV solves on a subset of the
measurements, so it cannot be a function of the full-batch output.

## Decision

1. `ucmf.train.objectives` holds one function per objective,
   `f(solve, y, q, t, operator, column_norms, *, w, ...)`. It runs the solve itself;
   `solve(y, operator, t)` returns all iterates `(t, B, n)`. `OBJECTIVES` maps
   `config.loss` to it; `build_loss(config)` binds `config.loss_kwargs` and the keywords
   of `SUPPLIED` the objective takes (`w`, the CV `generator` seeded with `config.seed`).
   `TrainingConfig` checks `loss_kwargs` against the signature.
2. `weighted_mean` weights iterate `k` of depth `t` with `w^(t-k)`, normalized to sum one,
   so the loss scale does not depend on `t` or `w` (FA2026: unnormalized sum).
   `w = 0` evaluates only `x_t`, so earlier non-finite losses cannot enter as `0 * inf`.
   Validation uses the same objective and `w`.
3. Objectives (`mse` and `relative_residual` were dropped):
   - `nmse`: `||q_hat - q||^2 / ||q||^2` with `q_hat = x_hat / u` (targets nonzero).
   - `lasso`: `||D x - y||^2 / (2 m) + lam ||x||_1`; `lam` relative to `lambda_max = 1`.
     The model coupling `theta = gamma lam` of the paper is not part of ucmf.
   - `bic`: `m log(||D x - y||^2 / m) + log(m) sum_j (1 - exp(-x_j^2 / (2 eps^2)))`,
     Gaussian family, `eps = alpha ||x||_inf` detached (a constant), `alpha = 1e-2`
     (20 dB below the maximum for power-like `x`). FA2026 named this setting
     `40db` with an amplitude conversion that was later changed to power; ucmf takes `alpha`.
     `||x||_inf` is clamped to `1e-12`, since zero solutions are common; the residual
     variance is not clamped, as measurements are always noisy.
   - `cv`: split, normalize the columns and `lambda_max` on the train subset, apply both
     to the validation rows, solve on the train subset, and return
     `||D_v x - y_v||^2 / (2 m_v)`. `D_v x` is the full forward of `x / u_t`, gathered
     at the validation rows, so no validation operator is formed. Requires a nonnegative model.
4. CV splits (`split`, `train_ratio = 0.8`), random per problem:

   | split | unit | Re/Im together | train / val / unused at 0.8 | train operator |
   | --- | --- | --- | --- | --- |
   | `rows` (paper) | scalar rows | no | 80 / 20 / 0 % | dense |
   | `complex_rows` | CSM entries, diagonal stratified | yes | 80 / 20 / 0 % | `CMFDictionary` |
   | `mics` | microphones, val block only | yes | 64 / 4 / 32 % | CMF |
   | `mics_cross` | microphones, val incl. cross terms | yes | 64 / 36 / 0 % | CMF |

   For the microphone splits, `train_ratio` refers to microphones. On a row subset `P`
   (`rows`, `complex_rows`), the dense weights use `B[rows, rows]`; the reduced models
   (R, SWR) use the same principal submatrix of their congruence,
   `P hvec(Br hvec*(P^T v) Br^H)`: zero-fill the other rows, apply the congruence, keep the
   train rows. `CMFDictionary.select_rows` therefore keeps its CMF coordinates (`hvec_rows`).
5. Training compiles by default (`compile=True`): `compile_unrolled(model, stages)`
   computes all `stages` iterations once per batch; stage `t` enters the graph as a tensor,
   and the iterations after `t` keep `x_t` (`active_iterations`), so one graph serves all
   stages without overflowing into the gradient. Batches are full (`drop_last`), so the
   shapes stay fixed.

## Consequences

- New objectives are new functions in `OBJECTIVES`; parameters go to `loss_kwargs`.
- Early stages pay for all `stages` iterations when compiled.
- `rows`/`complex_rows` form dense per-problem train matrices `(B, m_t, n)`.
- `mics` with `r_diag` needs at least two validation microphones.
- The CV `val_loss` varies with each random split.
