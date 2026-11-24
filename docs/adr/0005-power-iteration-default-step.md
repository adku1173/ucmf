# 0005: Default step size from power iteration with a safety factor

Status: Accepted — requested by the author. `default_gamma_from_A` and
`estimate_squared_spectral_norm` were later merged into `default_gamma_from_operator`
(same algorithm, applied through `operator.forward` / `adjoint`).

## Context

The default step `gamma = m / ||A||_2**2` used the largest eigenvalue of the
smaller Gram matrix (`eigvalsh`). For the CMF example dictionary (4032 x 1681)
this cost 24 ms (CUDA float32) to 212 ms (CPU float64) per frequency bin,
i.e. 13–60 % of a 1000-iteration solve.

Reference libraries: PyLops uses ARPACK (`eigsh`) with NumPy and 10 power
iterations with CuPy; CIL uses 10 power iterations (`ISTA`: `0.99 * 2 / L`,
`FISTA`: `1 / L`); PyProximal needs a given step or uses backtracking. None
applies a safety factor for FISTA, and all start from a random vector.

## Decision

`default_gamma_from_A` returns `1 / L` with
`L = 1.05 * estimate_squared_spectral_norm(A) / m`. The estimate runs 50 power
iterations on `A.T @ A` (shared or batched A) and returns the Rayleigh quotient.
The random start vector comes from a dedicated CPU generator with a fixed
seed: deterministic, without consuming the global RNG. A fixed iteration
count avoids per-iteration host syncs; a tolerance would not stop earlier for
close singular values anyway.

## Consequences

- The Rayleigh quotient is a lower bound. For the CMF dictionary
  (sigma_2 / sigma_1 = 0.99), the implementation underestimates by 0.4–0.9 %;
  10 iterations reached up to 12 % (start vector of ones). For generic random
  square matrices, 0.5–1 % of cases underestimate by more than 5 % (max 7.6 %).
- FISTA on the CMF problem tolerated `gamma * L <= 1.12` without change and
  diverged (signed) at 1.5. The factor 1.05 is a practical margin, not a proof.
- Cost per bin: 3.9 ms (CUDA float32), 8.3 ms (CUDA float64), 44 ms (CPU
  float64), i.e. 5–14x cheaper than `eigvalsh`.
- Default-step results change slightly. Tests take the ucmf default step as
  NumPy reference input; tests needing one exact step pass `gamma` explicitly.
  The PyLops benchmark passes the same step (`alpha = gamma / m`).
