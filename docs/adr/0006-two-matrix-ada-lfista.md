# 0006: Two-matrix Ada-LFISTA with the paper's weight numbering

Status: Accepted — requested by the author.

## Decision

`AdaLFISTA(FISTA)` implements Algorithm 1 of Aberdam et al. like
[0004](0004-single-weight-ada-lfista.md) (own forward, learned `gamma`/`theta`,
`msm_subset`, identity init), with two auxiliary weights numbered as `W1`/`W2`:

```text
R = B1 @ A   (recurrence, W1)     M = B2 @ A   (measurement branch, W2)
x_next = shrink(z + gamma_k * (M.T @ y - R.T @ (R @ z)) / m, theta_k)
```

FA2026 swapped the numbering (`B1` was the measurement branch). `M.T @ y` is
computed once per solve, so an iteration needs two matrix-vector products like
Ada-LFISTA-SW (FA2026: three). With `msm_subset`, each factor is restricted
first (`B1[I, I]`, `B2[I, I]`); `R.T @ R` is never formed. End callbacks get the
physical `A`, `W = M` and `W_recurrent = R`.

## Consequences

- `B1 = B2 = I` reproduces FISTA. Weights taken over from FA2026 must be
  swapped: its `B1` / `measurement_weight` is `B2` here, its `B2` /
  `recurrence_weight` is `B1`.
- Properties shared by all unfolded models are tested once, parametrized over
  every model (`unfolded_model_type` fixture); model files test specifics only.
