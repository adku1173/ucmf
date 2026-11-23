# 0004: Single-weight Ada-LFISTA as a subclass with its own forward

Status: Accepted — requested by the author.

## Context

FA2026 combined an `AdaptiveGradientWrapper` (trainable `B`, `W = B @ A`) with
a learned FISTA (`ALFISTA`, model name `ADA-ALFISTA`, paper name
`Ada-LFISTA-SW`). Since [0002](0002-classical-forward-without-weights.md), the
classical forward accepts no W; weighted models own their forward.

## Decision

`AdaLFISTASW(FISTA)` owns `forward(y, x0=None, t=1, A=None, msm_subset=None)`
instead of wrapping a solver. It learns:

- `B` (`m x m`, identity init): effective adjoint `W = B @ A`, forward stays `A`.
  `B = I` reproduces FISTA.
- `_gamma`, `_theta` (unconstrained, per iteration). The applied values are
  the properties `gamma = softplus(_gamma)`, `theta = softplus(_theta)`;
  softplus is smooth, injective and strictly positive, unlike `abs`.

`msm_subset` names the original rows `I` of an already reduced measurement set
(cross-validation, dictionary reduction): `W_sub = B[I, I] @ A_sub`, shared
`(m_sub,)` or per sample `(B, m_sub)`. Only dtype, device and shape are checked.
Index range is a precondition guaranteed by the index generator: a value check
forces a GPU sync per forward (7–28 % per training step at exp000 sizes).

`W = B @ A` is formed once per solve. The factored form `A.T (B.T r)` per
iteration saves FLOPs only for small batches; at exp000 sizes (m = n = b = 256,
RTX 3090) it was 1.0–1.6x slower, as per-iteration matrix-vector products are
memory bound. The auxiliary-weight helpers live in `weights.py`;
`validate_unfolded_iterations` and `softplus_inverse` in `utils.py`.

`compile_unrolled(model, t_max)` compiles one graph for `t_max` iterations and
returns all iterates `(t_max, B, n)`; stage `t` uses `out[t - 1]`. Compiling the
model per `t` would recompile for every training stage (5–60 s each).

## Consequences

- Iterations after `t` are computed but do not affect `x_t` (zero gradient);
  non-finite values there could still produce NaN gradients.
- Model callbacks must be compilable (no `.item()` host syncs).
- Measured at exp000 sizes, T = 60, CUDA graphs: the step time is independent
  of `t` (4.9 ms shared A, 18 ms per-sample CV vs. 3–59 ms eager), i.e. up to
  11x (shared A) and 3.2x (CV) faster at `t = 60`, but slower than eager for
  small `t` with CV. Estimated over all stages: ~5x (shared A), ~1.8x (CV).
  One-off compilation ~1 min; the inductor cache reuses compiled graphs
  across processes (measured for a full-model compile: 8.2 s -> 2.0 s).
- No automatic step-size guarantee holds for learned `gamma` and `W != A`.
