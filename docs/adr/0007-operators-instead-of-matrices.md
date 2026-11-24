# 0007: Linear operators instead of sensing matrices

Status: Accepted — requested by the author; implemented.

## Context

All solvers take a dense sensing matrix `A`. For CMF, column `j` of `A` is
`hvec(h_j h_j^H)` with the propagation matrix `H` (`c x n`, complex), so `A`
can stay implicit: `A x = hvec(H diag(x) H^H)`,
`A^T r = Re(diag(H^H hvec*(r) H))`. For Ada-LFISTA-SWR, `W = B A` uses
congruence in the active measurement coordinates. With the CSM diagonal
included, `W` is exactly the dictionary of `G = Br H`. A microphone subset
`M1` restricts the congruence to `Br[M1, M1]`.

Measured (RTX 3090, float32, t = 15 training steps fwd+bwd unless noted;
`b` = number of problems, "distinct A" = one operator per problem):

| Case | dense A | propagation (H only) |
| --- | --- | --- |
| b = 1, c = 64, n = 1681, 1000 it., eager | 118 ms | 689 ms |
| b = 1, same, compiled (CUDA graphs) | 77 ms | **46 ms** |
| b = 256, shared A, c = 16 (exp000): FISTA / SW / LFISTA, eager | **1.9 / 7.2 / 15 ms** | 10 / 40 / 58 ms |
| b = 256, distinct A, c = 16: FISTA / SW / LFISTA, eager | **2.7 / 12 / 20 ms** | 11 / 46 / 60 ms |
| b = 32, distinct A, c = 64: FISTA / SW / LFISTA / SWR, eager | 30 / 286 / 524 / 253 ms | **15 / 40 / 49 / 47 ms** |
| b = 256, mic-CV, c = 64: FISTA | 155 ms, 4.5 GB | **55 ms** |
| b = 256, mic-CV, c = 64: SWR | out of memory | **169 ms** |
| b = 256, distinct A, c = 64 | A alone ~7 GB | feasible |

SWR with `W` from `G = Br H` ("dense-G") is never slower than the current
column transform and up to 1.7x faster with 1/3 of the memory (mic-CV).
The `G` shortcut does not implement the existing zero-diagonal SWR update
when `r_diag=True`.
Neither representation wins everywhere: dense wins for small CMF problems and
shared A in eager mode; matrix-free propagation wins compiled at b = 1, for distinct A with
many microphones, and for per-sample microphone CV.

## Decision

1. Solvers and models take an operator instead of `A`:
   `forward(y, x0=None, t=1, operator=None, ...)`. Explicit matrices are
   wrapped once by the caller; no tensor fallback.
2. `LinearOperator` (shared or per-problem batch): `forward(x)` `(B, n) -> (B, m)`,
   `adjoint(r)` `(B, m) -> (B, n)`, `shape`, `dtype`, `device`, `batch_size`
   (`None` = shared).
3. `MatrixMult(A)`: dense matrix, exposes `matrix`; `select_rows(I)`.
4. CMF operators share `H`, `hvec`, `r_diag` and two derived operators:
   `select_mics(M1)` (shared `(c1,)` or per problem `(B, c1)`) and
   `transformed(Br)` (`G = Br H`). Representations: `CMFDictionary` (builds the
   dense dictionary, also a `MatrixMult`) and `CMFPropagation` (H only).
   All live in `operator.py` next to `HermitianVectorization`.
5. Subsets belong to the operator; the forward argument `msm_subset` is
   removed. Every operator knows its measurement rows `rows` relative to the
   full set (`None` = all), CMF operators also `mics`; models restrict their
   auxiliary weights with them (`B[rows, rows]`, `Br[mics, mics]`). Both
   measurement subsets (rows / microphones, e.g. CV) and source subsets
   (`select_columns(J)`, dictionary reduction) are supported.
6. Default step size: power iteration on `operator.forward` / `adjoint`.
7. Cross-validation selects microphones (`select_mics`), not rows of `A`;
   real and imaginary parts and all entries of a microphone pair stay together.
   A time-block split would be a different objective and is out of scope.
8. Models express their update through the operator: SW `A^T (B^T r)`,
   LFISTA `A^T B1^T B1 A z`, SWR `A^T (B^T r)` with `B` the congruence
   in the active CMF coordinates. SWR preserves the existing rule:
   `B v = hvec(Br hvec*(v) Br^H)`. When `r_diag=True`, `hvec*` zero-fills
   the diagonal before congruence and `hvec` drops it afterwards. Transforming
   `H` with `Br` and dropping the diagonal only afterwards is not equivalent.

## Resolved and remaining points

- SW/LFISTA support both `weight_mode='factored'` (default) and `'precomputed'`
  (dense operators only); timings are in the README benchmark.
- Callback keyword `W` (and `W_recurrent`) is the effective operator, not a matrix.
- Open: reduced models (SWR/R) accept only CMF operators, so arbitrary row subsets are unsupported.
- Open: `ucmf.acoular` wraps Acoular's dense dictionary; matrix-free propagation needs the Acoular PR.
