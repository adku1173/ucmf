# 0009: Column normalization outside the solvers

Status: Accepted — requested by the author; implemented.

## Context

Ada-LISTA-type models assume a dictionary with unit column norms (FA2026,
Sec. LASSO): `D = A U^-1` with `U = diag(||a_1||_2, ..., ||a_n||_2)`. The
solvers then estimate the normalized solution `x = U q`; source strengths are
`q = U^-1 x`. Cross-validation normalizes both subsets with the norms of the
training subset (`D_v = A_v U_t^-1`), so `D_v` is not unit-normalized.
The supervised MSE is defined on `q`, while LASSO, CV, and BIC use `x` and `D`.

Three places for the back-transformation were considered: inside the solver,
as an attribute `U` on the operator, or in a separate scaling layer. A solver
that normalizes itself cannot apply foreign norms (CV) and would hide `D` from
the losses. Storing `U` on every operator adds state that does not fit future
CMF-C operators, whose unknowns are covariance entries, and the later data
scaling (`lambda_max`, per measurement) would still live elsewhere.

## Decision

1. Operators provide two capabilities on the concrete classes (`MatrixMult`,
   `CMFDictionary`, `CMFPropagation`), not on `LinearOperator`:
   `column_norms()` (`(n,)` or `(B, n)`) and `scale_columns(scale)`, which returns
   `A diag(scale)` of the same type with unchanged `rows` and `mics`. A per-problem
   scale makes a shared operator per problem. `scale > 0` is a precondition.
2. CMF operators work matrix-free: `||hvec(h_j h_j^H)||_2 = ||h_j||_2^2`
   (`hvec` is an isometry), with `r_diag` `sqrt(||h_j||^4 - sum_i |h_ij|^4)`;
   `scale_columns` scales `H` by `sqrt(scale)`.
3. Column scaling commutes with `select_rows`, `select_mics`, and `transformed`.
   CV computes `U_t` on the training subset, scales the full operator, and then
   selects the subsets.
4. `ucmf.scaling` (functions, not exported from `ucmf`) holds the problem layer:
   `normalize_operator(A) -> (D, u)` and `to_source(x, u) = x / u`.
5. Solvers only see `D` and return `x`. Ada models require unit column norms as a
   documented, unchecked precondition; ISTA/FISTA stay general.
6. Training always normalizes the configured operator; `mse`/`nmse` compare
   `to_source(x_hat, u)` with the targets `q_train`/`q_val`.

## Consequences

- Users of trained models call `normalize_operator` and `to_source` themselves.
- The NMSE on `q` is invariant to a global scale of `A`, not to per-column scaling.
- Acoular keeps its own `dictionary_scaling`; the capabilities are ucmf-only for now.
- Open: data scaling by `lambda_max` (equiregularization) extends `ucmf.scaling`.
