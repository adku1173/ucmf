---
description: Adapt and implement IR-FISTA-B (Wang et al. 2022) for the nonnegative LASSO in ucmf via a gated multi-agent workflow
---

Task: Adapt and implement IR-FISTA-B for the nonnegative LASSO in the ucmf repository.

Objective: Investigate, mathematically adapt, implement, and validate the IR-FISTA-B algorithm described by Wang et al., "An Iterative Reduction FISTA Algorithm for Large-Scale LASSO", SIAM J. Sci. Comput. 44(4), A1989–A2017 (2022), DOI: 10.1137/20M1374328, for the nonnegative LASSO problem

    min_{x >= 0}  ||Ax - y||^2 / (2m) + lambda * ||x||_1

in the ucmf repository (batch dimension B, solutions of shape (B, n); solvers in `src/ucmf/optim/`, tests in `tests/`). A draft mathematical specification with paper deviations already exists at `docs/ir_fista.md` — cross-check it against the original paper, do not trust it blindly. Use `.pi/research/ir-fista-b/` for all inter-agent artifacts. Use the specialized subagents; model families are preset in settings.json.

## Phase 1: Parallel investigation

Launch three independent subagents in parallel (async), none of them modifying source code:

1. `literature-researcher`: Original paper (local PDF `~/mendeley_papers/Wang/Wang2022.pdf`, also possibly `~/Desktop/ir-fista.pdf`; Zotero MCP available; more papers under `~/mendeley_papers/<Lastname>/<Lastname|year>.pdf`) plus related work (FISTA-B, working-set methods for LASSO, nonnegative LASSO / NNLS / active-set solvers). Extract the algorithm anatomy with exact equation/algorithm/lemma/theorem numbers: objective and scaling, proximal step, KKT/optimality residual, working-set selection, variable elimination and reactivation, inner solver, backtracking, stopping criteria, convergence theorems, complexity. Cross-check `docs/ir_fista.md` (IDs A/D/E). Write `.pi/research/ir-fista-b/literature.md`.
2. `mathematician`: Independently derive the adaptation to the nonnegative LASSO: proximal operator `S^+_{gamma*lambda}(z) = max(z - gamma*lambda, 0)`, one-sided KKT conditions (`x_i>0 => g_i = -lambda`; `x_i=0 => g_i >= -lambda` with `g = A^T(Ax-y)/m`) and a suitable KKT residual, working-set selection/elimination/reactivation, inner FISTA-B solver, momentum, backtracking validity, global stopping criteria, scaling conversion (`lambda_p = m*lambda`, `delta_p^k = m*delta^k`, `L_p = m*L`), which convergence results transfer (label Unchanged/Adapted/Conditional/Unproven/Invalid), complexity. Write `.pi/research/ir-fista-b/derivation.md`.
3. `scout`: Analyze the existing ucmf implementations (`src/ucmf/optim/ista.py` with `ISTA`/`FISTA` and the `positive=True` shrinkage, `ada_lfista*.py`, `operator.py` `MatrixMult`, `callbacks.py`, `preprocessing.py`), objective scaling `||Ax-y||^2/(2m)`, interface `solver(y, operator=..., lam=..., t=...)`, tensor shapes (B, n), device/dtype, existing tests (`tests/test_ista.py`, ...), and integration points for a new solver. Write `.pi/research/ir-fista-b/repository_analysis.md`.

## Phase 2: Mathematical specification + Gate 1

Synthesize the three reports into `.pi/research/ir-fista-b/algorithm_specification.md`: problem formulation, notation/scaling, optimality conditions, KKT residual, working-set selection, reduced problem, inner FISTA-B solver, outer iteration, stopping criteria, complete pseudocode, convergence analysis, complexity, differences from the original paper, unresolved theoretical questions. Distinguish proven results from assumptions/conjectures.

Delegate an independent review to `mathematical-reviewer` (Codex, different model family than `mathematician`). It classifies each statement Unchanged/Adapted/Conditional/Unproven/Invalid and findings as Critical/Major/Minor/Suggestion. Resolve all critical and major findings. Write `.pi/research/ir-fista-b/mathematical_review.md`.

**Gate 1:** Only when the specification is mathematically consistent and no unresolved implementation-relevant errors remain may implementation start. Missing theoretical guarantees must be documented explicitly. Stop and request human input if fundamental theoretical uncertainties cannot be resolved.

## Phase 3: Implementation

Delegate to `scientific-coder` (the only writer in this working tree):

- Reuse existing ISTA/FISTA components (`shrink_positive`, `MatrixMult`, callbacks, stoppers) where possible; keep the `solver(y, operator=..., lam=..., t=...)` interface so the new solver is a drop-in next to `FISTA`.
- Support `positive=True` (nonnegative LASSO) and `positive=False` if the spec covers both; keep batch dim B, device/dtype handling consistent.
- Consistent scaling (`1/(2m)` objective, `lam`); no materialized Gram matrices; matrix-vector products via the operator; GPU-aware.
- Implement working-set logic exactly as specified: selection, reduced subproblem, inner FISTA-B, elimination/reactivation, backtracking, global stopping on the KKT residual.
- Tests in `tests/test_ir_fista_b.py`: `A=I` vs analytical solution, `lam=0` vs NNLS, `lam >= lam_max` => zero solution, small random problems vs CVXPY or a long reference FISTA run, high-coherence dictionaries, rank-deficient A, different sparsity levels and initializations.
- Run `uv run pytest` and `uv run ruff check .`.
- Document deviations/trade-offs in `.pi/research/ir-fista-b/implementation_notes.md`. Do not silently deviate from the specification.

## Phase 4: Validation + Gate 2

Independent review by `scientific-reviewer` plus numerical validation:

- Correctness tests (above) must pass; compare objective values `F(x) - F(x_ref)` and `||r_KKT||_inf` — not only NMSE of reconstructed coefficients (solutions may be non-unique).
- Benchmarks vs the existing `FISTA`/`ISTA`: runtime, memory, working-set sizes, number of matrix-vector products, large dictionaries, realistic CMF problems.
- Distinguish correctness tests from performance benchmarks. Write `.pi/research/ir-fista-b/validation.md`.

**Gate 2:** numerical reference tests must pass before any performance claims. Resolve critical findings and rerun affected tests.

## Deliverables

- `.pi/research/ir-fista-b/literature.md`
- `.pi/research/ir-fista-b/derivation.md`
- `.pi/research/ir-fista-b/repository_analysis.md`
- `.pi/research/ir-fista-b/algorithm_specification.md` (reviewed)
- `.pi/research/ir-fista-b/mathematical_review.md` (Gate 1)
- IR-FISTA-B implementation for nonnegative LASSO + unit/regression tests
- `.pi/research/ir-fista-b/implementation_notes.md`
- `.pi/research/ir-fista-b/validation.md` (Gate 2)
- Documented remaining theoretical limitations

Do not claim theoretical guarantees that have not been established. Only one agent may modify the working tree at a time; use git worktrees for parallel implementations.
