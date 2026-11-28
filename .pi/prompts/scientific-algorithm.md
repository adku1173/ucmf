---
description: Adapt and implement a scientific algorithm in this repo via a gated multi-agent workflow
argument-hint: "<algorithm> <target problem>"
---

Task: Adapt and implement ${1:-<algorithm>} for ${2:-<target problem>} in this repository.

Use the specialized subagents (see AGENTS.md "Multi-Agent Orchestration"; model families are preset in settings.json: research=Mistral, math/review=Claude, coding/independent math review=Codex). Use topic directory `.pi/research/${1:-topic}/` for all inter-agent artifacts.

## Phase 1: Parallel investigation

Launch three independent subagents in parallel (async), none of them modifying source code:

1. `literature-researcher` (or `scientific-researcher`): analyze the original paper(s) and directly related work. Full papers are local under `~/mendeley_papers/<Lastname>/<Lastname|year>.pdf`; Zotero MCP is available. Extract exact equation/algorithm/lemma/theorem numbers, assumptions, convergence results, complexity. Write `literature.md`.
2. `mathematician`: independently derive the adaptation to the target problem — proximal operator, KKT/optimality residual, variable selection/elimination/reactivation if applicable, inner solver, backtracking, stopping criteria, convergence transfer (label each statement Unchanged / Adapted / Conditional / Unproven / Invalid). Write `derivation.md`.
3. `scout`: analyze the existing codebase — solvers, interfaces, objective scaling conventions, tensor/batch dimensions, device handling, tests, reusable components, integration points. Write `repository_analysis.md`.

## Phase 2: Mathematical specification + Gate 1

Synthesize the three reports into `.pi/research/${1:-topic}/algorithm_specification.md`: problem formulation, notation and scaling, optimality conditions, KKT residual, reduced problem, inner solver, outer iteration, stopping criteria, complete pseudocode, convergence analysis, computational complexity, differences from the original algorithm, unresolved theoretical questions. Explicitly distinguish proven results from assumptions and conjectures.

Delegate an independent review to `mathematical-reviewer` (different model family). Resolve all critical and major findings.

**Gate 1:** implementation may only start once the specification is mathematically consistent and no unresolved implementation-relevant errors remain. Stop and request human input if fundamental theoretical uncertainties cannot be resolved.

## Phase 3: Implementation

Delegate to `scientific-coder` (the only writer in the working tree). Follow existing repo conventions, reuse existing components, do not silently deviate from the specification; document deviations and trade-offs in `implementation_notes.md`. Write unit and regression tests.

## Phase 4: Validation + Gate 2

Independent review by `scientific-reviewer` plus numerical validation: analytical cases (e.g. A=I), independent reference solver (e.g. CVXPY) or existing implementation, edge cases (lambda=0, lambda>=lambda_max, rank-deficient, high coherence), multiple seeds. Metrics: objective gap F(x)-F(x_ref) and KKT residual norm (not only reconstruction error — solutions may be non-unique), plus runtime, memory, working-set sizes, number of matrix-vector products. Distinguish correctness tests from performance benchmarks. Write `validation.md`.

**Gate 2:** the algorithm must pass the numerical reference tests before any performance claims are made. Resolve critical findings and rerun affected tests.

## Deliverables

- `.pi/research/${1:-topic}/literature.md`, `derivation.md`, `repository_analysis.md`
- `.pi/research/${1:-topic}/algorithm_specification.md` (reviewed)
- `.pi/research/${1:-topic}/mathematical_review.md` (Gate 1 verdict)
- implementation + tests in the repo
- `.pi/research/${1:-topic}/implementation_notes.md`
- `.pi/research/${1:-topic}/validation.md` (Gate 2 verdict)
- documented remaining theoretical limitations

Do not claim theoretical guarantees that have not been established. Only one agent may modify the working tree at a time; use git worktrees for parallel implementations.
