# Research artifacts: IR-FISTA-B → nonnegative LASSO (ucmf)

This directory is the persistent interface between the subagents of the IR-FISTA-B workflow
(see `.pi/prompts/ir-fista-b.md` and the generic template `.pi/prompts/scientific-algorithm.md`).
Agents write their reports here; the orchestrator reads them in later phases. The files survive
session restarts and document why algorithmic decisions were made.

Expected files:

- `literature.md` — `literature-researcher`: algorithm anatomy of Wang et al. 2022
  (DOI 10.1137/20M1374328, local PDF `~/mendeley_papers/Wang/Wang2022.pdf`) with exact
  equation/algorithm/lemma/theorem numbers, related work, and a cross-check of `docs/ir_fista.md`
  (IDs A/D/E).
- `derivation.md` — `mathematician`: adaptation to `min_{x>=0} ||Ax-y||^2/(2m) + lam*||x||_1`
  (prox `max(z-gamma*lam, 0)`, one-sided KKT conditions and residual, working-set rules,
  convergence transfer labeled Unchanged/Adapted/Conditional/Unproven/Invalid).
- `repository_analysis.md` — `scout`: ucmf solver landscape (`src/ucmf/optim/`), interfaces,
  scaling conventions, tensor shapes, tests, integration points.
- `algorithm_specification.md` — orchestrator: synthesized, reviewed specification + pseudocode.
- `mathematical_review.md` — `mathematical-reviewer`: Gate 1 verdict (Critical/Major/Minor/Suggestion).
- `implementation_notes.md` — `scientific-coder`: deviations and computational trade-offs.
- `validation.md` — `scientific-reviewer`: Gate 2 verdict, numerical validation and benchmarks
  (objective gap, KKT residual, runtime/memory/working-set sizes — not only NMSE).
