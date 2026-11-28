# Safeguarded IR-FISTA-B for normalized nonnegative LASSO

Status: reviewed specification v2; **Gate 1 PASSED** by independent mathematical-reviewer (see mathematical_review.md §7). Changes address review M1 and N1–N4; no algorithmic scope expansion. Source changes prohibited until independent review resolves all implementation-relevant Critical/Major findings.

## 1. Evidence, scope, and identity

Synthesis of `literature.md`, `derivation.md`, and `repository_analysis.md`, repository baseline `ab87210`. Primary source: Wang et al., SISC 44(4), A1989–A2017 (2022), DOI 10.1137/20M1374328. Original PDFs independently checked by literature investigator. `docs/ir_fista.md` is preserved as an historical, untrusted draft, not the implementation contract.

This is a **safeguarded adaptation**, not a literal reproduction of Algorithm 5. It retains iterative working-set reduction, support retention, proximal reduced subproblems, backtracking FISTA, and global reactivation. It deliberately replaces printed formulas that are inconsistent or require expensive spectral constants. Nonnegativity is a new extension; signed mode is also specified. No published nonnegative IR-FISTA-B theorem or speed guarantee is asserted.

Alternatives considered: (a) literal Algorithm 5: rejected because (2.19) is inconsistent with (2.15) and (2.22) can be unreachable; (b) corrected spectral decrease threshold: needs eigenvalues/condition numbers and additional proof; (c) first-step descent safeguard: selected because it avoids Gram matrices and has a short independent residual-convergence argument. A capped working set and safe permanent screening are out of scope.

## 2. Problem, notation, scaling [Adapted]

For each independent batch row, A maps R^n to R^m, m>0, lambda>=0:

F_C(x) = f(x) + h_C(x), f(x)=||Ax-y||²/(2m), h_C(x)=lambda||x||_1+I_C(x).

C=R_+^n for positive mode, R^n otherwise. g(x)=A^T(Ax-y)/m; L_f=||A||²/m. Output always (B,n); no column normalization or change of measurement denominator when selecting sources.

Paper units: F_p=m F, lambda_p=m lambda, delta_p=m delta, L_p=m L, sigma_p=m sigma. Steps gamma_p=gamma/m give identical prox thresholds gamma_p lambda_p=gamma lambda. Paper residual (2.4) and E1's min(x,g+lambda) mix coefficient and gradient units and do not admit a universal paper-tolerance conversion. All implementation tolerances are newly defined in normalized units below.

On C=R_+^n, h_C=lambda 1^T x+I_C. Prox_{gamma h_C}(v)=max(v-gamma lambda,0). On R^n it is signed soft-thresholding. Reuse ISTA.shrink_nonnegative (the actual repository name, not shrink_positive) and ISTA.shrink.

F>=0. Minimizers exist for lambda>0 by coercivity; for lambda=0 both LS and NNLS attain their minima (polyhedral constrained quadratic). Coefficients need not be unique; compare objectives and KKT, not coefficient NMSE alone.

## 3. KKT and global residual [Adapted]

Positive: s=g+lambda 1. KKT: x>=0, s>=0, x_j s_j=0. Equivalently x_j>0 => g_j=-lambda, x_j=0 => g_j>=-lambda.

Signed: g_j+lambda sign(x_j)=0 for x_j!=0; |g_j|<=lambda for x_j=0.

Define the fixed-unit natural KKT residual (implicit reference step gamma_ref=1 in normalized units):

R(x)=x-prox_{h_C}(x-g(x)).

Positive: R_j=min(x_j,s_j). Evaluate directly as min(x,s), avoiding cancellation of two large x values. Signed: **must** evaluate the cancellation-free identity R_j=min(max(x_j,g_j-lambda),g_j+lambda). Literal x-S_lambda(x-g) subtraction is prohibited for certification and inner stopping: at x=2^60,g=0,lambda=1 in float64 it falsely computes zero instead of one. In the subproblem substitute grad q for g. This equals paper (2.4) in magnitude in the same normalized units. Global certificate r_inf=||R||_inf. R is continuous, and R=0 iff KKT. It is a stationarity measure, **not an objective-gap or coefficient-error certificate**. A small positive x with large positive s can have small natural residual; document this, and also report the raw subgradient residual for numerical validation:

positive e_j=|s_j| if x_j>0 else max(-s_j,0);
signed e_j=|g_j+lambda sign(x_j)| if x_j!=0 else max(|g_j|-lambda,0).

Raw residual e is discontinuous at zero and is diagnostic, not required for the convergence status. Validate both natural and raw infinity norms against references where practical. No support epsilon is used to redefine x_j=0.

Stop a row iff r_inf <= tol + rtol*r_inf_initial, with fixed initial scale per solve; default tol=1e-6, rtol=0. Recompute full Ax-y and g on every accepted outer row update and before final status. Report the residual of the **returned** point. No reduced-only certificate.

Zero solution: positive lambda>=max_j(A^T y)_j/m, signed lambda>=||A^T y||_inf/m. Negative positive-mode lambda_max implies zero optimal for every allowed lambda. No separate potentially inconsistent branch is needed: initial full KKT test handles it, including A=0.

## 4. Working-set selection, elimination/reactivation [Adapted]

Given full x,g, tau in [0,1):

positive W=supp(x) union {j:x_j=0, g_j<-(1-tau)lambda};
signed W=supp(x) union {j:x_j=0, |g_j|>(1-tau)lambda}.

Support means x_j>0 in feasible positive mode, x_j!=0 signed. Use exact comparisons; never drop tiny nonzeros. All support and all zero-coordinate KKT violators belong to W. Outside W, x=0 and the natural residual is exactly zero. No top-K cap, permanent screening, or irreversible elimination. Recompute W from the full gradient after each accepted update; excluded coordinates can re-enter. Coordinate order is ascending global index order; preserve mapping explicitly. Empty W implies global KKT, so do not construct an empty reduced operator.

lambda=0: positive W=supp union {g<0}; signed W=supp union {g!=0}. This may be nearly full-width; no sparsity claim.

## 5. Reduced problem and curvature [Adapted]

At outer row iterate x, W, a=x_W, define

q(z)=||A_W z-y||²/(2m)+(delta/2)||z-a||²,
Phi(z)=q(z)+lambda||z||_1+I_{C_W}(z).

Since supp(x) subset W, A_W a=Ax and Phi(a)=F(x). Gradient grad q(v)=A_W^T(A_W v-y)/m+delta(v-a); Hessian H=A_W^T A_W/m+delta I (notation only: **never materialize H or Gram matrices**). delta>=0 fixed per problem during a solve. Public delta=None uses delta=1e-3/m, matching the paper's numerical proximal weight in paper units; an explicit delta is in normalized units. delta=0 is allowed.

New full point x_new has x_new_W=z_out, other entries zero. Then
F(x_new)=Phi(z_out)-(delta/2)||z_out-a||².

The default is not the paper's full-spectrum policy delta^k=max(sigma-lambda_min(A^T A),0). That policy requires costly spectral information; fixed positive delta ensures reduced strong convexity but not full outer iterate convergence for the implemented inexact solves.

## 6. Reduced FISTA-B and backtracking [Unchanged composite structure; Adapted formulas]

Reset momentum every outer row solve: z_prev=a, v=a, theta=1. For an inner step, with d_q=grad q(v):

z=prox_{(lambda/L)||.||_1+I_C}(v-d_q/L),
D=z-v.

Quadratic backtracking test:

||A_W D||²/m+delta||D||² <= L||D||².

This is exactly smooth majorization q(z)<=q(v)+<grad q(v),D>+(L/2)||D||². It correctly includes delta(v-a) in the gradient. The nonsmooth terms cancel. v can be infeasible; the prox always restores feasibility. Use eta>1 (default 2); multiply L by eta on rejection. Within a reduced run L never decreases. Across outer runs start at max(L_floor,L_last/eta); initial L is max(L_floor,1/gamma) if supplied, otherwise max(L_floor,1). Validate that the initial curvature is positive, finite, and representable in the solve dtype. Support float32/float64 for the numerical solver; reject low-precision floating dtypes rather than silently underflow energy tests. gamma is an initial line-search guess, not a promise of fixed steps. L_floor default 1e-12. Accepted L values are bounded in exact arithmetic by max(initial L, eta*(L_f+delta),L_floor); an initially oversized L may exceed eta*(L_f+delta), unlike the paper's claimed unconditional upper bound.

For numerical stability use the quadratic test rather than subtraction of large objective values. A relative roundoff allowance at most 32*finfo.eps times the maximum absolute left/right energy is allowed, **no additive O(eps)*max(1,energy) slack** which could accept an arbitrarily tiny L. Finite max_backtracking (default 60) and finite-value checks; exhaustion raises RuntimeError, never reports convergence. Finite arithmetic safeguards do not constitute an exact-arithmetic theorem.

After accepted prox: theta_new=(1+sqrt(1+4 theta²))/2,
v_new=z+((theta-1)/theta_new)*(z-z_prev). Use existing FISTA momentum helper where possible. Momentum and L are per row, never coupled to batch neighbors.

### Inner stopping and first-step safeguard

After each accepted step evaluate Phi(z), and track the lowest Phi among ALL accepted feasible candidates (including the first, not the anchor). Return that best candidate, retaining an earlier candidate on ties. This guarantees Phi(z_out)<=Phi(z_first) in exact arithmetic. Do not return a later accelerated candidate with a higher Phi merely because its residual passes.

For a practical early exit compute the natural residual of the **proximal subproblem** with grad q(z), using lambda and C as in section 3. Exit when its infinity norm <= inner_tol*r_inf_outer, default inner_tol=0.1 in (0,1), or after inner_max_iter (default 100, a strictly positive integer). max_backtracking is also a strictly positive integer trial budget. Track whether the best candidate itself met this target; do not mislabel inner-budget output as an exact reduced solution. The relative rule is a work-saving heuristic; neither it nor a fixed budget implies summable proximal errors. A finite budget still returns the safeguarded point, with inner-budget diagnostic. The descent/residual argument needs only the first-step safeguard, not inner target attainment, eigenvalues, condition numbers, or Theorem 3.5.

## 7. Batch, operators, callbacks, interface [Project-specific Adapted]

New `src/ucmf/optim/ir_fista_b.py`, public `IRFISTAB`, exported through optim and top-level ucmf. Subclass FISTA or ISTA for shared preparation/prox/callback/momentum helpers. Existing solver signatures/behavior unchanged.

Proposed constructor: positive=True, callbacks=None, keyword nonnegative=None compatibility alias, tol=1e-6, rtol=0, tau=1e-4, delta=None, inner_max_iter=100, inner_tol=0.1, backtracking_factor=2, max_backtracking=60, min_lipschitz=1e-12. If nonnegative is supplied, it sets the effective positivity flag; document alias priority. Forward: (y,x0=None,t=100,operator=None,lam=None,gamma=None). t counts maximum OUTER sweeps (one reduced solve per unconverged row), not inner prox steps. Explain this difference prominently. Validate t>=0 integer, all tolerances/curvatures finite and appropriate signs/ranges, finite y/x0, lambda>=0, gamma>0. Existing strict dtype/device/batch validation reused. Positive warm starts are projected with clamp_min(0) and this projection is documented; signed warm starts unchanged. t=0 returns this prepared point plus its full residual/status; not falsely converged unless KKT criterion passes.

Each batch row has independent W, L, delta, initial tolerance scale, budget and convergence status. Iterate row solves sequentially inside each outer sweep (GPU math still on input device, but host-driven branching incurs synchronization). Do not promise torch.compile or graph-stable unrolling. Algorithm is an eager optimization solver, not a learned adaptive solver. No automatic no_grad requirement: preserve repository autograd conventions where practical, but no differentiability promise across discrete selection/stopping boundaries.

Operator adaptation remains solver-local to avoid altering the base operator API:

- Shared operator batch_size=None: apply directly to the one row; use native select_columns(W) when present.
- Batched operator: a one-problem wrapper injects the row into a zero (B,n) tensor, calls the original operator, and extracts that output row; similarly for adjoint. For restriction, wrap operator.select_columns(W) if native selection exists (same W for all original rows); this is correct but may compute unused batch rows. No dense dictionary construction or assumptions about private operator data.
- If no native select_columns exists, scatter restricted input into original source coordinates and gather adjoint results. This preserves matrix-free correctness but forfeits reduced-width product savings. Count/report fallback use.
- Wrapper supports LinearOperator contracts, dtype/device and detached semantics. No numpy/cpu conversion inside solver math. MatrixMult native column restriction copies selected matrix columns, CMFPropagation selects propagation columns without materializing its measurement dictionary. Generic fallback may allocate B*n workspaces; report this trade-off.

Callbacks: one begin/end callback pair per outer sweep, full global (B,n) iterates with stable shape. Begin sees current x and solver.iter=k. End gamma/theta are (B,1) tensors of the most recent per-row accepted inverse L / lambda/L (initial guesses for untouched rows), plus A=full operator, W=full operator, y=full measurements. Metadata may include detached KKT, row status, working-set sizes. Reset stop each solve; existing stoppers can request early exit but never override independent convergence diagnostics. On callback stop, unfinished rows have callback_stopped status, not converged. Repeated calls reset diagnostics; callbacks maintain their own existing state as per repository contract.

Diagnostics stored separately, forward returns only x: per-row status (converged/max_iter/callback_stopped), natural/raw final KKT residual, objective, outer/inner iterations, per-sweep working-set sizes, backtracking rejections, inner target/budget information, full/restricted forward/adjoint call counts, fallback use. Counters count actual operator calls including rejected trials and final certification; distinguish logical row products from underlying B-wide work. Public layout may use a documented diagnostics dictionary; no API dependency in existing solvers.

## 8. Complete pseudocode

```
prepare y,x0,detached A; validate scalar/per-row lambda,gamma
x <- projected x0 (positive) or x0 clone
for each b: cache full residual, gradient, F, R, initial R norm; initial L
reset stop and diagnostics
for k in range(t):
    mark currently certified rows converged
    if all rows converged: break
    run begin callbacks(k,full x); if stop: break
    for each uncertified row b:
        W <- support union one-/two-sided near violators from FULL gradient
        if empty: certify full KKT and skip
        a <- x_b[W]; construct restricted row operator
        v <- a; z_prev <- a; theta <- 1; best <- undefined
        L <- initial L, or max(floor,last_L/eta)
        for j in range(inner_max_iter):
            d_q <- reduced.adjoint(reduced.forward(v)-y_b)/m+delta*(v-a)
            backtrack prox(v-d_q/L,lambda/L) until quadratic test passes
            z <- accepted candidate; evaluate Phi(z)
            if first candidate or Phi(z)<Phi(best): best <- z
            compute subproblem residual at z
            if subproblem residual <= inner_tol*global_R_before: break
            theta_new <- FISTA recurrence
            v <- z+(theta-1)/theta_new*(z-z_prev)
            z_prev <- z; theta <- theta_new
        x_b <- zeros with x_b[W]=best; remember accepted L
        recompute FULL residual/gradient/objective/KKT at x_b
    run end callbacks(full x, full operators, per-row gamma/theta,...)
    if stop: break
recompute full quantities at returned x; set honest per-row statuses; return x
```

Candidate residual and objective calculations may share cached images/adjoints; equivalence to these rules, honest counts, and stable backtracking are required. No hard thresholding, unreviewed cap, restart variant, or changed inner criterion.

## 9. Convergence claims and proof boundary

### C1 [Adapted, proved here subject to independent review]: exact-arithmetic outer descent

Let L_1 be first accepted curvature; z_1=prox_{h/L_1}(a-grad q(a)/L_1). Majorization and strong convexity of its proximal quadratic imply
Phi(a)-Phi(z_1)>=L_1||a-z_1||²/2.

The best-candidate safeguard yields
F(x)-F(x_new)>=L_1||a-z_1||²/2+delta||x_new-x||²/2.

Support retention and inclusion of all zero violators mean off-W full proximal displacement is zero, hence ||a-z_1||=||R_{1/L_1}(x)|| with R_gamma=x-prox_{gamma h}(x-gamma g). This holds for positive and signed modes.

### C2 [Adapted, proved here subject to independent review]: residual convergence, not accelerated global rate

If first accepted curvatures are uniformly bounded 0<L_min<=L_1^k<=L_max<infinity, then per coordinate
|R_{1/L_1}(x)| >= (1/L_max)|G_{1/L_max}(x)|,
where G_gamma=R_gamma/gamma. For positive mode this follows from |min(L_1 x,s)|>=(L_1/L_max)|min(L_max x,s)|; signed soft threshold has the analogous coordinate case split. Thus
F(x^k)-F(x^{k+1}) >= L_min/(2L_max²)||G_{1/L_max}(x^k)||².

Telescoping against F>=F* gives square summability, G->0, and best-so-far residual O(K^-1/2), with constants depending on initial gap/curvature bounds. R_1 shares the zero set and tends to zero by coordinate comparisons with fixed L_max. Every finite cluster point is optimal (continuity). For lambda>0 bounded level sets ensure existence of cluster points. At lambda=0 boundedness is not assumed and full sequence convergence is not claimed. Backtracking with bounded initialization/eta and fixed delta supplies the bounds in exact arithmetic. Finite max_backtracking can instead raise an honest error.

### Other claims

| Claim | Classification / implementation consequence |
|---|---|
| KKT zero-set equivalence, scaled prox, source restriction keeps m | Adapted, elementary proof in sections 2–4 |
| Composite FISTA-B O(1/j²) reduced objective gap | Unchanged structure / Adapted indicator and L constant; nondecreasing inner L and finite upper bound needed |
| Reduced strong convexity for delta>0 | Unchanged; H>=delta I; no Gram computation needed |
| Whole outer coefficient sequence converges | Conditional for fixed delta>0, bounded iterates, exact or summably distance-inexact proximal solves and the working-set/safeguard assumptions; **Unproven for this fixed-budget/relative-inner default** |
| Local linear convergence for implemented solver | Unproven; do not inherit paper Theorem 3.7 or arbitrary contraction factors |
| Fixed support identification / permanently safe elimination | Unproven / not implemented |
| Sparsity-dependent global speedup or paper O(mp log(1/epsilon)) total work | Invalid as an unconditional claim; global scans and conditioning cannot be omitted |
| Printed (2.19), literal (2.22) | Invalid implementation rules: objective-gradient inconsistency; unreachable scalar threshold |
| Positive analogue of paper (3.7) | Invalid constant per derivation counterexample; not needed by C1/C2 |
| Signed paper (3.7) itself | Invalid: independently reviewed counterexample H=[[.51,.50],[.50,.51]], r=(-1.51,-.50), lambda=1, x*=(1,0), x=(1,1) gives paper residual (.5,1), ||H||=1.01, contradicting (3.7). This does not prove (3.8) false. |
| Paper (3.8)/(3.19) constants | Unproven for extension; not used |

The reports contain stronger proposed statements (including raw-residual bounds, whole-sequence/local rates) not needed by implementation. They remain research notes and are not adopted without proof. In particular do not adopt a coordinatewise bound from a norm-only subgradient argument without checking it.

## 10. Complexity and storage

Per row, dense native products: each outer full scan O(mn); T inner steps with W width p cost O(T*m*p), with extra forward trials for backtracking and adjoints for subproblem residuals. Total O(sum_k[mn+T_k*m*|W_k|+backtracking work]); no unconditional small-p guarantee. Matrix-free CMF costs are those of the operator on selected propagation columns; avoid expressing them as dense mn if inappropriate. Generic scatter/gather restriction still pays full-width operator work, batched row wrapper can pay additional original batch work.

No n*n or p*p Gram matrices or identity-probing dictionary construction. Incremental per-row workspace O(n+m+p), plus batch state/output O(B*n+B*m), excluding stored operators, diagnostics, native selected matrix copies, and batched fallback O(B*n)/O(B*m) scratch. Per-row processing trades GPU utilization and host sync for unambiguous independent working sets. Autograd graphs can grow with iterations; benchmarks use no_grad. Benchmark memory separately from correctness and disclose CPU/GPU measurement scope.

## 11. Validation and implementation plan

Only scientific-coder modifies code/tests/docs during Phase 3. Implementation notes must record any deviations and gate review must cover material changes before claiming compliance. Preserve existing untracked .pi/ and docs/ir_fista.md; no commits, pushes, dependency changes or unrelated refactoring.

Files: new src/ucmf/optim/ir_fista_b.py and tests/test_ir_fista_b.py; small exports in src/ucmf/optim/__init__.py and src/ucmf/__init__.py; README usage and diagnostics/t semantics; new reproducible benchmark script may be added under examples/ by scientific-coder. Existing operator API remains unchanged.

Test-first coherent increments: (1) analytic identity/KKT and input API, observe failure then implement prox/outer core; (2) reduced FISTA-B/backtracking/descent, scalar D2 regression, observe failure then add safeguards; (3) batching/restriction/callbacks and diagnostics with missing-feature failures; (4) numerical oracles, challenging dictionaries and reactivation; (5) docs/benchmark instrumentation. Record actual failing/passing commands in implementation_notes.md. Reuse helpers, no unrelated refactors.

Correctness cases: identity (positive/signed) x*=prox(y,m lambda), lambda=0 vs scipy.optimize.nnls (already indirect dev dependency; no new runtime dependency), lambda>=lambda_max => zero, deterministic random vs long independently certified FISTA or CVXPY if available (no dependency required), sparse/dense supports, zero/random/negative warm starts, high coherence, rank deficiency, scaling, different B/operators/parameters, dtype/device CPU/CUDA-if-available, CMFPropagation vs CMFDictionary, generic fallback, warm support retention and reactivation, empty sets/A=0, signed x0=+/-2^60 with A=0, lambda=1, float64 (natural/raw residual exactly 1, never converged at t=0), excessive initial gamma backtracking, delta scalar optimum, final certificate/t=0/iteration exhaustion/callback stop/repeated calls, no Gram/no matrix-free dictionary materialization.

Compare F(x)-F(x_ref), natural/raw ||r_KKT||_inf, feasibility; do not require coefficient agreement for nonunique cases. Existing test suite baseline: `uv run pytest -q`: 1912 passed,21 skipped,15 warnings (104.39 s); `uv run ruff check .`: passed. Initial setup attempt timed out downloading dependencies; second completed successfully, tracked files unchanged.

Phase 4 independent scientific-reviewer checks theory vs code and reproduces correctness, followed by benchmark runs designed by experiment-designer. Gate 2 requires numerical references to pass before performance claims. Benchmarks report seed/shapes/dtype/device, stopping quality and budgets, runtime/setup scope, peak memory with measurement caveats, working-set histories, full/reduced operator-call counts (including line-search and certificate work), large dictionaries and realistic CMF. Existing ISTA/FISTA baselines use the same objective and tolerance (or disclose unmet tolerance/censoring). Limited measurements are not general speed guarantees.

## 12. Gate 1 acceptance record

Initial independent mathematical-reviewer approved C1/C2 and blocked v1 on M1 (optional signed stabilization could cause a false floating-point certificate). v2 mandates the cancellation-free signed expression in both residual sites and the +/-2^60 regression; N1 initial floor, N2 positive-delta qualification, N4 batch memory are corrected; N3 is corrected in the supporting derivation. Focused re-review independently verified v2 in full and PASSED: no unresolved Critical or Major implementation-relevant findings. Implementation may start against v2; Gate 2 remains outstanding. All future Critical/Major implementation-relevant findings must be resolved and reviewed before claiming compliance. Fundamental mathematical uncertainty in the safeguarded descent or KKT rule triggers human escalation; missing local acceleration/whole-sequence guarantees are explicit limitations, not a blocker to the bounded, globally certified numerical solver.
