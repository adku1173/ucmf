# Independent mathematical review — Gate 1

**Current verdict: PASS for specification v2.**

**Initial review:** `algorithm_specification.md` v1 in full, supported by `derivation.md`, `literature.md`, `repository_analysis.md`, and historical `docs/ir_fista.md`.  
**Focused closure review:** specification v2 independently reread in full; derivation’s review errata and corrected zero-displacement caveat inspected.  
**Reference baseline:** `ab87210`. Read-only review; no files modified or commands executed.

Primary-source equations were checked during the initial review against `~/mendeley_papers/Wang/Wang2022/Wang2022.md`. Visual PDF typography checks remain attributable to the literature investigator’s two-PDF verification.

## Executive result

**No unresolved Critical or Major implementation-relevant mathematical findings remain in v2.**

The exact-arithmetic algorithm is sound:

- The nonnegative and signed proximal steps and KKT systems are correct.
- Support retention, global reactivation, and the first-step safeguard establish C1.
- Bounded accepted curvatures establish C2, including at `lambda=0`.
- Neither exact reduced solves nor summable inner errors are needed for C1/C2.
- The specification appropriately declines whole-sequence, local-linear, and sparsity-dependent runtime guarantees.

The initial review blocked v1 because stable evaluation of the **signed natural residual was optional**. V2 mandates the correct cancellation-free formula at both residual sites, prohibits literal proximal subtraction for certification, and requires the counterexample regression. **M1 is closed at the specification level.** N1–N4 are also addressed.

This gate approves the mathematical implementation contract, not an implementation or its numerical behavior. Required regressions and numerical references remain Gate 2 obligations.

## 1. Original findings and disposition

### M1 — Major, CLOSED in v2: optional stable signed residual permitted false convergence

**Location:** specification §3, line 37; subproblem stopping in §6, line 95; regression in §11, line 205.

The mathematical definition
\[
R=x-S_\lambda(x-g)
\]
is correct, but evaluating the subtraction literally is unsafe. V1 recommended a stable expression only “if useful.”

**Counterexample, IEEE float64, checked analytically:**
\[
A=[0],\quad y=0,\quad \lambda=1,\quad x^0=2^{60},
\quad \texttt{positive=False}.
\]
Then \(g=0\), the exact natural and raw residual norms are both \(1\), and the unique optimizer is zero. However, \(2^{60}-1\) rounds to \(2^{60}\), so literal evaluation gives
\[
\operatorname{fl}\bigl(x-S_1(x)\bigr)=0.
\]
The initial certificate could report `converged`, including with `t=0`, while the objective gap is \(2^{60}\). All quantities remain finite, so finite-value checks do not catch this.

**Verified v2 correction:**
\[
\boxed{R_j=\min\{\max\{x_j,g_j-\lambda\},\,g_j+\lambda\}.}
\]
This identity is now mandatory for outer certification and signed subproblem stopping, with \(\nabla q\) substituted for \(g\) in the latter. Literal \(x-S_\lambda(x-g)\) subtraction is expressly prohibited at those sites.

For \(g=0,\lambda=1\), the corrected expression returns \(+1\) at \(x=2^{60}\) and \(-1\) at \(x=-2^{60}\), without subtracting nearly equal large coefficients. Both natural residual norms and raw residual norms are exactly \(1\).

**Verified regression requirement:** float64, \(A=0,\lambda=1,x^0=\pm2^{60},t=0\), residual norms \(1\), and no convergence at the default tolerance.

**Disposition:** closed. The required regression has been specified, not executed. This was a certification error, not an absent convergence theorem.

### N1 — Minor, CLOSED in v2: initial curvature floor

**Locations:** specification §6, line 84; pseudocode line 132.

V1 used \(L=1/\gamma\) initially without explicitly applying `L_floor`. Subsequent runs applied the floor. This did not invalidate C2, since a positive fixed initial guess still permitted
\[
L_{\min}=\min\{L_{\mathrm{initial}},L_{\mathrm{floor}}\}>0,
\]
but the advertised floor was not consistently enforced.

**Verified v2 correction:**
\[
L=\max\{L_{\mathrm{floor}},1/\gamma\},
\]
or \(\max\{L_{\mathrm{floor}},1\}\) when `gamma=None`. The specification additionally requires positive, finite, representable initial curvature and restricts the numerical solver to float32/float64.

**Disposition:** closed. These checks do not eliminate all possible floating-point underflow; that remains an implementation-validation concern rather than an exact-arithmetic claim.

### N2 — Minor, CLOSED in v2: conditional whole-sequence theorem requires positive proximal weight

**Location:** specification §9, line 180; derivation §9.1, Theorem 2.

The derivation assumes \(\delta>0\), but v1’s conditional summary omitted it. Exact or summably inexact reduced minimization plus boundedness alone is insufficient when \(\delta=0\).

For example, with \(A=[1,1]\), \(y=2\), \(\lambda=1\), both \((1,0)\) and \((0,1)\) minimize the positive problem. For \(\tau>0\), the working set at either point contains both coordinates. Arbitrary exact reduced minimizers can alternate between them.

This is **not** a counterexample to the specified best-candidate algorithm: its first-step/tie rule fixes an optimal anchor.

**Verified v2 correction:** the conditional statement now requires fixed \(\delta>0\), bounded iterates, exact or summably distance-inexact proximal solves, and the working-set/safeguard assumptions. Whole-sequence convergence remains explicitly **Unproven** for the implemented fixed-budget/relative-inner default.

**Disposition:** closed.

### N3 — Minor, CLOSED in supporting derivation: directional curvature inference requires nonzero displacement

**Location:** `derivation.md`, §6, line 237.

The asserted Rayleigh-quotient lower bound requires \(D\ne0\). When \(D=0\), the backtracking test is \(0\le0\), regardless of \(\delta\).

**Verified correction:** the derivation now explicitly conditions the inference on \(d\ne0\) and states that \(d=0\) supplies no curvature lower bound.

**Disposition:** closed. C1/C2 never depended on this invalid zero-displacement inference.

### N4 — Minor, CLOSED in v2: per-row workspace versus batch storage

**Location:** specification §10, line 195.

The original \(O(n+m+p)\) estimate described per-row workspace but did not separately account for the returned tensor and batch-wide state.

**Verified v2 correction:** incremental per-row workspace is distinguished from batch state/output \(O(Bn+Bm)\), with operator storage, diagnostics, selected matrix copies, and batched fallback scratch separately excluded or disclosed.

**Disposition:** closed.

### Remaining nonblocking implementation guidance

- V2 now explicitly requires positive-integer `inner_max_iter` and `max_backtracking`; C1 requires at least one accepted candidate.
- Diagnostics must distinguish “current candidate met inner target,” “returned best candidate met target,” and “budget exhausted.” These events need not coincide.
- Relative energy slack and float32/float64 support do not prevent every underflow or operator-evaluation error.
- Retain objective/reference comparisons in validation. A natural-residual certificate is deliberately not an objective-gap certificate.

## 2. Mathematical classification of specification v2

| Statement | Classification | Independent assessment |
|---|---|---|
| Normalized objective, gradient, measurement denominator | **Adapted** | Correct: \(g=A^\top(Ax-y)/m\). Source restriction does not change \(m\). |
| Positive proximal map | **Adapted** | Correct: minimization over \(z\ge0\) gives \([v-\gamma\lambda]_+\). |
| Signed soft threshold | **Unchanged** | Same separable proximal operator as paper (1.2), with normalized parameters. |
| Positive KKT system | **Adapted** | Exactly \(x\ge0,\ g+\lambda\ge0,\ x\odot(g+\lambda)=0\). |
| Signed KKT system | **Unchanged** | Paper (1.7)–(1.8), after normalization. |
| Natural-residual zero set and continuity | **Adapted** | Correct in both modes; v2 mandates stable evaluation. |
| Raw residual | **Adapted** | Correct minimum-norm subgradient residual on feasible points; discontinuous at zero. |
| Support-retaining working set | **Unchanged / Adapted** | Signed structure matches (2.1); the positive rule correctly becomes one-sided. |
| Reactivation and empty-set handling | **Adapted** | Correct. Excluded coordinates are reconsidered using the full gradient. |
| Reduced objective, gradient, Hessian | **Adapted** | Correct, including \(\delta(v-a)\). |
| Reduced strong convexity for \(\delta>0\) | **Unchanged** | \(H\succeq\delta I\), without forming \(H\). |
| Quadratic backtracking inequality | **Adapted** | Exactly equivalent to smooth majorization; constraints do not invalidate it. |
| FISTA momentum recurrence | **Unchanged** | Matches paper (2.20)–(2.21). Resetting between reduced solves is correct. |
| Curvature upper bound and finite exact-arithmetic line search | **Conditional** | Correct with positive initialization, fixed finite \(\delta\), finite operator norm, and \(\eta>1\). |
| First-step/best-candidate safeguard | **Adapted** | Correct; retaining the first accepted candidate is essential. |
| C1 outer descent | **Adapted** | Independently proved below. |
| C2 residual convergence | **Conditional** | Independently proved under the stated curvature bounds and exact arithmetic. |
| Reduced FISTA objective rate \(O(j^{-2})\) | **Conditional** | Valid for nondecreasing accepted inner curvatures with a finite upper bound. |
| Relative inner stopping/budget | **Adapted heuristic** | Does not imply summable errors; nevertheless sufficient with the safeguard for C1/C2. |
| Whole outer coefficient sequence convergence | **Unproven** for default | Correctly excluded; the alternative conditional statement now includes fixed positive \(\delta\). |
| Local linear rate/support identification | **Unproven** for default | Correctly excluded. |
| Unconditional sparsity-based speedup | **Invalid** | Full scans, conditioning, fallback products, and batch wrapper overhead cannot be omitted. |
| Row wrapper and source restriction | **Conditional / Adapted** | Correct under the repository’s independent-problem operator contract. |
| Finite-budget success guarantees | **Unproven**, not asserted | Honest `max_iter`, callback termination, and line-search errors are appropriate. |

## 3. Independent verification of C1 and C2

### C1: first-step descent is valid

Let
\[
Q_L(z,a)=q(a)+\langle\nabla q(a),z-a\rangle+
\frac L2\|z-a\|^2+h_{C_W}(z).
\]
The first proximal candidate \(z_1\) minimizes this \(L\)-strongly convex model. Since \(a\) is feasible,
\[
Q_L(a,a)-Q_L(z_1,a)\ge\frac L2\|a-z_1\|^2.
\]
Accepted backtracking gives \(\Phi(z_1)\le Q_L(z_1,a)\), hence
\[
\Phi(a)-\Phi(z_1)\ge\frac L2\|a-z_1\|^2.
\]

The safeguard gives \(\Phi(z_{\rm out})\le\Phi(z_1)\). Support retention gives
\[
\Phi(a)=F(x),\qquad
F(x^+)=\Phi(z_{\rm out})-\frac\delta2\|x^+-x\|^2.
\]
Therefore
\[
\boxed{
F(x)-F(x^+)\ge
\frac L2\|a-z_1\|^2+
\frac\delta2\|x^+-x\|^2.}
\]

At the anchor, \(\nabla q(a)=g_W(x)\). Outside \(W\), \(x_j=0\) and the original KKT inequality holds, so the full proximal displacement is zero there. Thus
\[
\|a-z_1\|=\|R_{1/L}(x)\|.
\]

No exact reduced solve, strong convexity of \(F\), paper residual bound, or spectral calculation is needed.

### C2: the signed comparison is valid too

For positive mode,
\[
R_{\gamma,j}=\min(x_j,\gamma s_j).
\]
For signed mode,
\[
R_{\gamma,j}
=\min\{\max\{x_j,\gamma(g_j-\lambda)\},\gamma(g_j+\lambda)\}.
\]

A scalar case split gives, for \(0<\beta\le\alpha\),
\[
|R_{\beta,j}|\le |R_{\alpha,j}|
\le \frac{\alpha}{\beta}|R_{\beta,j}|
\]
in both modes.

Taking \(\beta=1/L_{\max}\), \(\alpha=1/L_1^k\), gives
\[
F(x^k)-F(x^{k+1})
\ge\frac{L_{\min}}{2L_{\max}^2}
\|G_{1/L_{\max}}(x^k)\|^2.
\]
Consequently,
\[
\sum_k\|G_{1/L_{\max}}(x^k)\|^2
\le
\frac{2L_{\max}^2}{L_{\min}}(F(x^0)-F^*).
\]
This proves residual convergence and the stated best-so-far \(O(K^{-1/2})\) residual rate. Fixed-step residual comparisons also imply \(R_1(x^k)\to0\).

Every finite cluster point is optimal. For \(\lambda>0\), coercivity ensures bounded level sets and cluster points. At \(\lambda=0\), bounded coefficients are not established by this proof.

**Cycling assessment:** a nonoptimal finite cycle is impossible in exact arithmetic: C1 gives strict descent at every non-KKT anchor. This does not establish convergence of the complete coefficient sequence.

### Backtracking bounds

Put \(M=\|A\|^2/m+\delta\). Every restricted Hessian has norm at most \(M\). Therefore every trial \(L\ge M\) passes. Inductively, across all reduced runs,
\[
L\le \max\{L_{\rm initial},L_{\rm floor},\eta M\}.
\]
V2 applies the positive floor at initialization and subsequent restarts, providing the intended uniform lower bound in exact arithmetic.

Acceptance can occur below the spectral norm because the test is directional. An oversized initial \(L\) also need not fall below \(\eta M\). The specification handles both points correctly.

A finite trial budget does not guarantee successful acceptance for every valid input. Raising an explicit error is mathematically honest; claiming convergence would not be.

These are exact-arithmetic results. The allowed floating-point slack and finite-value safeguards do not establish an identical floating-point theorem.

## 4. Working sets, reduced optimality, and re-entry

The proposed sets include every nonzero coordinate and every zero-coordinate KKT violator, including when \(\lambda=0\). Consequently, omitted coordinates have zero natural residual **at the current anchor**, not necessarily after the reduced update.

An exact reduced solution is globally optimal only when the **updated full point** satisfies original KKT:

- With \(\delta=0\), reduced KKT supplies the on-set conditions, but off-set inequalities must still be checked.
- With \(\delta>0\), reduced KKT contains \(\delta(z-a)\); it is not automatically original KKT on \(W\).
- If the anchor itself solves its proximal reduced problem, the proximal correction vanishes and the current off-set conditions complete global KKT.

**Concrete re-entry example:**
\[
A=\begin{pmatrix}1&-1\\0&1\end{pmatrix},\quad
y=(1,1),\quad m=2,\quad\lambda=0.1,\quad\tau=0,\quad x^0=0.
\]
Initially only coordinate 1 enters. Its unregularized reduced solution is \((0.8,0)\). At that point \(g_2=-0.4<-\lambda\), so coordinate 2 must re-enter. The full optimizer is \((1.4,0.6)\).

The specified full-gradient reselection handles this correctly. Permanent exclusion or reduced-only certification would fail.

## 5. Residuals, scaling, batching, and edge cases

### Natural versus raw residual

Both residual definitions have exactly the correct zero set. Their small-value behavior differs. For example, in positive mode, \(x_j>0\) tiny and \(s_j>0\) large gives \(R_j=x_j\) but raw residual \(s_j\).

Thus finite-tolerance `converged` means **natural stationarity tolerance attained**, not small objective gap or coefficient error. That limitation is explicit and acceptable. Original M1 concerned a different problem: computing zero when the true natural residual was not small.

For signed mode, direct case analysis also shows
\[
g_j^{\rm paper}(f,x)=|R_{1,j}(x)|.
\]
The natural residual’s magnitude is precisely paper (2.4) in the same units, not a different signed stationarity measure. The specification changes the norm and stopping convention. V2 now records this identity.

### Scaling

The conversions
\[
\lambda_p=m\lambda,\quad\delta_p=m\delta,\quad
L_p=mL,\quad\sigma_p=m\sigma,\quad\gamma_p=\gamma/m
\]
are correct.

They preserve corresponding proximal steps. They do not automatically preserve an entire adaptive run: curvature initialization/floors and stopping rules must also be converted. Fixed-reference natural residuals mix coefficient and gradient scales, so there is no universal exact paper-tolerance conversion. Raw residuals, in contrast, scale by \(m\).

### Batch row wrapper

For independent batched operators, injection into row \(b\), operator application, and extraction of row \(b\) implement \(A_b\). Applying the corresponding measurement injection and source extraction to the adjoint implements \(A_b^\top\). Restriction then gives
\[
A_{b,W}=A_bS_W,\qquad A_{b,W}^\top=S_W^\top A_b^\top.
\]
This validates both native selection and scatter/gather fallback. Batch-coupled maps would violate the independent-problem assumption and are outside this contract.

Per-row state and full-coordinate callback iterates are appropriate. Underlying \(B\)-wide work must remain visible in performance accounting.

### Edge-case checks

| Case | Result |
|---|---|
| \(A=I,\ m=n\) | Positive solution \([y-m\lambda]_+\); signed solution \(S_{m\lambda}(y)\). With \(\delta=0,L=1/m\), the full proximal step is exact. |
| \(A=I,\delta>0\) | On working coordinates, positive reduced solution \([(y-m\lambda+m\delta a)/(1+m\delta)]_+\); signed counterpart \(S_{m\lambda}(y+m\delta a)/(1+m\delta)\). |
| \(\lambda=0\) | Correct NNLS/LS rules; C1/C2 remain valid. No sparsity or coefficient-boundedness guarantee follows. |
| \(\lambda\ge\lambda_{\max}\) | Zero satisfies KKT. Immediate termination applies to a certified initial point, ordinarily the default zero start—not every warm start. |
| \(A=0,\lambda=0\) | Every feasible point is optimal. Return the prepared/projected point. |
| \(A=0,\lambda>0\) | Zero is the unique optimizer; v2’s stable signed residual addresses the large-warm-start false-certificate example. |
| Rank-deficient \(A\) | Existence and residual convergence remain valid. Coefficient uniqueness must not be assumed. |
| Multiple batch rows | Mathematical independence holds with the proposed wrapper; validate against separate solves. |

## 6. Supporting derivation and source corrections

These findings do not undermine C1/C2. V2 and the derivation’s top-level errata now record the relevant corrections and precedence over superseded research statements.

### Paper (3.7) is invalid in the signed case as well

Take
\[
H=\begin{pmatrix}.51&.50\\.50&.51\end{pmatrix},\quad
r=(-1.51,-.50),\quad\lambda=1.
\]
For \(\phi(x)=\tfrac12x^\top Hx+r^\top x+\|x\|_1\), \(x^*=(1,0)\) satisfies signed KKT because \(Hx^*+r=(-1,0)\).

At \(x=(1,1)\), paper (2.4) gives
\[
\widetilde g(x)=(.5,1),\qquad
\|x-x^*\|=1,\qquad\|H\|=1.01.
\]
Equation (3.7) would require
\[
1\ge \frac{\sqrt{1.25}}{1.01}>1,
\]
which is false. Since \(H\) is positive definite, this quadratic can also be represented as a least-squares smooth term.

**Classification: Invalid.** This confirms the concern about the norm-combination step (3.13). The replacement
\[
\|R_1(x)\|\le\sqrt{1+\|H\|^2}\,\|x-x^*\|
\]
is valid; it is not asserted to be sharp. Failure of (3.7) does not itself prove (3.8) false.

**V2 disposition:** explicitly incorporated in specification §9 and the derivation errata.

### Paper (2.19) and literal (2.22)

The corrected proximal gradient follows from differentiating (2.15), independently of the printed formula.

The scalar example \(A=2,y=1,\lambda=1,\delta=.001,a=0\) gives available decrease
\[
1/(2\cdot4.001)\approx .124969,
\]
versus the printed threshold \(4.001^3/4\approx16.012003\).

**Classification: Invalid as implementation rules.** Replacing these rules, rather than inheriting their constants, is justified.

### Qualification of the wide-matrix rate counterexample

**Location:** derivation §9.4 and its top-level errata.

With \(\sigma=\delta=1\), the claimed exact proximal contraction \(1/2\) is correct, and the paper’s inference that reduced full rank implies \(\delta^k=0\) is wrong. However, that example does not satisfy literal (2.22): its decrease requirement is unreachable.

A repaired example uses \(A=[1,.5]\), \(y=1\), \(0<\lambda<1\), and \(\delta=\sigma=.001\). For sufficiently small \(\tau\) and iterates near \(x^*=(1-\lambda,0)\), the working set is \(\{1\}\), and exact reduced solves contract by
\[
\delta/(1+\delta)\approx .000999.
\]
With \(M=1.001\), the literal decrease threshold is \(M^3e^2/4\), below available decrease \(e^2/(2M)\). The iteration-count requirements can also be met. A claimed arbitrary contraction \(\rho<.000999\) still fails.

**Classification:** arbitrary inherited contraction **Invalid**; a rate for a particular corrected algorithm **Conditional**.

**V2 disposition:** the derivation errata expressly supersede the original \(\delta=1\) counterexample as a counterexample satisfying the literal stopping assumptions.

### Other research-note qualifications

- **Derivation Proposition 2:** its coordinatewise raw-residual bound is valid because the nonsmooth subdifferential is a Cartesian product. A norm inequality alone would not prove the displayed coordinatewise claim.
- **Finite inner stopping:** strong convexity is not needed to establish eventual attainment of a positive **natural-residual** target in an unlimited FISTA run. Objective-gap convergence and a fixed global proximal-gradient descent bound suffice. This does not promise attainment within `inner_max_iter`.
- **Dual gap:** the displayed positive dual and feasible-ray construction are correct. Calling the dual gap the “only” rigorous objective certificate is too strong; other bounds are possible with additional information.
- **Complexity:** the specification correctly restores full-gradient scans and conditioning-dependent work. Paper (3.41) is not an unconditional total-work bound for this implementation.

The updated derivation’s opening errata record these qualifications. Its unrevised historical passages must not override the reviewed specification.

## 7. Focused v2 closure

The closure review was based on the updated documents themselves, not solely the parent’s change summary.

| Item | Independently verified evidence | Closure |
|---|---|---|
| M1 | Specification line 37 mandates the signed clamp identity, prohibits literal subtraction, and explicitly applies the rule to \(\nabla q\) for inner stopping. | **Closed** |
| M1 regression | Specification line 205 requires float64 \(A=0,\lambda=1,x^0=\pm2^{60}\), residual norms \(1\), and no convergence at `t=0` under the intended default tolerance. | **Closed at contract level; execution pending** |
| N1 | Specification line 84 applies the floor initially and requires positive, finite, representable curvature; float32/float64 only. | **Closed** |
| N2 | Specification line 180 includes fixed positive \(\delta\), boundedness, summable distance errors, and working-set/safeguard assumptions. | **Closed** |
| N3 | Derivation line 237 explicitly distinguishes \(d\ne0\) from \(d=0\). | **Closed** |
| N4 | Specification line 195 separately accounts for incremental row workspace and \(O(Bn+Bm)\) batch state/output. | **Closed** |
| Inner budgets | Specification line 95 requires strictly positive integer iteration/trial budgets. | **Addressed** |
| Source/research qualifications | Specification §9 and derivation opening errata record the signed counterexample and other reviewed qualifications. | **Addressed** |

The revisions preserve the reduced objective, working-set rules, first-step safeguard, momentum recurrence, and C1/C2 assumptions. No new Critical or Major implementation-relevant mathematical error was identified.

### Residual risks and approval boundary

- The specified regression has not been implemented or executed in this review.
- Finite-precision gradients, energy tests, objective comparisons, overflow/underflow, and operator adjoint accuracy require numerical validation.
- Fixed outer/inner budgets can return an uncertified point; honest status reporting is required.
- Whole-sequence coefficient convergence, local linear rates, permanent support identification, and general speedup guarantees remain unproved for the default.
- Natural residual tolerance is not an objective-gap or coefficient-error tolerance.
- Material deviations from v2 require review before being represented as compliant.

These are explicit limitations and implementation-validation obligations, not unresolved Gate 1 blockers.

## Final verdict

**Gate 1 PASSED — implementation gate PASS for specification v2.**

**Unresolved Critical findings:** none.  
**Unresolved Major implementation-relevant findings:** none.

The original M1 blocker and N1–N4 are closed at the specification level. C1/C2 remain approved under their stated exact-arithmetic assumptions. Implementation may proceed against v2; numerical correctness and performance remain subject to independent implementation review and Gate 2.