# IR-FISTA-B: mathematical specification and paper deviations

Status: review draft; approval covers this document, not solver implementation or the proposals below.
Approved requirements: signed and nonnegative LASSO, independent processing of each batch problem.
Source: Wang et al., *An Iterative Reduction FISTA Algorithm for Large-Scale LASSO*,
SIAM J. Sci. Comput. 44(4), A1989–A2017 (2022), DOI: 10.1137/20M1374328.
Paper equation numbers below refer to that publication; (2.19), (2.22), and (3.19) were checked in the PDF.
IDs identify equivalent reformulations (A), correction proposals (D), and an extension (E).

## A1 — Objective, units, and residuals

Let $m$ be the number of active measurements, $y=b$, and $C=\mathbb R^n$ (`positive=False`)
or $C=\mathbb R_+^n$ (`positive=True`). With $I_C$ the indicator of $C$, ucmf solves
$$F_C(x)=f(x)+\lambda\|x\|_1+I_C(x),\qquad f(x)=\|Ax-y\|^2/(2m).$$
All vector norms are Euclidean; matrix norms are spectral unless stated otherwise.
Write $d=\nabla f(x)=A^\top(Ax-y)/m$; this is the negative of ISTA's prepared direction.
Dividing the paper's objective (1.1), (2.15)–(2.16) by $m$ gives
$$\lambda_p=m\lambda,\qquad\delta_p^k=m\delta^k,\qquad L_p=mL.$$
Thus minimizers and proximal steps agree when parameters are converted; $L$ includes the proximal curvature.
A curvature-floor parameter $\sigma$ must likewise satisfy $\sigma_p=m\sigma$ if used.
This is an exact rescaling, not a new regularizer; $m$ does not change on selecting source columns.
Paper (2.4), evaluated with normalized $d,\lambda$, defines the proposed signed residual:
$$g_j=\begin{cases}\max\{|d_j|-\lambda,\min(|d_j+\lambda\operatorname{sign}(x_j)|,|x_j|)\},&x_j\ne0,\\
\max(0,|d_j|-\lambda),&x_j=0.\end{cases}$$
Its zero set is the signed KKT system; proposed outer accuracy is $\|g\|_2\le\varepsilon$.
Because (2.4) mixes gradients with $|x_j|$, generally $g_p\ne m g$; no universal tolerance conversion follows.
Reusing the paper's numerical tolerance or inner-bound constants after rescaling needs separate justification.

## D1 — Derive the proximal step from (2.15)–(2.16), not printed (2.19)

Set $W=W^k$, $a=x_W^k$, and $q_k(z)=\|A_Wz-y\|^2/(2m)+\delta^k\|z-a\|^2/2$.
The printed (2.19) uses $v_p=\hat z+\delta_p a/L_p-A_W^\top(A_W\hat z-y)/L_p$
and minimizes $(L_p/2)\|z-v_p\|+\lambda_p\|z\|_1$: the norm square is absent.
Differentiating the defining quadratic instead gives
$$\nabla q_k(z)=A_W^\top(A_Wz-y)/m+\delta^k(z-a).$$
Proposal: $v=\hat z-\nabla q_k(\hat z)/L$, then minimize
$$\frac L2\|z-v\|^2+\lambda\|z\|_1+I_C(z).$$
Here $C$ is restricted to the working coordinates. The minimizer is soft-thresholding at $\lambda/L$,
or $\max(v-\lambda/L,0)$ for nonnegativity; the extra $-\delta^k\hat z/L$ is essential.
Backtracking must check $q_k(z^+)\le q_k(\hat z)+\langle\nabla q_k(\hat z),z^+-\hat z\rangle+L\|z^+-\hat z\|^2/2$.
The nonsmooth terms cancel in this majorization test; use the same $q_k$ in the objective decrease test.
Impact: restores consistency with the stated subproblem; tests must check its gradient and analytic scalar minimizer.

## D2 — Conflicting inner-decrease thresholds in (2.22) and (3.19)

In paper units let $H_p=A_W^\top A_W+\delta_p^k I$, $\mu_p=\lambda_{\min}(H_p)$, $M_p=\|H_p\|$,
and $g_{p,0}=g(q_{k,p},a)$, where $q_{k,p}=m q_k$. For $\mu_p>0$ the printed thresholds are
$$T_{2.22}=\frac{\mu_p\|g_{p,0}\|^2}{4\min(1,M_p^{-2})},\qquad
T_{3.19}=\frac{\mu_p\|g_{p,0}\|^2}{4\max(1,M_p^2)}.$$
They are not equivalent. Proposal: investigate the latter, obtained by subtracting (3.15) from (3.16).
Counterexample to the former: $A=[2],y=[1],\lambda_p=1,\delta_p=10^{-3},a=0$ gives $g_{p,0}=1$.
The minimizer is $z_*=1/4.001$; the maximal possible decrease is $1/(2\cdot4.001)\approx0.124969$,
but $T_{2.22}\approx16.012003$; its condition cannot hold. By contrast $T_{3.19}\approx0.062484$.
Impact: literal (2.22) can loop forever. This counterexample does not validate the rest of the proof.
Neither replacement in normalized units nor transfer to nonnegativity is approved or proved here.
Required tests: this counterexample and finite termination for the eventually justified stopping rule.

## E1 — Nonnegative extension: KKT, working set, and residual

For feasible $x\ge0$, convex optimality is $s=d+\lambda\mathbf1\ge0$ and $x_js_j=0$ for every $j$.
Use the proposed residual $g_j^+=|x_j-\max(x_j-s_j,0)|=|\min(x_j,s_j)|$.
If $x_j>0$, this vanishes exactly when $s_j=0$; if $x_j=0$, it vanishes exactly when $s_j\ge0$.
Thus its zero set is precisely the constrained KKT system; proposed accuracy is $\|g^+\|_2\le\varepsilon$.
Signed (2.1) becomes $W=\{j:x_j\ne0\}\cup\{j:x_j=0,|d_j|>(1-\tau)\lambda\}$ after A1.
For nonnegativity propose $W^+=\{j:x_j>0\}\cup\{j:x_j=0,-d_j>(1-\tau)\lambda\}$.
Reason: at zero only $d_j<-\lambda$ violates constrained KKT; large positive $d_j$ does not.
For $\lambda>0$ and $0<\tau<1$, both sets retain support and include all KKT-violating zero coordinates.
These formulas extend the paper, not merely its shrinkage; no inherited convergence rate is asserted.
Required tests: active/inactive KKT cases, including opposite gradient signs, and constrained scalar solutions.

## A2 — Execution and proposed code boundary

Process batch problems independently, including distinct operators, warm starts, parameters, and stopping states.
Return full source vectors in original order; use `select_columns(W)` for inner operator applications.
This changes scheduling, not a single problem's mathematics; GPU tensor operations remain possible.
Proposed `optim/ir_fista.py` owns explicit outer/inner loops and reuses ISTA/FISTA mathematical helpers.
Existing solvers remain unchanged; CMFPropagation must not silently materialize its measurement dictionary.
Per-problem extraction of batched operators needs a defined implementation; current operators have no batch selector.
Required tests: batched input versus independent calls, warm starts, and dense versus matrix-free CMF results.

## Decisions required before implementation

- Spectral quantities: exact reduced Gram spectrum versus justified bounds; cost and guarantees differ.
- Proximal curvature: (2.15) uses full $A^\top A$, while later criteria use $A_W^\top A_W$; fix the policy explicitly.
- Inner stopping: justify D2, its normalized/nonnegative versions, and the iteration requirement in (2.22).
- Outer stopping: check the returned iterate; Algorithm 5 tests $x^k$ but returns $x^{k+1}$; document any change.
- Lipschitz warm start: reconcile extra $L$ in (2.17) with (2.14); choose initialization and zero-denominator handling.
- Working-set cap/top-K (p. A2000): preserve existing support or redefine the reduced target; never silently drop sources.
- Define feasible warm-start handling, $\lambda=0$ support, empty working sets, and numerical/backtracking tolerances.
- Specify iteration budgets, exhaustion reporting, callback granularity/reset/batch identity, and public parameter names.
No approximation, safeguard, changed selection rule, or convergence claim is implied by this draft.
