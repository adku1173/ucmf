# IR-FISTA-B for the nonnegative LASSO: Phase 1 derivation

Original report ref ab87210, read-only. Nothing was run or written by its author. **Independent-review corrections:** `mathematical_review.md` and specification v2 supersede this report where they differ. The signed (3.7) counterexample is now verified; the wide-A example with delta=1 does not meet literal (2.22) and is replaced in review by delta=.001; coordinatewise Proposition 2 uses the Cartesian-product subdifferential; natural-residual inner termination does not require strong convexity for an unlimited run; dual gap is not the only possible objective certificate with additional information. No stronger claims below are adopted by implementation without the reviewed specification. Sources read:
- `docs/ir_fista.md`.
- Wang et al. 2022 (SISC 44(4), A1989–A2017), full text through §3.2, plus Table 4.1 and 4.2 and the parameter list in §4.
- `ucmf/optim/utils.py::default_gamma_from_operator` and `ista.py::shrink_nonnegative` (convention check only).

**Labels.** Unchanged / Adapted / Conditional / Unproven / Invalid. "Proved" means I give a proof below. "Verified" means a hand-checked numerical example. Nothing here was executed.

---

## 0. Headline findings

1. **Reduction.** On C = R₊ⁿ, ‖x‖₁ = 𝟙ᵀx, so the problem is a smooth convex QP on the orthant:
   F₊(x) = f(x) + λ𝟙ᵀx. The "soft threshold" is `max(z − γλ, 0) = Π_{≥0}(z − γλ𝟙)`.
   - The one-sided KKT conditions are exactly those of a bound-constrained QP.
   - E1's residual and working-set rule are correct (Proved, §2–§3).

2. **A sound IR-FISTA-B⁺ exists.** Global convergence of the KKT residual is proved under two conditions:
   - W ⊇ supp(x) ∪ {j : xⱼ = 0, sⱼ < 0}.
   - The inner solver never returns an F^k-value worse than its first (prox-gradient) step.

   It needs no strong convexity, no eigenvalue/cond information, and no (2.22)-type threshold. The paper's route to this result (Thms 3.5–3.6 with (2.22)) rests on an invalid constant (§9). A one-line majorization argument replaces it (Prop. 4, Thm 1).

3. **Errors found in the paper** (details §9–§11):
   - **(3.7) is false.** The constant min(1, 1/‖H‖) fails. A 2×2 counterexample is given in §3.3. The correct constant is √(1+‖H‖²) (Proved). The printed proof step (3.13) mixes index sets V₁/V₂.
   - **(2.22) has a typo** (min and max swapped relative to (3.15)/(3.19)). The doc's D2 is right about the typo, but T₍₃.₁₉₎ is also not proved, because it inherits (3.7).
   - **Theorem 3.7(iii) is false as stated** for wide A (m < n). With δᵏ = max(σ − λ_min(AᵀA), 0) = σ > 0, the claimed "δᵏ = 0" and "any ρ ∈ (0,1)" fail.
     - Explicit counterexample in §9.4: an exact proximal contraction of 1/2 against a claimed arbitrary ρ.
   - **The complexity O(mp ln 1/ε) (3.41) omits the O(mn) working-set selection per outer iteration.**
     - It also hides cond(H_{W*}), applies only locally, and needs cond to be known.
   - **The top-ℓ cap (p. A2000) can drop support coordinates, which breaks monotonicity.**
     - Fix: retain supp(x), or use a frozen-variable block formulation (§4.4).

4. **No fundamental blocker to a correct, monotone, globally residual-convergent algorithm.** What cannot be guaranteed is listed in §14:
   - No global complexity bound beyond O(1/√K) for min‖G‖.
   - No a-priori bound on |W|.
   - No tolerance-to-accuracy certificate other than the duality gap (§3.5).
   - Full-sequence convergence needs δ > 0 and summable inner errors.
   - Local linear rate and the O(mp ln) cost need strict complementarity / full column rank on the final W / a known rate.

5. **Signed mode is sound** under the same unified framework (§12). Differences: the 𝟙ᵀx identity is lost, and λ = 0 degenerates (every coordinate with dⱼ ≠ 0 enters W).

---

## 1. Problem formulation and scaling

**Data.** A ∈ ℝ^{m×n}, y ∈ ℝᵐ, λ ≥ 0. Each batch row is an independent problem: its own x ∈ ℝⁿ, W, L, θ, tolerances, status. Nothing is shared.

**Objective.** F_C(x) = f(x) + λ‖x‖₁ + I_C(x), f(x) = ‖Ax−y‖²/(2m), g(x) = ∇f(x) = Aᵀ(Ax−y)/m.
- Positive mode: F₊(x) = f(x) + λ𝟙ᵀx on x ≥ 0.
- Define s(x) = g(x) + λ𝟙, the reduced gradient of the smooth surrogate on C.

**Scaling (paper → ucmf)** (F_p = m·F):

| Paper | ucmf |
|---|---|
| λ_p = mλ | λ |
| δ_p = mδ | δ |
| σ_p = mσ | σ |
| L_p = mL | L |
| d_p = m·d | d |
| s_p = m·s | s |
| H_p = m·H | H = AₓᵀAₓ/m + δI |

Consequences:
- Minimizers, proximal steps (thresholds γ_pλ_p = γλ with γ_p = γ/m) and iterates coincide under the conversion.
- The paper's residual (2.4) is not scale-covariant. In positive form it is min(x, s_p) = min(x, m·s) ≠ m·min(x, s). Compared with the natural scale, the paper's implicit step γ_p = 1 corresponds to ucmf step γ = m, i.e. ‖A‖² times the natural step 1/L.
  - **Scale-covariant family:** R_γ(x) := min(x, γ s(x)) is invariant under conversion with γ = mγ_p. Its gradient-mapping form G_γ := R_γ/γ = min(x/γ, s) scales by m.
  - Tolerances must be restated in ucmf units; there is no universal conversion of the paper's 10⁻⁸ ⇒ any recommended ε is a new parameter.
- The proximal weight must be m-scaled to keep the same proximal ratio (see the A = I example, §13).

**λ_max.**
- Signed: λ_max = ‖Aᵀy‖_∞/m.
- Positive: λ_max⁺ = max_j (Aᵀy)_j/m, which may be ≤ 0.
- x* = 0 is optimal iff s(0) = λ𝟙 − Aᵀy/m ≥ 0 iff λ ≥ λ_max⁺ (Proved from §2).
- If λ_max⁺ ≤ 0, x* = 0 for all λ ≥ 0.

**Existence.**
- λ > 0: F₊ ≥ λ‖x‖₁ is coercive, so a minimizer exists.
- λ = 0: F₊ is a convex quadratic bounded below on a polyhedron, so it attains its minimum (Frank–Wolfe/Eaves).
- Uniqueness: Ax* and s* are unique, because f is strongly convex in Ax. Hence θ* = (Ax*−y)/m is unique. x* itself is unique only if the columns on its support are independent; do not assume uniqueness for correlated CMF dictionaries.

---

## 2. Optimality conditions (positive mode)

**Prop. 1 (KKT, Proved).** For x ≥ 0, x minimizes F₊ iff s(x) ≥ 0 and xⱼsⱼ = 0 ∀j. Equivalently:
- xⱼ > 0 ⇒ gⱼ = −λ.
- xⱼ = 0 ⇒ gⱼ ≥ −λ.

Equivalent fixed-point form: x = max(x − γs(x), 0) ∀γ > 0.

*Proof.* F₊ is convex and differentiable on the cone, so the first-order condition is ⟨s, z−x⟩ ≥ 0 ∀z ≥ 0, which is the stated system. The fixed-point form is the projected-gradient/prox form with threshold γλ. ∎

**Violator sets.** V(x) = {j : xⱼ = 0, sⱼ < 0} (zero coordinates that violate KKT). Large positive dⱼ at xⱼ = 0 is not a violation, unlike the signed case (|dⱼ| > λ).

---

## 3. Residuals

### 3.1 Definitions (three objects with distinct roles)

1. **Exact (subgradient) residual**, for x ≥ 0:
   ρ^ex_j(x) = |sⱼ| if xⱼ > 0, and (−sⱼ)₊ if xⱼ = 0.
   - ρ^ex = dist(0, ∂F₊(x)) coordinatewise (∂F₊ = s + N_{ℝ₊ⁿ}(x)).
   - ρ^ex = 0 ⟺ KKT.
   - It is discontinuous in x: xⱼ = 10⁻³⁰ with sⱼ = 5 gives ρ^ex_j = 5.
   - Use it for certificates on prox outputs (§3.2), not as a robust stopping measure.

2. **Natural (scale-γ) residual:** R_γ(x) = x − max(x − γs, 0) = min(x, γ s).
   - Valid for any sign of s (Proved: if x − γs ≥ 0, both sides equal γs; otherwise both equal x).
   - Continuous. R_γ = 0 ⟺ KKT.
   - E1's g⁺ is |R₁| (γ = 1). The claim "zero set = KKT" is Proved (Prop. 1).
   - Gradient-mapping form: G_γ(x) = R_γ/γ = min(x/γ, s), in gradient units.

3. **Duality-gap certificate** (§3.5).

**Relations (Proved).**
- |G_γ,ⱼ| ≤ ρ^ex_j (check cases).
- The reverse fails.
- For L ≥ 1 and any x ≥ 0: |min(Lx, s)| ≥ |min(x, s)|.
- For L ≤ L_ref: |min(Lx, s)| ≥ (L/L_ref)|min(L_ref x, s)| (case split on sign of s).
- Off-W coordinates (xⱼ = 0, sⱼ ≥ τλ ≥ 0) have zero natural residual: min(0, sⱼ/L) = 0.

### 3.2 Residual of a prox-gradient point

**Prop. 2 (Proved).** Let z = max(ẑ − (∇q(ẑ)+λ𝟙)/L, 0) for a quadratic q with Hessian H. Then
ρ^ex(z) ≤ |(H − L·I)(z − ẑ)| coordinatewise.

*Proof.* Prox optimality gives ν ∈ N(z) with ν = −∇q(ẑ) − L(z−ẑ) − λ𝟙. Then
u := ∇q(z) + λ𝟙 + ν = ∇q(z) − ∇q(ẑ) − L(z−ẑ) = (H − LI)(z−ẑ) ∈ ∂Φ(z),
so dist(0, ∂Φ(z)) = ‖ρ^ex(z)‖ ≤ ‖u‖. ∎

### 3.3 Error bound (replaces paper Thm 3.5)

**Prop. 3 (Proved).** Let x* be any minimizer and H a Hessian bound with ‖g(x) − g(x*)‖ ≤ ‖H‖·‖x − x*‖ (in ucmf units M ≤ ‖A‖²/m for the full problem). Then for every γ > 0:
|R_γ(x)ⱼ| ≤ max(|xⱼ − x*ⱼ|, γ|gⱼ(x) − gⱼ(x*)|).
Hence ‖R_γ(x)‖² ≤ (1 + γ²M²)‖x − x*‖², and with γ = 1/M: ‖min(Mx, s)‖² ≤ 2M²‖x−x*‖².

*Proof.* The map (a, b) ↦ min(a, b) is 1-Lipschitz in the ∞-norm:
|min(a,b) − min(a′,b′)| ≤ max(|a−a′|, |b−b′|).
Apply it with (a, b) = (xⱼ, γsⱼ) and (a′, b′) = (x*ⱼ, γs*ⱼ), using min(x*ⱼ, s*ⱼ) = 0 (from KKT) and sⱼ − s*ⱼ = gⱼ(x) − gⱼ(x*). Sum squares: Σ max(uⱼ², γ²vⱼ²) ≤ ‖u‖² + γ²‖v‖². ∎

**Consequence (replaces (3.8)).** For a μ-strongly convex reduced objective with Hessian H:
Φ(z) − Φ* ≥ (μ/2)‖z − z*‖² ≥ μ‖R₁(z)‖²/(2(1+M²)).

**The paper's (3.7) is Invalid, even in the nonneg analog.**
- Counterexample (Verified by hand): n = 2, H = [[0.51, 0.50],[0.50, 0.51]] (eigenvalues 1.01 and 0.01, so ‖H‖ = 1.01).
- Take x* = (1,0), s* = (0,1) (so x*ⱼs*ⱼ = 0 and s* ≥ 0, a valid KKT pair; the linear term is c = s* − Hx* − λ𝟙).
- Take x = (1,1). Then s = s* + H(x−x*) = (0.5, 1.51), R₁ = min(x, s) = (0.5, 1), ‖R₁‖ = √1.25 = 1.118.
- ‖x − x*‖ = 1, but (3.7) claims ‖x−x*‖ ≥ min(1, 1/1.01)·1.118 = 1.107. False.
- (3.8) with the paper's constant is Unproven: the chain through (3.7) is broken, and I found no counterexample to (3.8) itself. The signed-case (3.7) is similarly unjustified; I verified only the nonneg analog.

### 3.4 Dimensional note

The mixed form min(x, s) (E1, paper (2.4)) compares x-units to gradient-units. Prefer scale-aware R_γ/G_γ with γ = 1/L̂ (L̂ is a Lipschitz proxy). `default_gamma_from_operator` already gives a heuristic 1/(1.05‖A‖²/m): the Rayleigh quotient is a lower bound on ‖A‖², so L̂ is not guaranteed ≥ L_f.

### 3.5 Duality-gap certificate (optional, rigorous; Proved)

Dual: max D(θ) = −(m/2)‖θ‖² − yᵀθ subject to Aᵀθ ≥ −λ𝟙.
- Weak duality: P(x) ≥ D(θ) for feasible x ≥ 0 and feasible θ. By Fenchel, ‖w‖²/(2m) ≥ θᵀw − (m/2)‖θ‖², giving P(x) ≥ (Aᵀθ+λ𝟙)ᵀx + D(θ).
- Strong duality: at x*, with θ* = (Ax*−y)/m, D(θ*) = P(x*) iff (g*+λ𝟙)ᵀx* = 0 (complementarity).
- From x, take θ₀ = (Ax−y)/m and θ = αθ₀ with 0 ≤ α ≤ α_max = min over {j : gⱼ < 0} of λ/(−gⱼ), capped at 1 (α_max = 1 if no such j). D(αθ₀) is concave in α, so use α = clip(−yᵀθ₀/(m‖θ₀‖²), 0, α_max).
- Gap G(x, θ) = P(x) − D(θ) ≥ F₊(x) − F*. Cost: one gradient, already available.

This is the only bound from the iterate to the objective value that does not need an unknown error-bound constant.

---

## 4. Working-set rules

### 4.1 Selection (complete rule)

Given x ≥ 0 with s = g + λ𝟙, and τ ∈ [0,1):
- W(x) = supp(x) ∪ N_τ(x), where N_τ(x) = {j : xⱼ = 0, sⱼ < τλ} ⟺ {xⱼ = 0, dⱼ < −(1−τ)λ}.
- This matches E1's W⁺: −dⱼ > (1−τ)λ ⟺ sⱼ < τλ ✓.

**Prop. 5 (Proved).** W(x) ⊇ supp(x) ∪ V(x), and the reduced residual equals the full residual: coordinates outside W have R_γ = ρ^ex = 0 (xⱼ = 0, sⱼ ≥ τλ ≥ 0).

**Corollaries.**
- If x is a minimizer of F^k on the face supp ⊂ W(x), then x is a global minimizer. (Off-W: KKT holds with sⱼ ≥ τλ ≥ 0. On-W: reduced KKT = full KKT, with δ(z−a) = 0 at z = a.)
- W = ∅ ⇒ x = 0 and no violators, so x = 0 is optimal. Check KKT and stop before forming the reduced problem.
- τ = 0 is admissible for correctness: it selects exactly supp ∪ V. τ > 0 adds near-active coordinates (0 ≤ sⱼ < τλ), which matters only for identification and local theory (§9).
- λ = 0: τλ = 0, so W = supp ∪ {dⱼ < 0}, well-defined. In the signed rule the analog (|dⱼ| > 0) degenerates (§12).

### 4.2 Elimination

Elimination of j ∉ W is heuristic (not safe in general): coordinates may be excluded that are positive in x*. Correctness is restored by re-selection each outer iteration (reactivation, §4.3).

**Optional safe elimination (Proved).** Given a dual feasible θ and gap G = P(x) − D(θ):
- ‖θ − θ*‖ ≤ √(2G/m), because D is m-strongly concave and D* − D(θ) ≤ G.
- If a_jᵀθ + λ > ‖a_j‖·√(2G/m), then s*ⱼ > 0 and xⱼ* = 0 for every minimizer. (Complementarity holds for any primal solution paired with the unique θ*.)
- Eliminated columns never need reactivation (Proved).
- Cost: needs column norms ‖a_j‖, which may be expensive for CMF operators.

### 4.3 Reactivation

W is recomputed each outer iteration from the full gradient s(x^{k+1}) (needed anyway for the global stopping test). There is no persistent inactive set, so reactivation is implicit: j re-enters when xⱼ = 0 and sⱼ < τλ. After reduced solves, new violators of the full problem are exactly the zero coordinates with sⱼ < 0 at x^{k+1}.

### 4.4 Cap on |W| (paper p. A2000: "top-ℓ")

- **Paper's rule (choose ℓ indices of W with largest gⱼ): Invalid for monotonicity.** It can set xⱼ = 0 for a support coordinate with small residual. Then F(x^k) ≠ F^k(z⁰) and the descent argument breaks.
- **Option A (recommended): retain support, cap only additions.** W = supp(x) ∪ Sel, where Sel are the ℓ most negative sⱼ (violators first, ranked by −sⱼ; near-active entries 0 ≤ sⱼ < τλ only if capacity remains).
  - Descent constant degrades by a factor ≥ ℓ/|V| ≥ ℓ/n: the top-ℓ sum of squares is ≥ (ℓ/|V|) of the total. Thm 1 still holds with constant ×ℓ/n.
- **Option B (block coordinate form, as in PBC-ℓ₁-Hom Alg. 2):** coordinates outside W are frozen at their current (possibly positive) values. Reduced gradient: ∇q(z) = A_Wᵀ(r_k + A_W(z−a))/m + δ(z−a), with r_k = Ax^k − y. Both options use this residual form. Monotonicity holds (Gauss–Southwell block descent).

---

## 5. Reduced problem

For W = W(x^k), a = x_W^k ≥ 0, r_k = Ax^k − y:

  Φ_k(z) = q_k(z) + λ𝟙ᵀz + I_{z≥0}, z ∈ ℝ^{|W|},
  q_k(z) = ‖r_k + A_W(z − a)‖²/(2m) + (δ/2)‖z − a‖².

- Hessian: H = A_WᵀA_W/m + δI ⪰ δI.
- Gradient: ∇q_k(z) = A_Wᵀ(r_k + A_W(z−a))/m + δ(z−a). At z = a: ∇q_k(a) = g_W(x^k), reusing the outer gradient.
- F^k(a) = F(x^k) exactly (support retained).
- F(x^{k+1}) = Φ_k(z) − (δ/2)‖z − a‖² when x^{k+1} = (z, 0).

**Strong convexity.** λ_min(H) ≥ δ.
- For the paper's δᵏ = max(σ − λ_min(AᵀA), 0), interlacing gives λ_min(A_WᵀA_W) ≥ λ_min(AᵀA), hence λ_min(H) ≥ σ. (Unchanged, Proved.)
- The paper's δᵏ is not computable cheaply. Recommendation: fixed δ ≥ 0, with δ > 0 only if needed (§9: Thm 2 and finite inner termination need it).

---

## 6. Inner solver: FISTA-B on the reduced problem

**Prox (positive).** z⁺ = max(ẑ − (∇q_k(ẑ) + λ𝟙)/L, 0) = S⁺_{λ/L}(ẑ − ∇q_k(ẑ)/L), where S⁺_{γλ}(v) = max(v − γλ, 0). The −δẑ/L correction is the D1 fix (the extra δ(ẑ − a) term in ∇q_k) and is essential.

**Backtracking test** (exact for a quadratic):
  q_k(z⁺) ≤ q_k(ẑ) + ⟨∇q_k(ẑ), z⁺−ẑ⟩ + (L/2)‖z⁺−ẑ‖²
  ⟺ ‖A_W d‖²/m + δ‖d‖² ≤ L‖d‖², d = z⁺ − ẑ.
- It needs only A_W d; the nonsmooth terms cancel, so the test is on q_k alone.
- L ≥ L_prev within an inner run (monotone nondecreasing, required by FISTA-B theory); η > 1; L_floor > 0.
- If d ≠ 0, any accepted L ≥ Rayleigh quotient of H along d ≥ λ_min(H) ≥ δ. If d = 0, the test is 0 ≤ 0 and implies no curvature lower bound (independent review N3).
- Extrapolated ẑ may be negative; harmless, since q_k is defined on all ℝ^{|W|}, and the prox restores feasibility.

**Momentum:** θ₁ = 1, z⁰ = a, ẑ¹ = a; θ_{t+1} = (1 + √(1+4θ_t²))/2; ẑ^{t+1} = z^t + ((θ_t − 1)/θ_{t+1})(z^t − z^{t−1}).
Images are affine in ẑ: res(ẑ) = (1+β)res(z^t) − β res(z^{t−1}), so no extra forward products are needed.

**Cost per iteration:** one forward (A_W d, inside the backtracking test) + one adjoint (A_Wᵀ res/m). The paper's O(m|W|) per iteration holds.

**Monotone safeguard S1 (new, recommended).** Track F^k(z^t) = ‖res‖²/(2m) + (δ/2)‖z−a‖² + λ𝟙ᵀz (O(m + |W|) per iteration) and return z_best = argmin over visited, or at least require F^k(z_out) ≤ F^k(z¹). Alternatively use MFISTA.

**Lipschitz init.** The paper's (2.14) has S⁰ = 0 ⇒ L₀¹ = 0, and the extra L in (2.17) is garbled (doc item). Backtracking tolerates any L₀ > 0. Suggestions:
- L₀ᵏ = max(L_floor, L_end^{k−1}/η), or
- the Rayleigh quotient dᵀHd/‖d‖² along the last inner step (free from A_W d), times a safety factor.

---

## 7. Stopping criteria

**Inner (S2).** Stop when ‖ρ^ex_red(z^t)‖ ≤ tol_k, with tol_k = max(ω·‖G_γ(x^k)‖, tol_floor), ω ∈ (0,1) (ω = 0.1–0.3), plus a budget t_max(k) and the safeguard S1.
- Cheap trigger: ν_t = L_t‖z^t − ẑ^t‖ (free). When ν_t ≤ tol_k, verify with the exact ρ^ex (one extra adjoint). Prop. 2 gives ρ^ex(z^t) ≤ ‖(H−L)(z^t−ẑ)‖ ≤ max(L−δ, M−L)·‖z^t−ẑ‖.
- The reduced ρ^ex differs from the full-problem ρ^ex on W by at most δ‖z−a‖ (the δ(z−a) term).
- For summable errors (Thm 2) use tol_k ≤ δ·ε_k with Σε_k < ∞ (e.g. c/(k+1)²), since ‖z − z*‖ ≤ ‖ρ^ex(z)‖/λ_min(H) ≤ ‖ρ^ex(z)‖/δ.
- If tol_k is not reached by t_max, return the S1-safeguarded point (status "inner budget").

**Global.** Computed at the returned iterate x^{k+1} (fixes Alg. 5, which tests x^k but returns x^{k+1}); the full gradient is needed for the next W anyway:
- (G1) ‖G_γ(x)‖₂ ≤ ε_abs + ε_rel·scale (scale e.g. ‖G_γ(x⁰)‖).
- (G2) optional: duality gap G(x, θ) ≤ ε_gap·max(1, |F(x)|).
- (G3) outer budget exhausted; (G4) stagnation: |F(x^{k+1}) − F(x^k)| below a numerical noise floor, flagged with a status. Per-row status flags are independent.
- Test before the first iteration: if x = 0 and s(0) ≥ 0 (λ ≥ λ_max⁺), return immediately.

**Warm start.** Project x⁰ ← max(x⁰, 0); do not silently drop. Tiny positive entries enter supp and inflate W (no safe hard-thresholding).

---

## 8. Pseudocode (per batch row; positive mode)

```
Input: A (m×n), y, λ ≥ 0, x0, τ ∈ [0,1), δ ≥ 0, η > 1, ω ∈ (0,1),
       L_floor > 0, t_max(k), ℓ (optional cap, retain support), ε, γ_ref = 1/L_ref
x ← max(x0, 0); res ← A x − y; g ← Aᵀ res / m; s ← g + λ
for k = 0, 1, ...:
    # global stop (full-problem quantities at x)
    if ‖min(x/γ_ref, s)‖ ≤ tol  or gap criterion:  return x, status = converged
    W ← supp(x) ∪ {j : x_j = 0, s_j < τλ}   (cap: keep supp, add ≤ ℓ most negative s_j)
    if W = ∅: return x (x = 0, KKT holds)
    a ← x_W; ĝ_W ← g_W; z⁰ ← a; ẑ ← a; θ ← 1; res_ẑ ← res; L ← L_start
    F1 ← F^k(z¹) (after first step), z_best ← z¹
    for t = 1, 2, ...:
        ∇ ← A_Wᵀ res_ẑ/m + δ(ẑ − a)                 # for t=1 equals g_W
        repeat:                                      # backtracking (L nondecreasing)
            z⁺ ← max(ẑ − (∇ + λ)/L, 0); d ← z⁺ − ẑ
            Ad ← A_W d
            if ‖Ad‖²/m + δ‖d‖² ≤ L‖d‖²: break
            L ← ηL
        res_{z⁺} ← res_ẑ + Ad; evaluate F^k(z⁺); update z_best
        β ← (θ − 1)/θ_new; ẑ ← z⁺ + β(z⁺ − z_prev); res_ẑ ← affine combination
        θ ← θ_new
        if ν_t = L‖d‖ ≤ tol_k and ‖ρ^ex(z⁺)‖ ≤ tol_k, or t ≥ t_max(k): break
    z_out ← z⁺ if certified and F^k(z⁺) ≤ F^k(z¹), else z_best
    x_W ← z_out  (support-retained: x_{W^c} = 0 stays; frozen variant leaves x_{W^c} unchanged)
    res ← res + A_W(z_out − a)                       # incremental; recompute from A x − y periodically
    g ← Aᵀ res / m; s ← g + λ                         # one full adjoint: needed for next W and stop test
```

---

## 9. Convergence analysis

### 9.1 Statements and proofs

**Prop. 4 (one-step descent; Proved).** Suppose first inner step z¹ = prox-gradient step from z⁰ = a passes the backtracking test at a with accepted L₁. Then
Φ_k(a) − Φ_k(z¹) ≥ (L₁/2)‖a − z¹‖² = (L₁/2)‖R_{1/L₁}(x^k)‖².

*Proof.* The accepted test gives Φ_k(z¹) ≤ Q_{L₁}(z¹, a). Q_{L₁}(·, a) is L₁-strongly convex with minimizer z¹, so Q_{L₁}(a, a) ≥ Q_{L₁}(z¹, a) + (L₁/2)‖a − z¹‖². Also Q_{L₁}(a, a) = Φ_k(a). By Prop. 5 the off-W entries contribute zero. ∎

**Theorem 1 (global residual convergence; Adapted, Proved).** Assume:
- λ ≥ 0, F bounded below (F ≥ 0).
- W^k ⊇ supp(x^k) ∪ V(x^k) (or the capped variant of §4.4 with factor ℓ/n).
- The inner solver returns z_out with Φ_k(z_out) ≤ Φ_k(z¹) (S1).
- Accepted L₁^k ∈ [L_min, L_max] (L_min ≥ L_floor, L_max ≤ max(L₀, η‖A‖²/m + ηδ)).

Then F(x^k) is nonincreasing and
F(x^k) − F(x^{k+1}) ≥ (L₁^k/2)‖R_{1/L₁^k}(x^k)‖² + (δ/2)‖x^{k+1} − x^k‖²
 ≥ (L_min/(2L_max²))‖G_{1/L_max}(x^k)‖² + (δ/2)‖x^{k+1} − x^k‖².
Hence Σ_k ‖G_{1/L_max}(x^k)‖² ≤ 2L_max²(F(x⁰) − F*)/L_min, so ‖G(x^k)‖ → 0 and min_{k<K}‖G(x^k)‖ ≤ O(1/√K). Every limit point is a minimizer (R continuous). No strong convexity, no eigenvalues, no (2.22).

*Proof.* Combine Prop. 4 with F(x^{k+1}) = Φ_k(z_out) − (δ/2)‖z_out − a‖² ≤ Φ_k(z¹) − (δ/2)‖z_out − a‖² and F(x^k) = Φ_k(a); then use the residual comparison from §3.1 (L-scaling, L ≤ L_max). ∎

*Note:* this improves the paper's (3.18), which drops the δ/2 term, and avoids (3.7).

**Theorem 2 (convergence of the whole sequence; Adapted, new proof).** Assume λ > 0, δ > 0, and x^{k+1} = (z_out, 0) with ‖z_out − T_k(x^k)‖ ≤ e_k, T_k the exact proximal solution of Φ_k, Σe_k < ∞. Then x^k → x̄, a minimizer.
- For exact solves (e_k = 0) this is the paper's Thm 3.7(i), obtained here without strict complementarity or PD assumptions.

*Proof.*
1. F ≥ λ‖x‖₁ gives bounded iterates, hence a cluster point x̄. By Thm 1, ‖G(x̄)‖ = 0, so x̄ is a minimizer. Let m₁ = min positive entry of x̄ (∞ if x̄ = 0).
2. Choose K with ‖x^K − x̄‖ + Σ_{k≥K} e_k < m₁ (possible: a subsequence → x̄ and tails → 0).
3. Induction: if ‖x^k − x̄‖ < m₁ then supp(x̄) ⊂ supp(x^k) ⊂ W^k. So x̄ is feasible for Φ_k's face and minimizes F on the cone, hence on the face, so T_k(x̄) = x̄.
4. T_k is a proximal map (single-valued, nonexpansive, for δ > 0), so ‖T_k(x^k) − x̄‖ ≤ ‖x^k − x̄‖, giving ‖x^{k+1} − x̄‖ ≤ ‖x^k − x̄‖ + e_k.
5. This quasi-Fejér property plus a subsequence → x̄ gives x^k → x̄. ∎

- Without δ > 0 (δ = 0): Unproven (T_k is not a prox map).
- λ = 0: Conditional on bounded iterates (iterates may be unbounded along null directions).
- The relative rule S2 (tol_k = ω‖G‖) alone does not give summable e_k: Unproven. Use the summable variant of §7 for the theory.

**Theorem 3 (finite inner termination; Proved).** If δ > 0, FISTA-B on Φ_k gives z^t → z*, ẑ^t → z*, hence ‖z^t − ẑ^{t−1}‖ → 0, hence ρ^ex_red(z^t) ≤ ‖(H − L_t)(z^t − ẑ^{t−1})‖ → 0 (Prop. 2). So S2 holds in finitely many iterations for any tol_k > 0.
- Also: positive coordinates of z^t at strictly complementary zeros of z* must have |s| tiny. Hence the iterates identify zeros automatically once they converge; no separate identification result for FISTA is needed.
- Without strong convexity (δ = 0, singular A_W): convergence of the iterates of original FISTA is open; the budget t_max is the only guarantee. Unproven.

**Theorem 4 (local linear rate; Conditional).**
- (a) Exact proximal solves, δ > 0, strict complementarity (xⱼ* = 0 ⇒ sⱼ* > 0). Then:
  - x^{k+1}_j = 0 for strictly complementary zeros for k ≥ K: if zⱼ > 0 then sⱼ(x^{k+1}) = −δ(zⱼ − aⱼ) ≈ 0, contradicting sⱼ* > 0 for x^{k+1} close to x*.
  - On the support, (H_{W₁*}+δI)(x^{k+1} − x*) = δ(x^k − x*), so contraction ≤ δ/(δ + s_min), s_min = smallest positive eigenvalue of A_{W₁*}ᵀA_{W₁*}/m (paper (3.39)), as the paper's Lemma 3.2 applies to powers of a fixed operator. Adapted from paper (ii).
- (b) Inexact solves with δ = 0, W^k ≡ W̄ := supp(x*) ∪ {j : s*ⱼ < τλ} (needs no coordinate with s*ⱼ = τλ), and A_W̄ᵀA_W̄ ≻ 0. Then FISTA-B (Thm 3.4 analog: ‖z^t − z*‖ ≤ 2√(η·cond(H))/(t+1)·‖z⁰ − z*‖) gives ‖x^{k+1} − x*‖ ≤ ρ‖x^k − x*‖ if t̄ ≥ 2√(η cond)/ρ − 1.
  - Degenerate coordinates (x* = 0, s* = 0) are automatically in W̄ for λ, τ > 0. Strict complementarity is needed only for identification of the excluded zeros, and the inner accuracy tol_k < min over those excluded sⱼ* − τλ suffices.
- Requires: knowing cond(H_{W̄}) to enforce the iteration count, and |W̄| ≤ m for full column rank.

### 9.2 Classification of the paper's statements

| Statement | Class | Note |
|---|---|---|
| (2.4) residual as KKT measure | Adapted | Positive version E1/R_γ; zero set proved; dimensionally mixed (§3.4) |
| Prop 2.1 (descent with δ/2 term) | Unchanged | Needs F^k(z_out) ≤ F^k(a); indicator is harmless (z ≥ 0) |
| Lemma 3.1 (strong-convexity growth) | Unchanged | 0 ∈ ∂Φ(z*) |
| Lemma 3.2 (powers of H) | Unchanged | |
| Lemma 3.3 / Thm 3.4 (FISTA-B rates) | Adapted | Composite prox of λ𝟙ᵀz + I_{≥0}; constant max(L₀, ηM) (the paper ignores L₀) |
| Thm 3.5 (3.6) | Adapted | New 1-Lipschitz proof |
| Thm 3.5 (3.7) | **Invalid** | Counterexample §3.3; correct constant √(1+‖H‖²) |
| Thm 3.5 (3.8) | Unproven (paper constant); Adapted with (1+M²) | |
| (2.22) | **Invalid as printed** (min/max typo) | Doc D2 correct on the typo; T₍₃.₁₉₎ also unproved |
| (3.15)–(3.19), Thm 3.6 | Adapted | With corrected constants; (3.20) should read 4(1+(‖A‖²/m+δ)²)/σ, not 4max(1,‖AᵀA‖)/δᵏ |
| Thm 3.7(i) | Adapted / Conditional | Exact (or summably inexact) solves, δ > 0, bounded iterates; new proof (Thm 2) |
| Thm 3.7(ii) | Conditional | Strict complementarity; exact solves |
| Thm 3.7(iii) | **Invalid as stated**; Conditional | §9.4 |
| (3.41) complexity | **Invalid as stated**; Conditional | §10 |
| (3.42) FISTA comparison | Not a convergence claim | Compares objective-gap vs distance error; FISTA is also linear when strongly convex |
| Top-ℓ cap | **Invalid** for monotonicity | §4.4 |
| δᵏ ⇒ μ ≥ σ | Unchanged | Interlacing |

### 9.3 Non-obvious points about δ

- Exact proximal solves contract at best δ/(δ + s_min). If δ ≫ s_min (ill-conditioned dictionary), the outer iteration is nearly stalled. A small δ ≪ s_min is needed for speed, but then strong convexity is weak and cond(H) ~ ‖H‖/δ is large. There is no universal choice.
- With δ = 0: Thm 1 holds unchanged; Thm 2 and Thm 3 are lost.

### 9.4 Counterexample to (iii) for wide A (Verified by hand)

m = 1, n = 2, A = [1, 0.5], y = 1, 0 < λ < 1, σ = 1.
- AᵀA is rank 1, so λ_min = 0 and the paper's δᵏ = σ = 1 > 0.
- x* = (1−λ, 0): a₁ᵀ(Ax*−y) = −λ ✓ (active); |a₂ᵀ(Ax*−y)| = 0.5λ < λ ✓ (strictly complementary). W* = {1}, A_{W*}ᵀA_{W*} = 1 ≻ 0.
- Near x*, W^k = {1}. The reduced prox step is exact (one-dimensional quadratic with L = 1+δ = 2): x^{k+1}₁ = (1−λ+δx^k₁)/(1+δ).
- So x^{k+1}₁ − x*₁ = 0.5(x^k₁ − x*₁): exact contraction 1/2, contradicting (3.22) for ρ < 0.5.
- The "δᵏ = 0" in the paper's proof of (iii) is wrong for wide A. The statement is valid only with δ → 0 or ρ ≥ δ/(δ+s_min).
- With the paper's default σ = δ = 10⁻³ (paper units), the floor is ≈ δ/(δ+s_min): small for well-conditioned, large for ill-conditioned problems.

---

## 10. Complexity

**Per-iteration costs** (dense A, per batch row):
- Outer (selection + stopping): one full adjoint, O(mn). Ax^{k+1} is updated incrementally with the O(m|W|) reduced forward.
- Inner: O(m|W|) per iteration (forward + adjoint). With a precomputed Gram matrix the setup is O(m|W|²) and the per-iteration cost O(|W|²).

**Local complexity (Conditional on Thm 4(b) and cond known).**
Inner iterations per outer: T_in = ⌈2√(η·κ)/ρ⌉ − 1, κ = cond(A_W̄ᵀA_W̄/m + δI).
Outer iterations to reduce the error from ε₁ to ε: K = ln(ε₁/ε)/ln(1/ρ).
Total flops:
  O( m·(n + |W̄|·√(ηκ)/ρ)·ln(ε₁/ε)/ln(1/ρ) ).

- The paper's (3.41) O(mp ln 1/ε) drops the O(mn) selection per outer iteration, hides √κ, and is local only. The selection term dominates unless √κ ≳ n/p.
- Minimizing 1/(ρ ln(1/ρ)) over ρ gives ρ = 1/e (a constant-factor remark).
- There is no global (pre-identification) complexity statement; the only global rate is min_{k<K}‖G‖ ≤ O(1/√K) (Thm 1).
- The size |W̄| = |supp x*| + |degenerate zeros| + |{0 < s* < τλ}|, not p alone, and is not bounded a priori (§14).

---

## 11. Differences from the original / the doc

| Item | Original | Here |
|---|---|---|
| Prox | soft threshold S_{γλ} | max(z − γλ, 0) (D1 correction carried over) |
| KKT | two-sided | one-sided: xⱼ = 0 ⇒ gⱼ ≥ −λ |
| Residual | (2.4), mixed units | R_γ / G_γ (scale-covariant), ρ^ex (certificate), gap |
| W rule | (2.1) | supp ∪ {sⱼ < τλ} (E1, Proved correct) |
| Global convergence | via (2.22), cond-based thresholds | via monotone safeguard + Prop. 4 (no cond needed) |
| (3.7) | claimed | Invalid; replaced by Prop. 3 |
| Cap | top-ℓ of W | retain supp, or frozen-variable block |
| δᵏ | max(σ − λ_min(AᵀA), 0) | fixed δ ≥ 0 |
| Stop test | tests x^k, returns x^{k+1} | tests the returned iterate |
| Complexity | O(mp ln 1/ε) | O(m(n + |W̄|√(ηκ)/ρ) ln(ε₁/ε)/ln(1/ρ)), local |

**Doc remarks.**
- D1: correct.
- D2: the (2.22) counterexample is verified (A = [2], y = 1, λ_p = 1, δ_p = 10⁻³: decrease ceiling 1/(2·4.001) = 0.124969 vs T₍₂.₂₂₎ ≈ 16.012). But T₍₃.₁₉₎ ≈ 0.0625 also relies on (3.7); the rigorous replacement threshold is μ‖R₁‖²/(4(1+M²)) = 4.001·1/(4·17.008) ≈ 0.0588 (feasible; decrease available 0.1250).
- E1: correct and complete for the zero set and the W⁺ claims; no convergence claims in E1 are supplied, which this report provides.
- A1: the claim "generally g_p ≠ m·g" is correct (§1).
- A2 and "Decisions required" items are addressed in §6–§8 and §14.

---

## 12. Signed mode (sound under the same framework)

Same structure with C = ℝⁿ:
- Prox: soft threshold S_{γλ}.
- Exact residual: ρ^ex_j = |dⱼ + λ sign(xⱼ)| if xⱼ ≠ 0; (|dⱼ| − λ)₊ if xⱼ = 0.
- Natural residual: R_γ = x − S_{γλ}(x − γg). The map (a, b) ↦ a − S(a − γb) has partial derivatives (1−σ, γσ) with σ ∈ [0,1], so |R_γ(x)ⱼ − R_γ(x*)ⱼ| ≤ max(|uⱼ|, γ|vⱼ|) and Prop. 3 carries over.
- W = supp(x) ∪ {xⱼ = 0, |dⱼ| > (1−τ)λ} ⊇ supp ∪ violators (|dⱼ| > λ).
- Thms 1, 2, 3 carry over. Thm 2's step 3 uses the nonzero pattern in place of the sign.
- Differences: the identity ‖x‖₁ = 𝟙ᵀx is unavailable; λ = 0 degenerates (|dⱼ| > 0 selects almost every coordinate).
- Local theory requires |d*ⱼ| < λ at zeros (strict complementarity).

---

## 13. Special and limiting cases

- **A = I (m = n):** f = ‖x − y‖²/(2n); x* = max(y − nλ, 0) = S⁺_{γλ}(y) with γ = 1/L = n.
  - From x⁰ = 0: W = {yⱼ > n(1−τ)λ}.
  - With δ = 0 and L = 1/n the first inner step is exact (one outer iteration).
  - With δ > 0 the exact subproblem is max((y − nλ + nδ·x^k)/(1+nδ), 0); outer contraction nδ/(1+nδ). In paper units nδ = δ_p, so δ must be m-scaled to match the paper's ratio.
  - λ = 0: x* = max(y, 0).
- **λ ≥ λ_max⁺:** x* = 0, immediate return at k = 0.
- **λ = 0 (NNLS):** W = supp ∪ {dⱼ < 0} is well-defined. The iterates may be unbounded along null directions of A on the cone (Thm 2 Conditional); residual convergence (Thm 1) still holds.
- **Scalar check** (A = [2], y = 1, m = 1, λ = 1): s(0) = −1 < 0, a violator; x* = 1/4; R₁(0) = −1, so ‖R₁‖ = 1.
- **m < n (typical):** A_WᵀA_W is singular once |W| > m, so strong convexity comes only from δ.
- **Batch:** all per-row state (W, L, θ, tolerances, status) is independent; only the tensor operations are batched (padding/masks).

---

## 14. Missing guarantees and blockers

No blocker prevents an implementable, correct, monotone algorithm with proved residual convergence (Thm 1). What cannot currently be guaranteed:

1. **Global rate/complexity.** Only min‖G‖ = O(1/√K); no global iteration count that exploits sparsity.
2. **|W| bound.** No a-priori bound on |W^k| (initial iterates, degenerate zeros, correlated dictionary). A cap changes the constants (ℓ/n) and can slow convergence arbitrarily.
3. **Accuracy certificate.** Residual → iterate error needs a Hoffman constant, which is unknown. The only rigorous certificate is the duality gap (§3.5).
4. **Full-sequence convergence with inexact inner solves** needs δ > 0 and summable errors; the relative rule alone is unproven. δ = 0: unproven.
5. **Local linear rate and the O(mp ln) cost** need: full column rank of A on W̄ (hence |W̄| ≤ m), strict complementarity (identification), known/estimated cond, and δ small relative to s_min. For correlated CMF dictionaries these are unlikely to hold with usable constants.
6. **FISTA iterate convergence/identification without strong convexity** is open.
7. **ρ-driven stop count in (2.22)** requires cond(H), which is expensive; replaced by S2 (certificate) plus budget.
8. **Floating-point:** residual noise floor ≈ u·‖A‖‖res‖/m; tiny positive entries; Lipschitz heuristics (Rayleigh lower bound) are not guarantees; backtracking covers correctness of the inner steps.
9. **Unproven (never established):** the paper's heuristic for L₀ (S(A_W) ratio), and the claim that the top-ℓ working-set cap preserves any convergence statement.

---

## 15. Tests recommended (numerical, from the derivations)

- D2 scalar: A = [2], y = 1, λ = 1, δ = 10⁻³: reduced minimizer 1/4.001, max decrease 0.124969; check the inner stop is reachable.
- (3.7)-analog counterexample (§3.3): H, x*, s*, x as given; verify ‖R₁‖ = √1.25.
- §9.4 wide-A example: contraction exactly 1/2 for σ = δ = 1.
- A = I closed form; λ ≥ λ_max⁺; λ = 0.
- KKT cases at xⱼ = 0 with sⱼ ≥ 0 and sⱼ < 0; support coordinates with sⱼ ≠ 0.
- Scaling invariance of R_γ under (λ, δ, L, γ) → (mλ, mδ, mL, γ/m).
- Monotone F^k and F with the S1 safeguard; no increase when the cap retains support.
- Batch vs independent calls; dense vs matrix-free results.

---