# Literature report: IR-FISTA-B (Wang et al. 2022) and transfer to nonnegative LASSO

Phase-1 read-only investigation for the ucmf IR-FISTA-B adaptation (ref ab87210).
Prepared by the literature-researcher role. No source, test, or `docs/` files were modified; this file is the only artifact written.

## 0. Sources, access, and verification legend

- **Primary paper (local):** `~/mendeley_papers/Wang/Wang2022.pdf` (read via its cached OCR translation `~/mendeley_papers/Wang/Wang2022/Wang2022.md`, 29 pages, A1989–A2017).
- **Primary paper (cross-check):** author-hosted official PDF `https://numbda.cs.tsinghua.edu.cn/papers/sisc22.pdf` (fetched and text-extracted independently; identical content; used to confirm exact printings of (2.17), (2.19), (2.22)).
- **Other local PDFs found by probing `~/mendeley_papers/`:** `Beck/Beck2009.pdf`, `Efron/Efron2004.pdf`, `Parikh/Parikh2014.pdf`. Probed but **not present** under the probed filenames: Wang2020 (PBC-ℓ1-Hom), Li2018 (SSNAL), Bro1997, Ma2023, Nadisic2022, Rakotomamonjy2022, Esser2013, Benvenuto2009, Asif2014, Fang2020, Berg2008, Scheinberg2014, Lawson1974, Cao2024, Tibshirani1996, Wright2009, Daubechies2004, Wen2010, Figueiredo2007, Kim2018 (absence of a probed filename does not prove absence under another name).
- **Zotero MCP status this run:** search endpoints respond but item-data endpoints (`get_item_metadata`, `get_recent`, advanced search, fulltext) return `[Errno 111] Connection refused`; `add_by_doi` is blocked ("local-only mode … add ZOTERO_API_KEY"). Search returned only unverifiable "Untitled" placeholder keys, which I do **not** cite. **Consequence: no Zotero item keys could be confirmed this run; all Zotero-key fields below are marked "not retrievable this run".** DOIs were verified via web sources instead.
- **Verification tags:** `[PDF×2]` = verified verbatim in both PDF copies of Wang et al. 2022; `[PDF]` = verified in the local PDF only; `[MATH]` = verified by my own derivation/arithmetic (not a quotation); `[CODE]` = verified by reading ucmf source (read-only); `[WEB]` = metadata verified via web (DOI/journal pages); `[INF]` = my inference, not a source statement; `[OPEN]` = unresolved.

---

## 1. Primary source: Wang, Yu, Liang, Wu, Yu (2022) — complete algorithm anatomy with exact numbers

**Reference:** G. Wang, W. Yu, X. Liang, Y. Wu, B. Yu, "An Iterative Reduction FISTA Algorithm for Large-Scale LASSO," *SIAM J. Sci. Comput.* 44(4), A1989–A2017 (2022), DOI 10.1137/20M1374328. Zotero item key: not retrievable this run. Local PDFs: `~/mendeley_papers/Wang/Wang2022.pdf`; official copy `numbda.cs.tsinghua.edu.cn/papers/sisc22.pdf`.

### 1.1 Problem, scaling, conventions `[PDF×2]`

- **(1.1):** `min_{x∈R^n} F(x) := f(x) + λ Ω(x)`; LASSO when `f(x) = ½‖Ax − b‖²`, `Ω(x)=‖x‖₁`, `A ∈ R^{m×n}` (m samples/rows, n features/columns), `λ > 0` pregiven. `‖·‖` is the l2-norm throughout; `f` is L-Lipschitz smooth, i.e. `‖AᵀA‖ ≤ L`.
- KKT conditions: **(1.7)** `A_jᵀ(A x̄ − b) + λ sign(x̄_j) = 0, j ∈ W`; **(1.8)** `|A_jᵀ(A x̄ − b)| ≤ λ, j ∈ W^c`.
- LARS solution path **(1.5)–(1.6)**; l1-homotopy **(1.9)**; PBC-ℓ1-Hom block selection **(1.10)–(1.12)**, Algorithm 2.
- Key reduction identity **(1.13):** for any `V ⊇ W* = {j : x*_j ≠ 0}`, `x*_V = argmin_{z∈R^{|V|}} ½‖A_V z − b‖² + λ‖z‖₁` — the formal justification for working-set reduction.

### 1.2 IRA outer loop (working set, elimination, reactivation, stopping) `[PDF×2]`

- Working-set selection **(2.1)**: `W₁(x^k) = {j : x_j^k ≠ 0}`, `W₂(x^k) = {j : x_j^k = 0 and |A_jᵀ(Ax^k − b)| > (1−τ)λ}`, `W(x^k) = W₁ ∪ W₂`, with `τ` a small positive number. (W₁ = retained support; W₂ = **reactivation** of zero coordinates whose KKT inequality (1.8) is nearly violated.)
- Reduced subproblem **(2.2)**: `x^{k+1}_{W^k} = argmin_{z∈R^{|W^k|}} ½‖A_{W^k} z − b‖² + λ‖z‖₁`; **elimination**: `x^{k+1}_{(W^k)^c} = 0`.
- **Algorithm 3 (IRA):** loop with **(2.3)** `W₁^k, W₂^k, W^k, L_k := ‖A_{W^k}ᵀ A_{W^k}‖`; approximately solve (2.2); set `x^{k+1}_{W^k} = z^t`, complement 0; **outer stop:** `if ‖g(f, x^k)‖ < ε_tol then Break`; **`Return x* = x^{k+1}`** — i.e. the test is applied to `x^k` while the returned iterate is `x^{k+1}` (verified print; also in Algorithm 5).
- KKT/optimality residual **(2.4)**: for `x_j ≠ 0`, `g_j(f,x) = max{ |∂f/∂x_j(x)| − λ, min{ |∂f/∂x_j(x) + λ sign(x_j)|, |x_j| } }`; for `x_j = 0`, `g_j(f,x) = max{0, |∂f/∂x_j(x)| − λ}`. Paper states: "`g_j(f,x)=0` is equivalent to (1.7)–(1.8)"; `‖g(f,x*)‖` measures solution precision.
- Extensions: l1−2 minimization **(2.5)–(2.6)**; adaptive LASSO **(2.7)**. **No nonnegative/box-constrained variant exists in this paper** `[PDF×2]`.

### 1.3 Inner solver: FISTA-B (backtracking FISTA) `[PDF×2]`

- **Algorithm 1 (FISTA):** **(1.2)** `x^t = P_L(y^t) := argmin_{x} (L/2)‖x − (y^t − (1/L)Aᵀ(Ay^t − b))‖² + λ‖x‖₁` (note: the square **is present** here); **(1.3)** `θ_{t+1} = (1+√(1+4θ_t²))/2`; **(1.4)** `y^{t+1} = x^t + ((θ_t−1)/θ_{t+1})(x^t − x^{t−1})`; `y¹=x⁰, θ₁=1`.
- Majorization model **(2.8)**: `Q_L(x,y) = f(y) + ⟨x−y, ∇f(y)⟩ + Ω(x) + (L/2)‖x−y‖²`; `P_L(y) = argmin_x Q(x,y)`.
- **Backtracking test (2.9)** (Algorithm 4, FISTA-B): find smallest nonnegative integer `i_t` such that with `L̄ = η^{i_t} L_{t−1}`, `F(P_L̄(y^t)) ≤ Q_L̄(P_L̄(y^t), y^t)`; set `L_t = L̄`, `x^t = P_{L_t}(y^t)`, momentum **(2.10)**. Guaranteed once `L̄ ≥ L(f)`: `L(f) ≤ L̄ ≤ η L(f)`.
- Lipschitz warm start across outer iterations: `L_t^k` = value of `L̄` at inner iteration `t` of subproblem `k`; **(2.11)** `‖A_{W^k}ᵀA_{W^k}‖ ≤ L^k_{t̄^k} ≤ η‖A_{W^k}ᵀA_{W^k}‖`; column-sum heuristic **(2.12)** `S(A) = Σ_{i=1}^{|A|} ‖A_i‖`, **(2.13)** `‖A_{W^{k+1}}ᵀA_{W^{k+1}}‖ ≈ (|S(A_{W^{k+1}})|/|S(A_{W^k})|)‖A_{W^k}ᵀA_{W^k}‖`; **(2.14)** `L_0^{k+1} = S^k L^k_{l^k}` with `S^k = (1−α)S^{k−1} + α |S(A_{W^{k+1}})|/(η|S(A_{W^k})|)`, `S⁰ = 0`, `α > 0`.

### 1.4 Proximal regularization of subproblems `[PDF×2]`

- **(2.15)** `f^k(z) = ½‖A_{W^k} z − b‖² + (δ^k/2)‖z − x^k_{W^k}‖²`, with **`δ^k = max(σ − λ_min(AᵀA), 0)`** — note: uses the **full** `AᵀA`, `σ > 0` small.
- **(2.16)** `x^{k+1}_{W^k} = argmin_{z∈R^{|W^k|}} F^k(z) := f^k(z) + λ‖z‖₁`; then `x^{k+1}_{(W^k)^c} = 0`.
- **Proposition 2.1:** `F(x^k) − F(x^{k+1}) ≥ (δ^k/2)‖x^{k+1} − x^k‖²`.

### 1.5 Algorithm 5 (IR-FISTA-B) — exact printings, including anomalies `[PDF×2]`

Inputs: `x⁰ ∈ R^n, L_0⁰ > 0, γ > 1, η > 1, T > 0, τ > 0, σ > 0, δ > 0` (the input `T` does not appear again in the printed algorithm `[PDF×2]`).

- Outer **(2.17)**: `W₁^k, W₂^k, W^k` as in (2.1); `ẑ⁰ = x^k_{W^k}`; **`L_0^k = L · S^k · L^{k−1}_{t^{k−1}}`** — printed with a leading factor `L` that does **not** appear in (2.14) (`L_0^{k+1} = S^k L^k_{l^k}`); both printings verified in the same official PDF `[PDF×2]`.
- Inner loop: backtracking **(2.18)** printed as `F(P_L̄(y^t)) ≤ Q_L̄(P_{L̄}(y^t), y^t)` — it literally uses the **outer** `F, Q`, not `F^k` (extraction shows a subscript glitch `P_{L^k}` in place of `P_{L̄}` in one argument; treat as OCR-level ambiguity, the `F` vs `F^k` point stands) `[PDF×2]`.
- Prox step **(2.19)** printed as `z^t = argmin_{z∈R^{|W^k|}} (L_k/2) ‖ z − ( ẑ^{t−1} + (δ/L_k) x^k_{W^k} − (1/L_k) A_{W^k}ᵀ(A_{W^k} ẑ^{t−1} − b) ) ‖ + λ‖z‖₁` — verified in **both** PDF copies: (i) the **square is absent** from the norm (contrast (1.2), where the square is present); (ii) it uses the input **`δ`**, not `δ^k`; (iii) the argument lacks the `−(δ/L_k)ẑ^{t−1}` term that the true prox-gradient of (2.15) requires: `v = ẑ − ∇f^k(ẑ)/L = (1−δ^k/L)ẑ + (δ^k/L)x^k_{W^k} − (1/L)A_{W^k}ᵀ(A_{W^k}ẑ−b)` `[MATH]`.
- Momentum **(2.20)** `θ_{t+1} = (1+√(1+4θ_t²))/2`, **(2.21)** `ẑ^t = z^t + ((θ_t−1)/θ_{t+1})(z^t − z^{t−1})`.
- Inner stop **(2.22)** (for given `γ>1`, `0<ρ<1`): `t > min{ 2√(η·cond(A_{W^k}ᵀA_{W^k} + δ^k I))/ρ − 1, 5·γ^k }` **and** `F^k(z⁰) − F^k(z^t) ≥ (λ_min(A_{W^k}ᵀA_{W^k}) + δ^k)‖g(f^k, z⁰)‖² / ( 4 min(1, 1/(δ^k + ‖A_{W^k}ᵀA_{W^k}‖)²) )`, with `z⁰ = x_{W^k}`.
- After inner stop: `x^{k+1}_{W^k} = z^{t̄_k}`, `x^{k+1}_{(W^k)^c} = 0`; outer stop `if ‖g(f, x^k)‖ < ε_tol then Break`; **`Return x* = x^{k+1}`** `[PDF×2]`.
- Working-set cap (p. A2000, text after (2.22)–(2.23)): keep only `ℓ` indices `J = {j₁,…,j_ℓ} ⊂ W^k` with the `ℓ` largest `g_{W^k}(f, x^k)`; **(2.23)** discusses near-degenerate complementarity; "once W* is identified, IR-FISTA-B degenerates to restart FISTA [1,24,33]".
- Experiment parameters (p. A2008) `[PDF×2]`: `η=1.1, γ=1.5, τ=10⁻⁴, δ=10⁻³, ℓ=min(m,3000)`, `x⁰=0`, `λ = λ_c‖Aᵀb‖_∞`, `0<λ_c<1`; automatic τ adjustment: increase τ if more than half of `W₂^k` becomes nonzero, decrease if less than a quarter. Deblurring model **(4.1)–(4.2)** (note (4.2) prints `λ/2‖x‖₁`, inconsistent with (1.1)'s `λ‖x‖₁` — minor print observation `[PDF×2]`).

### 1.6 Convergence theory `[PDF×2]`

- Generic problem **(3.1)**: `min_z φ(z) := ½ zᵀH z + rᵀz + λ‖z‖₁`, H SPD.
- **Lemma 3.1:** `φ(z) − φ(z*) ≥ (λ_min(H)/2)‖z − z*‖²`.
- **Lemma 3.2:** if `lim ‖H^k v‖ = 0` then `‖H^k v‖ ≤ μ‖H^{k−1}v‖` for some `μ∈(0,1)`; `v` in eigenspace with eigenvalues < 1 (uses **(3.2)**).
- **Lemma 3.3** (cited to [3] = Beck–Teboulle): for FISTA-B on (3.1), **(3.3)** `φ(z^t) − φ(z*) ≤ 2η‖H‖‖z⁰−z*‖²/(t+1)²`.
- **Theorem 3.4:** **(3.4)** `‖z^t − z*‖ ≤ (2√(η·cond(H))/(t+1))‖z⁰−z*‖`.
- Residual for (3.1) **(3.5)**: `g̃(z) = g(½zᵀHz + rᵀz, z)` (same structure as (2.4)).
- **Theorem 3.5:** **(3.6)** `max{|z_j−z*_j|, |H_jᵀ(z−z*)|} ≥ g̃_j(z)`; **(3.7)** `‖z−z*‖ ≥ min(1, 1/‖H‖)‖g̃(z)‖`; **(3.8)** `φ(z)−φ(z*) ≥ (λ_min(H)/2)·min(1, 1/‖H‖²)·‖g̃(z)‖²` (proof uses **(3.9)–(3.13)**).
- **Theorem 3.6:** **(3.14)** `lim_{k→∞} ‖g(f, x^k)‖ = 0` for IR-FISTA-B on (1.1). Proof: **(3.15)** (exists `t̄_k`) `F^k(z^{k,t̄_k}) − F^k(z*^k) ≤ (λ_min(A_{W^k}ᵀA_{W^k})+δ^k)‖g(f^k,x_{W^k})‖² / (4 max(1, (δ^k+‖A_{W^k}ᵀA_{W^k}‖)²))`; **(3.16)** `F^k(z^{k,0}) − F^k(z*^k) ≥ (…same numerator…) / (2 max(1, (…)²))`; "(2.22) would be satisfied in finite numbers"; **(3.17)** `z^{k,0}=x^k_{W^k}`, `x^{k+1}_{W^k}=z^{k,t̄_k}`; **(3.18)** `F(x^k)=F^k(x^k_{W^k})`, `F(x^{k+1}) ≤ F^k(x^{k+1}_{W^k})`; **(3.19)** `F(x^k)−F(x^{k+1}) ≥ (λ_min(A_{W^k}ᵀA_{W^k})+δ^k)‖g(f,x^k)‖²/(4 max(1,(δ^k+‖A_{W^k}ᵀA_{W^k}‖)²))`; **(3.20)** summability `Σ_k‖g(f,x^k)‖² < ∞` (note: the final bound divides by `δ^k`, which can be 0 when `σ ≤ λ_min(AᵀA)` — print observation `[PDF×2]`, likely intends the input `δ>0`; `[OPEN]`).
- **Theorem 3.7:** (i) `x^k` converges to a minimum `x*` and for `k ≥ K₁`, `‖x^{k+1}−x*‖ ≤ ‖x^k−x*‖`; (ii) under **strict complementarity** at `x*`, for `k ≥ K₂`, **(3.21)** `‖x^{k+1}−x*‖ ≤ μ‖x^k−x*‖` with `μ ≤ δ/(δ + min{s>0 : s ∈ Λ(A_{W₁*}ᵀA_{W₁*})})`, via **(3.23)** (definitions of `W₁*,W₂*,W₃*,m₁*,m₂*`), **(3.24)–(3.40)**, key recursion **(3.38)** `(A_{W₁*}ᵀA_{W₁*} + δ^kI)(x^{k+1}_{W₁*} − x*_{W₁*}) = δ^k(x^k_{W₁*} − x*_{W₁*})` and **(3.39)**; (iii) if `A_{W₁*}ᵀA_{W₁*}` is positive definite and FISTA-B terminates under (2.22) with `t̄_k > 2√(η·cond(A_{W₁*}ᵀA_{W₁*}))/ρ − 1`, then for any `0<ρ<1`, for `k > K₃`, **(3.22)** `‖x^{k+1}−x*‖ ≤ ρ‖x^k−x*‖` (via **(3.40)** and Theorem 3.4).

### 1.7 Complexity statements `[PDF×2]`

- **(3.41):** IR-FISTA-B needs `O(mp ln(ε₁/ε))` flops to reduce `‖x^k−x*‖` from `ε₁` to `ε` (abstract: `O(mp ln 1/ε)`), `p` = number of nonzeros of `x*`; inner FISTA-B iteration cost `O(mp)` (computing (2.19)–(2.21)).
- **(3.42):** FISTA-B needs `O((ε₁/ε)^{1/2}·mn)` flops to reduce `F(x^k)−F(x*)` from `ε₁` to `ε` (cost `O(mn)`/iteration).
- Comparisons (Section 3.2): SSNAL linear, `O(m²p)`/iteration → `O(m²p ln(ε₁/ε))` total; LARS and l1-homotopy `O(p²+(n+p)m)`/iteration, generally `O(p)` iterations → `O(p³+(n+p)mp)`; PBC-ℓ1-Hom `O(((m+n)q²+q³) ln(ε₁/ε))` with block size `q`.

---

## 2. Skeptical cross-check of every A/D/E ID in `docs/ir_fista.md`

| ID | Draft claim | Paper anchor | Verdict |
|---|---|---|---|
| A1 | ucmf objective `F_C(x)=f(x)+λ‖x‖₁+I_C(x)`, `f=‖Ax−y‖²/(2m)`; paper (1.1) is `½‖Ax−b‖²+λ‖x‖₁`; dividing by m gives `λ_p=mλ`, `δ_p^k=mδ^k`, `L_p=mL`; exact rescaling, minimizers/prox agree | (1.1), (2.15)–(2.16) `[PDF×2]` | **Scaling equivalence: [MATH] correct** (paper never states it; derived). `m` unchanged by column selection: [MATH] trivially true. |
| A1 | Signed residual from (2.4) with `d=∇f=Aᵀ(Ax−y)/m`; zero set = signed KKT; outer accuracy `‖g‖₂≤ε` | (2.4) `[PDF×2]`; Alg 3/5 stop `‖g(f,x^k)‖ < ε_tol` `[PDF×2]` | **Quote of (2.4): verified verbatim [PDF×2]**. Zero-set equivalence = paper's own sentence after (2.4) `[PDF×2]`. Stop uses strict `<` on `x^k` `[PDF×2]`. |
| A1 | `(2.4)` mixes gradient with `|x_j|`, so generally `g_p ≠ m·g`; no universal tolerance conversion | (2.4) `[PDF×2]` | **[MATH] verified structurally**: for `x_j≠0`, `g_p,j = max{m(|d_j|−λ), min{m|d_j+λsign(x_j)|, |x_j|}}` vs `m·g_j = max{m(|d_j|−λ), min{m|d_j+λsign(x_j)|, m|x_j|}}`; the unscaled `|x_j|` breaks homogeneity. Correct. Same caveat applies to the nonnegative residual (see §4): `g⁺_{p,j}=|min(x_j, m s_j)|` vs `m·g⁺_j=|min(m x_j, m s_j)|` `[MATH]`. |
| A1 | "L includes the proximal curvature" | Alg 1 `(1.2)` uses `L` for the prox quadratic; subproblem curvature enters via `δ^k` in (2.15) and via `L_t^k ≈ ‖A_{W^k}ᵀA_{W^k}‖` in (2.11) | Wording is loose; the paper keeps `L` (prox curvature) separate from `δ^k` (proximal regularization) `[PDF×2]`. `[INF]` |
| D1 | Printed (2.19) lacks the norm square | (2.19) vs (1.2) | **Verified at source in both PDF copies [PDF×2]**: (1.2) has `‖·‖²`, (2.19) does not. Draft's flag is confirmed. |
| D1 | Printed `v_p = ẑ + δ_p a/L_p − A_Wᵀ(A_Wẑ−y)/L_p`; the `−δ^k ẑ/L` term is missing and is essential | (2.19) `[PDF×2]`; (2.15) `[PDF×2]` | **Verified [PDF×2] + [MATH]**: printed (2.19) uses input `δ` (not `δ^k`) and omits `−(δ/L_k)ẑ^{t−1}`; the prox-gradient of (2.15) is `v = ẑ − ∇f^k(ẑ)/L = (1−δ^k/L)ẑ + (δ^k/L)x^k_{W^k} − A_Wᵀ(A_Wẑ−b)/L`. Author intent unresolvable from text `[OPEN]`. |
| D1 | Backtracking must check `q_k(z⁺) ≤ q_k(ẑ) + ⟨∇q_k(ẑ), z⁺−ẑ⟩ + (L/2)‖z⁺−ẑ‖²`; nonsmooth terms cancel | (2.18) prints `F(P_L̄(y^t)) ≤ Q_L̄(P_{L̄}(y^t), y^t)` with outer `F,Q` `[PDF×2]` | **Deviation from literal print confirmed**: (2.18) uses `F`, not `F^k`; the draft's proposal matches the theory (Lemma 3.3 is stated for the subproblem (3.1)) but not the printed line `[PDF×2]`+`[INF]`. Cancellation of `Ω` on both sides: [MATH] correct. |
| D1 | Prox minimizer is soft-threshold at `λ/L`, or `max(v−λ/L,0)` for nonnegativity | (1.2) `[PDF×2]`; ucmf `ista.py` | **[MATH] correct**; matches ucmf `shrink` / `shrink_nonnegative = relu(x−θ)` with `θ=γλ` `[CODE]`. |
| D2 | Printed thresholds: `T_{2.22} = μ‖g‖²/(4 min(1,M⁻²))`, `T_{3.19} = μ‖g‖²/(4 max(1,M²))`, `M_p = δ^k+‖A_WᵀA_W‖ = ‖H_p‖`; not equivalent | (2.22) `[PDF×2]`; (3.15),(3.16),(3.19) `[PDF×2]` | **Verified verbatim [PDF×2]**: (2.22) denominator `4 min(1, 1/(δ^k+‖A_WᵀA_W‖)²)`; (3.15)/(3.19) denominator `4 max(1, (δ^k+‖A_WᵀA_W‖)²)`; (3.16) `2 max(…)`. Since H_p PSD, `‖H_p‖ = δ^k+‖A_WᵀA_W‖` `[MATH]`. |
| D2 | Counterexample `A=[2], y=[1], λ_p=1, δ_p=10⁻³, a=0`: `g_{p,0}=1`, `z*=1/4.001`, max decrease `=1/(2·4.001)≈0.124969`, `T_{2.22}≈16.012`, `T_{3.19}≈0.062484`; literal (2.22) can loop forever | printed formulas `[PDF×2]` | **Arithmetic independently re-verified [MATH]**: H=4.001, r=−2, `g=|r|−λ=1`; `z*=(2−1)/4.001`; `F^k(0)−F^k(z*) = ½ − (1/(2·4.001) − 2/4.001 + ½) − z* = 1/(2·4.001) ≈ 0.12497`; `T_{2.22} = 4.001/(4·(1/4.001²)) = 4.001³/4 ≈ 16.012`; `T_{3.19} = 4.001/(4·4.001²) ≈ 0.06248`. All numbers check out. Conclusion "literal (2.22) can be unsatisfiable ⇒ non-termination if enforced literally" is mathematically sound on the printed formulas `[MATH]`; whether min↔max is a typo is `[OPEN]` (the proof's (3.15)/(3.16) use the max-form, supporting the draft's suspicion `[INF]`). |
| D2 | Iteration requirement in (2.22): `t > min{2√(η cond(A_WᵀA_W+δ^kI))/ρ − 1, 5γ^k}` | (2.22) `[PDF×2]` | **Verified verbatim [PDF×2]**. |
| E1 | Constrained KKT for `x≥0`: `s = d + λ1 ≥ 0`, `x_j s_j = 0`; residual `g⁺_j = |x_j − max(x_j−s_j,0)| = |min(x_j,s_j)|`, zero set = constrained KKT; `W⁺ = {x_j>0} ∪ {x_j=0, −d_j>(1−τ)λ}` | **Not in the paper** (no nonnegativity anywhere; extensions only (2.5)–(2.7)) `[PDF×2]` | **Extension proposal, mathematically verified [MATH]**: on `R₊ⁿ`, `‖x‖₁=1ᵀx`, so KKT is `d+λ1−μ=0, μ≥0, μ∘x=0`, i.e. `s=d+λ1≥0, x∘s=0`. `g⁺=0 ⟺ (x_j=0, s_j≥0) or (x_j>0, s_j=0)` — case-checked. At `x_j=0` the only violation is `−d_j>λ`, so the one-sided `W⁺` rule is correct and retains support `[MATH]`. No inherited convergence rate — the draft says exactly this, correctly `[INF]`. |
| A2 | Batch problems independent; `select_columns(W)`; `optim/ir_fista.py` proposal; CMFPropagation must not materialize the dictionary | paper has a single problem only `[PDF×2]`; ucmf `[CODE]` | **Project-specific, not paper-derived**. Code facts verified by read-only inspection: `src/ucmf/optim/__init__.py` exports `ISTA, FISTA, AdaLFISTA(+R/SW/SWR, NA_*)`; `ista.py` solves `‖y−Ax‖²/(2m)+lam‖x‖₁`, `nonnegative=True` default, `shrink_nonnegative(x,θ)=relu(x−θ)`, step `x⁺=shrink(x+γ·Aᵀ(y−Ax)/m, θ=γ·lam)`, default `γ` from `1.05‖A‖₂²/m`; `FISTA._nesterov_momentum` implements exactly paper (2.20)–(2.21) (`t_next=(1+√(1+4t_k²))/2`, `z_next=x_next+(t_k−1)/t_next·(x_next−x)`) `[CODE]`. |

### 2.1 Draft "Decisions required before implementation" — each mapped to verified paper text

1. *Spectral quantities* — paper uses exact `L_k=‖A_WᵀA_W‖` in Alg 3 (2.3) and backtracking `L_t^k` with (2.11) bounds in Alg 5; `cond(A_WᵀA_W+δ^kI)` appears only in (2.22) `[PDF×2]`.
2. *Proximal curvature policy* — **verified inconsistency in the paper**: (2.15) defines `δ^k` with full `AᵀA`; (2.22)/(3.15)/(3.19) use `A_WᵀA_W`; Alg 5 inputs contain both `σ>0` and `δ>0`, and (2.19) uses `δ` while (2.22) uses `δ^k` `[PDF×2]`.
3. *Inner stopping (D2)* — see above; (2.22) min-form vs (3.15)/(3.16)/(3.19) max-form `[PDF×2]`.
4. *Outer stopping tests `x^k`, returns `x^{k+1}`* — **verified verbatim in Alg 3 and Alg 5 `[PDF×2]`**.
5. *Lipschitz warm start* — **verified print discrepancy**: (2.14) `L_0^{k+1}=S^k L^k_{l^k}` vs (2.17) `L_0^k = L·S^k·L^{k−1}_{t^{k−1}}` (extra leading `L`) `[PDF×2]`.
6. *Working-set cap/top-ℓ* — verified on p. A2000: keep `ℓ` indices of `W^k` with largest `g_{W^k}(f,x^k)`; experiments use `ℓ=min(m,3000)` (p. A2008) `[PDF×2]`.
7. *λ=0 / empty W / tolerances* — paper assumes `λ>0` (1.1) `[PDF×2]`; no guidance for `λ=0` or empty `W^k` `[OPEN]`.

---

## 3. Related and follow-up literature (full references)

All Zotero item keys: **not retrievable this run** (Zotero local API connection refused; see §0). DOIs web-verified `[WEB]` unless noted.

### 3.1 Directly related working-set / reduction methods for LASSO
1. **Wang, Wei, Yu, Xu (2020)**, "An Efficient Proximal Block Coordinate Homotopy Method for Large-Scale Sparse Least Squares Problems," *SIAM J. Sci. Comput.* 42(1), A395–A423, DOI 10.1137/19M1243828 — the direct predecessor (PBC-ℓ1-Hom, paper's ref [36], summarized as Algorithm 2 with (1.10)–(1.12) `[PDF×2]`). Not in local library under probed name.
2. **Li, Sun, Toh (2018)**, "A Highly Efficient Semismooth Newton Augmented Lagrangian Method for Solving Lasso Problems," *SIAM J. Optim.* 28(1), 433–458, DOI 10.1137/16M1097572 — SSNAL (paper's ref [26]); working set `J = {j : |x_j| > σλ}` and Sherman–Morrison reduction of `(I_m + σA_J A_Jᵀ)⁻¹` described in paper intro `[PDF×2]`.
3. **Efron, Hastie, Johnstone, Tibshirani (2004)**, "Least Angle Regression," *Ann. Stat.* 32(2), 407–499, DOI 10.1214/009053604000000067 — LARS (paper's ref [14], path (1.5)–(1.6) `[PDF×2]`). **Local PDF present:** `~/mendeley_papers/Efron/Efron2004.pdf`.
4. **Asif & Romberg (2014)**, "Sparse Recovery of Streaming Signals Using ℓ₁-Homotopy," *IEEE Trans. Signal Process.* 62(16), 4209–4223, DOI 10.1109/TSP.2014.2328981 — l1-homotopy (paper's ref [2], (1.9) `[PDF×2]`).
5. **Fang, Fan, Sun, Friedlander (2020)**, "Greed Meets Sparsity: Understanding and Improving Greedy Coordinate Descent for Sparse Optimization," *PMLR* 108 (AISTATS), 434–444 — GCD (paper's ref [17]). Metadata `[WEB]` (PMLR volume is 108; the paper's own ref prints "pp. 434–444" `[PDF×2]`).

### 3.2 FISTA and backtracking foundations
6. **Beck & Teboulle (2009)**, "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems," *SIAM J. Imaging Sci.* 2(1), 183–202, DOI 10.1137/080716542 — paper's ref [3], cited for Lemma 3.3 (3.3) (the `2η‖H‖` backtracking bound). **Local PDF present:** `~/mendeley_papers/Beck/Beck2009.pdf`. Its general model is (1.6) `min f+g` with arbitrary closed convex `g` (so indicator functions / box constraints are covered by the prox); ISTA step (1.4) with shrinkage (1.5); a backtracking line-search variant with `L0, η` is in the paper (p. 194; also in the authors' FOM `fista` solver options) `[WEB]`+(`[PDF]`).
7. **Scheinberg, Goldfarb, Bai (2014)**, "Fast First-Order Methods for Composite Convex Optimization with Backtracking," *Found. Comput. Math.* 14(3), 389–417, DOI 10.1007/s10208-014-9189-9 `[WEB]` — rigorous backtracking FISTA/FALM preserving `O(1/t²)`.

### 3.3 Follow-ups of IR-FISTA-B (citing works found; signed LASSO only)
8. **Ma, Xu, Qu (2023)**, "An Improved Iterative Reduction FISTA-Barzilai-Borwein Algorithm for Large-Scale LASSO," *Proc. Int. Conf. New Trends in Computational Intelligence (NTCI)*, pp. 165–170, DOI 10.1109/NTCI60157.2023.10403676 `[WEB]` — replaces the FISTA-B inner step by a Barzilai–Borwein step; still unconstrained LASSO.
9. **Cao, Tang, Gao, Wang, Chen, Gao, Zhao, Su (2024)**, "SODL-IR-FISTA: sparse online dictionary learning with iterative reduction FISTA for cone-beam X-ray luminescence computed tomography," *Biomedical Optics Express* 15(9), 5162–5179, DOI 10.1364/BOE.531828 `[WEB]` — applies IR-FISTA-B inside online dictionary learning; cites Wang et al. as [26]/[37]; unconstrained.
- Web search 2021–2025 found **no published extension of IR-FISTA-B to `x ≥ 0` / box-constrained LASSO** `[WEB, search-based, not exhaustive]`. The closest working-set-transfer reference is FireWorks (below), which is nonconvex-regularizer, not nonnegativity.

### 3.4 Nonnegative / constrained LASSO and NNLS-type active-set literature (for the E1 transfer)
10. **Lawson & Hanson (1974)**, *Solving Least Squares Problems*, Prentice-Hall, Ch. 23, p. 161 (SIAM Classics reprint, DOI 10.1137/1.9781611971217) `[WEB]` — the NNLS active-set algorithm: passive set `P` with dual `w = Aᵀ(b − Ax)`, `w_j ≤ 0` on `P`, `w_j > 0` on the complement; add the most positive `w_j`, solve LS on `P`, remove indices that hit zero. This is the canonical **one-sided KKT / reactivation** mechanism for `x ≥ 0`.
11. **Bro & de Jong (1997)**, "A Fast Non-Negativity-Constrained Least Squares Algorithm," *J. Chemometrics* 11(5), 393–401, DOI 10.1002/(SICI)1099-128X(199709/10)11:5<393::AID-CEM483>3.0.CO;2-L `[WEB]` — FNNLS, a block/Cholesky-based variant of Lawson–Hanson for many RHS.
12. **Benvenuto, Zanella, Zanni, Bertero (2009)**, "Nonnegative Least-Squares Image Deblurring: Improved Gradient Projection Approaches," *Inverse Problems* 26(4), 045014, DOI 10.1088/0266-5611/26/4/045014 `[WEB]` — gradient-projection NNLS; also paper's ref [4] (whose printed "pp. 4–25" is an OCR garble of 045014 `[PDF×2]`).
13. **Esser, Lou, Xin (2013)**, "A Method for Finding Structured Sparse Solutions to Nonnegative Least Squares Problems with Applications," *SIAM J. Imaging Sci.* 6(4), 2010–2046, DOI 10.1137/13090540X `[WEB]` — paper's ref [16]; NNLS with ℓ1−ℓ2 penalties solved via active sets — closest published analogue of "nonnegative + ℓ1-like penalty + working set".
14. **van den Berg & Friedlander (2008)**, "Probing the Pareto Frontier for Basis Pursuit Solutions," *SIAM J. Sci. Comput.* 31(3), 890–912, DOI 10.1137/060646721 `[WEB]` — SPGL1 (paper's ref [5]); solves ℓ1-regularized LS **with general box/bound constraints** (incl. `x ≥ 0`) via a root-finding formulation on the dual `‖Aᵀ(Ax−b)‖`-type quantity — relevant to one-sided KKT screening.
15. **Nadisic, Cohen, Vandaele, Gillis (2022)**, "Matrix-wise ℓ0-constrained Sparse Nonnegative Least Squares," *Machine Learning* 111(12), 4453–4495, DOI 10.1007/s10994-022-06260-2 (arXiv:2011.11066) `[WEB]` — homotopy method for sparse multiple-RHS NNLS driven by the **one-sided KKT** conditions (`x≥0`, `x_i=0 ⟹ (Aᵀ(b−Ax))_i ≤ 0`); support changes add/remove one index at a time, analogous to IRA's W₂ reactivation but for nonnegativity.
16. **Friedman, Hastie, Tibshirani (2010)**, "Regularization Paths for Generalized Linear Models via Coordinate Descent," *J. Stat. Softw.* 33(1), 1–22, DOI 10.18637/jss.v033.i01 `[WEB]` — glmnet: coordinate descent for LASSO/elastic net **with bound constraints** (`x ≥ 0` supported); coordinate update clamps at the bound instead of soft-thresholding through zero.
17. **Parikh & Boyd (2014)**, "Proximal Algorithms," *Found. Trends Optim.* 1(3), 123–239, DOI 10.1561/2400000003 — **local PDF present:** `~/mendeley_papers/Parikh/Parikh2014.pdf`. Framework: `prox_{λf}(v) = argmin f(x) + (1/2λ)‖x−v‖²`; ℓ1 prox and nonnegative-orthant indicator are both separable examples.
18. **Rakotomamonjy et al. (2022)**, "Convergent Working Set Algorithm for Lasso with Non-Convex Sparse Regularizers," *PMLR* 151 (ICML 2022), paper ID rakotomamonjy22a, proceedings.mlr.press/v151/rakotomamonjy22a.html `[WEB; page numbers not verified]` — FireWorks: convergent working-set algorithm with inexact inner solver for (nonconvex) sparse regularizers; relevant as a template for "working set + inner stopping" convergence arguments outside Wang et al.
19. **Tibshirani (1996)**, "Regression Shrinkage and Selection via the LASSO," *JRSS-B* 58(1), 267–288, DOI 10.1111/j.1467-9868.1996.tb00074.x `[WEB]` — paper's ref [34].

---

## 4. Transfer analysis: IR-FISTA-B → nonnegative LASSO `min_{x≥0} ‖Ax−y‖²/(2m) + λ‖x‖₁`

1. **Prox of `λ‖·‖₁ + I_{R₊ⁿ}`** `[MATH]`: on `R₊ⁿ`, `‖x‖₁ = 1ᵀx`, so `prox_{λ‖·‖₁+I_{≥0}}(v) = argmin_{x≥0} (L/2)‖x−v‖² + λ1ᵀx = max(v − λ/L, 0)` elementwise. This is exactly ucmf's `shrink_nonnegative` (`relu(x−θ)`) with `θ = γλ` `[CODE]`, and the separable composition is standard in the prox literature (ref 17). The signed soft-threshold `sign(v)max(|v|−λ/L,0)` is **not** the constrained prox `[MATH]`.
2. **One-sided KKT / residual** `[MATH]` (draft E1, verified): with `d = ∇f(x) = Aᵀ(Ax−y)/m` and `s = d + λ1`, optimality of `min_{x≥0} f + λ‖x‖₁` is `s ≥ 0, x∘s = 0`. Residual `g⁺_j = |min(x_j, s_j)|` has exactly this zero set. Working set `W⁺ = {j : x_j>0} ∪ {j : x_j=0, −d_j > (1−τ)λ}` retains the support and all KKT-violating zeros `[MATH]`. **Caveat not in the paper:** under the paper→ucmf rescaling, `g⁺` is also not homogeneous (`g⁺_{p,j}=|min(x_j, m s_j)|` vs `m g⁺_j`), so no tolerance conversion either `[MATH]`.
3. **Elimination/reactivation under nonnegativity:** elimination `x_{(W^k)^c} := 0` stays feasible (`0 ∈ R₊ⁿ`), so the paper's structure carries over; reactivation must use the **one-sided** test (`−d_j > (1−τ)λ`), because a large **positive** `d_j` at `x_j=0` is *not* a violation when `x≥0` `[MATH]`. This matches NNLS active-set practice (Lawson–Hanson dual sign condition, refs 10–11, 15) `[WEB]`.
4. **Inner subproblem** becomes `min_{z≥0} ½‖A_W z − b‖² + λ1ᵀz + (δ^k/2)‖z − a‖²` (paper units); the inner prox is `max(v − λ/L, 0)` `[MATH]`; the backtracking majorization test is structurally unchanged (smooth quadratic model + nonsmooth terms cancel `[MATH]`), but the printed-paper anomalies of §1.5 ((2.19) square / `−δẑ/L` / `δ` vs `δ^k`; (2.18) `F` vs `F^k`; (2.22) min vs (3.19) max) must be resolved **before** any constrained variant is specified, because the constrained theory would inherit them `[INF]`.
5. **What does NOT transfer automatically:** Theorem 3.5's residual bounds (3.6)–(3.8) are proved for the signed residual (2.4)/(3.5); a one-sided analogue of `g̃` and of (3.6)–(3.8) for `g⁺` is **not established in any cited source** `[OPEN]` — this is a math task, not a literature fact. Theorem 3.7's support-identification argument (3.23)–(3.40) relies on signed KKT (1.7)–(1.8) and on `sign(x*)` stability; under `x≥0` the sign pattern is fixed (`+1` on the support) and the complementarity sets become one-sided (`W₂* = {x*=0, −Aᵀ(Ax*−b) ≥ λ}`-type), so a re-proof is needed; **no inherited linear-rate claim is justified** (the draft E1 says exactly this — correct `[INF]`).
6. **Scaling:** `λ_p = mλ`, `δ_p = mδ`, `L_p = mL` is an exact objective rescaling `[MATH]`; inner thresholds like (2.22)/(3.19) involve `λ_min`, `‖·‖`, and `‖g‖²` and do **not** rescale homogeneously once the `|x_j|` term in the residual is present `[MATH]` — consistent with draft A1/D2 cautions.

## 5. Open questions / missing fundamental evidence

- **[OPEN]** Author intent for the printed (2.19) (missing square, missing `−(δ/L)ẑ`, `δ` vs `δ^k`) and for the extra `L` in (2.17) vs (2.14): both PDF copies agree on the printings; only an erratum/preprint/author contact can resolve intent. Affects D1 directly.
- **[OPEN]** Which inner decrease threshold is intended: (2.22) `min(1, M⁻²)` (can be unsatisfiable — D2 counterexample verified `[MATH]`) vs (3.15)/(3.16)/(3.19) `max(1, M²)`. Affects D2 and any finite-termination test.
- **[OPEN]** Whether printed (2.18) means `F` or `F^k` in the inner backtracking test.
- **[OPEN]** One-sided (nonnegative) analogues of Theorem 3.5 (3.6)–(3.8) and of Theorem 3.7's support-identification proof — no published source found; requires new math (delegate to mathematician), not literature.
- **[OPEN]** Zotero item keys for all references — Zotero API was down this run (connection refused; local-only mode); DOIs web-verified instead. Re-run Zotero when the API is reachable.
- **[OPEN]** Local `~/mendeley_papers/` contains only Wang2022, Beck2009, Efron2004, Parikh2014 under probed names; PBC-ℓ1-Hom (Wang2020), SSNAL (Li2018), Bro–de Jong, Ma2023, Nadisic2022 etc. are not locally available and were cited from web-verified metadata only.
- **[SEARCH-BASED, NOT EXHAUSTIVE]** No 2021–2025 publication extending IR-FISTA-B to nonnegative/box LASSO was found; if one exists outside indexed web results, it was missed `[WEB]`.
- **[PDF×2 observation]** (3.20) divides by `δ^k` (can be 0); deblurring (4.2) prints `λ/2‖x‖₁` vs `λ‖x‖₁` in (1.1).

## 6. Summary for the adaptation spec

- The paper's skeleton — outer working set (2.1) + elimination + one-shot full solve of the reduced problem (2.2), inner FISTA-B (Alg 4/5) with backtracking (2.9)/(2.18), momentum (2.20)–(2.21), proximal term (2.15)–(2.16), residual (2.4), stops (2.22) and `‖g(f,x^k)‖<ε_tol`, theory Thm 3.4–3.7, complexity (3.41)–(3.42) — is verified verbatim `[PDF×2]` except the specific print anomalies flagged in §1.5/§2.
- `docs/ir_fista.md` is **faithful to the paper** on A1's quotations of (2.4) and the rescaling, and its D1/D2 flags about (2.19)/(2.22)/(3.19) are **confirmed at source** (both PDF copies; D2 arithmetic re-verified). E1 is a mathematically sound **extension** (verified `[MATH]`), not paper content — the paper contains no constrained variant, and the needed one-sided residual bounds are an open math task.
- For the nonnegative adaptation, the literature supplies: the constrained prox `max(v−λ/L,0)` (refs 17 + ucmf code `[CODE]`), one-sided KKT and active-set/reactivation practice (refs 10–16), and working-set-with-inexact-inner-solver convergence templates (ref 18) — but **no** published constrained analogue of IR-FISTA-B's theory.
