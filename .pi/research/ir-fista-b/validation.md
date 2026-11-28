# Independent Phase 4 scientific/code review, Gate 2 verdict, and post-Gate-2 benchmark/memory audit: safeguarded IRFISTAB

**Verdict:**
- **Gate 2 numerical correctness is PASSED**, scoped to CPU float64 and float32 and to the evidence in §2–§7. This is unchanged by the benchmark audit.
- **Bounded exploratory CPU benchmarks and isolated-process memory probes are now recorded** in `benchmark_notes.md` and the four JSONL artifacts. §9 audits them. They are single-seed, single-thread, CPU float64, with no warm-up, no statistical study, no GPU and no Acoular capture. Nothing in them supports a general speed, memory or convergence claim.
- No Critical or Major finding exists, in either the correctness review or the benchmark audit.

**Review basis**
- Baseline commit: `ab87210`.
- Source hashes, unchanged throughout (correctness review, recovery, benchmark runs, this audit):
  - `src/ucmf/optim/ir_fista_b.py`: `889fe15fca27912c67a9139e865e87155b7b099729077e63e7f33155776f2091`
  - `examples/benchmark_ir_fista_b.py`: `44745d48a1c687ae01418a7aba82adef53a9d073db9d75c9aac24e24b384a60e`
  - `tests/test_ir_fista_b.py`: `e83e9c420e28230ed36bdceffd3967065758d9ac40afe2c2f2dfb244960567e1`
- All 27 benchmark records carry `script_sha256` 44745d48….
- I made no working-tree changes. Scratch scripts are in `/tmp/rev/`. This report is the final message and the runtime persists it.
- Reviewed contract: `algorithm_specification.md` v2 and `mathematical_review.md` §7.
- `coder_report.md` and `implementation_notes.md` were treated as claims only.

## 1. Correctness gate versus performance

| Item | Status |
|---|---|
| Spec-v2 compliance (code review) | Done. No deviation affecting correctness. |
| Focused tests, harness reproduction, independent numerics (CPU) | Done. Passed within the scope in §7. |
| CUDA correctness | Not run. No GPU; 2 tests skipped. |
| Bounded exploratory benchmarks (CPU f64, 27 records) | Run by the parent after this PASS and audited in §9. Descriptive only. |
| Isolated cold-process memory probes (6 records) | Run and audited in §9. Descriptive only. |
| Performance on GPU, multi-seed, warm-up-controlled statistics, Acoular captures, generic/B-wide fallback performance, full multi-hour experiment plan | Not measured. |

## 2. Commands I ran and results

| Command | Result |
|---|---|
| `uv run pytest -q tests/test_ir_fista_b.py` | 108 passed, 2 skipped (CUDA only), 4.4 s |
| Parent's `uv run pytest -q` | 2020 passed / 23 skipped / 14 existing warnings (I did not rerun it) |
| Parent's `uv run ruff check .` and `git diff --check` | clean (I did not rerun them) |
| `benchmark_ir_fista_b.py --mode correctness` (positive f64) | 6 records, all `success_oracle=True` |
| Same, `--signed` (f64) | 6 records, all `success_oracle=True` |
| Same, `--dtype float32 --tol 1e-4` | 6 records, all `success_oracle=True` |
| `/tmp/rev/num.py` (broad independent sweep) | 3124 probes, completed in 1133 s |
| `/tmp/rev/num2.py` (bounded subset) | 718 probes, 136 s |
| `/tmp/rev/bad.py` (one severe case) | converged |
| `/tmp/rev/bad2.py` (extended-t reruns) | Aborted with no output; counted as no evidence |
| `/tmp/aud.py`, `/tmp/aud2.py` (post-Gate-2 artifact parsing) | Read the saved JSONL only; no solver runs |

The correctness harness runs were printed to stdout; I wrote no `--out` files.

### Harness records (IR rows, from the correctness-mode reproduction)

Every reference was certified. Reference natural residuals were about 1.7e-12 (dense) and 2.4e-12 to 3.4e-12 (CMF).

| Mode | Family | IR natural = raw | Tolerance | Objective gap F − F_ref |
|---|---|---:|---:|---:|
| positive f64 | dense | 1.04e-8 | 4.02e-8 | 4.6e-15 |
| positive f64 | CMF | 6.5e-8 | 1.11e-7 | 1.2e-13 |
| signed f64 | dense | 1.01e-8 | 3.6e-8 | 4.2e-15 |
| signed f64 | CMF | 3.9e-9 | 3.3e-8 | 6.9e-16 |
| positive f32 | dense | 3.0e-6 | 4.0e-6 | 2.8e-10 |
| positive f32 | CMF | 2.1e-6 | 1.1e-5 | 8.0e-11 |

- IR status was `converged` in all six cases.
- ISTA and FISTA also passed the common natural-residual check (objective gaps 1e-15 to 1.3e-9).
- The working-set histories took the form [10, 5, 3, 3, 3, 3].
- The counts check `counted.total == full + restricted` held. Baselines ran as one public call, so no FISTA restarts.
- The float32 records are validated in float64 on the same quantized inputs.

## 3. Specification-v2 compliance (equation to code)

| Spec item | Code | Result |
|---|---|---|
| Normalized objective and gradient (1/(2m), 1/m) | `_certificate` and `_reduced_solve` use `y.shape[1]`. Restriction keeps m. | ✔ `test_measurement_stacking_and_scaling` passes. |
| Positive and signed prox | `get_shrink_func`, threshold `lam/L` | ✔ |
| Natural KKT, positive: `min(x, g+λ)` | `_natural` | ✔ |
| Natural KKT, signed: cancellation-free `min(max(x, g−λ), g+λ)` | `_natural` | ✔ I checked the three cases algebraically against `x − S_λ(x−g)`. |
| Stable residual at both sites (global and inner) | `_certificate` and `_reduced_solve` both call `_natural` (inner with ∇q) | ✔ The ±2^60 regressions pass at both sites: residual 1, status `max_iter`. |
| Raw residual | `_certificate` | ✔ Natural and raw are both reported and compared with oracles. |
| Working set (support ∪ near-violators, exact comparisons, ascending) | `_working_set` | ✔ Exact selection is tested. |
| No empty reduced operator | `columns.numel()==0 → continue` | ✔ |
| Reactivation after each full gradient | Full certificate after every row update | ✔ `test_reactivation_review_example` shows working-set size 1 then 2. |
| Reduced gradient `A_Wᵀ(A_W v − y)/m + δ(v − a)` | `_reduced_solve` | ✔ |
| F(x_new) = Φ − δ/2‖z−a‖² | Φ includes `δ‖z−a‖²/2` | ✔ |
| Default δ = 1e-3/m; explicit δ not rescaled | `forward` | ✔ |
| Momentum reset per outer solve; standard FISTA recurrence | `v=a, previous=a, momentum=1`; `_nesterov_momentum` reused | ✔ |
| Backtracking: quadratic test, factor η, L never decreases within a solve | `_reduced_solve` | ✔ |
| Homogeneous displacement normalization; relative 32·eps slack; no additive slack; finite checks | `scale = max(abs(D))` rescales both sides; `slack = 32·eps·max(|left|,|right|)`; `_finite` | ✔ Algebraically the spec inequality. Subnormal and underflow regressions pass; exhaustion raises `RuntimeError`. |
| First-step safeguard: best Φ over all accepted candidates, first included, earlier kept on ties | `best is None or phi < best_phi` | ✔ `test_best_candidate_including_first_not_last…` checks against an independent NumPy replay. |
| Inner stop: natural residual of ∇q ≤ `inner_tol`·global R, budget `inner_max_iter` | `hit`, `budget_exhausted`, `best_target_met` | ✔ Best-candidate and exit-target flags are tracked separately. |
| Warm L `max(floor, L_last/η)`; initial `max(floor, 1/γ)` | `forward` | ✔ |
| Full-return certificate; status from the returned point; `t=0` | Final certificate after the loop | ✔ |
| Callback semantics | Once per sweep; full (B,n) tensors; (B,1) gamma/theta; stop never overrides KKT status | ✔ Mutation is re-certified, with a conservative extra scan in inference mode. |
| Input validation; per-call diagnostics reset | Validation before work, even at `t=0` | ✔ |
| No Gram matrix or dictionary creation | Static source scan plus a runtime dispatch guard | ✔ |

**Per-row independence.**
- Working sets, L, tolerance scale and status are all per row.
- `test_batch_independence_and_actual_counts` checks batched output against per-row solves for native and fallback operators.
- It also checks the counters against independent counting wrappers, including B-wide `batch_fallback_calls`.

## 4. Findings

### Critical

None.

### Major

None.

I found no false convergence certificate and no spec-v2 deviation affecting correctness. Every `converged` status in the sweeps matched an independent natural and raw KKT evaluation.

### Minor

**M-1. Weak (tautological) oracle.**
- `tests/test_ir_fista_b.py::test_delta_scalar_d2_and_safeguard` asserts `info['target_met'] != info['budget_exhausted']`.
- `budget_exhausted` is defined as `not hit`, so the assertion always holds. The meaningful assertions in that test are fine.
- Correction: drop the line, or assert iterations ≤ budget when `target_met`.

**M-2. Small test dimensions and single seeds.**
- Most reference tests use n=8 and one seed.
- This is mitigated by my 3124 + 718 independent probes and by the benchmark artifacts.
- Correction: add an extra seed or two to the parametrization.

**M-3. Per-step product overhead.**
- Each accepted inner step costs 3 restricted forwards and 2 restricted adjoints (plus one forward per rejected trial). The tests count this.
- Cached images and adjoints could remove some of it; the spec permits that.
- This is a performance caveat, not a correctness issue. The benchmark counts show it is outweighed by narrow-width products in the sparse cases and not in the NNLS control (§9).

**M-4. Budget-limited honest `max_iter` on severe conditioning.**
- With default `gamma=None` (L starts at 1) and `inner_max_iter=100`, ill-conditioned or badly scaled problems can exhaust budgets.
- About 97% of inner solves exhausted the budget on the severe case I inspected.

**M-5. Absolute default tolerance.**
- The default `tol=1e-6, rtol=0` is in normalized units. For data of magnitude ≲1e-6, `t=0` can certify a poor point.
- This is specified behaviour and should be documented. The harness uses a scale-relative tolerance.

### Suggestions

- **S-1.** Add a wall-clock or product-budget stop to the solver. The harness only checks wall caps at outer-sweep boundaries (a censored "10 s" run can exceed 10 s slightly).
- **S-2.** Note in the README that diagnostics grow with total inner work. The README already says so.
- **S-3.** Memory reporting must use fresh processes (done in §9).

## 5. Harness audit (safe later use)

- **Reference certification.** `reference()` runs FISTA to `min(tol/100, 1e-10·scale)` in float64 and records `reference_certified`.
  - Correctness mode asserts it.
  - Benchmark mode only records it.
- **Objective gaps.** Correctness mode asserts the gap is within 1e-7·max(1, F_ref). Benchmark mode only records it.
- **Common KKT stopping.**
  - Baselines run in one public call with a callback evaluating the global natural KKT every `--check-every` steps.
  - `check_harness()` replays the trajectory with `atol=0` to show momentum is not restarted.
  - Baseline products include the 51 forward / 50 adjoint spectral setup, and the stopping-check products are counted.
- **Honest censoring.** `success_oracle`, `censored` and `wall_cap_hit` come from an external float64 oracle.
- **IR false-certificate guard.** In float64, the harness asserts that the IR residuals match the external oracle and raises on a disagreement. In float32 it only records `float64_oracle_disagrees`.
- **Gating.** Benchmark mode requires `--gate2-passed`. `--large` is rejected in correctness mode. Dense allocation guards check `--max-mib` and `/proc/meminfo`.
- **Dual gap.** I re-derived weak duality. For any u with Aᵀu + λ ≥ 0 (positive mode) or |Aᵀu|_∞ ≤ λ (signed mode), F(x) ≥ −u·y − (m/2)‖u‖².
  - With u = α r/m, the code's dual value, the feasibility bound `α_max` and the optimal α all match.
  - So the recorded `dual_gap` is a valid upper bound on F(x) − F\*, up to floating point. This holds even when the reference is uncertified.
- **Caveats.**
  - The acoustic problems are synthetic free-field propagation, not Acoular captures.
  - Wall time includes instrumentation, selection, callbacks, practical certification and native column copies. Setup, reference and external oracle are excluded.
  - Repeats are exploratory.

## 6. Independent numerical probes (bounded; completed work only)

Metrics were the objective gap F(x) − F_ref and the natural and raw infinity KKT norms of the returned point, computed by my own NumPy code. Coefficient error was never used. The reference was a separately certified NumPy FISTA (≤1e-11 natural residual; skipped if uncertified). The test-suite oracles also compare against scipy `nnls` and `lstsq`.

**Broad sweep (`num.py`, 1133 s, 3124 probes).**
- Grid:
  - positive and signed modes;
  - Gaussian, coherent, rank-deficient, duplicated/zero-column, ill-conditioned (σ 1…1e-5) and badly scaled (1e-3…1e3) dictionaries;
  - shapes 30×12, 12×40, 40×40;
  - seeds 0–5;
  - λ/λ_max ∈ {0, 0.02, 0.2, 0.7};
  - warm starts none, random, negative, optimum;
  - tol 1e-9, t=1000.
- The minimum objective gap was −5.8e-11, so IR was never meaningfully better than the certified reference.
- 3071 probes (98.3%) passed my strict criterion: `converged`, natural ≤ 1e-9, gap ≤ 1e-8·max(1, F_ref), feasible.
- 53 probes were flagged. The first 25 printed were all `max_iter` with small errors (gap ≤ 2.6e-6, natural ≤ 1.6e-6) in coherent or ill-conditioned λ=0 cases.
- The summary maxima (gap 10.2, natural 1.66) came from the other 28 flagged probes, whose rows I did not print. The bounded rerun localised the severe cases to bad scaling.

**Bounded rerun (`num2.py`, 718 probes, 136 s; seeds 0–2, warm starts none/random).**
- 6 probes were flagged. All were `max_iter`; none was a false `converged`.
  - 2 ill-conditioned λ=0 probes had natural about 2e-8 and gap 5e-9.
  - 4 severe bad-scaling probes with a random warm start (cond ≈ 1.5e6):

| Mode | seed | λ/λ_max | Natural | Gap |
|---|---|---|---:|---:|
| positive | 2 | 0.02 | 0.26 | 0.23 |
| positive | 2 | 0.7 | 0.35 | 0.38 |
| signed | 1 | 0.2 | 0.65 | 1.11 |
| signed | 2 | 0.02 | 0.37 | 1.08 |

**Targeted inspection (`bad.py`).**
- The same family with a different random warm start converged by sweep ~400 (natural 9.9e-10, F − F_ref = 0).
- F − F_ref fell from 332 to 1.0e-3 and the natural residual from 1203 to 5e-3 over sweeps 0–200. About 97% of inner solves exhausted the 100-step budget, with accepted L up to about 1e6.
- That warm start is not the same draw as the four failing probes. It shows slow monotone progress rather than a stall. It is not proof that those four probes converge with more sweeps.
- `bad2.py` (larger t) was aborted without output, so I have no evidence about those four at larger t and do not claim it.

**Interpretation.**
- All four severe probes returned `max_iter` with a correct (not false) certificate, natural equal to raw, and a feasible point.
- This is a budget-limited stress limitation. The spec allows finite budgets to return an uncertified point and makes no conditioning-free guarantee.

**Not covered.**
- No random float32 sweeps.
- No large CMF sweeps beyond the benchmark artifacts.
- No GPU.
- No cross-seed statistics beyond the above.

## 7. Scope of the Gate 2 PASS and residual limitations

1. **Scoped PASS.** The CPU float64/float32 numerical correctness gate passes for the implementation as written. It is not a comprehensive guarantee.
2. **Budget-limited stress cases are explicit limitations.**
   - Bad column scaling (cond ≳ 1e6), ill-conditioned λ=0 and coherent λ=0 problems can return honest `max_iter` at the default budgets (`inner_max_iter=100`, `t=1000`).
   - The natural residual is not an objective-gap bound: the severe cases show gaps of order 1 at natural 0.3–0.65.
   - Users should check `diagnostics['status']`.
3. **No unproven theory is claimed.** There is no whole-sequence convergence, no local linear rate, no permanent support identification, and no speed guarantee. The code, README and docstrings respect this. Only the specified exact-arithmetic descent/residual argument applies, and the finite-precision engineering safeguard is not claimed as a global proof.
4. **CUDA** was not run. The 2 tests skip here.
5. **Counters** are work proxies, not FLOPs. Callback products are outside them.
6. **Sequential host-driven rows** and B-wide wrapper overhead for batched operators are known costs and are counted. Dense native `select_columns` copies columns.

## 8. Stale wording (parent to update; I cannot edit)

- `benchmark_notes.md`, final section: "Independent post-measurement audit pending" is now stale; this audit completes it.
- The notes' Products and working sets table omits the NNLS control row. I computed it from the JSONL: IR 646.2, FISTA 321.0, ISTA 585.0 width-weighted units, so IR does more work there.
- It also omits that IR inner-solve budget exhaustion was nonzero in the CMF cases (§9.3).
- The harness docstring and the README gating text ("Before Gate 2 … After Gate 2") remain accurate. Gate 2 CPU correctness has now passed, but the performance-related wording must stay "bounded exploratory".

## 9. Post-Gate-2 benchmark and memory audit

I inspected `benchmark_notes.md`, `exploratory.jsonl` (18 records), `exploratory_large.jsonl` (6), `exploratory_nnls.jsonl` (3) and `memory_isolated.jsonl` (6) by parsing the saved JSONL.
- I did not rerun any measurement.
- I could not inspect the scratch memory-probe script, which is not in the repository or the artifacts. For the memory probe I audited only its method as described in the notes and its saved output.

### 9.1 Provenance and consistency

- All 27 benchmark records share:
  - `script_sha256` 44745d48… (the unchanged harness);
  - torch 2.12.0+cu130, 1 thread, seed 7957, float64, CPU, benchmark mode, batch 1.
- The three repeats of each of the first two cases produce identical quality and identical counts and working-set histories (checked by comparing the counters, iteration counts and working-set lists across repeats). Only the wall time varies.
  - Repeat wall times agree within 2%.
  - These are three repeats of the same instance, not independent instances.
- The notes' table values match the JSONL. I checked:
  - times (median of three where applicable);
  - natural and raw residuals;
  - gaps;
  - the 18/6/3 record counts.

### 9.2 Accuracy, certification and censoring

| Case | Reference | IR | FISTA | ISTA |
|---|---|---|---|---|
| Dense 512×8192 | certified (ref natural 1.3e-13, 860 steps) | converged, natural 7.5e-10, gap 2.4e-16, dual 4.9e-10 | reached (1.2e-9), gap 9.8e-16 | reached (1.5e-9), gap 9.4e-16 |
| CMF 240×1024 | certified (3.8e-13, 1030 steps) | converged, 5.7e-10, gap 5.3e-17 | reached (3.8e-9), 1.5e-15 | reached (4.4e-9), 3.1e-15 |
| Dense 1024×32768 | **uncertified** (ref natural 2.93e-8, 481 steps, 10 s cap) | converged, natural 1.70e-10, dual gap 1.38e-10 | wall-capped, natural 2.93e-8 | wall-capped, natural 8.15e-8 |
| CMF 992×4096 | certified (7.5e-14, 3210 steps) | converged, 6.3e-11, gap 1.7e-18 | reached (1.25e-9), 9.5e-16 | wall-capped, natural 6.44e-6, gap 4.5e-6, support 16 |
| NNLS 256×1024 (λ=0, support 128) | certified (5.8e-13, 220 steps) | converged, 7.2e-9, gap 9.0e-13 | reached (3.6e-9), 1.6e-13 | reached (6.3e-9), 5.4e-13 |

- In every record natural equals raw, and each record's `success_oracle` and `censored` flags agree with the external float64 natural residual against `tol_abs`.
- **Large dense reference.**
  - The reference is itself the FISTA trajectory capped at the same 481 steps and 10 s (its natural residual equals FISTA's, 2.93e-8).
  - The recorded FISTA gap of exactly 0.0 is an artifact of that identity.
  - The IR gap of −8.2e-13 and the ISTA gap of 7.0e-12 are therefore not verified objective gaps.
  - The only valid evidence for the IR point is the external natural target (1.7e-10 against tol 1.18e-9) and the weak-duality dual-gap bound (1.38e-10, derived in §5).
  - The notes already say this; I confirm it.
- Cap behaviour:
  - Only the baseline runs hit the cap.
  - IR never did.
  - The capped times are right-censored (≥10.0 s); I computed only lower bounds on ratios.
- Other references were certified, so their gaps are valid. All reported gaps are small, and there is no certified instance where IR's objective is worse than a certified reference by more than 9e-13.
- **NNLS control.**
  - IR returns support 749, FISTA 1024 and ISTA 722 on a non-unique problem, so coefficients are not comparable.
  - The objective gaps and KKT residuals above are the valid comparison.

### 9.3 Counters and working sets

- I recomputed the notes' call totals from the JSONL. Public-call totals (`solve` plus `stopping` phases) are:

| Case | Solver | Forward / adjoint calls | Source-width-weighted products, in units of n |
|---|---|---|---:|
| Dense 512×8192 | IRFISTAB | 743 / 498 | 49.0 |
| same | FISTA | 480 / 479 | 959 |
| same | ISTA | 491 / 490 | 981 |
| CMF 240×1024 | IRFISTAB | 777 / 521 | 77.4 |
| same | FISTA | 557 / 556 | 1113 |
| same | ISTA | 1547 / 1546 | 3093 |
| Dense 1024×32768 | IRFISTAB | 1011 / 677 | 30.9 |
| CMF 992×4096 | IRFISTAB | 1929 / 1290 | 68.6 |
| NNLS 256×1024 | IRFISTAB | 578 / 388 | 646 |
| same | FISTA | 161 / 160 | 321 |
| same | ISTA | 293 / 292 | 585 |

  - These match the notes' table; the NNLS rows were the reviewer's addition from JSONL. **Parent post-audit reconciliation:** the original NNLS F/A cells inadvertently counted only `solve`, omitting 10+10 FISTA and 22+22 ISTA stopping products. Cells above now include both phases (161/160 and 293/292), as independently recomputed from raw JSONL; weighted totals 321 and 585 were already correct.
  - The FISTA/ISTA totals include the 51/50 spectral setup and the stopping checks.
  - The IR counters (full 8–12 and restricted counts per case) agree with the notes (e.g. 8+8 full and 735+490 restricted for the first dense case) and with an independent recomputation of width-weighted work from the `counts` phase table.
- Working sets for IR:

| Case | Working-set sizes (first → last, max) | Solves |
|---|---|---|
| Dense 512×8192 | 599 → 3 | 6 |
| CMF 240×1024 | 174 → 3 | 7 |
| Dense 1024×32768 | 824 → 3 | 7 |
| CMF 992×4096 | 371 → 3 | 10 |
| NNLS | 569 → 754 | 6 |

  - In the sparse problems the set shrinks to the true support of 3.
  - In the NNLS control it grows and the final support is 749.
- No native-restriction or batch fallbacks were used (`restriction_fallbacks` and `batch_fallback_calls` are 0 in all IR records).
- **Not stated in the notes:**
  - Zero backtracking rejections occurred and the accepted L never exceeded 1 in any record, so the benchmark never exercised curvature growth. The initial default L=1 happened to be a valid majorizer on these instances.
  - Inner-budget exhaustion was nonzero in some solves. CMF 240×1024: 2 of 7; large dense: 1 of 7; large CMF: 6 of 10. The dense 512×8192 case and the NNLS control had none.

### 9.4 Runtime claims

| Case | IR | FISTA | ISTA | IR advantage |
|---|---:|---:|---:|---|
| Dense 512×8192 | 0.078 s | 0.998 s | 1.017 s | 12.7× vs FISTA, 13.0× vs ISTA (median of 3, same instance) |
| CMF 240×1024 | 0.132 s | 0.156 s | 0.427 s | 1.18× vs FISTA, 3.2× vs ISTA |
| Dense 1024×32768 | 0.287 s | ≥10.01 s | ≥10.00 s | ≥35× lower bound (right-censored) |
| CMF 992×4096 | 0.400 s | 3.23 s | ≥10.0 s | 8.1× vs FISTA (single run), ≥25× vs ISTA (lower bound) |
| NNLS control 256×1024 | 0.062 s | 0.0166 s | 0.0246 s | IR is **slower**: 3.8× vs FISTA, 2.5× vs ISTA |

- The notes' table values are correct and censoring is handled honestly (lower bounds only, with no ratio from censored runs).
- The accuracy at stop is not identical across solvers. IR certifies after every outer sweep and typically ends 2–8× below the target, while baselines check every 10 steps. If anything this favours the baselines.
- The work proxy is a poor predictor of CMF runtime.
  - CMF 240×1024: 14× fewer width-weighted products for IR, but only a 1.18× time gain.
  - CMF 992×4096: 54× fewer products, 8× faster.
  - Per-call overhead and the cost of `CMFPropagation` column selection evidently dominate.
  - The notes already say the proxy is not a CMF cost model.

### 9.5 Memory probes

- `memory_isolated.jsonl` has 6 fresh-process records for the first two cases. Each reports scope "fresh process; after instance generation; public solve + final external oracle", `reset_peak: True`, `success_oracle: True`, and the same natural residuals as the main runs.

| Problem | Solver | RSS before (KiB) | Peak after reset (KiB) | Increment (KiB) |
|---|---|---:|---:|---:|
| Dense 512×8192 | IRFISTAB | 536796 | 580308 | 43512 |
| same | FISTA | 536836 | 540080 | 3244 |
| same | ISTA | 536960 | 540076 | 3116 |
| CMF 240×1024 | IRFISTAB | 511112 | 551100 | 39988 |
| same | FISTA | 511160 | 511736 | 576 |
| same | ISTA | 511052 | 511648 | 596 |

- The notes' table matches the JSONL exactly, and the harness lifetime-RSS field (611404 KiB for the first run's solvers) is correctly not used for comparison.
- The measured increments are higher for IR by about 40 MB. There is no memory-reduction claim, and the notes make none.
- Interpretation limits:
  - One cold run per solver at two shapes. The increment is nearly the same (40–44 MB) for dense and CMF despite roughly 10× different problem size.
  - The dense native column copy at working-set size 599 is only about 2.5 MB.
  - Most of the 40 MB is therefore plausibly shape-independent cold-start cost (first-use kernels or allocator pages in the IR code paths).
  - I cannot establish that from the artifacts. It is a plausible explanation only, so scaling with problem size is not inferable.
  - The increments include the external final oracle, common to all three solvers.
  - No large-shape or CUDA memory measurement exists.
- At audit time the probe script and earlier parser error (empty `/proc/self/status` fields) were described only in the notes. **Parent closure:** the exact successful probe is now preserved in `memory_probe.md`.
  - The fix was scoped to `VmRSS` and `VmHWM`, and all six probes were rerun.
  - That is a harness-scratch error, not a solver issue.

### 9.6 Findings from this audit (all Minor or Suggestion)

- **B-1 (Minor, closed by parent documentation update).** The notes' stale "audit pending" status is replaced with completed audit; NNLS public-call/weighted counts, inner-budget exhaustion and absence of backtracking are now included.
- **B-2 (Minor).** The large dense reference is the capped FISTA trajectory itself, so the FISTA gap 0.0 is an artifact and the ISTA/IR numerical gaps there are unverified. The dual bound is the valid certificate.
- **B-3 (Minor, reproducibility portion closed by parent).** Memory probes remain one cold run per solver at two shapes. The exact successful parent probe is now preserved in `memory_probe.md`; increments cannot establish memory scaling.
- **B-4 (Minor).** No benchmark instance exercised backtracking growth (all accepted L ≤ 1). The default L=1 start was a valid majorizer by luck of problem scaling. Instances with larger ‖A‖²/m would exercise the safeguard and could change the timing picture.
- **B-5 (Minor).** The width-weighted work proxy does not predict CMF runtime (14× fewer products, 1.18× time).
- **S-4 (Suggestion).** For any claim beyond "bounded exploratory", add independent instances and seeds with warm-up and repeated order randomisation. Preserve the memory-probe script in the repository, and add a GPU measurement before any GPU claim.
- **S-5 (Suggestion).** The NNLS (λ=0, m<n) and ill-conditioned problems show IR's behaviour is not uniformly favourable. Any summary should keep the NNLS control next to the sparse wins.

### 9.7 What the artifacts do and do not support

- **Supported (bounded exploratory, CPU float64, one thread, seed 7957, positive mode, three generated sources, λ/λ_max = 0.1, tolerance 1e-6 × initial correlation scale):**
  - On these sparse instances the IR solver reached the common natural-KKT target with 12.7× (dense 512×8192), 8.1× (CMF 992×4096) and 1.18× (CMF 240×1024) lower wall time than FISTA, and ≥35× (dense 1024×32768, right-censored) lower than the wall-capped baselines.
  - At the same time it used 14–54× fewer width-weighted products.
  - Its returned points have natural and raw residuals and objective gaps (certified where the reference is certified) comparable to or better than the baselines.
- **Not supported:**
  - Any general speed, convergence-rate or memory claim.
  - Any advantage on non-sparse or NNLS-type problems (IR is slower on the NNLS control).
  - Any memory-reduction claim (isolated increments are higher for IR).
  - Any claim about GPU, batches, generic/B-wide fallbacks, signed or float32 performance, other seeds, Acoular-captured data, or the full experiment-plan menu.

## 10. What the parent must do and what to rerun

- **Nothing blocks the Gate 2 PASS.** No source correction is required.
- **Optional documentation/test fixes:**
  - M-1 (tautological assertion) and M-2 (extra seeds) in the tests.
  - `benchmark_notes.md` has been updated by the parent for B-1; `memory_probe.md` preserves the successful measurement script. Remaining optional test improvements are unchanged.
- If any source or test changes are made, rerun `uv run pytest -q tests/test_ir_fista_b.py`, `uv run pytest -q`, `uv run ruff check .` and the three `--mode correctness` harness commands.
- Any future benchmark must keep its exploratory framing and the §9.7 limits. It should state product counts per step (3+2 for IR), censoring, and whether the reference was certified.

**Final verdict.** Gate 2 numerical correctness is PASSED (scoped CPU). The bounded exploratory benchmark and memory artifacts are audited as honest descriptive evidence with Minor findings only. No Critical or Major findings are open.