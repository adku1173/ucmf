# Experiment plan: bounded correctness and performance protocol for IR-FISTA-B against ISTA/FISTA

Baseline: `ab87210`. Specification: `.pi/research/ir-fista-b/algorithm_specification.md` v1, Gate 1 pending. Status: **design only**. No code was changed or run, and no file was written. I had no shell tool, so every command below is proposed and unexecuted.

**Governing rule: no performance claim before Gate 2.** Only Stages H, M and C (§0) may run before the Gate 2 verdict. Stage C is Gate 2 evidence. All speed, work and memory comparisons are Stages P1/P2, and they are released only after Gate 2 passes.

---

## 0. Staging and claim policy

| Stage | Purpose | Class | Allowed before Gate 2? | Output allowed |
|---|---|---|---|---|
| **H** | Validate the harness using ISTA/FISTA only: oracle, counters, stopping, replay, reference pipeline | correctness of the *instrument* | Yes | Harness pass/fail. No IR data. |
| **M** | Operator micro-benchmark: matvec time versus column width `p` for `MatrixMult` and `CMFPropagation`, including `select_columns` | characterization, no IR | Yes | Cost curves `t_F(p)`, `t_A(p)` |
| **C** | IR-FISTA-B correctness against independent references (this is the Gate 2 evidence) | confirmatory correctness | Yes, on the candidate implementation | Pass/fail table |
| **P1** | Performance exploration: ablations, scaling, regime mapping | exploratory | **No. Only after Gate 2 passes.** | Descriptive only, no hypothesis tests |
| **P2** | Pre-registered performance comparison | confirmatory | **No. Only after Gate 2 passes and P1 is frozen.** | Claims restricted to the measured regimes |

- **Claim policy.** Permitted wording is "in this measured regime, on this host, with this accuracy target, IR-FISTA-B reached the target in X of N instances, with time ratio R [CI]". Forbidden: general speed guarantees and sparsity-dependent speed-up claims outside the sampled regimes. Also forbidden are claims that the spec marks unproven: local linear rate, support identification, whole-sequence convergence.
- **NMSE** is reported only as an information column for uniquely identifiable instances. It is never a pass/fail or stopping metric. Coefficients are not compared for non-unique instances (rank-deficient, `n>m`, `λ=0`, duplicate columns).

---

## 1. Environment, reproducibility, hardware (as observed read-only)

**Host (from `/proc`, `/sys`):**
- CPU: AMD Ryzen 7 8845HS, 8 cores / 16 threads, a mobile part.
- Memory: 15.6 GB total, about 8.9 GB available at inspection.
- Governor: `powersave`. Expect frequency and thermal noise.
- **No NVIDIA driver node** (`/proc/driver` lists only `rtc`). The iGPU is a Radeon 780M.
- GPU rows are therefore expected to be **not-run (device unavailable)**. The harness probes `torch.cuda.is_available()` and `torch.version.hip` and records the result. A GPU row is never silently replaced by a CPU row.

**Pre-run environment capture.** The runner must write these to `$RUN/env.json` and `$RUN/env.txt`:
```bash
RUN=.pi/research/ir-fista-b/runs/$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $RUN
git rev-parse HEAD; git status --porcelain=v1                       # baseline ab87210 + uncommitted candidate
git diff --stat ab87210 -- src tests examples                       # candidate = uncommitted working tree
sha256sum src/ucmf/optim/ir_fista_b.py examples/benchmark_ir_fista_b.py uv.lock
uv sync --frozen                                                    # network-dependent; the earlier attempt timed out, pre-fetch/retry
uv pip freeze > $RUN/pip_freeze.txt
uv run python - <<'PY'
import acoular, torch, numpy, scipy, platform, sys
import sklearn  # optional reference; record absence rather than failing
print(sys.version, platform.platform(), torch.__version__, numpy.__version__, scipy.__version__,
      torch.get_num_threads(), torch.cuda.is_available(), torch.version.cuda, getattr(torch.version,'hip',None))
PY
lscpu -e; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
grep -E 'MemTotal|MemAvailable' /proc/meminfo; nvidia-smi -L || echo "no-nvidia"
```
Import Acoular before NumPy and `ucmf` (repo convention).

**Determinism and pinning:**
- Master seed 7957. Each case seed is `SeedSequence([7957, crc32(case_id), instance_idx])`.
- Generate data in float64 NumPy, then cast.
- Torch RNG uses seeded CPU `Generator`s.
- Never consume the global RNG.
- Store every seed in the result record.
- Threads: run `torch.set_num_threads(k)`, `threadpoolctl` for BLAS, and `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=k`. Acoular sets `OPENBLAS_NUM_THREADS=1` on import.
- Thread configurations: **k=1** (`taskset -c 2`) and **k=8** (`taskset -c 0-7`). First verify with `lscpu -e` that CPUs 0–7 are 8 distinct physical cores. Otherwise choose one thread per core.
- Governor cannot be changed (no privileges). Mitigation:
  - Randomized/interleaved solver order per repeat.
  - Discard the first call as warm-up.
  - Insert a 10 s cooldown between cases longer than 60 s.
  - Record CPU MHz before and after each timed run from `/proc/cpuinfo`.
  - Flag runs whose repeat CV exceeds 10%.
- Determinism check: two CPU runs at k=1 must give identical iteration counts and identical working-set histories. Returned `x` should be bitwise identical (within 1 ulp-level if BLAS is multithreaded).
- Dtype: float64 is primary (matches `ucmf.acoular` defaults). float32 is an exploratory subset only. Disable TF32 and use `torch.set_float32_matmul_precision('highest')` on GPU, as `benchmark_operators.py` does.
- Run under `torch.no_grad()` (spec §10).
- Memory guard before each case: require `MemAvailable > 3×(dictionary bytes) + 1 GiB`. Otherwise record `not-run (memory guard)`.

---

## 2. Metrics and independent oracle

All evaluation is by a **harness oracle**, independent of the solver's own diagnostics. It is a separate float64 implementation using the full operator.

For `f = ‖Ax−y‖²/(2m)`, `g = Aᵀ(Ax−y)/m`, positive mode `s = g+λ`:

| Symbol | Definition |
|---|---|
| `F(x)` | `‖Ax−y‖²/(2m) + λ‖x‖₁` (`+I_C`) |
| `r_nat` | `‖min(x, s)‖_∞` (positive); signed: `‖x − S_λ(x−g)‖_∞` via stable piecewise form |
| `r_raw` | spec §3 `e_j` raw subgradient residual, infinity norm |
| `s0` | fixed scale `‖Aᵀy‖_∞/m` (positive: `max_j(Aᵀy)_j/m`). It does not depend on `x0` or on the solver. |
| feasibility | `min(x) ≥ 0` exactly (positive mode) |
| `F_ref` | Best certified reference objective, from the pipeline below |
| `ĝ` (normalized gap) | `(F(x)−F_ref)/(F(0)−F_ref)` when `F(0)−F_ref > 1e-12·F(0)`. Otherwise (`x*≈0`, trivial instance) `(F(x)−F_ref)/max(F(0),tiny)` flagged "trivial". |
| `ĝ_dual` | Reference-free certified gap upper bound for `λ>0`, from one extra vector (below). |
| `supp` | `|{x_j>0}|/n` of the returned and reference points (the covariate for performance) |

**Dual gap certificate (λ>0, positive mode).**
- Dual feasible set: `Aᵀw ≥ −λ1`, `D(w) = −(m/2)‖w‖² − wᵀy`.
- Given `x`, set `w₀ = (Ax−y)/m` and `α_max = min(1, λ/max_j(−(Aᵀw₀)_j)₊)`, with `α_max=1` if no `g_j<0`.
- Set `α* = clip(−w₀ᵀy/(m‖w₀‖²), 0, α_max)` and `w = α* w₀`.
- Then `F(x)−F* ≤ F(x)−D(w) =: gap_dual`, so `ĝ_dual = gap_dual/(F(0)−F_ref)`.
- Signed mode uses `|Aᵀw|_∞ ≤ λ`.
- For `λ=0` the bound is vacuous unless `g≥0`, so `F_ref` alone is used.
- The certificate is computed by the harness, outside timing, from the already-needed `g`.
- This gives gap results that do not depend on a reference solver. The interval `[D, F(x)]` also bounds `F_ref` quality.

**Reference pipeline** (harness only, outside timing, float64):
1. Identity: closed form, `x* = prox(y, mλ)` (spec §2: threshold `mλ`).
2. `λ=0`, positive mode: `scipy.optimize.nnls`. Unique only when `A` has full column rank.
3. `λ>0`: `sklearn.linear_model.Lasso(alpha=λ, positive=True|False, fit_intercept=False, tol=1e-14, max_iter=10⁶)`. Its objective equals ucmf's normalized objective. Optional `cvxpy` if installed. No new dependency is required.
4. Polish: existing FISTA f64 started from step 3, with an oracle check every 100 steps, until `r_nat ≤ 1e-11·s0` or budget (2·10⁶ iterations or 30 min). Record `ref_quality ∈ {certified (r_nat≤1e-10 s0 and gap_dual ≤1e-10 F0), uncertified}`.
5. Uncertified references are excluded from gap-based pass/fail. KKT and cross-solver checks still apply.
6. `F_ref` comes **only** from references, never from the solver under test. If `F(x_IR) < F_ref − 1e-12·F(0)`, flag "reference deficient". That is a finding, not a failure.

**Target ladder (common accuracy), float64:** `ε_K ∈ {1e-4, 1e-6 (primary; spec default tol=1e-6 expressed relative to s0), 1e-8}`.
- A run **succeeds** iff the oracle `r_nat(x_returned) ≤ ε_K·s0` on the returned point.
- IR-FISTA-B is invoked with `tol = ε_K·s0`, `rtol = 0` (absolute normalized units, so the check matches the oracle).
- float32: `ε_K ∈ {1e-3, 1e-4}`. A target the f32 reference FISTA cannot reach within budget is dropped for both solvers as "unreachable at dtype".
- **Tier B (gap-matched, optimistic, disclosed):** target `ĝ ≤ 1e-8` as well as `ε_K`.
  - IR-FISTA-B is run on the ladder `tol = ε_K s0 · 4^{−j}`. The first run whose oracle `ĝ` passes is reported, and the search cost is disclosed separately.
  - FISTA uses first-hit on both criteria.
  - Reported as a secondary table because KKT residual does not bound the gap.
- **Honesty cross-check on every run:**
  - Solver status `converged` while the oracle says `r_nat > ε_K s0·(1+1e-9) + 32·eps·(‖A‖²/m·‖x‖_∞+‖g‖_∞)` is a **false certificate (correctness failure)**.
  - Oracle passes while the solver reports not converged: "conservative", info only.

---

## 3. Baselines and stopping in the common-accuracy protocol

ISTA/FISTA have no certificate. The harness supplies one **without source modification** via a harness-side `IterationCallback` subclass. It uses the existing `on_step_end(solver, x, A=…, y=…)` metadata and sets `solver.stop=True`. Checking through callbacks keeps FISTA's momentum intact. Re-calling `forward` with `x0` would silently restart momentum, so it is used only for the explicit "fixed-restart" baseline B3.

**Baselines:**
- **B1 FISTA** (`ucmf.FISTA`, `nonnegative=True`), default `gamma=None`.
- **B2 ISTA** (`ucmf.ISTA`). Expected to be censored on hard cases. Run it only on exploratory small and mid cases.
- **B3 (exploratory only)** FISTA with fixed restarts every K iterations, implemented harness-side as repeated `forward(t=K, x0=prev)`. It is a stronger baseline than any existing solver and addresses the momentum-reset confound in IR-FISTA-B.
- **External references (not timing baselines)** sklearn Lasso and scipy nnls are used only for references. Their timing is not scored.

**First-hit discovery for B1/B2** (exact, optimistic for the baseline):
1. *Stage 1.* A geometric check schedule `k_j = ⌈1.05^j⌉` finds the first checked `k` with oracle `r_nat ≤ ε_K s0`. This gives a bracket `(N_lo, N_hi]`.
2. *Stage 2.* Deterministically replay with the same inputs, checking **every** iteration only for `k > N_lo`. This yields the exact first-hit `N*`. FISTA's residual is non-monotone, so first-hit is optimistic for the baseline.
3. *Timed run T.* Replay with `t=N*`, no callbacks, bare operator. Its output is re-verified by the oracle after timing.
4. *Variant B1-cert, practical stopping.* Report `N₅₀ = ⌈N*/50⌉·50` plus the cost of `N₅₀/50` full certification checks (1 full F + 1 full A each). It is also timed on a bare operator, with explicit full-product checks in a timed loop wrapper.

Oracle check time is excluded from run T. It is charged in the **counting run C** and in B1-cert.

**IR-FISTA-B primary run:** `IRFISTAB(positive=True, tol=ε_K s0, rtol=0)`, other parameters default (`tau=1e-4`, `delta=None`, `inner_tol=0.1`, `inner_max_iter=100`, `backtracking_factor=2`).
- Outer budget `t=T_out` (cap below). Its internal certificate is part of the algorithm, so its time includes all certification products. This asymmetry is stated explicitly in every table.

**Censored budgets (apply to all solvers, recorded per run):**
- Baselines: `N_max` and a wall cap enforced by a harness callback that checks `perf_counter` and sets `stop`. IR-FISTA-B: outer `t` cap and wall cap enforced by an end-sweep callback.
- Dense CPU runs: `N_max = 5·10⁴` (FISTA), `10⁵` (ISTA), wall cap 300 s (P2), 120 s (P1/C), `T_out = 500` outer sweeps.
- Hard external kill: `timeout 15m` per process.
- If the cap is hit, the run is **right-censored**: record the best `r_nat` and `ĝ` reached and `time ≥ cap`. Never impute success.
- Summary statistics:
  - Success rate per solver.
  - Time-ratio only for pairs where both succeed.
  - One-sided bounds `ratio < T_IR/cap` or `> cap/T_FISTA` when one is censored. Pairs where both are censored are excluded from ratios and reported as "unresolved".
  - PAR2 (penalized mean with censored value `2×cap`) as a secondary scalar.
  - Dolan–Moré performance profiles on time and on full-matvec equivalents.
  - Equal-budget anytime curves (gap and `r_nat` versus cumulative work and versus time).

---

## 4. Scope definitions (setup versus runtime), operation and memory counting

**Scopes (every record carries a `scope` tag):**
- **S0 (excluded from solver time, but reported).** Instance generation, Acoular capture and assembly, H→dictionary build, reference pipeline, oracle evaluation, `float64` twin construction.
- **S1 (primary).** One public `solver(y, x0, t, operator, lam, gamma)` call as a user pays for it. It includes:
  - input preparation;
  - baselines' default spectral estimate (`gamma=None`: **exactly 51 forward and 50 adjoint** full products, from `default_gamma_from_operator`);
  - IR-FISTA-B's initial `L` handling and backtracking;
  - IR native `select_columns` copies;
  - final device synchronization.
  - Excluded: oracle checks, `.cpu()`/`.numpy()` transfers.
- **S2 (secondary, steady state).** `gamma` is passed explicitly to every solver, so the spectral setup is excluded. Variants:
  - IR with `gamma=None` (L starts at 1 and relies on backtracking).
  - IR with `gamma = 1/(1.05·‖A‖²/m)` (matched to the baseline).
  - IR with `gamma = 10/(…)` (oversized, to exercise backtracking, spec §6).
- **No S3.** The Acoular-adapter end-to-end path (`bf.synthetic`) is out of scope because no IR adapter exists. Acoular is used only for instance capture (§6.2).

**Three run types per (instance, solver, config), kept separate:**
- **T (timing):** bare operator, no callbacks, diagnostics on (internal, cannot be turned off by the spec), `cuda.synchronize` before and after. 1 warm-up (discarded), then 5 repeats if the median is under 20 s, 3 if under 60 s, otherwise 1 (disclosed). Report median and IQR.
- **C (counting and trace):** operator wrapped in a harness `CountingOperator(LinearOperator)`. Callbacks record the per-sweep trace and oracle values. Not timed.
- **M (memory):** fresh `spawn` subprocess, instance loaded from `.npy`/`.pt` in `$RUN/inst/`. Not timed.

**Operation counting (the wrapper is an independent cross-check of the solver's own counters):**
- The wrapper records for every `forward` and `adjoint` call: `kind`, `width p = x.shape[-1]` for forward (n or p), `underlying_rows = batch dim of the tensor`, and `native|fallback` for restricted calls.
- It exposes `select_columns` only if the wrapped operator has it. It returns a counting wrapper of width `p`. A second wrapper variant **without** `select_columns` forces the solver's scatter/gather fallback.
- Risk: if the solver uses `isinstance(op, MatrixMult)` instead of duck typing, the wrapper hides native restriction. In that case counts fall back to solver diagnostics only, and this must be recorded.
- Count taxonomy:
  - *Full* = width n. *Reduced* = width `p<n` via native restriction. *Fallback* = a reduced logical product executed at width n.
  - Attribution for IR (from solver diagnostics, cross-checked against wrapper totals by kind and width): initial scan; reduced gradient; **backtracking trial forwards (accepted plus rejected)**; sub-problem residual adjoints; **full post-update recompute (1 full F + 1 full A per accepted outer row update)**; final certification; and the initial `L` or `gamma` estimate if any.
  - FISTA/ISTA: setup `51F+50A`, then per iteration `1F+1A`. Check that the wrapper gives exactly `(N+51)F + (N+50)A` with `gamma=None` and `N F + N A` with `gamma` given. Stage H fails if not. Oracle/check calls carry a `phase=oracle` tag and are never merged into solver counts.
- **Work metrics:**
  - `FME` (full-matvec equivalents) `= Σ_calls (p_call/n)·rows_logical`, with fallback calls charged at `p=n`. It is the cost proxy for dense `A`.
  - `UW` = underlying-batch work `= Σ_calls (p_call/n)·underlying_rows`.
  - Raw call counts per type and width bucket.
  - Restriction copy volume `Σ_k m·p_k` elements (dense `MatrixMult`) or `c·p_k` complex (CMF), computed from working-set size history.
  - Number of host syncs is out of scope on CPU. On GPU, exploratory `torch.cuda.set_sync_debug_mode('warn')` counts only.
- **Cost-model check:** Stage M gives measured `t_F(p)`, `t_A(p)`, `t_select(p)`. `T_model = Σ_calls t(p_call) + Σ t_select`. The overhead fraction `(T_measured − T_model)/T_measured` quantifies Python, sync and allocation overhead. A large fraction explains why a work saving does not become a time saving.
- For `CMFPropagation` the product cost is about `c²·p` (forward) and `c²·p + c³` (adjoint), so reduced width helps linearly in `p` beyond the constant `c³` term. `FME` alone is therefore not a CMF time proxy. Always report the Stage M model alongside it.

**Working-set statistics (from per-sweep diagnostics plus oracle reconstruction in begin-callbacks):**
- `|W_k|/n` history.
- `p_max`.
- Final `|W|` versus `|supp x*|` (overshoot ratio).
- Number of outer sweeps.
- Inner iterations per sweep.
- Reactivation count (coordinates in `supp(x^k)` or `W_k` that leave and later re-enter), reconstructed by the oracle from the full gradient and `tau`.
- Backtracking rejections and `L` history.
- Fraction of sweeps ending on the inner target versus the inner budget.

**Memory protocol:**
- CPU: in the spawned child, record `VmRSS` before the solve, write `5` to `/proc/self/clear_refs` (resets `VmHWM`), run the solve, then read `VmHWM`. Incremental peak = `VmHWM − VmRSS_before`.
- GPU (if ever available): `reset_peak_memory_stats`, then report `max_memory_allocated` and `max_memory_reserved` minus the baseline. Report which scope was measured. Allocator caching and Python-level numbers are caveats.
- **Model:** IR incremental peak ≈ `8·m·p_max` bytes (dense native column copy) + `O(8(n+m))`·small constant + diagnostics. FISTA ≈ `O(8(n+m))`.
- **No-Gram test:** one case with `n≫m`, small `λ`, `p_max > m` (e.g. `m=500, n=20000`). It passes iff IR incremental peak `< 1.5·(8·m·p_max + 40·8·(m+n)) + 20 MiB`. A `p×p` Gram (`8p²` bytes) would break this.
- Report the baseline-subtracted increment, not total process RSS.

---

## 5. Stage H: harness validation (baselines only; allowed now)

Pass/fail (all required before any IR data is trusted):
1. `uv run pytest -q tests/test_ista.py tests/test_operator.py tests/test_matrix_mult.py` passes. This is a sanity check of the environment, not new data.
2. The oracle `r_nat` and `r_raw` match an independent NumPy/`longdouble` recomputation on 20 small instances within `64·eps·scale`.
3. The oracle agrees with `tests/test_ista.py`'s normalized-KKT notion on its fixtures.
4. Wrapper counts for FISTA/ISTA match the exact formulas in §4 (with and without `gamma`).
5. First-hit replay reproduces: running at `t=N*` gives oracle `r_nat ≤ ε_K s0`, and at `t=N*−1` it does not. The first-hit property holds because the same iterate sequence is deterministic.
6. Reference pipeline: on 20 random instances `F_ref ≤ F(x_FISTA)+1e-12 F0`, and `gap_dual ≥ F(x)−F_ref − 1e-12 F0` for every iterate checked (dual-bound consistency).
7. Timer sanity: for FISTA on `CK-RAND` mid case, repeat CV `<10%` under k=1. Otherwise lengthen runs and record.
8. Memory probe: a deliberately allocated 100 MB tensor in the child shows `ΔVmHWM ≈ 100 MB ±10%`.

```bash
uv run python examples/benchmark_ir_fista_b.py --suite harness --solvers FISTA ISTA --out $RUN/stage_h.jsonl   # ≤10 min
uv run python examples/benchmark_ir_fista_b.py --suite micro --out $RUN/stage_m.jsonl                          # ≤10 min
```
The harness file `examples/benchmark_ir_fista_b.py` is to be created by the implementer or runner (spec §11 allows a benchmark script under `examples/`). I have not created it. The CLI flags used throughout (`--suite {harness,micro,correctness,perf-explore,perf-confirm}`, `--solvers`, `--seeds`, `--dtype`, `--device`, `--threads`, `--eps`, `--cap`, `--out`) are proposals.

---

## 6. Instances

### 6.1 Normalization convention for generated instances

The spec says the solver does no column normalization. Normalization is therefore part of the instance and is recorded.
- Gaussian: `A_ij ~ N(0, 1/m)`, so `‖A‖² ≈ (1+√(n/m))²`.
- Acoular: `dictionary_scaling='unit_l2'`, captured as delivered.
- `λ` is set as a fraction `λ/λ_max` with `λ_max = ‖Aᵀy‖_∞/m` (positive: `max_j(Aᵀy)_j/m`). This is scale-free and records the realized solution support.

### 6.2 Acoustic CMF: existing Acoular assembly pattern

**Capture pattern** (from `examples/benchmark_solvers.py` and `ucmf/acoular.py`; no source change). The harness defines a `CaptureSolver(LeastSquaresSolver)` subclass (as `TimedSolver` does). Its `solve(self, problem, dictionary_matrix, data, start_value=None)` stores `dictionary_matrix` (m×n, real f64), `data`, `lam = problem.alpha*problem.unit_mult` and `problem.nonnegative`, and returns `SolverOutput(solution=np.zeros(n))`.
- Setup as in `sparse_recovery.py`:
  - 64-mic `array_64.xml`.
  - Three `WNoiseGenerator` sources (seeds 1, 2, 3; rms 1, 0.7, 0.5) at `(−0.1,−0.1,−0.3)`, `(0.15,0,−0.3)`, `(0,0.1,−0.3)`.
  - `PowerSpectra(block_size=128, Hanning)`, `sfreq=51200`, `ac.config.global_caching='none'`.
  - `BeamformerCMF(problem=L1RegularizedLeastSquaresProblem(solver=CaptureSolver(), alpha=1e-9, nonnegative=True, dictionary_scaling='unit_l2'))`.
  - `bf.synthetic(8000, 3)` yields one captured instance per frequency bin in the third-octave band (5 bins at `df=400 Hz`; the README states 5 bins).
- **Instance families:**
  - **CMF-S:** `RectGrid` increment 0.01 → 41²=1681 sources, `m=4032` (`r_diag`). Dictionary ≈ 54 MB f64.
  - **CMF-M:** increment 0.005 → 81²=6561 sources, ≈ 212 MB dense.
  - Replicates: simulation seeds `(1,2,3)+10·r`, `r∈{0,1,2}`; 5 bins each.
- **CMF-L** (n=16384, 128×128 on `[-0.2,0.2]²`): build `H` directly from `st.steer_vector(f)` on the fine grid. Do not use `BeamformerCMF` assembly, because intermediates would exceed memory (the complex `c²n` array is ≈ 1.07 GB). The dense arm is built once as `MatrixMult(CMFDictionary(H, r_diag=True).matrix)` (≈ 528 MB), and the transient is freed.
- **Matrix-free arm (`CMFPropagation(H, r_diag=True)`) and its equivalence gate (blocking):**
  - The `steer_vector` shape and conjugation conventions are not verified here. The harness must try `H = h.T` and `H = conj(h).T`.
  - It must also fit the per-column scale `s_j = ⟨dict_col, dict_col_unscaled⟩/‖dict_col_unscaled‖²` and set `H_j ← H_j·√s_j` (CMF columns are quadratic in `H_j`).
  - **Gate:** relative column residual `≤ 1e-10` and `‖MatrixMult(captured).forward(x) − CMFPropagation(H').forward(x)‖ ≤ 1e-10‖·‖` on 5 random `x`.
  - If neither convention passes, the Acoular-derived matrix-free arm is **not-run**. The fallback is a self-contained synthetic free-field `H_ij = exp(−j k r_ij)/r_ij` with seeded random mic and grid positions, recorded as "CMF-synth".
- **λ choices for CMF:** `λ/λ_max ∈ {0.3, 0.1, 0.03, 0.01}`, plus Acoular's actual `alpha·unit_mult` with its `λ/λ_max` reported. Positive mode (physical CMF). Signed is a Stage C check only.
- **Batched CMF:** the 5 bins of one replicate as `B=5` with `CMFPropagation(H_b)` of shape `(B,c,n)` (distinct operators). Baselines run batched, and success requires every row to pass (as `SolutionChangeStopper` does). IR processes rows sequentially, so batch behavior is a performance question. `UW` and `FME` are reported separately.

### 6.3 Synthetic and analytic families (Stage C and P)

| ID | Construction | Notes |
|---|---|---|
| **CK-ID** | `A=I_n`, `n∈{1,17,256}`, `y~N(0,1)` including negatives; `mλ ∈ {0, 0.25, 1, 1e3 (>max y → zero)}`; positive and signed | Analytic `x*=max(y−mλ,0)` or `S_{mλ}(y)`; `|x−x*|_∞ ≤ 64·eps·max(1,‖y‖_∞)` |
| **CK-NNLS** | `λ=0`: (80,40) full rank; (40,80) underdetermined consistent `y=Ax_true`; (60,60) with `κ(A)=1e4` via SVD | `scipy.optimize.nnls`. Compare `x` only when full column rank; otherwise `F` and KKT only. Also signed `λ=0` versus `lstsq` for full rank. |
| **CK-RAND** | `(m,n)∈{(50,100),(100,100),(100,300),(300,100)}`; truth sparse (5%) or dense; `λ/λ_max∈{0.5,0.1,0.01}`; pos and signed; 20 seeds | sklearn Lasso and polish reference |
| **CK-COH** (stress) | Common-factor columns `a_j=ρu+√(1−ρ²)ε_j`, `ρ²∈{0.9,0.99,0.999}`; AR(1) correlation 0.95; near-duplicate columns (`1e-6`, `1e-9` perturbation); `λ/λ_max∈{0.1,0.01}` | Non-unique-ish: `F`+KKT only |
| **CK-RD** (rank deficiency) | rank `r=m/4`; exact duplicated columns; zero columns; zero rows; `n>m`; `A=0`; `y=0`; `y⊥range(A)`; negative-only `y` (positive mode); `λ ≥ λ_max`; `λ=0` consistent | Must handle empty `W`, no empty reduced operator |
| **CK-SPARSE** | Realized support fraction `{0.5%, 2%, 10%, 50%, ~100%}` via `λ` path from the polished reference | The performance covariate |
| **CK-WARM** | `x0` ∈ {zero; random positive; random with negative entries (projection); exact optimum (expect no reduced work, certification only); optimum of nearby `λ` (`0.8λ`); optimum plus noise; wrong large support; support-only mismatch (reactivation)} | Same `x0` for all solvers. Compare only returned points. |
| **CK-SCALE** | `(A,y,λ) → (αA, αy, λα²?)` and the stacked-measurement test `[A;A], [y;y]` | The stacked test checks that the `m` denominator is preserved (normalized `F` and `x*` unchanged). Scaling relations are written out explicitly in the harness. |
| **CK-OP** | `MatrixMult`, `CMFDictionary`, `CMFPropagation`, generic fallback (no `select_columns`), counting wrapper; shared and batched `B=3` distinct operators; per-row `λ` tensor; mixed convergence in a batch (trivial zero-solution row plus hard row) | Per-row status honesty; counters equal wrapper counts |
| **CK-API** | `t=0`, `t=1`, callback stop, repeat call, iteration exhaustion, excessive initial `gamma` (backtracking) | Statuses honest (`converged` / `max_iter` / `callback_stopped`). Invalid-input tests belong to `tests/test_ir_fista_b.py`. |
| **CK-DTYPE** | float64 (required), float32 (floors per §2), CUDA if available | CPU/GPU agreement (f64, `F` within `1e-10` relative) is **N/A on this host** |
| **CK-CMF-S** | Captured Acoular bins (§6.2), `MatrixMult` and `CMFPropagation` | Equivalence of `x`-level `F`, KKT between the two operators |

Required versus stress: CK-ID, NNLS (full rank), RAND, SPARSE, WARM, SCALE, OP, API and CMF-S are **required**. CK-COH and CK-RD stress cases require *honesty* (a result is never labelled converged falsely) and `F`+KKT agreement when both solvers converge. If FISTA at the same budget also fails to reach `ε_K`, an IR non-convergence is recorded as "budget-limited", not a Gate 2 failure.

### 6.4 Large performance families (P1/P2)

**Large dense dictionaries** (CPU f64, `B=1`, same dictionary bytes ≈ 268 MB; guard against the memory limit):

| ID | `(m, n)` | Notes |
|---|---|---|
| D1 | (1024, 32768) | Support is bounded by `≈m` (3%), so the sparse regime |
| D2 | (2048, 16384) | |
| D3 | (4096, 8192) | |
| D4 | (512, 65536) | Optional; widest `n/m` |

`y = A x_true + noise` (SNR 20 dB), `x_true` k-sparse positive. `λ/λ_max ∈ {0.3, 0.1, 0.03, 0.01}`. Target realized `k* = |supp x*| ∈ {8, 32, 128, 512}`, selected by the λ path from the reference. Also report `λ/λ_max`. The scaling sweep fixes `m=1024` and uses `n ∈ {2048, 4096, 8192, 16384, 32768}` (16–268 MB). 3 instances (seeds) per cell, 5 in the confirmatory sparse-regime cells.

**Dense worst-case control:** `λ=0` (NNLS) and `λ/λ_max=1e-3` where the support and working set approach `n` (D2 and CMF-S). Expected IR slower than FISTA. It is mandatory to report the loss, not to hide it.

**Warm-start path (λ continuation):** 10 values geometrically from `0.9λ_max` to `0.01λ_max` on D2 and CMF-M. Each solver warm-starts from its own previous certified output. Report total path time and `FME`. Also run the neighbouring-bin warm start (bin `b` solution as `x0` for bin `b+1`) on CMF.

---

## 7. Stage C: correctness protocol (Gate 2 input)

**Runtime budget:** ≤ 20 min total on CPU k=1. The small-instance cases are ms each, and the CMF-S cases are seconds. Wall cap 60 s per run. `timeout 25m` outer.

```bash
timeout 25m taskset -c 2 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run python examples/benchmark_ir_fista_b.py --suite correctness --dtype float64 --device cpu \
  --seeds 0-19 --eps 1e-6 --cap 60 --out $RUN/stage_c.jsonl
uv run pytest -q tests/test_ir_fista_b.py            # implementer's unit tests, in addition (not a substitute)
uv run pytest -q                                      # existing suite regression (baseline: 1912 passed, 21 skipped)
uv run ruff check .
```
GPU row: `--device cuda` runs only if `torch.cuda.is_available()`. On this host, record `not-run (no CUDA device)`.

**Gate 2 numerical pass criteria (per instance; all required instances must pass; ε_K=1e-6 in f64):**

| # | Criterion | Threshold |
|---|---|---|
| C1 | Feasibility (positive mode) | `min(x) ≥ 0` exactly |
| C2 | Certificate honesty | status `converged` ⇒ oracle `r_nat ≤ ε_K s0 (1+1e-9) + 32·eps(‖A‖²/m·‖x‖_∞+‖g‖_∞)`. No false certificate allowed in any run. |
| C3 | Reported-versus-oracle KKT | Solver-reported `r_nat`, `r_raw` of the **returned point** agree with the oracle to relative `1e-8` or absolute `1e-12·s0`, whichever is larger |
| C4 | Objective | Well-posed required cases: `ĝ ≤ max(1e-8, 10·ĝ(FISTA at the same ε_K))`. All cases with `λ>0`: `F(x)−F_ref ≤ gap_dual + 1e-12F0` (consistency). Identity: analytic tolerance of CK-ID. |
| C5 | Non-unique cases | `F` and KKT only. NMSE reported as information. |
| C6 | Outer monotonicity | From end-callback traces: `F(x^{k+1}) ≤ F(x^k) + 1e-12·F(0)` in f64 |
| C7 | Working-set invariant | Reported `|W_k|` equals the oracle-reconstructed `|supp(x^k) ∪ violators(τ)|` exactly. Ties within `8·eps·scale` of the threshold are flagged "borderline", not failures. `|supp(x^k)| ≤ |W_k|` always. |
| C8 | Counter integrity | Wrapper totals by kind and width equal solver-reported counts exactly, including rejected backtracking trials and certification products. Fallback use is flagged consistently. |
| C9 | Status honesty (CK-API/OP) | `t=0` is not falsely converged. Callback stop gives `callback_stopped`. Budget exhaustion gives `max_iter`. Per-row statuses in a batch are independent. A backtracking cap raises `RuntimeError` rather than reporting convergence. |
| C10 | Determinism | Repeated call: same `x`, counters and status (bitwise at k=1). Repeated `forward` on one instance leaves no state leakage. |
| C11 | Operator equivalence | `MatrixMult` vs `CMFPropagation` vs `CMFDictionary` vs generic fallback give `|F−F'| ≤ 1e-10F0` and both pass C2. The fallback run additionally reports fallback use in the counters. |
| C12 | Scaling/units | CK-SCALE relations hold: stacked `[A;A]` gives the same `x*` and normalized `F` (`≤1e-10` relative) |
| C13 | No materialization | The memory no-Gram test of §4 passes. No call has width `>n`. No `n×n` or `p×p` allocation is observed. |

**Gate 2 requires:** 0 failures in C1–C3, C6–C13 over all required instances, C4 satisfied on all required certified-reference instances, and stress-class instances only honest (C2, C9). Any failure goes back to the implementer. A failing C4 on a well-conditioned case or any false certificate (C2) triggers escalation per spec §12.

---

## 8. Stage P1: exploratory performance (only after Gate 2)

Exploratory means no pre-specified hypothesis test. Report descriptive tables and plots, and label them "exploratory". Wall budget ≈ 90 min total, with a 120 s per-run cap.

**Cases:**
1. **Regime map:** D2 (and CMF-S) × `λ/λ_max ∈ {0.3, 0.1, 0.03, 0.01, 0}` × `ε_K ∈ {1e-4, 1e-6, 1e-8}`. Report time, `FME`, outer sweeps, `p_max/n`, success.
2. **Scaling in n** at fixed `m=1024`, fixed `k*` ∈ {32, 128}, `n` sweep as §6.4. Log-log slope of time and of `FME` per solver.
3. **Ablations** on two representative cases (D2 with `k*=32`; CMF-M with `λ/λ_max=0.03`); one factor at a time, then a small interaction grid:
   - `tau ∈ {0, 1e-4, 1e-2, 0.1, 0.5}`. The value `0.999` is an approximation of the full working set and acts as the "no reduction" control.
   - `inner_tol ∈ {0.5, 0.1, 0.01}`.
   - `inner_max_iter ∈ {20, 100, 500}`.
   - `delta ∈ {0, default 1e-3/m, 1e-1/m}`.
   - gamma initialization ∈ {`None`, `1/L_f`, `10/L_f`}.
4. **Operator arms:** `MatrixMult` versus `CMFPropagation` on CMF-S/M/L for FISTA and IR (matrix-free versus dense), plus the generic fallback (no `select_columns`) to quantify the lost savings.
5. **Warm starts:** CK-WARM variants and the λ-path (§6.4). Report total path work and time.
6. **Baselines B2 (ISTA) and B3 (fixed-restart FISTA)** on the regime map's small cells.
7. **Batch:** CMF `B=5` distinct, plus dense `B∈{1,8,32}` on a mid case. Report per-row sequential overhead against batched FISTA.
8. **float32:** CMF-S/M and D3 only, `ε_K ∈ {1e-3,1e-4}`.
9. **Threads:** k=1 versus k=8 for D2 and CMF-M. A reduced-width product may lose parallel efficiency, which matters for the time ratio.
10. **Memory (Run type M)** on D1, D2, CMF-L (dense and matrix-free) and the no-Gram case.

```bash
timeout 100m taskset -c 0-7 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  uv run python examples/benchmark_ir_fista_b.py --suite perf-explore --dtype float64 --device cpu \
  --threads 8 --eps 1e-4 1e-6 1e-8 --cap 120 --out $RUN/stage_p1.jsonl
```

---

## 9. Stage P2: confirmatory performance (pre-registered; only after Gate 2 and a frozen P1)

Freeze all parameters before running. Wall budget ≈ 4 h (`timeout 4h`), per-run cap 300 s. Host: CPU f64, `ε_K=1e-6`, scope S1 (S2 secondary), k=8 and k=1, run order randomized per repeat.

**Pre-registered hypotheses (the only claims P2 may support):**
- **H1 (sparse regime):** for cells with realized `supp ≤ 5%` and `n ≥ 8192` (dense D1–D3 `k*∈{8,32,128}`, and CMF-M/L `λ/λ_max ∈ {0.3,0.1}`), the geometric-mean time ratio `T_IR/T_FISTA` over instances where both succeed is `<1`, with a bootstrap 95% CI upper bound `<1`. The same test is applied to `FME` (IR/FISTA, including setup).
- **H2 (no-harm control):** for cells with `supp ≥ 50%` or `λ=0`, report the ratio descriptively. The expected outcome is ratio `>1`. A "no regression" statement is made only if the CI upper bound is `≤ 1.25`.
- **H3 (certification parity):** over all P2 cells, IR success rate at `ε_K` is not lower than FISTA's by more than 5 points, and IR has zero false certificates.
- Everything else (ablations, scaling slopes, warm starts, batch, float32, threads, ISTA, B3, matrix-free versus dense) is **exploratory** and not part of confirmatory inference.

**Design and replication:**
- Dense cells: 5 instances (seeds) in the sparse-regime cells, 3 elsewhere.
- CMF cells: 5 bins × 3 simulation seeds (CMF-S/M), 5 bins × 1 (CMF-L).
- The instance (seed or bin) is the unit of replication. Timing repeats are first reduced to a median per instance, then aggregated across instances.
- Aggregation: geometric mean of ratios, stratified by family. Bootstrap (10⁴ resamples) over instances within stratum. With 3–5 instances per cell the CI is crude and is labelled descriptive.
- Report instance counts, censored counts and the number of runs affected by CV `>10%`.
- Stage P2 tables report both S1 and S2 scopes, and both Tier A (KKT-matched) and Tier B (gap-matched, oracle-tuned, disclosed as optimistic for IR).

```bash
timeout 4h taskset -c 0-7 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  uv run python examples/benchmark_ir_fista_b.py --suite perf-confirm --dtype float64 --device cpu \
  --threads 8 --eps 1e-6 --cap 300 --repeats 5 --out $RUN/stage_p2.jsonl
# k=1 replicate of the primary cells, then analysis (non-timed):
uv run python examples/analyze_ir_fista_b.py --in $RUN/stage_p2.jsonl --out $RUN/report/   # proposed analysis script
```

**Expected outcomes (hypotheses, not facts):**
- **Sparse and well-conditioned (large `n/k*`):** IR wins in `FME` and likely in time. Each outer sweep costs one full F plus one full A (about one FISTA iteration of work), so the win requires few outer sweeps and `p ≪ n`.
- **CMF (strongly coherent, fine grids):** uncertain. FISTA may need `>10⁴` iterations. IR's inner solves may also hit the inner budget, and its working set may inflate (`p_max` ≫ `|supp x*|`).
- **`λ→0`, dense support:** IR loses by roughly the per-sweep scan plus the restriction-copy overhead.
- **GPU or large batch:** IR's per-row sequential loop and host-driven branching likely hurt (not measurable here).
- **Setup:** baseline `51F+50A` setup is small once iterations exceed about 10³. It matters for warm-start cases that finish in tens of iterations.

---

## 10. Result schema (JSONL, one record per run)

Fields: `stage, case_id, family, instance_idx, seeds, dtype, device, threads, scope, run_type, solver, config{...}, m, n, B, λ, λ/λ_max, s0, eps_K, tol_abs`, then:
- Outcome: `status, success_oracle, censored, cap_kind, wall_s (median + IQR + all repeats), cv, cpu_mhz_before/after`.
- Accuracy: `r_nat, r_raw, F, F_ref, ref_quality, ĝ, ĝ_dual, supp_frac, supp_ref_frac, nmse (info only)`.
- Solver-side: `iters{outer, inner, N*}, counts{F_full, A_full, F_red, A_red, fallback, by_attribution}, FME, UW, ws_history, p_max, reactivations, backtrack_rejections, L_history_summary, inner_target_hits`.
- Memory: `rss_before, rss_peak_delta, gpu_alloc_delta, gpu_reserved_delta`.
- Provenance: `git_sha, diff_hash, ir_source_sha256, harness_sha256, uv_lock_sha256, env_ref`.

---

## 11. Visualizations

1. Anytime curves: `r_nat/s0` and `ĝ` versus cumulative `FME` and versus wall time (IR from per-sweep callback traces, FISTA from check points; oracle time excluded).
2. Performance profiles (Dolan–Moré) on time and on `FME`, with censored runs as failures.
3. Ratio versus realized support fraction (log-log), coloured by family (dense / CMF), with censored points as arrows.
4. Scaling in `n`: time and `FME` versus `n` at fixed `k*`.
5. Working-set history `|W_k|/n` against the reference support, per case, with reactivation markers.
6. Stacked bar of IR cost by attribution (scan, reduced gradient, backtracking, sub-residual, recompute, certification, restriction copy), measured versus Stage M model time (overhead fraction).
7. Memory increment versus model `8·m·p_max`.
8. Ablation heatmaps (`tau × inner_tol`, time and `FME`).
9. KKT-versus-gap scatter (`r_nat` versus `ĝ`) showing that a small KKT residual does not bound the gap.

---

## 12. Confounds and threats

- **Reference accuracy:** the gap is measured against `F_ref`. The dual gap gives a reference-free bound for `λ>0`. It does not exist for `λ=0`.
- **Non-uniqueness:** the metrics are `F` and KKT. NMSE is info only.
- **Asymmetric certification cost:** IR pays for its own certificate. Baselines are credited first-hit (oracle), which favors them. Variant B1-cert gives the practical comparison.
- **Momentum reset:** IR resets FISTA momentum every outer sweep. B3 (fixed-restart FISTA) separates restart benefit from working-set benefit.
- **FISTA's heuristic step:** `L = 1.05·‖A‖²/m` is a power-iteration *lower* bound estimate, so it can undershoot. A baseline that diverges or stalls is reported, not tuned.
- **Timer noise (laptop, `powersave`, thermal):** interleaving, repeats, MHz logging and CV flags. Governor cannot be changed.
- **Memory-bound dense GEMV:** at ≥ 54 MB per matrix, narrower `p` should save time roughly in proportion to `p` until caches help. Cache effects may favor IR in a way that does not extrapolate. Stage M exposes the curve.
- **Oracle cache pollution:** oracle products can evict the matrix from the last-level cache. Negligible for ≥ 54 MB matrices. Oracle calls are only in C-runs, which are not timed.
- **Instance selection bias:** the `k*` targeting uses the reference solution's support (S0). Report realized support and `λ/λ_max`. Never tune on P2 data.
- **Python and host-sync overhead (per-row loop, eager):** reported through the model-versus-measured overhead fraction.
- **Memory measurement scope:** RSS includes allocator retention. The baseline-subtracted increment is only an approximation.
- **float32 floors:** `ε_K` at or below the attainable residual makes success impossible for both solvers. Such cases are dropped for both.
- **API dependence:** the harness reads IR diagnostics by documented keys, which are not fixed by the spec (only "a documented diagnostics dictionary"). Until the key names are known, the wrapper counts and oracle reconstruction are the only dependable sources. Counter integrity (C8) is conditional on diagnostics being exposed.

---

## 13. Go/no-go and routing summary

1. Run **Stage H and M** now (baselines only), then **Stage C** on the candidate implementation. Stage C is the Gate 2 evidence.
2. Gate 2 passes only if Stage C meets §7 in full. The independent reviewer reproduces the results using the commands in §7.
3. After Gate 2, run **P1** (exploratory), freeze, then **P2** (confirmatory H1–H3).
4. Only after P2 may any performance statement be made, and only in the regimes measured, on this host (CPU, f64). GPU claims need a CUDA device that is currently absent.