# IR-FISTA-B implementation notes

**Final parent status:** subsequent independent Gate 2 CPU correctness review and bounded exploratory benchmark audit passed with no unresolved Critical/Major findings; see `validation.md` and `benchmark_notes.md`. The evidence below remains the historical coder/recovery checkpoint, not a claim that later experiments were run by the coder. Source hashes remained unchanged.

## Provenance and recovery checkpoint

Required implementation artifact, populated during the same-agent recovery of run
`d9ac51f3-5cac-41a2-944e-dc9c0ae4eb65`. The implementation, TDD events and original
full-suite results below are actual observations from that run, retained in the
session context and `coder_report.md`; they are not newly invented or rerun red
states. The previous interpretation that report routing superseded this required
artifact was incorrect and is corrected here. Recovery findings are separately
routed to `coder_report.bbc35561.md`.

Recovery inspected file endings, parsed all three new Python files, checked source
hashes, and revalidated saved JSONL artifacts. No partial source/test/harness files
were found. No solver, test, harness, export or README changes were necessary.
The solver and harness hashes match those recorded at the original final full run:

- Solver SHA256: `889fe15fca27912c67a9139e865e87155b7b099729077e63e7f33155776f2091`
- Harness SHA256: `44745d48a1c687ae01418a7aba82adef53a9d073db9d75c9aac24e24b384a60e`

Recovery validation actually rerun:

- `uv run pytest -q tests/test_ir_fista_b.py`: **108 passed, 2 skipped in 4.78s**.
- `uv run ruff check .`: **All checks passed!**
- `git diff --check`: passed, no output.
- `git diff --cached --name-only`: empty, no staged files.
- Python syntax parsing: complete solver, tests and benchmark script.
- Saved correctness JSONL parsing: three files, six certified solution/reference
  records each (18 total). These were inspected, not remeasured during recovery.
- Fresh `import ucmf`: did not import Acoular.

The prior full run was **2020 passed, 23 skipped, 14 warnings in 23.13s**; it was
not rerun during this documentation-only recovery. No performance comparisons
were executed in either run. Independent Gate 2 scientific review remains pending.

## Original implementation/TDD/evidence record

## Implementation notes / specification mapping

- `IRFISTAB(FISTA)` reuses strict operator/input preparation, signed/nonnegative shrinkage, Nesterov momentum, callback lifecycle, tensor validation and batch-parameter reshaping. Constructor `positive=True`; supplied `nonnegative` overrides it.
- Forward signature `(y, x0=None, t=100, operator=None, lam=None, gamma=None)` returns only `(B,n)`. `t` counts outer sweeps, not proximal steps. Positive warm starts are projected. Each row has independent working sets, curvature, tolerance scale and stopping state.
- Objective and all gradients retain `1/(2m)` / `1/m` normalization under source restriction. `delta=None` means `1e-3/m`; explicit scalar/per-row delta is already normalized. The reduced gradient includes **delta*(v-anchor)**.
- Support-retaining selection, exact zero comparisons, ordered source mapping and global re-entry match v2. No top-K cap, permanent screening, source Gram matrix, eigenvalue calculation or identity-probing dictionary construction.
- Reduced FISTA resets momentum each outer solve; accepted curvature never decreases inside that run. Outer warm-start curvature is `max(floor,last_L/eta)` after the first solve. Every accepted candidate is considered for minimum Phi, including the first, with earlier ties retained.
- Both inner and global residuals use mandatory cancellation-free identities. The signed +/-2^60 float64 regression reports natural/raw residual exactly 1 and not converged at t=0; inner stopping is similarly tested.
- A later candidate reaching the inner target does not imply the returned earlier best candidate reaches it. These are separate diagnostics. Budget-limited reduced solves are not called exact solutions. Full returned-point certificates determine final `converged` / `max_iter` / `callback_stopped` status, independently of callbacks.
- Native column selection is used without inspecting operator internals. Generic restriction scatters/gathers full-width vectors. Batched row wrappers perform B-wide zero injection/extraction, potentially doing unused work. Counted tests verify actual calls, actual underlying batch rows and source-width-weighted work against independent wrappers.
- Callbacks see stable full `(B,n)` tensors at outer boundaries, with full A=W/y and `(B,1)` gamma/theta. In-place edits trigger full re-certification. Solver stop/diagnostics reset between calls; callback-owned state follows the existing contract.

### Numerically equivalent safeguards / engineering choices

1. **Homogeneous energy normalization:** evaluate the specified quadratic backtracking inequality using `D/max(abs(D))` on both sides for nonzero D (zero D remains zero). This is algebraically the same inequality and relative allowance, not a changed acceptance rule. A float32 `A=2,y=1e-30,L=1,inner_max_iter=1,max_backtracking=1` regression initially accepted because both squared energies underflowed to zero. It now correctly raises line-search exhaustion. No additive energy slack was introduced. This numerical fix was reported to the supervisor.
2. **Inference-mode callbacks:** inference tensors do not expose a mutation version counter. They initially raised at callback bookkeeping; now callbacks conservatively trigger full scans in inference mode. These extra real products are counted. No algorithmic selection/stopping rule changes. No-callback inference runs avoid these extra scans.
3. Candidate objective/residual share the candidate forward image. Current implementation uses 3 restricted forwards + 2 restricted adjoints per accepted inner step, plus one forward for each rejected trial. Initial/post-update/final full scans each use one forward and one adjoint. There is no hidden spectral setup.
4. Diagnostics keep scalar histories (including accepted curvatures) and thus grow with total inner iterations; conversion to Python scalars entails device synchronization. Eager sequential row processing, selected dense copies, generic scratch and B-wide overhead are disclosed. No compile or boundary differentiability promise.

No material algorithmic deviations from reviewed v2 were made. The historical draft and unsupported paper constants are not used.

## TDD and development outcomes

Actual red/green commands were run before their corresponding additions/fixes; repeated test command was `uv run pytest -q tests/test_ir_fista_b.py` unless a filter is shown.

| Increment | Observed red | Subsequent green |
|---|---|---|
| Analytic/API/M1 tests before source/export creation | Collection ImportError: cannot import IRFISTAB | 9 passed after implementation/export |
| Reduced scalar/D2, safeguard/curvature diagnostics | 6 failed / 11 passed: missing `best_phi` | 17 passed after diagnostic additions |
| Batch/native/fallback counters and callback repeat tests | 7 failed / 18 passed, chiefly missing `counts` | 25 passed after independently checked counter instrumentation |
| Tiny-displacement energy safety (`-k subnormal`) | 1 failed: expected RuntimeError was not raised | regression green; aggregate 58 passed |
| Numerical references, coherence/rank deficiency, reactivation/scaling/CMF | Additional oracle tests passed against existing implementation | 92 passed / 2 skipped |
| Best-candidate replay and raw/natural/autograd coverage | First test fixture had not yet reached acceleration overshoot (30 steps); corrected deterministic fixture to 50 steps | 95 passed / 2 skipped |
| Benchmark harness test (`-k benchmark`) before script creation | FileNotFoundError for new script | 1 passed / 97 deselected after script creation |
| Inference-mode callback regression (`-k inference`) | RuntimeError: inference tensors do not track version counter | 1 passed / 102 deselected after conservative scan handling |
| Final expanded D2 anchors, LS, counts, callback mutation and selection tests | — | **108 passed / 2 skipped**, 3.55 s before final full run |

Test corrections are disclosed rather than presented as solver fixes:
- Identity float32 coefficient tolerance originally tighter than the configured natural tolerance times m; changed to a mathematically consistent coefficient tolerance. Natural convergence assertions remained.
- An initial assertion incorrectly equated best-candidate inner-target attainment with exit-target attainment. Corrected to the specified separate flags, without changing the best-candidate tie rule.
- Native versus scatter products can differ at roundoff and lead to different objective tie decisions. Batch independence compares the same operator representation against independent row solves; cross-representation tests compare certified objective/KKT, not unrealistically exact coefficients.
- The first underflow fixture permitted many inner steps, eventually triggering a later rejection; limiting it to one step isolates the actual unsafe first acceptance.
- Ruff initially identified import ordering and long lines; only new/authorized files were formatted/fixed. Final repository lint is clean.

## Validation evidence

Final commands:

```text
uv run pytest
2020 passed, 23 skipped, 14 warnings in 23.13s

uv run ruff check .
All checks passed!

git diff --check
(no output, passed)

git diff --cached --name-only
(no output: no staged files)
```

The 14 warnings are existing Torch JIT deprecation warnings in compile tests. Baseline was 1912 passed / 21 skipped; new suite adds 108 passing tests and 2 CUDA-unavailable skips. An earlier complete run also passed (2012 / 23 / 14) before the final eight additional passing cases and inference compatibility changes.

Test coverage includes positive/signed identity, positive lambda=0 vs scipy NNLS, signed lambda=0 vs NumPy least squares, lambda_max zero, A=0, warm-start projection, independent NumPy FISTA-certified random/sparse/dense/coherent/rank-deficient references, objective/raw/natural checks, outer descent, exact support/near-violator selection, review reactivation example, scaling and measurement stacking, shared/distinct batches with row lambda/gamma/delta, dense/native/generic/matrix-free CMF equivalence, counter formulas, callback/stopper/repeat/mutation/inference behavior, t=0 and budgets, D2 with zero/nonzero anchors, excessive gamma, tiny energy/backtracking exhaustion, best-not-last candidate replay, M1 both signs and inner site, dtype/device validation, autograd operator detachment, static no-Gram guards and runtime forbidden dictionary/Gram allocation shapes.

### Independent harness correctness runs (not performance experiments)

All three commands succeeded, each emitting six records (dense and free-field CMF, each with IR/FISTA/ISTA). Every returned result and every independent tighter reference met its respective natural-residual tolerance. Product count checks passed. No timing/memory comparison fields are emitted in correctness mode.

```bash
uv run python examples/benchmark_ir_fista_b.py --mode correctness \
  --out .pi/research/ir-fista-b/correctness_results.jsonl
uv run python examples/benchmark_ir_fista_b.py --mode correctness --signed \
  --out .pi/research/ir-fista-b/correctness_signed.jsonl
uv run python examples/benchmark_ir_fista_b.py --mode correctness --dtype float32 --tol 1e-4 \
  --out .pi/research/ir-fista-b/correctness_float32.jsonl
```

IR-only accuracy summaries (objective differences against independently certified FISTA, not performance claims):

| Mode | Family | Natural infinity residual | Objective minus reference |
|---|---|---:|---:|
| positive float64 | dense | 1.03588136e-8 | 4.56579219e-15 |
| positive float64 | free-field CMF | 6.45643478e-8 | 1.15352172e-13 |
| signed float64 | dense | 1.00684193e-8 | 4.16160162e-15 |
| signed float64 | free-field CMF | 3.94080329e-9 | 6.85215773e-16 |
| positive float32 | dense | 3.04507341e-6 | 2.79554871e-10 |
| positive float32 | free-field CMF | 2.08707119e-6 | 7.96342159e-11 |

Float32 final validation uses a float64 evaluation/reference of the **same quantized inputs**, not the original unquantized problem. Relative common tolerance is scaled by initial correlation (explicit `tol_abs` in each record). Natural/raw residuals coincide in these six IR examples; full records retain both.

## Public diagnostic key layout

`solver.diagnostics`:

```text
status: list[str] per row
natural_residual, raw_residual, objective, initial_residual, threshold: detached Tensor[B]
counts: sum of row counters
rows[b]:
  outer_iterations, inner_iterations
  working_set_sizes: list[int], per performed reduced solve
  counts
  inner: list[dict]:
    iterations, backtracking_rejections
    first_phi, best_phi, best_residual, lipschitz, accepted_lipschitz
    target_met, best_target_met, budget_exhausted
```

Counter names: `full_forward`, `full_adjoint`, `restricted_forward`, `restricted_adjoint`, `fallback_forward`, `fallback_adjoint`, `restriction_fallbacks`, `batch_fallback_calls`, `underlying_forward_rows`, `underlying_adjoint_rows`, `underlying_forward_elements`, `underlying_adjoint_elements`.

Logical restricted calls include p=n; actual source widths and B-wide work are reflected in the underlying element counters. These are work proxies, not FLOPs or CMF runtime models. User callback products are outside solver counters. Already-converged rows have empty working-set/inner histories.

## Independent review / later benchmark commands

Reproduce unit/full tests and the three correctness commands above. Inspect source/specification mapping, the homogeneous energy test, distinct best/current inner-target flags and wrapper count checks. Gate 2 remains required.

Only **after independent Gate 2 approval**, planned/not executed here:

```bash
uv run python examples/benchmark_ir_fista_b.py --mode benchmark --gate2-passed --large \
  --family both --m 512 --n 8192 --mics 16 --grid 32 --cap 30 --repeats 3 \
  --out .pi/research/ir-fista-b/exploratory.jsonl
# Larger explicit configurations: --m 1024 --n 32768 --mics 64 --grid 81
# Probe help for step/reference/outer/inner budgets, threads, seed, lambda and device.
```

The CLI requires explicit Gate-2 attestation for benchmark mode. Each baseline is one uninterrupted public solve, stopping via full-KKT callbacks (no momentum-reset loop). Setup power-iteration counts (51 forward / 50 adjoint), per-step products and callback certificates are independently checked. Runtime includes practical certification and instrumentation; references/construction/external final oracle are excluded. Baseline check interval is explicit (default 10). Rotated repeat order is exploratory only, with no warm-up/statistical guarantee. Reference failure and budget censoring remain visible.

Memory reporting is intentionally limited: Linux process-lifetime high-water RSS is **not isolated incremental solver peak**; CUDA allocator peaks omit driver allocations. Fresh-process experiments are needed for meaningful memory comparisons. Large dense allocation guards are present. Acoustic cases are self-contained 8kHz free-field CMF propagation, not falsely labelled Acoular captures.

## Remaining risks / limitations

- Independent Gate 2 scientific/code review is outstanding; this is checked implementation evidence, not approval.
- CUDA unavailable here: two tests skipped, no GPU correctness or performance results.
- Large/post-Gate-2 benchmark mode, isolated memory experiments, Acoular-captured datasets and full P1/P2 statistical studies are planned/not run. The script is a bounded exploratory tool, not the entire experiment-plan menu.
- Extreme conditioning, finite precision, objective ties and finite outer/inner budgets can yield honest nonconvergence. Natural stationarity is not an objective-gap/coefficient-error certificate. Raw residual can remain larger near tiny support.
- Whole coefficient-sequence convergence, local linear rates, permanent support identification and general speed guarantees remain unproved for the defaults; no such claims were added.
- Sequential host-driven batching and generic/B-wide fallbacks can be expensive; diagnostics/autograd graphs can grow. These are documented rather than hidden.

No known unresolved correctness blocker was found in this implementation pass. Reviewer approval remains mandatory.

