# Post-Gate-2 bounded exploratory measurements

These measurements were run by the parent **after** the independent CPU numerical-correctness PASS in `validation.md`. They do not establish general speed, memory, convergence-rate, or statistical claims. Solver/harness source remained unchanged throughout.

## Provenance and scope

CPU float64, one PyTorch thread, seed 7957, positive mode, no_grad. Synthetic Gaussian dense dictionaries; physically specified synthetic free-field acoustic CMFPropagation at 8 kHz, sound speed 343 m/s, spiral microphone array, off-diagonal covariance vectorization. These are not Acoular-captured or experimental measurements. See harness make_problem for geometry and amplitude scaling. Sparse cases have three generated sources and lambda/lambda_max=0.1. Tolerance is 1e-6 times initial correlation scale, shared across solvers.

Runtime includes the public solver call, counting instrumentation, practical full-KKT stopping, native column copies, and final internal IR certification. It excludes instance generation, reference solves, and external final oracle. Baseline default spectral setup is included (51 forward + 50 adjoint products); stopping checks run every 10 steps in one uninterrupted FISTA/ISTA call. IR inner steps cost 3 restricted forwards + 2 restricted adjoints, plus rejected trial forwards. Calls and source-width/batch-work are retained independently in JSONL. Repeats rotate solver order; **no warm-up, independent-instance replication, confidence interval, or statistical study**.

## Exact harness commands (completed, exit 0)

```bash
timeout 240 uv run python examples/benchmark_ir_fista_b.py \
  --mode benchmark --gate2-passed --large --family both \
  --m 512 --n 8192 --mics 16 --grid 32 --cap 10 --repeats 3 \
  --out .pi/research/ir-fista-b/exploratory.jsonl

timeout 180 uv run python examples/benchmark_ir_fista_b.py \
  --mode benchmark --gate2-passed --large --family both \
  --m 1024 --n 32768 --mics 32 --grid 64 --cap 10 --repeats 1 \
  --out .pi/research/ir-fista-b/exploratory_large.jsonl

timeout 60 uv run python examples/benchmark_ir_fista_b.py \
  --mode benchmark --gate2-passed --large --family dense \
  --m 256 --n 1024 --support 128 --lam-fraction 0 --cap 5 --repeats 1 \
  --out .pi/research/ir-fista-b/exploratory_nnls.jsonl
```

Results: 18 + 6 + 3 = 27 records. A cap is only checked at iteration boundaries, so a reported 10-second censored run can slightly exceed 10 seconds. Right-censored times are **not** times to the target.

## Runtime and accuracy

First two cases: median of three repeats of the SAME instance (all repeats had identical quality and iteration counts). Remaining cases: single observations. Natural and raw residuals coincide in these reported results; both are independently evaluated and saved.

| Problem (m,n) | Solver | Seconds | Target attained | Natural/raw infinity KKT | F minus reference |
|---|---|---:|---|---:|---:|
| Dense (512,8192), tol=1.5241e-9 | IRFISTAB | 0.07829 | 3/3 | 7.46e-10 | 2.35e-16 |
| same | FISTA | 0.99749 | 3/3 | 1.22e-9 | 9.81e-16 |
| same | ISTA | 1.01744 | 3/3 | 1.47e-9 | 9.41e-16 |
| CMF (240,1024), tol=4.4343e-9 | IRFISTAB | 0.13230 | 3/3 | 5.69e-10 | 5.30e-17 |
| same | FISTA | 0.15567 | 3/3 | 3.78e-9 | 1.54e-15 |
| same | ISTA | 0.42700 | 3/3 | 4.39e-9 | 3.11e-15 |
| Dense (1024,32768), tol=1.1774e-9 | IRFISTAB | 0.28659 | yes | 1.70e-10 | NOT a certified-reference gap |
| same | FISTA | >=10.013 | no, wall cap | 2.93e-8 | reference itself is uncertified |
| same | ISTA | >=10.002 | no, wall cap | 8.15e-8 | reference uncertified |
| CMF (992,4096), tol=1.4882e-9 | IRFISTAB | 0.39979 | yes | 6.30e-11 | 1.68e-18 |
| same | FISTA | 3.22997 | yes | 1.25e-9 | 9.47e-16 |
| same | ISTA | >=10.001 | no, wall cap | 6.44e-6 | 4.49e-6 |
| NNLS (256,1024), tol=9.7320e-9 | IRFISTAB | 0.06237 | yes | 7.23e-9 | 8.95e-13 |
| same | FISTA | 0.01656 | yes | 3.63e-9 | 1.61e-13 |
| same | ISTA | 0.02458 | yes | 6.31e-9 | 5.44e-13 |

The large dense reference hit its own 10-second cap at 481 steps and did not meet the tighter reference tolerance. JSONL contains `reference_certified=false` and numerical objective differences, but those differences must **not** be presented as validated optimality gaps. The IR point independently attains the requested KKT tolerance and its feasible dual-gap bound is 1.38e-10. Other problem references met their tighter thresholds. No successful-pair time ratio may be formed with censored runs.

The NNLS control is important: IR is **slower**, with an expanding rather than sparse working set (569 -> 754 of 1024), and a final support of 749. Nonunique coefficients are not compared. The sparse timings do not imply a universal advantage.

## Products and working sets

Totals below include public-call setup/stopping products but exclude reference generation/external final oracle. IR logical F/A totals can exceed baseline totals despite much narrower products. Width-weighted products are sum(calls * source_width/n * underlying_batch_rows), **not FLOPs**; particularly not a complete CMF cost model.

| Case | Solver | Forward / adjoint calls | Width-weighted products | IR working-set first -> last (max) |
|---|---|---:|---:|---|
| Dense 512x8192 | IRFISTAB | 743 / 498 | 49.04 | 599 -> 3 (599) |
| same | FISTA | 480 / 479 | 959 | — |
| same | ISTA | 491 / 490 | 981 | — |
| CMF 240x1024 | IRFISTAB | 777 / 521 | 77.37 | 174 -> 3 (174) |
| same | FISTA | 557 / 556 | 1113 | — |
| same | ISTA | 1547 / 1546 | 3093 | — |
| Dense 1024x32768 | IRFISTAB | 1011 / 677 | 30.89 | 824 -> 3 (824) |
| same (censored) | FISTA | 580 / 579 | 1159 | — |
| same (censored) | ISTA | 580 / 579 | 1159 | — |
| CMF 992x4096 | IRFISTAB | 1929 / 1290 | 68.58 | 371 -> 3 (371) |
| same | FISTA | 1866 / 1865 | 3731 | — |
| same (censored) | ISTA | 5879 / 5878 | 11757 | — |
| NNLS 256x1024 | IRFISTAB | 578 / 388 | 646.22 | 569 -> 754 (754) |
| same | FISTA | 161 / 160 | 321 | — |
| same | ISTA | 293 / 292 | 585 | — |

All benchmark IR runs had zero backtracking rejections and accepted L <= 1; they therefore do not measure curvature-growth overhead. Backtracking growth/underflow correctness is covered by regression tests, not these timings. Inner-budget exhaustion occurred in 2/7 solves for CMF 240x1024, 1/7 for the large dense problem, and 6/10 for CMF 992x4096; none in the first dense case or NNLS control.

Full IR certificate scans in the first two cases: 8 forward + 8 adjoint and 9 + 9 respectively. Restricted calls: 735 + 490 and 768 + 512. No native-restriction or batch fallbacks used by these B=1 instances; generic/B-wide overhead remains covered by correctness tests, not measured as a performance study here.

## Isolated-process memory probes

The harness lifetime RSS values are contaminated by previous solvers and references (all ~611404 KiB in the first run), so they are not used for comparative memory conclusions. A separate parent numerical probe ran each (family,solver) in a fresh Python process, using the existing harness `make_problem` and `run_solver` functions on the first two configurations. No source files were changed.

Method: import harness with importlib, set torch threads=1/no_grad; make the same problem; gc.collect(); record VmRSS from /proc/self/status; write `5` to /proc/self/clear_refs to reset VmHWM; call run_solver with benchmark-mode cap10, inner100/outer200/baseline10000/check_every10; then read VmHWM. Increment is peak-minus-before. Each child had a 40-second process timeout, returned success, and confirmed the external KKT target. Saved as `memory_isolated.jsonl` (6 records). The exact successful probe is preserved in `memory_probe.md` for reproduction.

| Problem | Solver | Before RSS KiB | Reset peak RSS KiB | Increment KiB |
|---|---|---:|---:|---:|
| Dense 512x8192 | IRFISTAB | 536796 | 580308 | 43512 |
| same | FISTA | 536836 | 540080 | 3244 |
| same | ISTA | 536960 | 540076 | 3116 |
| CMF 240x1024 | IRFISTAB | 511112 | 551100 | 39988 |
| same | FISTA | 511160 | 511736 | 576 |
| same | ISTA | 511052 | 511648 | 596 |

Scope includes public solve AND the external final oracle, not just tensor workspace. OS high-water RSS includes lazy mapped pages, PyTorch allocations and allocator effects; one cold run per solver, not a precise allocation profiler. These measured increments are **higher for IR**; no memory reduction claim. They do not show a Gram matrix was constructed (source/runtime allocation guards test that separately). No CUDA memory measurements.

An initial probe failed before producing data because a scratch `/proc/self/status` parser assumed every field had a value; empty x86_Thread_features fields caused IndexError. Parser was restricted to VmRSS/VmHWM and the six probes rerun successfully. This was a measurement-script error, not a solver failure.

## Review and remaining limitations

Independent post-measurement audit completed in `validation.md` §9: no Critical/Major findings; descriptive measurements accepted with explicit limitations. The parent reconciled NNLS public-call product counts against raw JSONL (including stopping products). Gate 2 CPU correctness passed before these experiments; the benchmarks are additional descriptive evidence, not a new theoretical guarantee. GPU, Acoular captures, multiple independent seeds, warm-up-controlled statistical comparisons, robust allocation profiling, performance of generic/B-wide fallbacks, and the full proposed multi-hour experiment-plan study remain unmeasured.
