# Acoular sparse-recovery example: alpha and outer-loop tracking

## Changes and measurements

`examples/sparse_recovery.py` now uses alpha=1e-4 (previously 1e-9). Both alpha=1e-8 and alpha=1e-4 were actually run on CPU, float64, one PyTorch thread, using the unchanged 64-microphone/1681-source geometry, five frequency-bin solves, fixed 2000-step ISTA/FISTA baselines, and at most 100 IR outer sweeps. CUDA unavailable; execution substituted CPU in memory without changing the example's device configuration.

No solver algorithm or stopping rule changed. Reduction/timing collection was **run-time instrumentation**, not a new permanent adapter API or automatic tracing feature in the example. The example source change is only alpha. Raw trace artifacts and plots preserve the measurements.

## Weak regularization probe: alpha=1e-8

This is still only lambda/lambda_max = 6.85e-6 to 7.48e-6. Initial working sets 685–757 shrink to 83–127 (4.94–7.56% of the full dictionary). However, most inner solves use their entire 100-step budget, and three frequency bins exhaust the 100-outer-sweep budget. Two bins converge in 40 and 59 outer sweeps. Core IR total 34.16 s versus FISTA 37.47 s. Thus the modest timing gain is not primarily hidden assembly time; slow inner/outer convergence remains important.

Full diagnostics: `figures/sparse_recovery_alpha1e-8_diagnostics.json`.

## Meaningful stronger regularization probe: alpha=1e-4

This gives lambda/lambda_max = 0.0685–0.0748. Alpha changes the optimization problem, not merely execution speed.

| Frequency solve | First working set | First sweep with 3 variables | Final outer sweeps | Total inner steps | Final natural KKT infinity norm |
|---|---:|---:|---:|---:|---:|
| 1 | 221 | 10 | 17 | 903 | 7.31e-7 |
| 2 | 191 | 10 | 22 | 907 | 2.94e-7 |
| 3 | 173 | 10 | 20 | 876 | 5.51e-7 |
| 4 | 161 | 10 | 23 | 870 | 7.98e-7 |
| 5 | 149 | 9 | 23 | 862 | 8.05e-7 |

All five IR solves report converged at their configured absolute normalized natural-KKT tolerance 1e-6. Each final working set contains 3 out of 1681 variables (0.178%). Inner-budget hits occur 7–8 times per frequency; subsequent reduced solves become cheap. The raw and natural final KKT values, objective, product counts, accepted curvatures and status are retained in JSON.

| Solver | Core PyTorch solve total | Adapter solve total | Full example panel time |
|---|---:|---:|---:|
| LassoLars | not separately instrumented | 1.52 s | 6.2 s |
| ISTA | 40.41 s | 40.42 s | 43.4 s |
| FISTA | 39.95 s | 39.95 s | 42.9 s |
| IRFISTAB | 2.39 s | 2.41 s | 5.3 s |

One run, no warm-up/statistical inference. Baselines use fixed 2000 steps, whereas IR uses its KKT stopping test: **not a common-accuracy performance benchmark**. No general speedup claim. Core timing excludes conversion/adapter setup and a separate lambda_max adjoint per IR frequency solve. Adapter timing includes those components; total panel times additionally include CSM/dictionary assembly. Outer timing callbacks add no matrix-vector products.

## Existing tracking interface

For an actual IRFISTAB module instance after a solve:

```python
row = ir_solver.diagnostics['rows'][0]
working_set_sizes = row['working_set_sizes']
inner_steps = [step['iterations'] for step in row['inner']]
inner_budget_hits = [step['budget_exhausted'] for step in row['inner']]
counts = row['counts']
status = ir_solver.diagnostics['status'][0]
kkt = ir_solver.diagnostics['natural_residual'][0]
```

The existing Acoular TorchSolver creates a fresh module for each frequency bin; this run wrapped the module forward call to retain its completed diagnostics per bin. A simple IterationCallback measured each outer begin/end interval without evaluating extra gradients. The original callback list was restored after each call. Timings and all original diagnostics are stored per frequency solve.

Artifacts under `.pi/research/ir-fista-b/figures/`:

- `sparse_recovery_alpha1e-4.png` and `.pdf`: four acoustic maps.
- `ir_fista_reduction_alpha1e-4.png` and `.pdf`: working-set fraction, inner steps and wall time versus outer sweep, with one curve per frequency solve.
- `sparse_recovery_alpha1e-4_diagnostics.json`: raw traces and separated times.
- Corresponding `alpha1e-8` files retain the weaker-regularization probe.

Validation: full acoustic example completed at both alpha values; final five statuses/thresholds and history lengths checked from JSON; plots inspected; Ruff on the changed example and `git diff --check` passed. No core source changes or commits.
