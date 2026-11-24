# Migration from FA2026_code

Extracted from `FA2026_code` at `8bd9e64c35dbe65bb4e605529115fb530113e144`. `fa2026` is no
runtime dependency; ucmf needs only PyTorch and NumPy. Tests do not import `fa2026`.

## Mapping

| FA2026 (`src/fa2026/optim/`) | ucmf |
| --- | --- |
| `ista.py` | `ISTA`, `FISTA`: explicit loops, least squares only |
| general classes of `callbacks.py` | `ucmf.optim.callbacks` |
| `weights.py` | `weights.py`: weight operators |
| `preprocessing.py` | removed; normalization will be revisited |
| `ADA-ALFISTA` (`ada.py` + `alfista.py`) | `AdaLFISTASW` |
| `ADA-LFISTA` (`lfista.py`) | `AdaLFISTA`; `B1`/`B2` swapped ([0006](adr/0006-two-matrix-ada-lfista.md)) |
| `ADA-CMF-ALFISTA` (`congruence.py`) | `AdaLFISTASWR` |
| `ADA-CMF-LFISTA` (`congruence.py` + `lfista.py`) | `AdaLFISTAR` |
| `ADA[-CMF]-NA-ALFISTA`, `ADA[-CMF]-NA-LFISTA` (`na_alista.py`) | `NA_AdaLFISTA*` ([0008](adr/0008-neural-augmentation-mixins.md)) |

Not migrated: TLS, `NA_ALISTA`, `NormLSTMB`/`NormLSTMC`, `LinearTransform`, NA with
`theta = lam * gamma`, `SFISTA`/`ALFISTA` without Ada weight, `L-` variants, training,
checkpoints, acoustic callbacks, and physical models.

## Behavior

Preserved: objective `||y - A x||² / (2 m) + lam ||x||₁`, direction `A.T (y - A x) / m`,
threshold `gamma * lam`, shrinkage, Nesterov recurrence, batching, callbacks, and autograd.
Changed: the default step `1 / L`, `L = 1.05 ||A||₂² / m`, uses 50 power iterations
([0005](adr/0005-power-iteration-default-step.md)).

## Interface changes

- Operators instead of matrices ([0007](adr/0007-operators-instead-of-matrices.md)):
  `operator=MatrixMult(A)`, `CMFDictionary`, or `CMFPropagation`. `select_rows` /
  `select_mics` replace `msm_subset` / `measurement_indices`; align `y` with `operator.rows`.
- No silent conversion: inputs must share `y`'s dtype and device; place models with `model.to(...)`.
- Classical forward takes no `W`; unfolded models own their forward
  ([0002](adr/0002-classical-forward-without-weights.md)).
- Renamed: `positive` → `nonnegative`, `n_*` → `num_*` (`num_iterations`, `num_measurements`,
  `num_mics`), `n_steps` → `num_steps`. Unfolded constructors are keyword-only.
- Removed: `ls_mode`/TLS, `A_init`, `solver.y` (use the `y` callback keyword),
  `normalize_sensing_matrix_columns`, `default_gamma_from_A`, and the state framework
  ([0001](adr/0001-explicit-iteration-state.md), superseded by [0003](adr/0003-local-loops-and-ls-only.md)).
- NA models: call `init_normalizers` once before solving (FA2026 did this lazily).
- Custom subclasses override `_make_gradient(y, operator, weight_operator)` instead of
  `ls_grad` / `get_grad_func`, see [architecture](architecture.md).
