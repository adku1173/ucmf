# Training throughput benchmark

Measured with [`examples/benchmark_training.py`](../../examples/benchmark_training.py) in
November 2026. The raw tables (every case, both modes, memory) are in [`training/`](training/),
one file per GPU and array: `<gpu>_m16.md`, `<gpu>_m64.md`.

## Setup

- One training step as in `train_model`: draw the batch, select its dictionaries, objective
  (solve, `nmse`, `w = 0`), backward and Adam step. Batch 256, `T = 15`, float32 without TF32,
  `r_diag=True`, equiregularized data, median of 25 steps after warm-up.
- Arrays: 16 microphones with a 16 × 16 grid (m = 240, n = 256) and 64 microphones with a
  51 × 51 grid (m = 4032, n = 2601), as in `ucmf_data`.
- Dictionary layouts: `shared` (one operator, today's single-frequency runs), `per_batch`
  (bank of 205 dictionaries, one per batch: `batch_sampling='per_dictionary'`) and `distinct`
  (one per sample: `batch_sampling='random'`).
- Operators `CMFDictionary` (dense) and `CMFPropagation` (matrix-free); `MatrixMult` computes like
  `CMFDictionary`. `weight_mode` `factored`/`precomputed` for the dense-weight models (SW, LFISTA).
- GPUs: RTX 3090 24 GB (compute4, PyTorch 2.12.0+cu130), and on the HPC (Singularity image from
  [`examples/hpc/ucmf.def`](../../examples/hpc/ucmf.def), PyTorch 2.12.0+cu126) A100 PCIe 40 GB,
  V100S 32 GB, P100 16 GB and H200 NVL 141 GB. The P100 (compute capability 6.0) cannot run
  Triton, so `torch.compile` fails there and only eager mode was measured.

## Best variant per model

Throughput in samples/s at the last stage (`t = T`) for the fastest variant. Codes: `D` =
`CMFDictionary`, `P` = `CMFPropagation`; `f`/`p` = `weight_mode` factored/precomputed (none for
the reduced models SWR/R); `c`/`e` = compiled/eager. A dash: every variant ran out of memory.

#### 16 microphones, 16 × 16 grid, `shared`

| Model | RTX 3090 | A100 40 GB | V100S | P100 (eager) | H200 NVL |
|---|---:|---:|---:|---:|---:|
| AdaLFISTASW | 86.6 k Dfc | 63.3 k Dpc | 62.1 k Dpc | 11.8 k Dpe | 234.4 k Dpc |
| AdaLFISTA | 77.7 k Dfc | 81.3 k Dpc | 58.2 k Dfc | 11.5 k Dpe | 197.8 k Dpc |
| AdaLFISTASWR | 68.4 k Dc | 36.8 k Pc | 42.8 k Dc | 5.0 k De | 85.3 k Dc |
| AdaLFISTAR | 51.7 k Dc | 32.8 k Dc | 30.4 k Dc | 3.3 k De | 49.0 k Dc |
| NA_AdaLFISTASW | 61.6 k Dpc | 41.1 k Dpc | 39.2 k Dpc | 5.9 k Dpe | 86.4 k Dpc |
| NA_AdaLFISTA | 49.9 k Dfc | 45.4 k Dpc | 36.7 k Dfc | 5.1 k Dfe | 75.4 k Dpc |
| NA_AdaLFISTASWR | 48.4 k Dc | 35.2 k Dc | 29.8 k Dc | 3.2 k De | 51.6 k Dc |
| NA_AdaLFISTAR | 33.9 k Dc | 25.4 k Dc | 20.7 k Dc | 2.4 k De | 34.8 k Dc |

#### 16 microphones, 16 × 16 grid, `per_batch`

| Model | RTX 3090 | A100 40 GB | V100S | P100 (eager) | H200 NVL |
|---|---:|---:|---:|---:|---:|
| AdaLFISTASW | 79.7 k Dfc | 81.3 k Dfc | 75.6 k Dpc | 11.1 k Dpe | 197.4 k Dpc |
| AdaLFISTA | 74.0 k Dfc | 78.3 k Dpc | 62.4 k Dpc | 10.1 k Dpe | 173.4 k Dpc |
| AdaLFISTASWR | 62.3 k Dc | 37.7 k Pc | 43.8 k Dc | 4.6 k De | 83.2 k Dc |
| AdaLFISTAR | 45.6 k Dc | 25.1 k Pc | 30.7 k Dc | 3.1 k De | 48.0 k Dc |
| NA_AdaLFISTASW | 54.5 k Dpc | 39.7 k Dpc | 40.9 k Dpc | 5.5 k Dfe | 81.1 k Dpc |
| NA_AdaLFISTA | 50.1 k Dpc | 43.8 k Dfc | 37.6 k Dfc | 5.4 k Dpe | 73.9 k Dpc |
| NA_AdaLFISTASWR | 45.0 k Dc | 30.0 k Dc | 29.5 k Dc | 3.4 k De | 51.1 k Dc |
| NA_AdaLFISTAR | 34.2 k Dc | 24.9 k Dc | 21.6 k Dc | 2.4 k De | 34.5 k Dc |

#### 16 microphones, 16 × 16 grid, `distinct`

| Model | RTX 3090 | A100 40 GB | V100S | P100 (eager) | H200 NVL |
|---|---:|---:|---:|---:|---:|
| AdaLFISTASW | 43.7 k Dfc | 50.4 k Dpc | 43.7 k Dfc | 9.4 k Dpe | 114.9 k Dfc |
| AdaLFISTA | 40.7 k Dfc | 52.1 k Dpc | 41.4 k Dfc | 8.6 k Dfe | 107.8 k Dpc |
| AdaLFISTASWR | 39.5 k Pc | 32.8 k Pc | 31.0 k Dc | 4.4 k De | 64.7 k Dc |
| AdaLFISTAR | 32.9 k Pc | 25.1 k Dc | 20.9 k Dc | 3.0 k De | 41.2 k Dc |
| NA_AdaLFISTASW | 35.4 k Pfc | 36.4 k Dfc | 29.5 k Dfc | 5.5 k Dpe | 60.8 k Dfc |
| NA_AdaLFISTA | 28.9 k Pfc | 29.8 k Dpc | 25.0 k Dfc | 4.9 k Dfe | 51.9 k Dfc |
| NA_AdaLFISTASWR | 31.8 k Pc | 28.0 k Dc | 22.2 k Dc | 3.9 k De | 42.5 k Dc |
| NA_AdaLFISTAR | 24.0 k Pc | 19.2 k Dc | 15.7 k Dc | 2.5 k De | 29.2 k Dc |

#### 64 microphones, 51 × 51 grid, `shared`

| Model | RTX 3090 | A100 40 GB | V100S | P100 (eager) | H200 NVL |
|---|---:|---:|---:|---:|---:|
| AdaLFISTASW | 7.0 k Dpc | 6.8 k Dpc | 5.2 k Dpc | 3.0 k Dpe | 16.1 k Dpc |
| AdaLFISTA | 5.8 k Dpc | 5.5 k Dpc | 4.3 k Dpc | 2.4 k Dpe | 12.9 k Dpc |
| AdaLFISTASWR | 8.8 k Dc | 9.3 k Dc | 6.9 k Dc | 2.4 k De | 19.8 k Dc |
| AdaLFISTAR | 6.6 k Dc | 7.4 k Dc | 5.2 k Dc | 1.5 k De | 14.4 k Dc |
| NA_AdaLFISTASW | 6.7 k Dpc | 6.4 k Dpc | 4.9 k Dpc | 2.7 k Dpe | 14.3 k Dpc |
| NA_AdaLFISTA | 4.6 k Dpc | 4.4 k Dpc | 3.4 k Dpc | 1.9 k Dpe | 9.9 k Dpc |
| NA_AdaLFISTASWR | 8.2 k Dc | 8.6 k Dc | 6.4 k Dc | 2.2 k De | 16.9 k Dc |
| NA_AdaLFISTAR | 5.6 k Dc | 6.1 k Dc | 4.4 k Dc | 1.3 k De | 11.6 k Dc |

#### 64 microphones, 51 × 51 grid, `per_batch`

| Model | RTX 3090 | A100 40 GB | V100S | P100 (eager) | H200 NVL |
|---|---:|---:|---:|---:|---:|
| AdaLFISTASW | 1.5 k Pfc | 1.7 k Pfc | 1.2 k Pfc | 0.6 k Pfe | 16.0 k Dpc |
| AdaLFISTA | 1.3 k Pfc | 1.5 k Pfc | 1.0 k Pfc | 0.5 k Pfe | 12.9 k Dpc |
| AdaLFISTASWR | 1.7 k Pc | 2.0 k Pc | 1.4 k Pc | 0.6 k Pe | 19.6 k Dc |
| AdaLFISTAR | 1.5 k Pc | 1.8 k Pc | 1.3 k Pc | 0.5 k Pe | 14.2 k Dc |
| NA_AdaLFISTASW | 1.5 k Pfc | 1.8 k Pfc | 1.2 k Pfc | 0.6 k Pfe | 14.2 k Dpc |
| NA_AdaLFISTA | 1.1 k Pfc | 1.2 k Pfc | 0.9 k Pfc | 0.4 k Pfe | 9.8 k Dpc |
| NA_AdaLFISTASWR | 1.7 k Pc | 2.0 k Pc | 1.4 k Pc | 0.6 k Pe | 16.7 k Dc |
| NA_AdaLFISTAR | 1.3 k Pc | 1.5 k Pc | 1.0 k Pc | 0.4 k Pe | 11.5 k Dc |

#### 64 microphones, 51 × 51 grid, `distinct`

| Model | RTX 3090 | A100 40 GB | V100S | P100 (eager) | H200 NVL |
|---|---:|---:|---:|---:|---:|
| AdaLFISTASW | 1.3 k Pfc | 1.5 k Pfc | 1.1 k Pfc | 0.5 k Pfe | 4.1 k Pfc |
| AdaLFISTA | 1.2 k Pfc | 1.3 k Pfc | 0.9 k Pfc | 0.4 k Pfe | 3.4 k Pfc |
| AdaLFISTASWR | 1.5 k Pc | 1.7 k Pc | 1.2 k Pc | 0.5 k Pe | 4.5 k Pc |
| AdaLFISTAR | 1.3 k Pc | 1.6 k Pc | 1.1 k Pc | 0.4 k Pe | 4.0 k Pc |
| NA_AdaLFISTASW | 1.3 k Pfc | 1.5 k Pfc | 1.1 k Pfc | 0.5 k Pfe | 4.0 k Pfc |
| NA_AdaLFISTA | 1.0 k Pfc | 1.1 k Pfc | 0.8 k Pfc | 0.3 k Pfe | 2.8 k Pfc |
| NA_AdaLFISTASWR | 1.4 k Pc | 1.7 k Pc | 1.1 k Pc | 0.5 k Pe | 4.4 k Pc |
| NA_AdaLFISTAR | 1.1 k Pc | 1.3 k Pc | 0.9 k Pc | 0.3 k Pe | 3.2 k Pc |

## Findings

- **Compile.** Compiled steps cost the same at every stage; eager steps grow with `t`. At
  `t = T`, compiled steps are 3.5–14× faster at 16 microphones (H200: 1.5–3.7×) and 1–3× at 64
  microphones, where the large matrix products dominate. Without Triton (P100), full-depth
  training at 16 microphones is 5–15× slower than on the other GPUs.
- **Operator.** With a shared dictionary, `CMFDictionary` beats `CMFPropagation`: by up to 2.3×
  at 16 microphones (A100: 0.9–1.4×, a tie for some models) and by 3.3–5.2× at 64 microphones
  (batch 256; the operator benchmark shows 2.7–3.8× at batch 32). `CMFPropagation` only wins for `distinct` at 64 microphones, where
  every sample would need its own dense dictionary.
- **Dictionary layout.** With the same operator, a bank costs nothing per batch: `per_batch`
  reaches 90–100 % of `shared`, `distinct` about half at 16 microphones and 85 % at 64
  microphones (`CMFPropagation`). At 64 microphones the dense bank (205 × 4032 × 2601 float32,
  8.6 GB) fits only on the H200, whose allocator reserved up to 63 GB while building and
  normalizing it. Elsewhere,
  `per_batch` falls back to `CMFPropagation` and is 5× slower than `shared`.
- **`weight_mode`.** No measurable difference at 16 microphones (ratios 0.8–1.4 around 1, within
  noise). At 64 microphones with a shared dense operator, `precomputed` is 1.2–1.55× faster
  (largest for the two-weight LFISTA models). For `distinct` dense operators it loses: the
  operator benchmark measured 154 ms (factored) versus 586 ms and 2.4× the memory (precomputed).
- **GPUs.** H200 > RTX 3090 ≈ A100 > V100S ≫ P100. At 16 microphones the steps take only a few
  milliseconds; kernel launches dominate, so the A100 is not faster than the RTX 3090 and even
  slower for the reduced and NA models. Differences below about 20 % at 16 microphones are within
  the run-to-run noise.

## Recommended configuration

| Case | Operator | `weight_mode` | `compile` | GPU |
| --- | --- | --- | --- | --- |
| 16 mics, one frequency or `per_dictionary` | `CMFDictionary` | either | `True` | any with Triton |
| 16 mics, `random` (one dictionary per sample) | `CMFDictionary` | `factored` | `True` | any with Triton |
| 64 mics, one frequency | `CMFDictionary` | `precomputed` | `True` | any with Triton |
| 64 mics, bank, `per_dictionary` | `CMFDictionary` if the bank fits, else `CMFPropagation` | `precomputed` / `factored` | `True` | H200 (dense bank) |
| 64 mics, bank, `random` | `CMFPropagation` | `factored` | `True` | any with Triton |

## Reproduce

```bash
# local, one GPU per array
CUDA_VISIBLE_DEVICES=1 uv run python examples/benchmark_training.py --mics 16 > rtx3090_m16.md
# HPC: build the image locally (fakeroot), copy image and repository to scratch, then
singularity build --fakeroot ucmf.sif examples/hpc/ucmf.def
sbatch -p gpu,gpu_short --constraint=tesla_a100 examples/hpc/benchmark_training.sbatch --mics 16
sbatch -p gpu,gpu_short --constraint=tesla_p100 examples/hpc/benchmark_training.sbatch --mics 16 --mode eager
```
