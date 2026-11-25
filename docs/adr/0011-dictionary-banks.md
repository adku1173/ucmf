# 0011: Dictionary banks in training

Status: Accepted — requested by the author; implemented.

## Context

The stored datasets (`ucmf_data`) hold one dictionary per frequency bin (205 bins, 16
microphones, 16 × 16 grid; later 64 microphones, 51 × 51 grid). Band training draws
the rows per bin log-uniformly in He (`ucmf_data.stored.load_band`), so a batch mixes
frequencies. FA2026 keeps a bank of all matrices and selects `bank[bins]` per batch.
`train_model` supported one shared operator only.

Memory per bank (float32): `H` 6.6 MB / dense 52 MB at 16 microphones; `H` 273 MB /
dense 8.7 GB at 64 microphones. A dense batch of 256 distinct dictionaries at 64
microphones needs about 11 GB per step; the bank itself always fits.

## Decision

1. A bank is a per-problem operator with `K` entries: `operator_kwargs` with a leading
   `K` (`matrix` `(K, m, n)`, `H` `(K, c, n)`). `train_model(..., index_train, index_val)`
   gets the bank entry of every row of `y`/`q`; the index is required exactly for a bank.
2. `select_problems(index)` (all operators) returns the entries `index` `(B,)` as a
   per-problem operator, or one entry (int or 0-dim) as a shared operator. It gathers the
   stored data (also the cached `CMFDictionary` matrix) instead of rebuilding it.
3. The bank is normalized once; `batch_operator(D, u, problems)` selects `D` and `u` of a
   batch. Equiregularization uses the dictionary of every sample, processed per entry, so
   no per-sample copy of the bank is formed.
4. `config.batch_sampling`:
   - `random` (default): batches mix entries in proportion to their row counts; one
     dictionary per sample (per-problem operator).
   - `per_dictionary`: batches drawn from one entry; one shared operator per step.
     An entry yields `count // batch_size` batches.
   The sample distribution (e.g. log-uniform frequencies) belongs to the data, not to the
   sampling.
5. Validation runs in batches of `batch_size` after a fixed permutation (seed
   `config.seed`), dropping the incomplete batch, always with one dictionary per sample.
   The validation loss is the mean over the batches. This also applies without a bank.
   The permutation keeps the dropped samples random instead of the end of ordered data.
6. `prepare_data(config, operator, y, q, index)` handles one dataset; the check that
   `per_dictionary` has a full batch lives in `iterate_batches`.
7. `HermitianVectorization` is a plain object, not an `nn.Module`. CUDA graphs treat
   module buffers as static inputs and re-recorded every step in which a derived CMF
   operator brought new ones (`select_problems`, `select_mics` in CV), e.g. compiled
   `CMFPropagation` with one dictionary per batch fell from 51 k to 4.6 k samples/s
   (16 microphones, batch 256). Plain index tensors are ordinary graph inputs.
8. Loading the bank and the index belongs to `ucmf_data`; `ucmf` gets tensors.

## Consequences

- `per_dictionary` never trains entries with fewer than `batch_size` rows and drops the
  incomplete batch per entry, which favors frequent entries in small datasets.
- With `per_dictionary`, validation uses a per-problem operator unlike training: one
  more compiled graph.
- Validation drops up to `batch_size - 1` samples; it needs at least `batch_size` rows.
- Distinct dense dictionaries at 64 microphones do not fit; `CMFPropagation` does.
- Bank entries without rows (possible for small band draws) are harmless.
