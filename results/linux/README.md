# Measured results, Linux

Raw output of `bench/fair/` (see `bench/fair/README.md` for the systems, grids and rules).
`summary_tables.md` and `edges_tables.md` are generated from these files by
`bench/fair/summarise.py` and `bench/fair/edges_summary.py`.

## Conditions

| | |
|---|---|
| CPU | AMD Ryzen 7 6800H (8 cores / 16 threads, AVX2, no AVX-512), laptop on mains power |
| host | Windows 11, 15.3 GB RAM |
| Linux | WSL2, Ubuntu 24.04, kernel 5.15.167.4-microsoft-standard-WSL2, 7.5 GB RAM visible to it |
| software | Python 3.12.3, numpy 2.5.4, faiss-cpu 1.15.1 (`OPTIMIZE DD AVX2`), ScaNN 1.4.2, hnswlib 0.8.0 (built from source), USearch 2.26.5, Rust 1.92.0 |
| MAIVS | built inside WSL with the repository's flags (`target-cpu=native`, fat LTO); the commit is in `environment.json` |
| threads | 16 for every system (`OMP_NUM_THREADS`, `RAYON_NUM_THREADS`, hnswlib `set_num_threads`, USearch `threads=`) |
| data | HotpotQA, MiniLM-L6-v2, d = 384, unit norm; base = first N rows of `hotpot_5M.npy`, IND = its last 1,000 rows, OOD = 1,000 real questions |
| ground truth | exact brute force, k = 10 (5.2 s at 250k, 9.6 s at 500k) |
| peak memory | 4.4 GB (MAIVS churn), then 4.3 GB (faiss IVF-RaBitQfs4) |

## Files

| file | what |
|---|---|
| `environment.json` | versions, CPU, compile options |
| `<system>_250000.json`, `<system>_500000.json` | the main comparison, one file per system and size |
| `maivs-quality_500000_repeat.json`, `faiss-hnsw_500000_repeat.json` | the drift check: re-run last, in fresh processes |
| `summary_tables.md` | matched-recall tables for the main comparison |
| `scann-int8_500000.json`, `scann-soar-int8_500000.json`, `faiss-hnsw-sq8_500000.json`, `faiss-ivf-pqfs-sq8_500000.json` | compressed baselines for the same-memory comparison |
| `churn_<system>.json` | the churn runs (`maivs` is the `quality` profile) |
| `filtered_<system>.json` | filtered search |
| `edges_tables.md` | tables for the memory-matched, churn and filtered runs |

## Notes

* **Drift.** MAIVS moved -5% to +3% between the first run and the repeat, faiss HNSW -5% to -9%.
  MAIVS running first didn't favour it. Treat differences under about 10% as ties.
* **Grid limits.** ScaNN without SOAR and the faiss IVF indexes don't reach 0.99-0.995 at 500k
  within these grids (leaves to search <= 256, nprobe <= 512). ScaNN + SOAR reaches every target,
  so the comparison against the strongest baseline doesn't depend on that.
* **ScaNN batching.** ScaNN is measured through its own batch API, and `search_batched_parallel`
  splits a batch into chunks of `batch_size` queries. The default of 256 only uses 4 threads on
  1,000 queries, so the chunk size is picked per build from 4-256. ScaNN's single-query throughput
  wasn't measured, so the per-query MAIVS numbers only compare fairly against HNSW.
* **One corpus, one model, one laptop**, at most 500k vectors. MS MARCO and 1M+ weren't re-run
  here; those Windows numbers are in `results/windows/`.

### The ScaNN SOAR update crash

ScaNN 1.4.2 crashes when a SOAR index is updated, so ScaNN + SOAR is missing from the churn
results. `bench/fair/scann_soar_crash.py` reproduces it. Each run builds with `docids`, then does
10 rounds of deleting and upserting N/40 vectors:

| N | variant | result |
|---|---|---|
| 20k | SOAR, f32 reorder, int ids | segfault in `delete()`, round 0 |
| 20k | SOAR, f32 reorder, string ids | same |
| 20k | SOAR, f32 reorder, upsert batch size 1 | same (in delete) |
| 20k | SOAR, f32 reorder, deleting a contiguous id range | same |
| 20k | no SOAR, f32 reorder | finishes |
| 20k | SOAR, INT8 reorder (random or contiguous ids) | finishes |
| 100k | SOAR, INT8 reorder | finishes |
| 400k | SOAR, INT8 reorder, upsert batch size 256 | abort in `upsert()`, round 0 |
| 400k | SOAR, INT8 reorder, upsert batch size 1 | rounds 0-3 finish, then `free(): invalid next size (normal)` in round 4 |
| 400k | SOAR, f32 reorder (the full churn run) | SIGSEGV |

So it looks like memory corruption in the SOAR incremental-update path: every time with f32
reorder, depending on size with INT8. Without SOAR, updates work, and that's what
`churn_scann.json` uses.
