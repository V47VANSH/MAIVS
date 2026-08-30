# bench/fair: comparison on Linux

The earlier numbers in `results/windows/` were measured on Windows, and when I went back to check
them I found that the faiss there was the generic build, not the AVX2 one:

```text
INFO:faiss.loader:Loading faiss with AVX2 support.
INFO:faiss.loader:Could not load library with AVX2 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
INFO:faiss.loader:Loading faiss.
```

`faiss.get_compile_options()` was empty. For HNSW that barely matters (it's bound by memory
latency; AVX2 turned out to be 0-18% faster), but it made both faiss IVF baselines meaningless:
IVF-PQ FastScan did 681 QPS at R@10 0.95 on Windows and 22,712 here. ScaNN was also missing
entirely, and it's the obvious baseline since it ships SOAR. So this directory re-runs everything
on Linux, with every system built properly, in one harness.

## Systems

| system | build | search grid |
|---|---|---|
| MAIVS `quality` / `memory` | as in `bench/bench_vs_faiss.py` (polysemy + FastScan), N/128 cells | nprobe 16-4096 (up to the cell count) x rerank factor 8, 16, 32; batches both cell-major and per-query |
| faiss HNSW32 | `IndexHNSWFlat`, M 32, efConstruction 200, inner product | efSearch 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024 |
| hnswlib | M 32, ef_construction 200, `ip` | same |
| USearch | connectivity 32, expansion_add 200, f32, `ip` | same |
| ScaNN | tree (N/500 and N/250 leaves, training sample min(N, 250k)), AH (2 dims per block, threshold 0.2), exact f32 reorder | leaves_to_search 4-256 x reorder 40, 100, 250, 500; batch chunk size picked per build |
| ScaNN + SOAR | the same plus `soar_lambda` 1.5, `overretrieve_factor` 2.0 | same |
| faiss IVF-PQ FastScan | `IVF{nl},PQ192x4fs,RFlat`, nl 1024/2048 at 250k, 2048/4096 at 500k | nprobe 4-512 (up to nl/2) x k_factor 4, 16, 64 |
| faiss IVF-RaBitQ FastScan | `IVF{nl},RaBitQfs,RFlat` (1 bit) and `RaBitQfs4` (4 bits) | same |

For the memory-matched runs: ScaNN and ScaNN + SOAR with INT8 reorder, faiss `HNSW32,SQ8`, and
faiss `IVF{nl},PQ192x4fs,Refine(SQ8)`.

## Rules

The same as `bench/bench_vs_faiss.py`, for every system:

* build: wall clock from raw vectors to a searchable index, training included
* batch QPS: all 1,000 queries in one call, 16 threads, best of 3
* latency: p50 / p99 over 200 queries, one per call, one thread, Python call included
* frontier: the grid points no other point beats on both recall and QPS; values at a target
  recall are log-interpolated between frontier points and never extrapolated
* latency at a target recall: the same over a separate p50 frontier, so each system at its best
  latency setting
* memory: MAIVS `stats()["total_bytes"]`, USearch `memory_usage`, everything else the serialized
  index size (written to `/dev/shm` and deleted)
* one process per system and size, so thread pools and OpenMP runtimes never mix

Data: base = the first N rows of `hotpot_5M.npy`; IND = its last 1,000 rows (never indexed);
OOD = the 1,000 real HotpotQA questions. Exact top-10 ground truth by brute force.

## Running it

Linux or WSL2. On WSL2 clone the repository into the Linux filesystem; building from `/mnt/c` is
slow.

```bash
bash bench/fair/setup.sh            # venv in ~/maivs-fair with faiss-cpu 1.15.1, scann 1.4.2,
                                    # hnswlib 0.8.0, usearch 2.26.5, and maivs built in release
python bench/fair/prep_data.py      # slices + exact ground truth (needs hotpot_5M.npy, see bench/README.md)
FAIR_OS="..." bash bench/fair/run_all.sh
python bench/fair/summarise.py ~/maivs-fair/results > summary_tables.md
```

`run_all.sh` stops before each system if free memory drops below 2.5 GB or disk below 5 GB
(`FAIR_DISK`; on WSL2 point it at `/mnt/c`, because the virtual disk grows on the Windows drive).
Finished systems are skipped, so after a stop it can just be started again. At the end it re-runs
MAIVS and faiss-HNSW at 500k in fresh processes to measure run-to-run drift.

`fairbench.py SYSTEM N` runs one system on its own if you want to look at a single result.

## Edge experiments

```bash
bash bench/fair/run_edges.sh
python bench/fair/edges_summary.py ~/maivs-fair/results > edges_tables.md
```

* **Same memory** (~600 B/vector, 500k): the compressed configurations above against MAIVS
  `memory`.
* **Churn** (`churn.py`): build on rows [0, 400k), then 10 rounds of deleting 10k random original
  ids and inserting 10k new rows from [400k, 500k). Ground truth is recomputed over the final live
  set. Measured on the churned index, after each system's own maintenance step (MAIVS `compact()`,
  ScaNN `rebalance()`), and against a fresh build on the final set. faiss HNSW has no delete, so
  it only reports the rebuild.
* **Filtered search** (`filtered.py`): every vector gets a random tag in [0, 1000) and the filter
  keeps tag < t, for 50%, 10%, 1% and 0.1%. Ground truth is the exact top-10 among the vectors that
  pass. MAIVS filters inside the scan, faiss HNSW uses `IDSelectorBitmap`, ScaNN has no filter API
  so it over-fetches and drops the rest, and brute force scans the matching subset.

ScaNN + SOAR isn't in the churn results because ScaNN 1.4.2 crashes when a SOAR index is updated.
`scann_soar_crash.py` reproduces it at small scale; see the notes in
`results/linux/README.md`.

## Files

| file | what |
|---|---|
| `setup.sh` | venv with the baselines, maivs built in release mode |
| `prep_data.py` | base slices, IND/OOD queries, exact ground truth |
| `fairbench.py` | one system at one size, one process |
| `run_all.sh` | every system at 250k and 500k, plus the drift repeat |
| `summarise.py` | matched-recall tables |
| `churn.py`, `filtered.py`, `run_edges.sh`, `edges_summary.py` | the edge experiments |
| `scann_soar_crash.py` | small reproduction of the ScaNN SOAR update crash |
