# Measured results

Raw output of `bench/bench_vs_faiss.py` and `bench/bench_scale_ours.py`. Every table in
`docs/RESULTS.md` and `docs/RESULTS_tables.md` is generated from these files by
`bench/results_tables.py`; nothing is copied by hand.

## Conditions

| | |
|---|---|
| machine | AMD Ryzen 7 6800H (8 cores / 16 threads, AVX2), 15 GB RAM, laptop on mains power |
| OS | Windows 11 |
| software | Python 3.13, Rust 1.96 (release, `target-cpu=native`), faiss-cpu 1.14.3 (the generic build, without AVX2; see below), USearch 2.26.4 |
| data | MiniLM-L6-v2 embeddings, d = 384, L2-normalised (inner product = cosine) |
| ground truth | exact brute force, k = 10 |
| IND queries | the last 1,000 rows of the shuffled corpus, never indexed |
| OOD queries | 1,000 real HotpotQA questions, same model (`bench/data/`) |
| throughput | batch QPS over the whole query set, all 16 threads, best of 3 |
| latency | single-thread p50 / p99, one call per query, including the Python call; over 200 queries (30 in the 2M and 5.2M streaming runs) |
| memory | this index: every allocation by capacity (`stats()["total_bytes"]`); faiss / USearch: serialized index size |
| build | wall clock from raw vectors to a searchable index, including training |

## Files

| file | systems |
|---|---|
| `hotpot_250k.json` | `quality`, faiss HNSW32, USearch |
| `hotpot_500k.json` | `quality`, faiss HNSW32 |
| `hotpot_1M.json` | `quality`, faiss HNSW32 |
| `hotpot_1M_memory.json` | `memory` |
| `hotpot_1M_ivf-rabitq.json` | faiss `IVF4096,RaBitQ,RFlat` |
| `hotpot_1M_ivf-pqfs.json` | faiss `IVF4096,PQ192x4fs,RFlat` |
| `hotpot_1M_ablation_polysemy.json` | `memory` without FastScan, with and without orthogonal polysemy |
| `hotpot_2M.json`, `hotpot_2M_memory.json` | streamed from a memory-mapped corpus in 250k chunks (the build time includes the extra cost of chunked ingest) |
| `hotpot_5.2M_memory.json` | `memory`, all of HotpotQA (5,232,329 indexed), same streaming |
| `msmarco_1M.json` | `quality`, faiss HNSW32 (999,000 indexed, IND only) |
| `hnsw_2026-07/` | faiss HNSW32 measured in July 2026 with `bench/bench_scale_hnsw.py` on the same machine |

## Notes

* **HNSW parameters.** M = 32, efConstruction = 200 (faiss `IndexHNSWFlat`, inner product),
  efSearch swept from 16 to 512. The index is built once and cached (`bench/build_hnsw_cache.py`),
  and its build time is the time of that build.
* **HNSW build at 1M HotpotQA (412 s)** overlapped with a Rust compile on the same machine. The MS
  MARCO 999k build (324 s) didn't. Both are reported; the docs use 324 s as the clean figure and
  give the ratio for both.
* **HNSW query timings at 1M** were taken twice on the same cached index and differ by at most 3%
  (IND R@10 0.95: 8,756 vs 8,485 QPS; OOD 0.95: 5,496 vs 5,639). `hotpot_1M.json` has the later
  run. I removed the copies of the earlier run from `hotpot_1M_memory.json` and
  `hotpot_1M_ablation_polysemy.json`, so each system appears once.
* **2M HNSW** only exists from the July run (`hnsw_2026-07/`, IND only; its memory column is
  process RSS, not index size). At 250k and 1M, where both runs exist, July's recall at each
  efSearch matches the Windows run's within 0.01 and its QPS is 0-28% higher, so ratios against it at 2M
  are on the conservative side.
* **Build times at small N are noisy.** The 250k build (4.0 s) is slower than the 500k build
  (3.1 s). Fixed costs dominate there and I haven't pinned down why.
* **Laptop.** Run-to-run spread is up to ~10-35% under thermal load, so treat smaller gaps as ties.
* **faiss without AVX2.** Found later: the faiss wheel on this machine loaded its generic build
  (the AVX2 module failed to import and `get_compile_options()` was empty). HNSW is barely
  affected, AVX2 faiss being 0-18% faster on Linux, so the HNSW ratios here are 10-20% high. The
  two IVF files (`hotpot_1M_ivf-rabitq.json`, `hotpot_1M_ivf-pqfs.json`) are not valid
  measurements of those indexes and aren't used in the docs. The re-run with AVX2 faiss is in
  `results/linux/`.
