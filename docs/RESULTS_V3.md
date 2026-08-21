# Engine v3 against faiss and USearch

All numbers from one machine (Ryzen 7 6800H, 8 cores / 16 threads, 15 GB RAM, Windows), exact
brute-force ground truth, d = 384 (MiniLM embeddings). The full tables are generated from the raw
JSON by `bench/results_tables.py` into [`RESULTS_V3_tables.md`](RESULTS_V3_tables.md); the
engineering changes are in [`ENGINE_V3.md`](ENGINE_V3.md).

**Query sets.** IND = 1,000 held-out corpus passages (the usual ANN benchmark setup). OOD = 1,000
real HotpotQA questions embedded with the same model, which is what a RAG system actually serves.

**My configurations.** Both use orthogonal polysemy (2 homes), FastScan (the asymmetric 1-bit
estimator), polysemic runs and cell-major batch execution.

* `quality`: exact f32 rerank, 1,761 B/vector, slightly less than faiss-HNSW's 1,808.
* `memory`: SQ8 rerank, 609 B/vector, about a third of HNSW.

Memory is every allocation by capacity (`stats()["total_bytes"]`); for faiss it's the serialized
index size.

## 1. HotpotQA 1M at equal memory (`quality`) vs faiss-HNSW

Batch QPS (1,000 queries, all threads, best of 3) / single-thread p50 in us.

| R@10 | mine, IND | HNSW, IND | mine, OOD (questions) | HNSW, OOD |
|---|---|---|---|---|
| 0.95 | 23,543 / 285 | 8,485 / 636 | 18,616 / 431 | 5,639 / 1,044 |
| 0.98 | 11,890 / 783 | 4,740 / 1,168 | 11,481 / 1,066 | 3,176 / 1,928 |
| 0.99 | 10,300 / 937 | 3,155 / 1,734 | 7,600 / 1,337 | 1,978 / 2,923 |
| 0.995 | 6,351 / 1,674 | 1,890 / 2,702 | 4,792 / 2,311 | 1,159 / 4,470 |
| build | 6.0 s | 324 s clean (412 s in this run, overlapped with a compile) | | |

2.5-4.1x the throughput and 1.5-2.4x lower median latency at every matched recall from 0.95 to
0.995, on both query sets, at slightly lower memory and with a 54x faster build. On real questions
the lead grows with the recall target (3.3x at 0.95, 4.1x at 0.995).

Same comparison on MS MARCO 999k (IND): 2.6x at 0.98, 3.8x at 0.99, 4.6x at 0.995; p50 3.5-4.5x
lower; build 6.1 s vs 324 s.

## 2. Other baselines (HotpotQA, same rules)

| system (1M) | build | B/vector | QPS at IND 0.95 | best R@10 (IND / OOD) |
|---|---|---|---|---|
| mine, `quality` | 6.0 s | 1,761 | 23,543 | 0.9982 / 0.9970 (2M run) |
| mine, `memory` | 5.5 s | 609 | 26,176 | 0.9917 / 0.9885 |
| faiss HNSW32 | 324 s | 1,808 | 8,485 | 0.998 / 0.997 |
| faiss IVF-RaBitQ + exact refine | 97 s | 1,606 | 3,201 | 0.96 / 0.94 (grid) |
| faiss IVF-PQ4 FastScan + exact refine | 94 s | 1,652 | 681 | 0.97 / 0.96 (grid) |
| USearch HNSW (250k) | 92 s | 2,626 | 7,270 (250k) | |

USearch only at 250k (a separate HNSW implementation with the same graph parameters; slower than
faiss-HNSW here). hnswlib wouldn't install (no Python 3.13 wheel for Windows). The two faiss IVF
indexes use standard factory strings and an nprobe x k_factor grid; I didn't tune them further.

## 3. Scaling

QPS ratio, mine (`quality`) / faiss-HNSW at matched recall (HotpotQA):

| | 250k | 500k | 1M | 2M* |
|---|---|---|---|---|
| IND R@10 0.98 | 4.85x | 4.21x | 2.51x | 2.31x |
| IND R@10 0.99 | 4.99x | 4.60x | 3.26x | 2.78x |
| IND R@10 0.995 | 4.05x | 4.75x | 3.36x | 4.13x |
| OOD R@10 0.99 | 4.08x | 4.61x | 3.84x | |
| OOD R@10 0.995 | 4.14x | 4.88x | 4.13x | |
| build ratio | 17x | 62x | 54x | 46x |

\* The 2M HNSW number is from the July run on this machine (`bench_scale_hnsw.py`, same M and
efConstruction); at 1M that run agrees with this one within 2%. My 2M runs ingest in chunks from
a memory-mapped file.

What this does and doesn't show:

* On real questions the lead is ~4x and flat from 250k to 1M: my throughput falls with the same
  exponent as HNSW's (-0.41 vs -0.37 at 0.99, -0.59 vs -0.59 at 0.995). I haven't measured HNSW on
  questions beyond 1M.
* On held-out passages the lead narrows with N at 0.98-0.99 (about 5x down to 2.3-2.8x by 2M):
  HNSW's throughput falls more slowly there (-0.20 vs my -0.50 at 0.99). At 0.995 it stays near
  4x. If those trends continued, the 0.98 lead on held-out passages would close somewhere in the
  tens of millions. So "gets better with scale" isn't shown.
* At 5.23M (all of HotpotQA), `memory` builds in 60 s at 601 B/vector (3.1 GB) and reaches IND
  0.976 at 4,787 QPS, OOD 0.979 at 4,272 QPS. I didn't run HNSW: at ~1.8 KB/vector its index alone
  would be ~9.4 GB on a 15 GB machine.

## 4. Where the gains came from, in order

| step | effect |
|---|---|
| engine fixes (parallel ingest, bucket-queue top-k, O(P) bounds, SIMD rescore, prefetch, streaming cells) | build 12.1 -> 4.8 s; serial latency 2.2-2.4x lower at equal recall |
| cell-major batch execution (seeded join, shared cut-off) | batch QPS 1.5-2.5x at nprobe >= 128, identical results |
| batch routing (blocked flat Hamming, exact shortlist) | takes the tree out of the batch path |
| orthogonal polysemy (independent second home) | 1.6-2.7x fewer vectors at R@10 0.99, growing with N |
| FastScan asymmetric estimator | much higher recall at fixed probes (0.8735 -> 0.9465 at nprobe 8, rf 4); this is the step that put it ahead at every recall level |

## 5. Scope

What the data supports: on HotpotQA (250k-1M, held-out passages and real questions) and MS MARCO
(1M), at equal memory, higher throughput and lower median and p99 latency than faiss-HNSW at every
matched R@10 from 0.95 to 0.995, with a 17-62x faster build (54x at 1M); at a third of the memory,
higher throughput up to R@10 0.98. On real questions throughput falls with N at the same rate as
HNSW's up to 1M.

What it doesn't:

* Anything about being state of the art. I haven't compared against ScaNN, DiskANN/Vamana,
  SymphonyQG, CAGRA or other graph + quantization engines, or on the standard ann-benchmarks sets
  (SIFT, GIST, GloVe, Deep), or at other dimensions.
* Getting better with scale: not on held-out passages (the lead narrows), and unknown for
  questions past 1M.
* Small-batch throughput: the batch numbers use cell-major execution with 1,000-query batches.
  Single-query latency is reported separately and is also better than HNSW's at every matched
  recall, but I haven't measured small batches.

## 6. Known gaps

* AVX2 kernels only (no ARM/NEON), built and tested on Windows only, Python binding only; no
  server, write-ahead log or sharding.
* FastScan frees the flat codes after `compact()`. The first mutation after that decodes them back
  (once, in parallel) and the index runs on the slower non-FastScan path until the next
  `compact()`, and nothing schedules compaction yet.
* Deletes leave the two-stage routing codes stale until the next refresh or reload (small effect
  on routing).
* `ConcurrentIndex` doesn't use cell-major execution, and save/load doesn't store the polysemy or
  FastScan settings (set them again after loading).
* It's a laptop: run-to-run spread is up to ~10-35% under thermal load, and smaller gaps are ties.

## 7. Reproduce

```bash
KMP_DUPLICATE_LIB_OK=TRUE python tests/test_index.py                      # all checks
python bench/build_hnsw_cache.py hotpot_5M.npy 1000000                    # faiss-HNSW, cached
python bench/bench_vs_faiss.py hotpot_5M.npy --n 1000000 \
    --questions bench/_lab_cache/hotpot_questions_minilm.npy \
    --systems ours-quality+poly+fs,ours-memory+poly+fs,hnsw --rfs 8,16,32 \
    --probes 64,128,256,512,1024,2048,4096
python bench/bench_scale_ours.py hotpot_5M.npy --n 2000000 --variant quality+poly+fs \
    --questions bench/_lab_cache/hotpot_questions_minilm.npy
python bench/results_tables.py bench/results/vs_faiss_*.json > docs/RESULTS_V3_tables.md
```
