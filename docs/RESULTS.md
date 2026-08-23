# Results: v3 against faiss and USearch

All numbers come from one machine: Ryzen 7 6800H (8 cores / 16 threads), 15 GB RAM, Windows 11.
Ground truth is exact brute force, embeddings are MiniLM-L6-v2 (d = 384). Raw output and the full
conditions are in [`results/windows/`](../results/windows/). Every table here can be regenerated
from those files with `python bench/results_tables.py`; the full tables are in
[`RESULTS_tables.md`](RESULTS_tables.md).

Query sets:

* IND: 1,000 held-out corpus passages, the usual ANN benchmark setup.
* OOD: 1,000 real HotpotQA questions embedded with the same model, which is what a retrieval system
  actually gets asked.

My configurations both use orthogonal polysemy (2 homes), FastScan, polysemic runs and cell-major
batch execution:

* `quality`: exact f32 rerank, 1,761 B/vector (faiss-HNSW is 1,808).
* `memory`: SQ8 rerank, 609 B/vector, about a third of HNSW.

Memory for this index is every allocation by capacity (`stats()["total_bytes"]`); for faiss it's
the serialized index size. Throughput is batch QPS (1,000 queries, all threads, best of 3).
Latency is single-thread p50 / p99 including the Python call, for every system. Values at a target
recall are log-interpolated between measured points, never extrapolated.

## 1. HotpotQA 1M, equal memory (`quality`) vs faiss HNSW32

Batch QPS / single-thread p50 in us at matched R@10.

| R@10 | mine, IND | HNSW, IND | mine, OOD | HNSW, OOD |
|---|---|---|---|---|
| 0.95 | 23,543 / 285 | 8,485 / 636 | 18,616 / 431 | 5,639 / 1,044 |
| 0.98 | 11,890 / 783 | 4,740 / 1,168 | 11,481 / 1,066 | 3,176 / 1,928 |
| 0.99 | 10,300 / 937 | 3,155 / 1,734 | 7,600 / 1,337 | 1,978 / 2,923 |
| 0.995 | 6,351 / 1,674 | 1,890 / 2,702 | 4,792 / 2,311 | 1,159 / 4,470 |

| | mine, `quality` | faiss HNSW32 (M 32, efC 200) |
|---|---|---|
| build | 6.0 s | 324 s clean (MS MARCO 999k); 412 s for this corpus, with a compile running at the same time |
| memory | 1,761 B/vector | 1,808 B/vector |
| best R@10 on the grid (IND / OOD) | 0.9993 / 0.9978 | 0.9982 / 0.9968 (efSearch 512) |

At every matched recall from 0.95 to 0.995, on both query sets:

* throughput is 2.5-4.1x higher;
* median latency is 1.5-2.4x lower, and p99 1.5-2.8x lower;
* memory is slightly lower;
* the build is 54x faster than the clean HNSW build, 68x faster than this run's.

On real questions the lead grows with the recall target, from 3.3x at 0.95 to 4.1x at 0.995.

**MS MARCO 999k (IND).** At R@10 0.98, 0.99 and 0.995 this index does 25,682, 23,328 and 16,929
QPS, HNSW 9,774, 6,189 and 3,664: 2.6x, 3.8x and 4.6x, with 3.5-4.5x lower p50. Build 6.1 s vs
324 s.

## 2. Other baselines (HotpotQA, same harness)

| system | N | build | B/vector | QPS at IND R@10 0.95 | best R@10 on the grid (IND / OOD) |
|---|---|---|---|---|---|
| mine, `quality` | 1M | 6.0 s | 1,761 | 23,543 | 0.9993 / 0.9978 |
| mine, `memory` | 1M | 5.5 s | 609 | 26,176 | 0.9917 / 0.9885 |
| faiss HNSW32 | 1M | 324-412 s | 1,808 | 8,485 | 0.9982 / 0.9968 |
| faiss IVF4096-RaBitQ + exact refine | 1M | 97 s | 1,606 | 3,201 | 0.962 / 0.940 |
| faiss IVF4096-PQ192x4fs + exact refine | 1M | 94 s | 1,652 | 681 | 0.968 / 0.957 |
| mine, `quality` | 250k | 4.0 s | 1,759 | 40,818 | 0.9992 / 0.9983 |
| faiss HNSW32 | 250k | 68 s | 1,808 | 9,908 | 0.9981 / 0.9989 |
| USearch HNSW (M 32, efC 200) | 250k | 92 s | 2,626 | 7,270 | 0.9980 / 0.9977 |

* USearch is a separate HNSW implementation. I only ran it at 250k, where it was slower than
  faiss-HNSW. hnswlib has no Python 3.13 wheel for Windows, so it isn't here.
* The two faiss IVF indexes use standard factory strings with nlist ~ 4*sqrt(N), and a grid of
  nprobe in {16 ... 256} x k_factor in {4, 8, 16}. I didn't tune them beyond that, and their best
  recall is limited by the grid.

The `memory` profile against HNSW at 1M, with a third of the memory:

* IND: 3.1x the throughput at R@10 0.95 and 2.7x at 0.98.
* OOD: 3.3x at 0.95 and 2.5x at 0.98.
* At 0.99 IND it's a tie (3,750 vs 3,155 QPS, inside the run-to-run spread).
* It tops out at R@10 0.992 IND / 0.989 OOD.

## 3. Scaling

QPS ratio, mine (`quality`) / faiss HNSW32 at matched recall (HotpotQA):

| | 250k | 500k | 1M | 2M* |
|---|---|---|---|---|
| IND R@10 0.98 | 4.85x | 4.21x | 2.51x | 2.31x |
| IND R@10 0.99 | 4.99x | 4.60x | 3.26x | 2.78x |
| IND R@10 0.995 | 4.05x | 4.74x | 3.36x | 4.14x |
| OOD R@10 0.98 | 4.52x | 4.63x | 3.62x | |
| OOD R@10 0.99 | 4.08x | 4.61x | 3.84x | |
| OOD R@10 0.995 | 4.14x | 4.88x | 4.14x | |
| build time ratio | 17x | 62x | 54-68x | 46x |

\* There's no recent HNSW run at 2M. Those numbers come from the July run on the same machine
with the same M and efConstruction, IND only (`results/windows/hnsw_2026-07/`). At 250k and 1M,
where both runs exist, July's recall at each efSearch matches within 0.01 and its QPS is 0-28%
higher, so if anything the 2M ratios are conservative. My 2M runs stream from a memory-mapped
corpus.

Throughput decay exponent, QPS ~ N^beta:

| | mine | HNSW |
|---|---|---|
| OOD, 250k -> 1M, R@10 0.98 | -0.46 | -0.30 |
| OOD, 250k -> 1M, R@10 0.99 | -0.42 | -0.37 |
| OOD, 250k -> 1M, R@10 0.995 | -0.59 | -0.59 |
| IND, 250k -> 2M, R@10 0.98 | -0.55 | -0.19 |
| IND, 250k -> 2M, R@10 0.99 | -0.58 | -0.29 |
| IND, 250k -> 2M, R@10 0.995 | -0.44 | -0.45 |

Reading this:

* **Real questions.** At R@10 0.99-0.995 the lead is about 4x from 250k to 1M and both fall at
  about the same rate. At 0.98 it narrows from 4.5x to 3.6x. I haven't run HNSW on questions past
  1M.
* **Held-out passages.** At 0.98-0.99 the lead narrows with N, from about 5x to 2.3-2.8x by 2M,
  because HNSW slows down less there. At 0.995 it stays near 4x. If these exponents held, the 0.98
  lead on held-out passages would be gone somewhere in the tens of millions.
* So "gets better with scale" isn't shown.
* **All of HotpotQA, 5.23M.** The `memory` profile builds in 60 s at 601 B/vector (3.1 GB) and
  reaches IND 0.976 at 4,787 QPS and OOD 0.979 at 4,272 QPS. I didn't run HNSW: at ~1.8 KB/vector
  the index alone would be ~9.4 GB on a 15 GB machine.

## 4. Where the gains came from

Ablation in the result files (HotpotQA 1M, `memory` profile without FastScan,
`hotpot_1M_ablation_polysemy.json`):

| | no polysemy | orthogonal polysemy |
|---|---|---|
| bytes / vector | 480 | 604 |
| QPS at IND R@10 0.95 | 19,821 | 22,458 |
| best IND R@10 | below 0.98 | reaches 0.98 at 9,216 QPS |

Earlier development measurements (MS MARCO 1M unless noted; not part of `results/`):

| step | effect |
|---|---|
| engine fixes: parallel ingest, bucket-queue top-k, O(nprobe) bounds, SIMD rescore, prefetch, streaming cells | build 12.1 -> 4.8 s; serial latency 2.2-2.4x lower at equal recall |
| cell-major batch execution (seeded join, shared cut-off) | batch QPS 1.5-2.5x at nprobe >= 128, identical results |
| batch routing (blocked flat Hamming, exact shortlist) | takes the tree out of the batch path |
| orthogonal polysemy (cost model, vectors scanned) | MS MARCO 499k: 1.6x / 2.3x / 2.7x fewer vectors at R@10 0.95 / 0.98 / 0.99; HotpotQA 500k at 0.99: 1.8x (IND), 1.9x (questions); the gain grows with N ([`RESEARCH_LOG.md`](RESEARCH_LOG.md) section 4) |
| FastScan asymmetric first stage | much higher recall at fixed probes (0.8735 -> 0.9465 at nprobe 8, rf 4); the step that put it ahead at every recall level |

## 5. Scope

What the data supports, on HotpotQA 250k-2M (held-out passages and real questions) and MS MARCO 1M
(held-out passages), at equal memory against faiss-HNSW:

* higher batch throughput at every matched R@10 from 0.95 to 0.995;
* lower median and p99 single-thread latency at the same points;
* a 17-68x faster build;
* and with a third of the memory, higher throughput up to R@10 0.98.

What it doesn't support:

* Being state of the art. No comparison yet against ScaNN, DiskANN/Vamana, CAGRA or the newer
  graph + quantization engines, on the standard ann-benchmarks sets (SIFT, GIST, GloVe, Deep), or
  at other dimensions and metrics.
* Getting better with scale. The lead narrows on held-out passages, and real questions past 1M
  aren't measured.
* Small-batch throughput. The batch numbers use 1,000-query cell-major batches. Single-query
  latency is reported separately and beats HNSW at every matched recall, but batch sizes 1-100
  aren't measured.
* Linux. Everything so far is on Windows, on a laptop.

## 6. Known gaps

* AVX2 only, no NEON. Python binding only: no server, write-ahead log or sharding.
* After `compact()` with FastScan, the first mutation decodes the codes back (once, in parallel),
  and the index runs on the plain Hamming path until the next `compact()`. Nothing schedules
  compaction.
* Deletes leave the two-stage routing codes stale until the next refresh or reload (small effect
  on routing).
* `ConcurrentIndex` doesn't use cell-major execution, and save/load doesn't store the polysemy or
  FastScan settings.
* Laptop: run-to-run spread is up to ~10-35% under thermal load, so smaller gaps are ties.

## 7. Reproduce

```bash
python bench/data/build_hotpot_corpus.py --out hotpot_5M.npy     # a GPU helps
python bench/data/embed_hotpot_questions.py
python tests/test_index.py
python bench/build_hnsw_cache.py hotpot_5M.npy 1000000            # slow
python bench/bench_vs_faiss.py hotpot_5M.npy --n 1000000 \
    --questions bench/data/hotpot_questions_minilm.npy \
    --systems ours-quality+poly+fs,ours-memory+poly+fs,hnsw \
    --rfs 8,16,32 --probes 64,128,256,512,1024,2048,4096
python bench/bench_scale_ours.py hotpot_5M.npy --n 2000000 --variant quality+poly+fs \
    --questions bench/data/hotpot_questions_minilm.npy
python bench/results_tables.py bench/results/*.json
```
