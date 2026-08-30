# Results

Two sets of measurements:

1. **Linux** (sections 1-6): every system built properly (faiss with AVX2, ScaNN 1.4.2,
   hnswlib, USearch) and run in one harness, `bench/fair/`, on HotpotQA 250k and 500k. These are
   the main results. Raw output and conditions: [`results/linux/`](../results/linux/).
2. **Windows** (section 7): HotpotQA 250k to 5.2M and MS MARCO 1M against faiss-HNSW,
   in [`results/windows/`](../results/windows/). The faiss there turned out to be the generic
   build without AVX2. That barely changes HNSW, but the faiss IVF numbers from those runs were
   wrong and are left out here.

Both use MiniLM-L6-v2 embeddings (d = 384) and exact brute-force ground truth. IND = 1,000
held-out passages; OOD = 1,000 real HotpotQA questions. Batch QPS is 1,000 queries on 16 threads,
best of 3; latency is single-thread p50 including the Python call. Values at a target recall are
log-interpolated on each system's Pareto frontier, never extrapolated; "-" means not reached.
Differences under ~10% are within run-to-run drift.

MAIVS configurations: `quality` (f32 rerank, 1,761 B/vector) and `memory` (SQ8 rerank,
609 B/vector), both with orthogonal polysemy, FastScan and cell-major batches.

## 1. HotpotQA 500k, Linux

Batch QPS at matched R@10:

| system | IND 0.95 | IND 0.98 | IND 0.99 | IND 0.995 | OOD 0.95 | OOD 0.98 | OOD 0.99 | OOD 0.995 |
|---|---|---|---|---|---|---|---|---|
| MAIVS `quality` | 29,738 | 18,001 | 12,798 | 8,548 | 25,777 | 15,091 | 10,206 | 7,199 |
| MAIVS `quality`, per-query (no batching) | 16,814 | 8,936 | 6,513 | 3,961 | 17,134 | 8,417 | 4,984 | 3,191 |
| ScaNN + SOAR | 31,008 | 19,213 | 12,349 | 9,459 | 32,397 | 18,938 | 10,711 | 8,949 |
| ScaNN | 31,243 | 14,424 | - | - | 20,522 | 10,312 | - | - |
| faiss IVF-PQ4fs | 22,712 | 12,876 | 10,041 | - | 21,623 | 10,734 | 8,316 | - |
| faiss HNSW32 | 10,101 | 5,356 | 3,427 | 2,268 | 6,697 | 3,783 | 2,602 | 1,667 |
| hnswlib HNSW32 | 11,472 | 5,579 | 3,302 | 1,589 | 6,152 | 3,943 | 2,797 | 1,981 |
| USearch HNSW32 | 7,395 | 4,170 | 2,636 | 1,649 | 4,770 | 2,725 | 1,869 | 1,258 |

Single-thread p50 in us at matched R@10:

| system | IND 0.95 | IND 0.98 | IND 0.99 | IND 0.995 | OOD 0.95 | OOD 0.98 | OOD 0.99 | OOD 0.995 |
|---|---|---|---|---|---|---|---|---|
| MAIVS `quality` | 278 | 540 | 769 | 1,181 | 310 | 632 | 927 | 1,575 |
| ScaNN + SOAR | 240 | 482 | 758 | 1,096 | 278 | 450 | 721 | 978 |
| faiss HNSW32 | 342 | 679 | 1,061 | 1,393 | 437 | 747 | 1,083 | 1,762 |
| hnswlib HNSW32 | 352 | 919 | 1,326 | 2,429 | 656 | 1,050 | 1,489 | 2,050 |

| system | build | bytes / vector | best R@10 (IND / OOD) |
|---|---|---|---|
| MAIVS `quality` | 5.2 s | 1,761 | 0.9993 / 0.9970 |
| MAIVS `memory` | 4.0 s | 609 | 0.9897 / 0.9874 |
| ScaNN + SOAR | 19.9-27.1 s | 1,940 | 0.9981 / 0.9985 |
| ScaNN | 13.6-19.5 s | 1,744 | 0.9882 / 0.9857 |
| faiss IVF-PQ4fs | 28.8-60.7 s | 1,665 | 0.9908 / 0.9903 |
| faiss HNSW32 | 152.6 s | 1,808 | 0.9988 / 0.9990 |
| hnswlib HNSW32 | 147.5 s | 1,812 | 0.9988 / 0.9989 |
| USearch HNSW32 | 206.4 s | 2,659 | 0.9983 / 0.9989 |

(Ranges are over the two partition sizes tried for ScaNN and IVF.)

What this says:

* **Against HNSW** (faiss, hnswlib, USearch): 2.6-3.9x the batch throughput of the best HNSW at
  every point, 1.5-2.6x even without batching, 1.1-1.4x lower p50, and a build about 30x faster
  (5.2 s against 147-206 s).
* **Against ScaNN + SOAR**: level on held-out passages (0.90-1.04x) and behind on real questions
  (0.80-0.95x). ScaNN's p50 is equal or lower at every point. MAIVS builds 4-5x faster (5.2 s vs
  20-27 s) with about 10% less memory.
* faiss IVF-PQ FastScan with AVX2 is a real competitor too (about 0.75x of MAIVS here), unlike the
  681 QPS it showed on the Windows build.

## 2. HotpotQA 250k, Linux

| system | IND 0.95 | IND 0.99 | IND 0.995 | OOD 0.95 | OOD 0.99 | OOD 0.995 | build |
|---|---|---|---|---|---|---|---|
| MAIVS `quality` | 41,603 | 19,332 | 11,686 | 33,274 | 13,385 | 9,519 | 6.2 s |
| ScaNN + SOAR | 62,824 | 22,115 | 11,527 | 44,701 | 20,357 | 11,992 | 10.7-13.6 s |
| faiss IVF-PQ4fs | 32,860 | 15,027 | 11,547 | 33,529 | 15,122 | 11,653 | 9.2-15.8 s |
| faiss HNSW32 | 11,279 | 4,375 | 2,955 | 7,653 | 3,374 | 2,425 | 70.4 s |

At 250k ScaNN + SOAR is further ahead (MAIVS at 0.66-1.01x of the best baseline), and the lead
over the best HNSW is 3.3-4.2x. All the 250k tables, with latency, are in
[`results/linux/summary_tables.md`](../results/linux/summary_tables.md).

## 3. How much of it is batching

Cell-major batch QPS divided by per-query QPS, MAIVS `quality`:

| R@10 | 250k IND / OOD | 500k IND / OOD |
|---|---|---|
| 0.95 | 1.05x / 1.10x | 1.77x / 1.50x |
| 0.98 | 1.32x / 1.13x | 2.01x / 1.79x |
| 0.99 | 1.56x / 1.53x | 1.97x / 2.05x |
| 0.995 | 1.67x / 1.58x | 2.16x / 2.26x |

Against the best HNSW at 500k:

| R@10 | per-query, IND / OOD | cell-major, IND / OOD | p50 (faiss HNSW / MAIVS), IND / OOD |
|---|---|---|---|
| 0.95 | 1.47x / 2.56x | 2.59x / 3.85x | 1.23x / 1.41x |
| 0.98 | 1.60x / 2.13x | 3.23x / 3.83x | 1.26x / 1.18x |
| 0.99 | 1.90x / 1.78x | 3.73x / 3.65x | 1.38x / 1.17x |
| 0.995 | 1.75x / 1.61x | 3.77x / 3.63x | 1.18x / 1.12x |

So at 500k the lead over HNSW is roughly 2x from batching times 1.5-2.6x from the engine itself,
and the batching share grows with N and with the recall target.

## 4. Same memory budget (~600 B/vector), 500k

| system | B/vector | build | best R@10 IND / OOD | IND 0.95 | IND 0.98 | OOD 0.95 | OOD 0.98 | p50 us IND 0.95 / 0.98 |
|---|---|---|---|---|---|---|---|---|
| MAIVS `memory` (SQ8 rerank) | 609 | 4.0 s | 0.9897 / 0.9874 | 29,711 | 15,462 | 28,563 | 11,923 | 281 / 677 |
| ScaNN + SOAR, INT8 reorder | 788 | 20.3-28.3 s | 0.9890 / 0.9879 | 38,705 | 14,860 | 27,556 | 12,624 | 231 / 604 |
| ScaNN, INT8 reorder | 592 | 15.2-21.6 s | 0.9797 / 0.9759 | 24,889 | - | 17,628 | - | 390 / - |
| faiss HNSW32, SQ8 | 656 | 92.6 s | 0.9914 / 0.9901 | 19,950 | 7,986 | 9,947 | 5,969 | 290 / 674 |
| faiss IVF-PQ4fs + SQ8 refine | 513 | 46.8-85.6 s | 0.9835 / 0.9822 | 21,670 | 9,283 | 18,355 | 8,969 | 461 / 1,428 |

Against the systems that fit the same budget, MAIVS `memory` is 1.2-2.9x faster (ScaNN INT8 never
reaches 0.98). It's level with ScaNN + SOAR INT8, which needs 30% more memory, except at IND 0.95
where ScaNN is 1.3x faster. Only HNSW SQ8 reaches 0.99 in this budget.

## 5. Churn: 25% turnover of a 400k index

Build on rows [0, 400k), then 10 rounds of deleting 10k random original ids and inserting 10k new
rows. Ground truth over the final live set. Batch QPS at R@10 0.95 / 0.98:

| system | build | deletes | inserts | maintenance | churned, OOD | after maintenance, OOD | fresh build, OOD |
|---|---|---|---|---|---|---|---|
| MAIVS `quality` | 4.8 s | 0.24 s (418k/s) | 1.41 s (71k/s) | `compact()` 0.39 s | 5,924 / - | 21,595 / 15,050 | 22,235 / 14,816 |
| MAIVS `memory` | 4.3 s | 0.31 s | 1.03 s | `compact()` 0.14 s | 4,077 / - | 26,296 / 14,633 | 30,355 / 11,955 |
| ScaNN (no SOAR) | 17.6 s | 0.48 s | 1.60 s | `rebalance()` (retrain) 19.1 s | 24,043 / - | 19,175 / - | 24,557 / - |
| hnswlib HNSW32 | 125.9 s | 0.10 s (`mark_deleted`) | 427.4 s (`replace_deleted`) | none | 6,535 / 3,851 | | 6,765 / 3,923 |
| faiss HNSW32 | | no delete | no delete | full rebuild 121.8 s | | | 6,052 / 3,349 |
| ScaNN + SOAR | | crashes | crashes | | | | |

* MAIVS takes the full 25% turnover, cleanup included, in about 2 s and ends at fresh-build
  quality (best R@10 0.9993 / 0.9983 after `compact()`, the same as a fresh build). A complete
  rebuild (3.8 s) is also cheaper than any other system's update path.
* The cleanup is needed: before `compact()` it's 3.6-6.4x slower and its recall ceiling drops to
  0.9856 IND / 0.9664 OOD. Nothing schedules `compact()` yet.
* ScaNN without SOAR updates quickly without losing anything, but tops out around 0.98 in this
  grid. ScaNN + SOAR crashes on update (ScaNN 1.4.2; reproduction in `bench/fair/`).
* hnswlib's `replace_deleted` inserts took 3.6x longer than building from scratch, and faiss HNSW
  can't delete.
* Not tested: drift, where the new data doesn't fit the existing cells. MAIVS never retrains its
  cells and fits the SQ8 range on the first batch.

## 6. Filtered search, 500k

Random tag per vector, filter keeps 50% / 10% / 1% / 0.1%. Batch QPS at R@10 0.95, IND:

| system | 50% | 10% | 1% | 0.1% |
|---|---|---|---|---|
| MAIVS (filter inside the scan) | 14,154 | 12,659 | 8,334 | 2,943 |
| faiss HNSW32 (`IDSelectorBitmap`) | 8,107 | 4,388 | 968 | best R@10 only 0.83 |
| ScaNN + SOAR (over-fetch, then filter) | 30,075 | 14,970 | 3,771 | 448 |
| brute force over the matching subset | 453 | 2,327 | 22,759 | 172,535 |

MAIVS reaches R@10 >= 0.9987 at every selectivity and is the fastest index at <= 1% (at 1%,
1.4-3.2x ScaNN and 7-12x HNSW across the recall targets), while faiss HNSW falls apart at 0.1%. But it isn't the best option anywhere:
over-fetching ScaNN wins at 10-50%, and brute force over the matching subset wins at <= 1%. The
"filters make it faster" result from July compared MAIVS with itself.
All four selectivities, IND and OOD: [`results/linux/edges_tables.md`](../results/linux/edges_tables.md).

## 7. Windows runs, 250k to 5.2M

These are the runs in [`results/windows/`](../results/windows/) and
[`RESULTS_tables.md`](RESULTS_tables.md). The faiss on that machine was the generic build without
AVX2, which I only found later. For HNSW it hardly matters (it's limited by memory latency):

| N | queries | R@10 | faiss HNSW, Windows -> Linux | MAIVS, Windows -> Linux | MAIVS / HNSW, Windows -> Linux |
|---|---|---|---|---|---|
| 250k | IND | 0.99 | 4,184 -> 4,375 | 20,894 -> 19,332 | 5.0x -> 4.4x |
| 250k | OOD | 0.99 | 3,324 -> 3,374 | 13,577 -> 13,385 | 4.1x -> 4.0x |
| 500k | IND | 0.99 | 2,898 -> 3,427 | 13,319 -> 12,798 | 4.6x -> 3.7x |
| 500k | OOD | 0.99 | 2,279 -> 2,602 | 10,508 -> 10,206 | 4.6x -> 3.9x |

So the HNSW ratios from Windows are 10-20% too high, and the faiss IVF-RaBitQ and IVF-PQ FastScan
numbers from those runs (3,201 and 681 QPS at 1M) were wrong; I've dropped them.

What the Windows runs add is scale. MAIVS `quality` against faiss HNSW32 at matched recall:

| | 250k | 500k | 1M | 2M* |
|---|---|---|---|---|
| IND R@10 0.99 | 4.99x | 4.60x | 3.26x | 2.78x |
| IND R@10 0.995 | 4.05x | 4.74x | 3.36x | 4.14x |
| OOD R@10 0.99 | 4.08x | 4.61x | 3.84x | |
| OOD R@10 0.995 | 4.14x | 4.88x | 4.14x | |

\* July HNSW run, same machine and parameters, IND only.

Expect these to be 10-20% lower with AVX2 faiss. On held-out passages the lead narrows with N; on
real questions it stays roughly flat up to 1M. MS MARCO 1M looked the same (2.6-4.6x at 0.98-0.995
on Windows). At 5.23M the `memory` profile builds in 60 s at 601 B/vector and reaches IND 0.976 at
4,787 QPS; HNSW wouldn't fit on the machine.

## 8. Where the gains came from

From the development runs, HotpotQA 1M, same harness throughout (Windows; each row
already has the engine fixes and cell-major batching):

| variant | B/vector | IND 0.95 | IND 0.98 | IND 0.99 | OOD 0.95 | OOD 0.98 | best IND / OOD |
|---|---|---|---|---|---|---|---|
| `memory`, no polysemy, no FastScan | 480 | 19,821 | - | - | - | - | 0.9728 / 0.9480 |
| `quality`, no polysemy, no FastScan | 1,632 | 6,872 | 5,166 | 3,739 | 6,176 | 3,907 | 0.9921 / 0.9886 |
| `memory` + polysemy | 604 | 22,458 | 9,216 | - | 13,809 | 4,040 | 0.9894 / 0.9835 |
| `memory` + polysemy + FastScan | 609 | 26,176 | 12,572 | 3,750 | 18,888 | 8,000 | 0.9917 / 0.9885 |
| `quality` + polysemy + FastScan | 1,761 | 26,339 | 13,950 | 10,361 | 17,684 | 11,486 | 0.9993 / 0.9978 |
| faiss HNSW32 (generic build) | 1,808 | 8,756 | 4,929 | 3,228 | 5,496 | 3,101 | 0.9982 / 0.9968 |

Without polysemy and FastScan, `quality` was behind HNSW at 1M (6,872 vs 8,756), which is what the
earlier report found. Together they're worth 3.4-3.8x at IND 0.95 and ~2.9x on real questions, and
that's what turned it around. Earlier in the chain:

| change | effect | when |
|---|---|---|
| flat cells, correct quantizer scale, exact rerank (v1 -> v2) | R@10 0.9648 -> 0.9952, QPS 78 -> 390 | July, 100k |
| 1-bit rotated codes instead of int8 | 4.3x at R@10 0.966 | July, 100k |
| cluster-major arena (`compact()`) | 3.0x, same recall | July, 1M |
| SQ8 centroid re-order | 1.12-1.28x | July, 1M |
| engine fixes (parallel encode, bucket-queue top-k, O(nprobe) bounds, prefetch) | 1.23-1.50x per query; build 12.1 -> 4.8 s | 1M |
| cell-major batches | 1.26x at nprobe 64 up to 2.40x at 1024 | 1M |

These come from different corpora, sizes and engine versions, so they shouldn't be multiplied
together. What I don't have yet is a leave-one-out ablation of the final configuration on the
Linux harness.

The short version of the mechanism: a partition scan is bound by memory bandwidth, so fewer bytes
(1-bit codes, contiguous cells, reading each cell once per batch) means proportionally more speed;
a graph search is bound by memory latency, where compression just adds decode work to each hop.
1-bit codes lose recall on their own, and the exact rerank, the independent second home and the
FastScan estimator win it back.

## 9. Scope

What the data supports, for 384-d MiniLM embeddings of text passages on one machine:

* about 3x the batch throughput of HNSW (2.6-3.9x at 500k), 1.5-2.6x without batching, and lower
  median latency at every matched recall from 0.95 to 0.995;
* level with ScaNN + SOAR on held-out passages and 5-20% behind on real questions, with a 4-5x
  faster build and ~10% less memory; ScaNN has the lower single-query latency;
* updates: 25% turnover absorbed in about 2 s including cleanup;
* in a ~600 B/vector budget, the fastest or level with the fastest, up to R@10 ~0.99.

What it doesn't support:

* anything about other datasets (SIFT, GIST, GloVe, Deep), dimensions or embedding models;
* anything above 500k on Linux, or above 5.2M at all;
* a comparison with Quake, SPFresh or DiskANN, which matters most for the update results;
* filtered search being a strength (it holds up, but it's never the fastest).

## 10. Known gaps

* `compact()` isn't scheduled automatically, and until it runs a churned index is several times
  slower.
* In cell-major mode the exactly-once bitset is built before adaptive pruning, so with early
  termination on (the benchmark config) a replica whose lowest-id home got pruned can be skipped.
  Measured overlap with the per-query path: 0.9985-1.0000.
* The tests that cover FastScan, the asymmetric cascade and real-data polysemy need a corpus file,
  so they don't run without one.
* AVX2 only; Python binding only; save/load doesn't keep the polysemy and FastScan settings.

## 11. Reproduce

The Linux comparison, from `bench/fair/README.md`:

```bash
bash bench/fair/setup.sh
python bench/fair/prep_data.py
bash bench/fair/run_all.sh && bash bench/fair/run_edges.sh
python bench/fair/summarise.py ~/maivs-fair/results
python bench/fair/edges_summary.py ~/maivs-fair/results
```

The Windows runs: `bench/bench_vs_faiss.py` and `bench/bench_scale_ours.py`, as in
[`bench/README.md`](../bench/README.md).
