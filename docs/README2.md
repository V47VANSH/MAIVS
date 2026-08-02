# AIGVS v2: a partitioned vector index for churn

Research prototype. Everything below was measured on my machine against exact brute-force ground
truth (`bench/`). What isn't true yet is listed under [Open gaps](#open-gaps).

This replaces `README.md`, which describes the June architecture. Several claims in it didn't
hold up once I measured them properly; see [What changed](#what-changed).

## What it is

A partitioned vector index meant for continuous inserts and deletes: no rebuilds, no compaction,
no recall decay, and 2.6x less memory than HNSW.

It's ahead on memory, build time, delete and recall under churn. It's behind faiss HNSW on raw
throughput at the same recall (1.15-2.1x), and it doesn't reach HNSW's top recall.

## Architecture

```text
  centroids (f32 SIMD scan)
       |  rank cells, pick a capped set
       v
  cells: Vec<u32> offsets into one global arena
       |  flat SIMD scan, no graph inside a cell
       v
  exact rerank of the top k * rerank_factor
```

Code: `src/index_v2.rs` and `src/rabitq.rs`.

**One arena.** All vectors are in one contiguous store and cells hold u32 offsets into it. A
vector is never copied into a second cell, so an extra membership costs 4 bytes, not a vector.

**No graph in the cells.** At d=384 a flat int8 scan beats an NSW search with ef=100 up to about
10k vectors per cell, and NSW adjacency at M=64 is 544 B per node against 384 B of vector.
Binary codes push the crossover to ~25k. A graph in the cell would also rule out O(1) delete.

**O(1) delete.** `swap_remove` from the cell's member list and push the slot onto the arena's free
list. No tombstones, so nothing to compact and nothing for recall to decay from. 0.9 us at 100k,
1.14 us at 1M.

**Three scan modes**, so the cost of quantization is measured rather than assumed:

| mode | bytes/vector (d=384) | ranking |
|---|---|---|
| `f32` | 1536 | exact within the scanned cells |
| `i8` | 384 | globally scaled int8 dot, then rerank |
| `binary` | 64 | rotated sign bits, POPCNT Hamming, then rerank |

The binary mode rotates with a fast Hadamard transform and random sign flips (the RaBitQ-style
rotation) rather than a dense random orthogonal matrix. A d x d rotation is 147k FLOPs per query
(20-40 us), more than the scan it's meant to speed up; the FHT is d*log2(d) ~ 4.6k ops (1-2 us).
Without the rotation, sign bits are a poor code because a few high-variance dimensions dominate.

## Results

100k: 99,000 base vectors, 1,000 queries, k=100, batch throughput on all cores. v2 config: binary
scan, int8 rerank (rf=8), adaptive probing, 314 cells.

| | v2-binary | faiss-hnsw |
|---|---|---|
| memory | 65 MB (464 B/vector) | 172 MB |
| build | 2.0 s | 19.6 s |
| delete | 0.92 us, O(1) | not supported |
| recall change over 100% churn | -0.0004 | needs a full rebuild |
| throughput at matched recall | see below | 1.15-2.1x ahead |
| best recall | 0.9946 | 1.0000 |

Throughput at matched recall (HNSW interpolated to my recall points):

| R@10 | v2-binary | faiss-hnsw | gap |
|---|---|---|---|
| 0.963 | 31,585 | 37,131 | 1.18x |
| 0.982 | 23,442 | ~26,900 | 1.15x |
| 0.989 | 15,309 | ~22,000 | 1.44x |
| 0.995 | 7,264 | ~15,100 | 2.08x |

Against faiss IVFFlat it's 8.7x the throughput at equal recall, and 10.9x during churn (71,740 vs
6,558 qps).

### Churn

10 cycles, each replacing 10% of the index, so the whole index turns over once. Ground truth is
recomputed for the live set at every checkpoint.

| | R@10 start -> end | change | rebuilds |
|---|---|---|---|
| v2 | 0.9380 -> 0.9376 | -0.0004 | none |
| faiss-ivf | 0.9636 -> 0.9576 | -0.0060 | none |
| faiss-hnsw | 0.986 throughout | | 92 s of rebuilds |

v2 kept 71,740 qps during churn, with 1.6 us deletes and 10.8 us inserts. faiss HNSW has no
delete, so keeping its recall meant rebuilding every cycle.

### Adaptive probing

nprobe becomes a cap. A query stops once no unscanned cell can hold anything better than its
current k-th result, using each cell's angular radius as the bound.

| nprobe cap | fixed | adaptive | recall |
|---|---|---|---|
| 32 | 27,805 | 29,214 | 0.9626 both |
| 64 | 12,182 | 23,442 | 0.9816 / 0.9815 |
| 128 | 10,688 | 15,309 | 0.9893 / 0.9894 |

Up to 96% more throughput at the same recall. Easy queries stop early, so a high cap costs little
and the hard queries still get it.

**1M** (999,000 base, 999 cells): R@10 0.9872, 513 MB (464 B/vector), build 18.7 s, delete
1.11 us, so delete is still flat against 0.92 us at 100k.

### Two things worth pointing out

**Quantization costs nothing in recall when you rerank.** int8 and f32 gave identical recall at
every nprobe, and binary at rerank_factor=8 gives 0.9942 against int8's 0.9943. The old engine's
recall ceiling was a bug, not a trade-off.

**The scan is memory-bandwidth bound**, so throughput comes down to bytes per query:

| nprobe | v2-f32 | v2-i8 | speedup | byte ratio |
|---|---|---|---|---|
| 32 | 1,896 | 6,424 | 3.4x | 4x |
| 128 | 484 | 1,910 | 3.9x | 4x |

About 25.6 GB/s at nprobe=32, which is what this machine can move. That's why quantization
changes the cost model and not just the memory footprint.

## Can routing do better?

A query's true top-10 sit in only 3.5 cells on average, and an oracle that probes the right 8
cells gets R@10 0.9997, while ranking by centroid needs nprobe=128 for 0.9942. That's 6.4x of
headroom, so I tried to get at it.

It didn't work. A gradient-boosted model over eight query-time features (centroid cosine,
max/mean/std cell radius, cell size, reachability bounds, gap to the best cell) got AUC 0.9715
against 0.9664 for plain cosine, but in practice it was worse: 37 probes to reach what cosine
reaches in 32. Ranking by the classic reachability bound was worse again (42-45 probes), because
a max-radius bound is set by one outlier and makes every big cell look good.

Two follow-ups were also negative:

* **Sub-centroids** (rank a cell by max cos(q, s) over S sub-centroids) help steadily, R@10 at
  np=8 goes 0.8863 -> 0.9100 at S=16, but only 1.2-1.3x on bytes once the extra keys are paid
  for, even binarised.
* **kNN-graph refinement of the k-means assignment** moved fragmentation from 3.50 to 3.21 cells
  and didn't change recall.

So centroid cosine is already close to the best you can do with what's known at query time. The
rest is the geometry of a Voronoi partition: a query isn't an indexed point, so its neighbours
straddle cell boundaries however the cells are drawn. The 6.4x is still open and I don't claim
any of it.

### Redundant assignment doesn't pay

At equal recall, more nprobe beat putting vectors in two cells at every cell count I tried:

| cells | assign=1 | assign=2 |
|---|---|---|
| 314, np=32 | 0.9668 at 15,874 qps | 0.9782 at 11,185 qps |
| 32, np=4 | 0.8769 at 15,911 qps | 0.9176 at 9,806 qps |

It only nearly breaks even at ~8 cells, where missing a cell is very costly. Same conclusion as
Quake's choice of no replication. The code is still there, off by default.

## Open gaps

| # | Gap | Note |
|---|---|---|
| 1 | 1.15-2.1x behind faiss HNSW at matched recall | was 5.7x; next are binary routing keys (~1.2x) and further byte cuts |
| 2 | best recall 0.9946 vs 1.0000 | int8 rerank resolution; f32 rerank gets 0.9943 at 3.4x the memory |
| 3 | nothing below 10M tests the scaling idea | need MSTuring-10M |
| 4 | query/document mismatch not measured | the D2 experiment in the plan |
| 5 | split/merge (`maintain()`, `autotune()`) works but only moves toward the target size slowly | 179 cells vs sqrt(N) = 244 after one pass at 60k |
| 6 | no concurrent writes | a write during a search raises in pyo3 |
| 7 | Quake not reproduced yet | Ubuntu-only build, needs WSL |
| 8 | the 6.4x routing headroom | see above |

Limits I don't expect to change:

* **Low recall (0.80-0.92):** my minimum is one cell scan (~315 vectors); HNSW's is a few hops at
  ef=8. It wins there.
* **Single-query p50:** 288 us vs 140 us for HNSW. I do ~30x more distance computations and only
  make it up through sequential access, quantization and parallelism, so a tight interactive
  p50 favours graphs.
* **p99/p50:** adaptive probing makes the work per query vary, by design.
* **Low dimensions:** the binary codes rely on concentration of measure; below d~128 the edge
  shrinks.
* **Static data:** if nothing is ever deleted, DiskANN or SVS.

So it fits when you need high recall and have churn and memory pressure and care about throughput
more than p50 and have high-dimensional vectors.

## What changed

The old `README.md` describes a continuous Hopfield router, deep attractor trees and NSW graphs in
the leaves. Measurement retired all three.

| Old claim | What I measured |
|---|---|
| continuous Hopfield router | the code ran one softmax step over centroid cosines and never iterated, which is IVF probe selection |
| 98% recall | that was the MS MARCO qrel hit rate; ANN recall was 0.9648 and flat, because of a 6-bit quantizer and no rerank |
| 10x faster build than faiss | faiss IVFFlat built the same corpus 10x faster than I did (0.9 s vs 9.6 s); the comparison had only been against HNSW, where it was 2x |
| same or better memory | 2.1x HNSW, the boundary copies didn't show in `stats()` |
| deep attractor trees | routing error compounds as p^d and depth doesn't reduce the scan, only locality; one partition level |

On Hopfield: Ramsauer's bound says one softmax step retrieves the nearest stored pattern with
error around exp(-beta*Delta), so it bounds retrieval noise. IVF computes the centroid dots exactly
and takes an exact top-p, so there's no retrieval noise to bound. The real failure is that a
query's neighbours are in cells whose centroid isn't the nearest one, and centroid separation says
nothing about that. The idea led to the right structure (partition, then search), but the part
doing the work is routing, not associative memory.

## Build and run

```bash
# Windows: pyo3 0.23 for CPython 3.13
export PYO3_PYTHON=/c/Users/<you>/AppData/Local/Programs/Python/Python313/python.exe
cargo build --release
cp target/release/aigvs_rust.dll aigvs_rust.pyd     # .so on Linux

# exact ground truth (streams over the base set, fine for 1M+)
python bench/groundtruth.py msmarco_embeddings_cache_v2_100000.npy --k 100 --queries 1000

# recall vs QPS against faiss
python bench/run_ann_recall.py msmarco_embeddings_cache_v2_100000.npy \
    --queries 1000 --k 100 --cells 314 --rerank 8 --i8-rerank \
    --systems v2-binary,faiss-hnsw,faiss-ivf
```

`.cargo/config.toml` sets `target-cpu=native`, and `Cargo.toml` sets `lto = "fat"` and
`codegen-units = 1`. Without them the AVX2 kernels don't inline.

## Layout

| Path | What |
|---|---|
| `src/index_v2.rs` | flat-cell core: arena, cells, scan modes, O(1) delete |
| `src/rabitq.rs` | FHT rotation, 1-bit codes, POPCNT Hamming |
| `src/lib.rs` | pyo3 module, plus the old `AIGVSEngine` |
| `bench/groundtruth.py` | exact k-NN ground truth, streamed |
| `bench/run_ann_recall.py` | recall vs QPS against faiss |
| `AIGVS_MASTER_PLAN.md` | plan, literature, stopping criteria |

`src/aigvs/` is the old Python pipeline and engine, kept for comparison.

## Related work

* **SPANN** (NeurIPS'21): fine postings, graph over centroids, bounded replication. Static.
* **SPFresh** (SOSP'23): incremental in-place split/merge. SSD-resident.
* **Quake** (OSDI'25): in-memory adaptive partitioning, cost-model maintenance, per-query
  nprobe. f32 only, no quantization, no replication. The closest baseline.
* **RaBitQ / Extended RaBitQ** (SIGMOD'24/'25): 1-bit and multi-bit codes with bounds.
* **SOAR** (NeurIPS'23): redundant assignment with orthogonal residuals.
* **DARTH** (SIGMOD'26): per-query recall targets through early termination.

All of these rank cells by centroid proximity.
