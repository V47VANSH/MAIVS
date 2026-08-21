# AIGVS

A partitioned vector index in Rust, built to be measured against faiss.

```text
fixed-size cells over one flat arena
  -> tree routing over the cell centroids
  -> SIMD scan of quantized codes in the probed cells
  -> exact rerank of what survives
```

It's not a graph index, and most of its properties come from that: O(1) delete (no edges to
repair), filters that get faster the more selective they are, easy segmenting for concurrency
(cells don't reference each other), and a query cost you can work out in bytes before running it.

## Engine v3

Full report: [`docs/RESULTS_V3.md`](docs/RESULTS_V3.md). What changed in the engine:
[`docs/ENGINE_V3.md`](docs/ENGINE_V3.md). Research log:
[`docs/RESEARCH_POLYSEMIC_ATTRACTOR_DAG.md`](docs/RESEARCH_POLYSEMIC_ATTRACTOR_DAG.md).

HotpotQA 1M (d=384), exact ground truth, 16 threads, with slightly less memory than faiss-HNSW
(1,761 vs 1,808 B/vector). Batch QPS / single-thread p50 in us:

| R@10 | AIGVS, passages | faiss-HNSW | AIGVS, real questions | faiss-HNSW |
|---|---|---|---|---|
| 0.95 | 23,543 / 285 | 8,485 / 636 | 18,616 / 431 | 5,639 / 1,044 |
| 0.99 | 10,300 / 937 | 3,155 / 1,734 | 7,600 / 1,337 | 1,978 / 2,923 |
| 0.995 | 6,351 / 1,674 | 1,890 / 2,702 | 4,792 / 2,311 | 1,159 / 4,470 |
| build | 6.0 s | 324 s | | |

2.5-4.1x the throughput and lower p50/p99 at every matched recall from 0.95 to 0.995 on both query
sets, and the same on MS MARCO 1M (2.6-4.6x). A `memory` profile at 609 B/vector (a third of
HNSW) is 2.6-3.4x faster up to R@10 0.98.

Not shown: that it gets better with scale. On held-out passages the lead narrows from ~5x at 250k
to ~2.3-2.8x at 2M (R@10 0.98-0.99); on real questions it stays at ~4x up to 1M. I also haven't
compared against ScaNN, DiskANN or the other graph + quantization engines yet.

What changed: orthogonal polysemy (an independent second home for each vector), the FastScan
asymmetric 1-bit estimator, cell-major batch execution, and a set of engine fixes (parallel
ingest, integer bucket-queue top-k, O(P) bounds, prefetch).

## v2 results: 1M x 384d MS MARCO, exact ground truth, 16 threads

Kept for reference; the v3 numbers above replace these.

| | AIGVS | faiss-hnsw | faiss-ivf |
|---|---|---|---|
| max Recall@10 | 0.9998 | 0.9992 | 0.9935 |
| max MRR | 1.0000 | 1.0000 | 0.9990 |
| build | 9.0 s | 288 s | 8.1 s |
| memory (memory profile) | 553 MB, 464 B/vector | 1,738 MB | 1,519 MB |
| delete | 1.4 us | none | O(N) |
| recall change over 100% turnover | -0.0046 | rebuild | -0.0060 |
| filtered search at 1% selectivity | 2x faster | recall drops | |
| batch QPS at matched recall | 3-5x behind | | 6-30x ahead of IVF |

Two profiles:

| profile | arena | max R@10 | MRR | RSS |
|---|---|---|---|---|
| `memory`: binary scan + SQ8 rerank | 464 B/vector | 0.9937 | 0.9963 | 553 MB |
| `quality`: binary scan + f32 rerank | 1,616 B/vector | 0.9998 | 1.0000 | 1,697 MB |

The quality profile beats HNSW's recall ceiling with less memory than HNSW.

### Where v2 lost

Throughput. 3-5x behind faiss-hnsw at matched recall at 1M, and the scaling study
(`docs/SCALING_STUDY.md`) shows it's a trend: on HotpotQA from 10k to 5.2M, throughput at R@10 0.95
falls as N^-0.78 here against N^-0.21 for HNSW. This index is 3.6x faster at 50k and 2.15x slower
at 2M. The centroid tree is logarithmic, but the number of cells that have to be probed isn't.
More on why under [Open problem](#open-problem).

## Quick start

```bash
PYO3_PYTHON=/path/to/python cargo build --release
cp target/release/aigvs_rust.dll aigvs_rust.pyd     # .so on Linux/macOS

KMP_DUPLICATE_LIB_OK=TRUE python tests/test_index.py   # 30 invariant checks
```

`.cargo/config.toml` sets `target-cpu=native`, and `Cargo.toml` sets `lto = "fat"` and
`codegen-units = 1`. Without them the AVX2 kernels don't inline.

```python
import numpy as np, aigvs_rust

d, N = 384, 1_000_000
X = ...                                    # (N, d) float32, L2-normalised

ix = aigvs_rust.FlatCellIndex(d, "binary", 32, 0x5EED, 0)
ix.set_per_dim_sq(True)                    # 464 B/vector; leave out for f32 rerank
ix.set_two_stage(True, 2)
ix.set_tree_route(True, 2048, 16)
ix.set_adaptive(True, 0.02)                # for batch workloads

ix.train_centroids(X, N // 128)            # hierarchical k-means, in Rust
ix.add_batch(list(range(N)), X)
ix.compact()                               # cluster-major layout, worth 3x

ids, scores = ix.search(X[0], k=10, nprobe=512)
ids, scores = ix.search_batch(X[:1000], k=10, nprobe=512)
ix.remove(42)                              # ~1.4 us
```

Set the configuration before adding data; scan mode and rerank precision are fixed at ingest.

## How it works (v2 base)

The v3 changes are in `docs/ENGINE_V3.md`; this is the design they sit on.

All numbers measured on this machine against exact ground truth. Details in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), and the experiments behind them in
[`docs/`](docs/).

**1. Fixed cell size instead of sqrt(N) cells.** Classic IVF uses sqrt(N) cells, so sqrt(N)
vectors per cell and an O(sqrt(N)) scan. Holding the cell size fixed makes the scan O(1) in N and
moves the growth into routing. On MS MARCO 124k -> 999k the scan exponent at fixed recall was 0.213
with 128-vector cells against 0.488 for sqrt(N) cells. (On HotpotQA over a wider range it's higher,
see the scaling study.)

**2. Cluster-major arena (3.0x, same recall).** Slots are handed out in insertion order, so a
cell's vectors end up scattered and scanning a cell is a random gather. `compact()` reorders the
arena so each cell is one contiguous run: 3.02x / 3.22x / 2.94x at three probe depths, with
recall, R@100 and MRR unchanged. The biggest single win so far, and it's only moving data around.

**3. Binary codes and a two-stage rerank.** Vectors are rotated with a fast Hadamard transform and
reduced to sign bits (64 B/vector), scanned with POPCNT Hamming, and the top `k*rerank_factor`
are rescored exactly. That beats a single accurate scan: scanning per-dimension SQ8 directly
(400 B/vector) reaches the same recall 3x slower, because `dot_i8` is int8 SIMD while SQ8 needs an
f32 conversion per element.

**4. Per-dimension SQ8 for the rerank.** One affine scale per dimension instead of one global
scale: same 8 bits, ~3 KB of shared metadata, nothing extra per vector.

**5. Two-stage routing.** Hamming over centroid codes shortlists `nprobe*2` cells and an exact pass
re-orders only the shortlist (in SQ8, which turned out to cost nothing; see
`docs/SUMMARISER_STUDY.md` 6b).

**6. A dynamic-depth centroid tree.** Built bottom-up by clustering each level by `fanout` until
the top fits in 256 nodes, so depth comes from the data: 1 level at 1M (7,804 cells), 2 at 10M, 3
at 100M, 4 at 1B. Every level is a contiguous array scanned with SIMD. An HNSW over the same
centroids was 2x slower (1,874 vs 3,677 QPS at R@10 >= 0.95): routing wants the top 1-7% of
cells, where greedy search visits much of the graph anyway, by pointer chasing.

**7. Hierarchical k-means (training 54.5 s -> 2.5 s).** Flat k-means is O(N*k*d) and was 80% of
the build. Two levels (sqrt(k) groups, then subdivide each) is O(N*sqrt(k)*d), in process.

**8. Tree-routed ingest (96.5 s -> 11.0 s).** Placing a vector used to scan every centroid. Now it
descends the same tree as queries, with a wider shortlist (64) rescored exactly, because a badly
probed query loses one result but a badly placed vector stays wrong. Recall unchanged (0.9903 vs
0.9902 at np=1024).

**9. Adaptive early stop (1.5-1.8x batch).** A per-cell reachability bound and a suffix maximum
let probing stop once no remaining cell can beat the current k-th result. Same recall. Serial
latency gets 13% worse, since the bound only pays off across threads, so it's on for batch and off
for single queries.

**10. Parallel scan inside one query (1.3-1.56x latency).** Probed cells share nothing, so one
query can use every core. An HNSW query can't, every hop depends on the last. The rerank had to be
parallel too (at rf=32 it's 3,200 vectors x 384 B). Off by default, since it's overhead when all
cores are already busy.

**11. O(1) delete.** `swap_remove` from the cell plus a free-list push. No tombstone. 1.4 us,
independent of index size, recall -0.0046 over 100% turnover. Locality does decay: slot reuse
slowly undoes the cluster-major layout and throughput halves over a full turnover. `compact()`
fixes it in 0.19 s for 500k vectors.

**12. Filter pushdown.** A u32 tag per slot, checked in the scan loop. A rejected vector never
reaches the distance kernel, so a 1%-selective query is 2x faster. In a graph a selective filter
breaks up connectivity and recall drops.

**13. Segments for concurrency.** `ConcurrentIndex` is immutable sealed segments, one mutable head
and tombstones. Readers clone an `Arc` and scan sealed segments without a lock. Without it, a
write during a search raises and the write is lost (`len()` stayed at 94,000 when it should have
been 98,700).

**14. Byte-budget model.** Throughput follows bytes per query closely enough to solve for
parameters instead of grid-searching them:

```text
scan    = nprobe * cell_size * bytes_per_vec
routing = M * 64 B                          (flat)
          or depth * beam * fanout * 64 B   (tree)
        + nprobe * shortlist_mult * 384 B   (SQ8 re-order)
rerank  = k * rerank_factor * rerank_bytes
```

## Benchmarks

Everything is against exact brute-force ground truth (`bench/groundtruth.py`, cached in
`bench/_gt_cache/`), so recall tops out at 1.0 and doesn't depend on the embedding model.

```bash
python bench/run_all.py msmarco_embeddings_cache_v2_1000000.npy          # everything
python bench/run_all.py <corpus.npy> --quick                             # 100k slice
python bench/run_all.py <corpus.npy> --only recall,latency               # some of it
```

| script | what it measures |
|---|---|
| `bench_build.py` | build cost split into train / ingest / compact, optionally against faiss-hnsw |
| `bench_recall.py` | recall vs QPS against 5 faiss configs |
| `bench_latency.py` | p50 / p99 single-query latency against faiss-hnsw, with and without parallel scan |
| `bench_scaling.py` | fits `scanned ~ N^alpha` for fixed cells vs sqrt(N) cells |
| `bench_scale_full.py`, `bench_scale_hnsw.py` | the 10k-5.2M scaling study |
| `bench_churn.py` | recall, delete cost and locality under 100% turnover |
| `bench_filter.py` | whether selectivity buys throughput |
| `bench_concurrency.py` | writes during reads, reader scaling |

`bench/research/` has the one-off studies (summaries, scan cascade, routability), and the
`bench_*` scripts not in the table are the routing experiments in `docs/SCALABILITY_PLAN.md`.

How to read the numbers:

* `serial_qps` / `p50` / `p99`: one query at a time, one thread, including the Python -> pyo3
  call. Use it for latency, not for comparing across languages.
* `batch_qps`: the whole batch on all threads, best of 7. This is the number to compare.
* recall and MRR come from the batch pass, latency from the serial pass.
* `spread_pct` is printed, and anything inside it is a tie. I've seen up to 35% run to run on an
  idle machine.
* Both engines get `BENCH_THREADS` (default: all cores).

## Layout

```text
src/
  index.rs       FlatCellIndex: arena, cells, scan kernels, routing, rerank,
                 delete, compaction, filters, ingest
  tree.rs        dynamic-depth centroid tree, spherical/hierarchical k-means
  graph.rs       HNSW over centroids (slower, switched off)
  maintain.rs    online cell split / merge
  quantize.rs    Hadamard rotation, 1-bit packing, POPCNT Hamming
  persist.rs     save / load
  concurrent.rs  ConcurrentIndex: segments + head + tombstones
bench/           benchmarks (above); research/ for one-off studies
tests/           30 invariant checks
docs/            architecture, studies, plan and notes
```

## Open problem

In v2, routing was what stood between this and competitive throughput. At 999k with 128-vector cells the
true top-10 sit in 4.01 of 7,804 cells, but centroid ranking needs 60 probes for R@10 0.95 and 404
for 0.99. And the floor stays flat as N grows while the probes needed don't, which is the whole
scaling gap.

I ruled out a better per-cell summary: the best one is worth about 1.25x once it's priced
properly (`docs/SUMMARISER_STUDY.md`). What did move it in v3 was giving each vector a second,
orthogonal home, so the two homes miss independently; see the research log.

## Known limitations

* The v3 lead over faiss-HNSW narrows with N on held-out passages; not measured past 2M.
* "O(log N)" only holds for the router, not for the query.
* Compaction is needed under churn but nothing schedules it.
* `ConcurrentIndex` has no save/load, never reclaims tombstones in sealed segments, and a write
  still stalls readers while it runs.
* `autotune()` ends up at ~1.2x the target cell size.
* No tiered storage: 448 B/vector x 1B = 448 GB.
* AVX2 only, built and tested on Windows only.
* Save/load doesn't keep the polysemy and FastScan settings; set them again after loading.

## License

See [LICENSE](LICENSE).
