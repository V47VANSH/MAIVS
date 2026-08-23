# AIGVS codebase guide

How the code is put together: what each file does, how queries and writes run, every tunable and
where its value came from, how to build, and how to run the benchmarks.

## 1. Overview

A partitioned (IVF-style) vector index in Rust with a Python binding, built to be measured against
faiss. Not a graph index. In one line:

```text
fixed-size cells over one flat arena
  -> tree or flat routing over the cell centroids
  -> SIMD scan of quantized codes inside the probed cells
  -> exact rerank of the top k*rf
```

Most of the other properties follow from that: O(1) delete (no edges to repair), cheap filters
(a linear scan can just skip), easy to split into segments for concurrency (cells don't reference
each other), and a query cost you can work out in advance as bytes.

The original engine (Hopfield router + NSW leaves) has been removed from `src/`; it's in the git
history. Everything current is `FlatCellIndex`.

## 2. Files

### Rust (`src/`)

| File | Lines | What |
|---|---|---|
| `index.rs` | ~2,500 | the engine. `FlatCellIndex`: arena, cells, scan kernels, routing, rerank, delete, compaction, filters, ingest |
| `tree.rs` | ~600 | dynamic-depth tree over the cell centroids, spherical and hierarchical k-means. Routes queries at scale and ingest at every scale |
| `graph.rs` | ~360 | HNSW over centroids. Not used in practice: 2x slower than a flat scan at 7,804 nodes, so its threshold is 65,536. Kept for the untested range above that |
| `maintain.rs` | ~320 | online split/merge (spherical 2-means split, merge into the nearest centroid) |
| `quantize.rs` | ~230 | fast Hadamard rotation, 1-bit packing, POPCNT Hamming |
| `persist.rs` | ~270 | little-endian save/load; derived state is rebuilt, not stored |
| `concurrent.rs` | ~380 | `ConcurrentIndex`: immutable segments, a mutable head, snapshot reads |
| `lib.rs` | ~40 | module list and pyo3 registration |

### Python (`bench/`, `tests/`)

| File | What |
|---|---|
| `bench/groundtruth.py` | exact k-NN ground truth, streamed, cached in `bench/_gt_cache/`. All recall numbers come from this |
| `bench/bench_recall.py` | main benchmark: recall vs QPS for this index and 5 faiss configs |
| `bench/bench_churn.py` | churn benchmark |
| `bench/research/` | one-off studies (routability, summaries, centroid SQ8 A/B, scan cascade) |
| `tests/test_index.py` | 30 invariant checks, run after every change |

## 3. Architecture

### Data layout

```text
                       +---------------- one global arena -----------------+
  cells: Vec<Cell>     | bits      :   64 B/vec  1-bit RaBitQ-style codes  |
  +--------------+     | scan_i8   :  384 B/vec  per-dim SQ8 (rerank/scan) |
  | members:     |---->| rr_f16    :  768 B/vec  optional f16 rerank       |
  |   Vec<u32>   |     | vecs_f32  : 1536 B/vec  optional exact rerank     |
  | centroid     |     | ext_ids   :    8 B/vec  external id per slot      |
  | sum          |     | slot_cell :    4 B/vec  owning cell               |
  | r_bound,r_max|     | tags      :    4 B/vec  filter attribute          |
  +--------------+     +---------------------------------------------------+
```

A slot is an index into all of these arrays at once. A cell holds u32 slot numbers, not vectors.
After `compact()` each cell's slots are contiguous and ascending, so scanning a cell is one
sequential read. That alone was worth 3.0x, the biggest single win so far.

### Query path: `FlatCellIndex::search_one` (`index.rs`)

```text
 1. ROUTE     pick nprobe cells, first of these that exists:
                a. centroid tree   beam descent, one contiguous scan per level
                b. centroid graph  HNSW (off, threshold 65,536)
                c. two-stage flat  Hamming over all centroid codes, then exact
                                   re-order of the top nprobe*shortlist_mult
                d. flat            exact f32 over all centroids
 2. NUCLEUS   softmax over centroid scores, take cells until the cumulative
              mass reaches target_confidence (1.0 = use the whole budget)
 3. BOUNDS    if adaptive: per-cell reachability bound and suffix max, so
              probing stops once no remaining cell can beat the k-th result
 4. SCAN      score every member of each chosen cell:
                Binary -> POPCNT Hamming over 64 B codes
                I8     -> per-dim SQ8 asymmetric, or symmetric int8
                F32    -> exact dot
              tags filtered inline; optionally across cores (par_scan)
 5. SELECT    partial select of the top k*rerank_factor
 6. RERANK    rescore those against SQ8 / f16 / f32, return the top k
```

### Write path: `add_batch`

```text
 1. fit the per-dimension SQ8 ranges from the first batch (if per_dim_sq)
 2. assign every vector to a cell, in parallel:
      tree present -> descend, shortlist ASSIGN_CANDIDATES (64), rescore exactly
      otherwise    -> linear scan over all centroids (O(N*M), the slow path)
 3. encode (rotate + pack bits, and/or SQ8 / f16 / f32)
 4. put it in a free slot or append, add the slot to the cell's members
 5. recompute centroids exactly from (sum, count)
 6. refresh_derived(): radii, centroid codes, graph, tree if the partition changed
```

Routing ingest through the tree took a 1M ingest from 96.5 s to 11.0 s.

### Delete: `remove`

`swap_remove` from the owning cell and push the slot onto a free list. No tombstone. 1.4 us,
independent of index size. Recall doesn't decay (-0.0046 over 100% turnover), but locality does;
see `compact()`.

### The tree: `tree.rs`

Built bottom-up: cluster level L by `fanout` to get level L-1, stop once the top fits in `TOP_MAX`
(256) nodes. Depth is derived, never set:

| N | cells M | depth (fanout 64) |
|---|---|---|
| 1M | 7,804 | 1 |
| 10M | 78,125 | 2 |
| 100M | 781,250 | 3 |
| 1B | 7,812,500 | 4 |

Search keeps a beam per level and scans each level's codes as one contiguous run. Work is
`depth * beam * fanout * 64 B`, and `MIN_BEAM` is a constant; that's what makes it O(log N). After
building, `compact_layout()` renumbers every level so a node's children sit next to each other
(up to +34%).

### Concurrency: `concurrent.rs`

```text
 sealed:  [Arc<Segment>][Arc<Segment>]...   immutable, read without a lock
 head:    RwLock<FlatCellIndex>             takes the writes
 dead:    RwLock<HashSet<u64>>              tombstones
```

Readers clone an `Arc<Vec<Arc<Segment>>>` (one atomic increment) and scan the sealed segments
without locking. When the head reaches `seal_at` it's compacted, frozen and appended, and the new
head starts from the old centroids instead of retraining. Deletes are tombstones, so they never
modify a sealed segment.

Locks are always taken in the order head -> sealed -> dead. Only `seal_locked` holds two at once,
and nothing holds `sealed` while waiting for `head`, so there's no cycle.

Gaps: the head is on every read path, so a write stalls reads while it runs (110 ms for 5,000
inserts); tombstones in sealed segments are never reclaimed; no save/load; no segment merge.

## 4. Tunables and where they came from

### `index.rs`

| Constant | Value | Why |
|---|---|---|
| `TARGET_CELL_SIZE` | 128 | best of {128, 256, 512, 1024} at 999k on both the scan growth exponent (0.21 vs 0.23-0.46) and absolute scan at R@10 0.99 (3.3x better than 1024). This sets the complexity |
| `ASSIGN_CANDIDATES` | 64 | larger than a query beam on purpose: a badly probed query loses one result, a badly placed vector is wrong until it's deleted. All candidates are rescored exactly |

### `tree.rs`

| Constant | Value | Why |
|---|---|---|
| `DEFAULT_FANOUT` | 64 | 64 x 64 B = 4 KB per expansion, one sequential read |
| `TOP_MAX` | 256 | a scan that small is basically free; another level would only add routing error |
| `MIN_BEAM` | 32 | must not grow with M, or routing goes back to linear |
| `DEFAULT_BEAM_FRAC` | 0.5 | swept at 999k: at np=512, 823 (0.25) -> 883 (0.50) -> 893 (0.75) qps, and recall went up too |
| `MIN_CELLS_FOR_TREE` | 2048 | below this a flat centroid scan wins. The caller's `tree_min_cells` decides, not this constant |

### `graph.rs`

| Constant | Value | Why |
|---|---|---|
| `MIN_CELLS_FOR_GRAPH` | 65,536 | at 7,804 nodes the graph lost by 2x (1,874 vs 3,677 qps at R@10 >= 0.95). Set above the range where it's known to lose; the real crossover isn't measured |

### Runtime settings (Python)

| Call | Effect |
|---|---|
| `set_per_dim_sq(True)` | per-dimension affine SQ8 rerank, drops the f32 arena |
| `set_two_stage(True, mult)` | binary shortlist, then exact re-order of `nprobe*mult` centroids. mult=2 beats 4 and 1; at 1 the router needs 768 probes instead of 512 for the same recall |
| `set_centroid_sq8(True)` | on by default. Stage 2 reads an SQ8 copy of the centroids. 1.12-1.28x throughput, \|dR@10\| <= 1e-4, and less memory (12 MB -> 3 MB of centroids at 1M). Same probe counts at f32/f16/sq8/sq6/sq4 on two corpora |
| `set_tree_route(on, min_cells, fanout, beam_frac)` | tree routing |
| `set_graph_route(on, min_cells)` | HNSW routing; the tree takes precedence |
| `set_adaptive(True, 0.02)` | early stop. 1.5-1.8x batch, -13% serial. On for batch, off for single-query latency |
| `set_parallel_scan(True)` | scan and rerank across cores inside one query. 1.3-1.56x latency. Off for batch |
| `set_tags(ids, tags)` / `set_filter(lo, hi)` | filter pushdown |
| `compact()` | restore the cluster-major layout. Run it now and then under churn |
| `train_centroids(X, n_cells)` | hierarchical k-means in process, 21.8x faster than flat faiss k-means |

## 5. Build

```bash
PYO3_PYTHON="C:\Users\<you>\AppData\Local\Programs\Python\Python313\python.exe" \
  cargo build --release
cp target/release/aigvs_rust.dll aigvs_rust.pyd     # .so on Linux/macOS
```

`.cargo/config.toml` sets `target-cpu=native`, and `Cargo.toml` sets `lto = "fat"` and
`codegen-units = 1`. Without them the AVX2 kernels don't inline.

After every build:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python tests/test_index.py    # 30 checks
```

`KMP_DUPLICATE_LIB_OK=TRUE` because faiss and torch ship clashing OpenMP runtimes on Windows.

## 6. Usage

```python
import numpy as np, aigvs_rust

d, N = 384, 1_000_000
X = ...                                   # (N, d) float32, L2-normalised
ix = aigvs_rust.FlatCellIndex(d, "binary", 32, 0x5EED, 0)   # rerank_factor=32
ix.set_per_dim_sq(True)                   # 464 B/vector; leave out for f32 rerank
ix.set_two_stage(True, 2)
ix.set_tree_route(True, 2048, 16)
ix.set_adaptive(True, 0.02)               # for batch workloads

ix.train_centroids(X, N // 128)           # fixed cell size
ix.add_batch(list(range(N)), X)
ix.compact()                              # cluster-major layout, worth 3x

ids, scores = ix.search(X[0], k=10, nprobe=512)
ids, scores = ix.search_batch(X[:1000], k=10, nprobe=512)
ix.remove(42)                             # ~1.4 us
```

Set the config before `train_centroids` / `add_batch`. Changing scan mode or rerank precision
afterwards leaves a mixed index.

## 7. Benchmarks

### Ground truth

`bench/groundtruth.py` computes exact top-k by brute force, streamed over the base so 1M+ fits in
memory, and caches it in
`bench/_gt_cache/gt_<corpus>_<mode>_n<base>_q<queries>_k<k>_<hash>.npz`.

Two query modes: `holdout` (the last Q corpus vectors, left out of the base; the default) and
`external` (a separate `.npy`, e.g. embedded questions).

Recall here is ANN recall against exhaustive search: it tops out at 1.0 and doesn't depend on the
embedding model. The June version of this repo reported MS MARCO qrel hit rate, which measures the
embedder, not the index. They aren't comparable.

### Main benchmark

```bash
KMP_DUPLICATE_LIB_OK=TRUE python bench/bench_recall.py \
    msmarco_embeddings_cache_v2_1000000.npy \
    --queries 1000 --k 100 --cells 7804 --rerank 32 \
    --per-dim-sq --two-stage \
    --systems v2-binary,faiss-hnsw,faiss-ivf,faiss-hnsw-sq,faiss-ivf-sq
```

Systems: `v2-f32`, `v2-i8`, `v2-binary`, `faiss-hnsw`, `faiss-ivf`, `faiss-hnsw-sq`,
`faiss-ivf-sq`, `faiss-ivfpq`.

Main flags: `--cells`, `--rerank`, `--per-dim-sq`, `--two-stage`, `--adaptive`, `--tree`,
`--graph-min`, `--no-compact`, `--code-bits`, `--assign`.

Two harness bugs were fixed on 2026-07-27, and numbers from before that are affected:

1. It never called `set_tree_route`, so the tree routing I actually use wasn't the path being
   benchmarked. Now `--tree` (with `--tree-min`, `--tree-fanout`). It has to be set before
   `add_batch`, since the tree routes ingest too.
2. The adaptive probe grid jumped from 256 to n_cells, which at 1M skipped everything between
   R@10 0.98 and the exhaustive 0.9939, exactly where the faiss comparison is decided. It now
   steps through 512, 1024, 2048 and 4096.

### Build time: `add_batch` is O(N*d) per call

`bench_recall.py` ingests in 5 chunks of 200k and reports 74.7 s at 1M;
`bench/research/ab_centroid_sq8.py` builds the same index with one `add_batch` call in 12.2 s. The
tree isn't the difference: every `add_batch` ends in `refresh_derived()`, and its
`recompute_radii` is a scalar O(N*d) pass over the whole arena, so chunked ingest costs
`chunks * N * d`.

Ingest in as few `add_batch` calls as memory allows, and read the 9 s build in
`notes/2026-07-27.md` as the single-call number. The real fix (incremental `refresh_derived`, or
an explicit `finalize()`) isn't done.

Results are printed and written to `bench/results/<tag>.json`.

### How the numbers are produced

* `serial qps` / `p99` come from `_timed_queries`: one query at a time, one thread, including the
  Python/pyo3 call overhead, which matters at sub-millisecond scale.
* `batch qps` comes from `_timed_batch`: all queries at once, all threads, best of 7. This is the
  number to compare across systems.
* recall and MRR come from the batch pass, latency from the serial pass. Same index and nprobe, so
  recall is the same either way.
* The spread is printed. I've seen up to 35% run-to-run movement, so any gap smaller than the
  spread is a tie.
* Both engines get `BENCH_THREADS` (default: all cores). `RAYON_NUM_THREADS` is set at module
  level before `aigvs_rust` is imported, because rayon reads it when the pool is created.

### Churn

```bash
KMP_DUPLICATE_LIB_OK=TRUE python bench/bench_churn.py <corpus.npy> \
    --cycles 10 --churn 0.10 --cells 314 --nprobe 32
```

Uses the current configuration and tracks recall, delete cost and locality separately, because
only the first two need no maintenance. Locality decays and `compact()` restores it; the run
times that too.

### Tests

```bash
KMP_DUPLICATE_LIB_OK=TRUE python tests/test_index.py
```

30 checks: round trip, delete, save/load, maintenance, adaptive vs exhaustive, two-stage vs
single-stage, graph vs flat routing, delete under every storage mode, and a data check that
measures the uniform-random baseline for the same shape and fails if the fixture isn't clearly
better.

If a test is about a structure, assert that the structure exists
(`assert stats()['tree_depth'] >= 1`). Two bugs in July hid behind tests that passed without the
feature being on.

## 8. Performance model

Throughput follows bytes per query, so cost can be worked out before writing code:

```text
scan    = nprobe * cell_size * bytes_per_vec       (64 B binary, 384 B SQ8)
routing = M * 64 B                                 (flat two-stage)
          or depth * beam * fanout * 64 B          (tree)
        + nprobe * shortlist_mult * 384 B          (SQ8 re-order; 1536 B in f32)
rerank  = k * rerank_factor * rerank_bytes
```

The re-order used to be f32 (1,536 B per cell). At 1M / np=512 that was 1,536 KB per query, a
quarter of the budget, spent ordering cells whose order already has ~0.05 of error. Probe counts
don't change down to sq4; see `SUMMARISER_STUDY.md` 6b.

At 999k / np=512 / rf=8: scan 4.2 MB (64%), routing 2.1 MB, rerank 0.3 MB.

Two things measured:

* At a fixed cell size the cell scan is O(1) in N: the same 8,192-vector scan from 124k to 999k.
* Routing is the only part still growing with the corpus.

## 9. Gotchas

1. Run `compact()` after a bulk load and now and then under churn. It's worth 3.0x, churn halves
   throughput without it, and nothing calls it automatically.
2. `KMP_DUPLICATE_LIB_OK=TRUE`, or faiss + torch crash on Windows.
3. Build flags: without `target-cpu=native` and fat LTO the SIMD kernels don't inline.
4. Adaptive and parallel scan are for different workloads, and can't be combined in the scan
   (adaptive is sequential).
5. Config before data. Scan mode and rerank precision are fixed at ingest.
6. `ConcurrentIndex` has no save/load and never reclaims tombstones in sealed segments.
7. Vectors must be C-contiguous float32 and L2-normalised; everything is inner product on unit
   vectors.
