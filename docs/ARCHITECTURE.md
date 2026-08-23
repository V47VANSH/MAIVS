# Architecture

How the engine is put together: data layout, the build, query and write paths, the Python API,
and the cost model. The change-by-change record of the v3 engine, with the test that covers each
change, is in [`ENGINE_V3.md`](ENGINE_V3.md). The v2 guide, which explains where every v2 constant
came from, is in [`history/ARCHITECTURE_v2.md`](history/ARCHITECTURE_v2.md).

## 1. Overview

A partitioned (IVF-style) vector index in Rust with a Python binding (pyo3). Not a graph index:

```text
fixed 128-vector cells over one flat arena
  -> route: 1-bit centroid codes, exact SQ8 rescore of a shortlist
  -> scan:  each probed cell is one contiguous run of 1-bit codes,
            scored with an asymmetric FastScan estimator (4-bit query LUTs)
  -> exact rerank of the top k * rerank_factor
```

Two ideas on top of that:

* **Orthogonal polysemy.** Every vector gets a second home, chosen so that its residual there is
  orthogonal to its residual in the first. The two homes then miss independently, so a query that
  misses one home rarely misses both.
* **Cell-major batch execution.** A batch is routed first, then the probe lists are inverted to
  cell -> queries, so each cell is read once and scored for every query that wants it.

From not being a graph: O(1) delete (no edges to repair), filters that get faster as they get more
selective, segments that can be read without locks, and a query cost you can work out in bytes
before running it.

## 2. Files

| file | what |
|---|---|
| `src/index.rs` | `FlatCellIndex`, the engine: arena, cells, routing (`route_query`, `route_batch`), scan kernels, polysemy, cell-major join (`search_batch_cellmajor`), rerank, delete, compaction, filters, ingest |
| `src/quantize.rs` | randomised Hadamard rotation, 1-bit packing, POPCNT Hamming, `AsymQuery`, FastScan (`FsQuery`, block packing, AVX2 kernel) |
| `src/tree.rs` | dynamic-depth centroid tree; spherical and hierarchical k-means |
| `src/concurrent.rs` | `ConcurrentIndex`: immutable sealed segments plus one mutable head |
| `src/persist.rs` | save / load (derived state is rebuilt, not stored) |
| `src/maintain.rs` | online cell split / merge |
| `src/graph.rs` | HNSW over centroids; slower than a flat scan, switched off |
| `src/prof.rs` | opt-in phase timers (`aigvs_rust.profile`, `profile_read`) |
| `tests/test_index.py` | invariant suite, run after every change |
| `bench/` | benchmarks and data builders ([`bench/README.md`](../bench/README.md)) |

## 3. Data layout

```text
cells: Vec<Cell>                one global arena, indexed by slot
 +-------------------+          +--------------------------------------------+
 | members: Vec<u32> |--------> | bits     64 B   1-bit rotated codes        |
 | centroid, sum, n  |          | scan_i8  384 B  per-dimension SQ8 (rerank) |
 | r_bound, r_max    |          | vecs_f32 1536 B exact rerank (quality)     |
 +-------------------+          | ext_ids  8 B, slot_cell 4 B, tags 4 B      |
                                +--------------------------------------------+
after compact():
  cell c  = [ primary run | replica run ]     contiguous, ascending slots
  FastScan blocks: codes of each run interleaved by nibble, 32 vectors/block
```

A slot indexes every array at once, and a cell stores slot numbers, not vectors. `compact()`
renumbers slots so each cell's primaries are one contiguous run. With polysemy each cell also gets
a replica run: a copy of the 64-byte code of every vector whose second home is that cell
(`rep_bits`). The rerank data is stored once. So scanning a cell is two sequential reads, never a
gather.

With FastScan on, `compact()` repacks both runs into 32-vector blocks (`fs_codes`, 2 KB per block
at 512 bits) and frees the flat codes. The first mutation after that decodes them back
(`ensure_bits`), and the index uses the plain Hamming path until the next `compact()`.

Bytes per vector, every allocation by capacity (`stats()["total_bytes"]`):

| profile | rerank | B/vector at 1M |
|---|---|---|
| `quality` | f32 | 1,761 |
| `memory` | per-dimension SQ8 | 609 |

Both include two homes per vector (2 x 64 B of codes) and all the bookkeeping.

## 4. Build

```text
train_centroids(X, N/128)   hierarchical spherical k-means, in process
add_batch(ids, X)
  1. fit the per-dimension SQ8 range (parallel min/max)
  2. place: descend the centroid tree, rescore 64 candidates exactly
  3. polysemy: pick the extra home(s) from the best 16 candidates by
               ||r'||^2 + lam * sum_h (r' . r_h_hat)^2
  4. encode in parallel straight into the arena (append_parallel)
  5. exact centroids from (sum, count); radii; centroid codes; tree
compact()                   cluster-major renumbering; replica runs;
                            FastScan blocks
```

Build time is the wall clock for all three calls: 6.0 s for 1M x 384 in the `quality` profile on
16 threads, against 324 s for faiss HNSW32 on the same machine ([`RESULTS.md`](RESULTS.md),
section 1).

Every `add_batch` ends in an O(N*d) refresh of derived state, so many small batches cost more than
one large one (`bench_scale_ours.py` ingests in large chunks for that reason). Making the refresh
incremental is still open.

## 5. One query (`search`)

```text
1. ROUTE    route_query: Hamming over all centroid codes (or tree descent
            at scale) -> shortlist of nprobe * 2, ties at the cut-off kept
            -> exact SQ8 rescore -> nucleus (softmax mass >= target_confidence)
2. BOUNDS   if adaptive: per-cell reachability bound + suffix max, O(nprobe)
3. SCAN     for each cell in probe order: primary run, then replica run.
            A replica is scored only if this cell is that vector's
            first-ranked home among the probed cells, so every vector is
            scored once (crank + epoch stamps).
            Estimator: POPCNT Hamming, or FastScan block sums.
4. SELECT   integer bucket queue over the distances with a falling
            threshold; every tie at the cut-off is kept
5. RERANK   exact f32 or SQ8 over the survivors -> top k
```

## 6. A batch, cell-major (`search_batch(..., cell_major=True)`, the default)

```text
1. ROUTE    route_batch: all queries x all centroid codes in cache-blocked
            tiles (32 queries' codes stay in L1), counting selection, exact
            SQ8 rescore, same nucleus rule as route_query
2. SEED     each query scans its 8 best cells -> a valid cut-off tau
3. JOIN     invert probes to cell -> queries; split cells into balanced
            ranges; each worker streams a cell once and scores every
            query that wants it while the codes are in L1.
            Exactly once: a replica is scored at its lowest-id probed home
            (per-query probe bitset).
            Workers share a per-query AtomicU32 cut-off (fetch_min). Any
            worker's k-th best bounds the global k-th best from above, so
            rejecting above it never drops a winner.
4. MERGE    per query, exact cut-off with ties, exact rerank
```

The tests check that cell-major results equal the per-query path (overlap 1.0000 on real data).

## 7. Orthogonal polysemy

Let `r = x - mu` be a vector's residual in its home cell. A query misses x when the cell's score
`q.mu = q.x - q.r` ranks too low, so the miss is driven by `q.r`. Write `r = s + c`, where
`s = x - m(x)` is the deviation from x's local mean (the same in every partition) and `c` is the
partition-specific part.

* The next-nearest cell has a residual nearly parallel to `r`, so its misses are correlated with
  the first home's (measured 3.2x above independence).
* An independently trained second partition still shares `s` (5.7x correlated).
* A home whose residual is orthogonal to `r` misses independently (joint miss 0.018 against 0.019
  for the product of the marginals).

The rule `min ||r'||^2 + lam (r'.r_hat)^2`, evaluated over the 16 best of the 64 placement
candidates, finds such a home cheaply; on unit vectors every term comes from dot products already
computed for placement. It's SOAR's assignment loss (Sun, Guo & Kumar, NeurIPS 2023). The
derivation and measurements are in [`RESEARCH_LOG.md`](RESEARCH_LOG.md). Cost: +64 B/vector of
codes and one more member entry. Delete unlinks every home.

## 8. Estimators

| estimator | query | data | used |
|---|---|---|---|
| Hamming | 1 bit | 1 bit | default scan; centroid routing |
| `AsymQuery` | 4 bit-planes | 1 bit | optional cascade (`set_asym_cascade`, off: it costs more than it saves) |
| FastScan (`FsQuery`) | 16-entry u8 table per nibble | 1 bit, nibble-interleaved | first-stage scan with `set_fastscan(True)` |

FastScan treats each 4-bit group of a 1-bit code as an index into a per-query table of partial
inner products. One `pshufb` scores 32 vectors at one nibble position. Sums accumulate in u16 (even
and odd lanes separately). The distance is `maxk - sum`, bucketed by `d >> 6` for the bucket queue.
It's the FastScan layout of Andre et al. (VLDB 2015) applied to RaBitQ-style sign codes, which the
RaBitQ paper also does for batch estimation.

## 9. Writes, deletes, maintenance

* **Insert:** tree descent and exact rescoring of 64 candidates; extra homes come from the same
  candidates.
* **Delete:** `swap_remove` from every owning cell plus a free-list push. O(homes), no tombstones;
  recall moved -0.0046 over 100% turnover (v2 measurement).
* **Locality decays under churn:** slot reuse undoes the cluster-major layout. `compact()` restores
  it (0.19 s for 500k vectors in v2), but nothing calls it automatically.
* **Filters:** a u32 tag per slot, checked in the scan loop (`set_tags`, `set_filter`). A rejected
  vector never reaches the distance kernel.
* **Persistence:** `save` / `load`. Derived state is rebuilt on load. The polysemy and FastScan
  settings aren't stored and have to be set again.
* **Concurrency:** `ConcurrentIndex` keeps sealed segments that readers scan without locks, plus
  one mutable head. It uses the per-query path, not cell-major.

## 10. Python API

```python
import aigvs_rust
ix = aigvs_rust.FlatCellIndex(dim, scan_mode="binary", rerank_factor=32,
                              seed=0x5EED, code_bits=0)
```

| call | effect |
|---|---|
| `set_per_dim_sq(True)` | SQ8 rerank, no f32 (`memory` profile) |
| `set_orthogonal_polysemy(homes=2, lam=1.0)` | extra orthogonal homes; set before `add_batch` |
| `set_fastscan(True)` | FastScan first stage; takes effect at `compact()` |
| `set_two_stage(True, 2)` | 1-bit centroid shortlist of `nprobe * 2`, exact SQ8 rescore |
| `set_tree_route(True, min_cells, fanout)` | tree routing above `min_cells` cells (also routes ingest) |
| `set_adaptive(True, 0.02)` | early termination from per-cell bounds |
| `set_rerank_factor(rf)` | `k * rf` survivors reranked exactly; can change at query time |
| `set_parallel_scan(True)` | one query across all cores (latency mode) |
| `train_centroids(X, n_cells)` | hierarchical k-means; `n_cells = N / 128` |
| `add_batch(ids, X)`, `remove(id)`, `compact()` | writes |
| `search(q, k, nprobe, target_confidence, beta)` | one query |
| `search_batch(Q, k, nprobe, target_confidence, beta, cell_major=True)` | a batch |
| `stats()` | sizes, depth, `total_bytes`, `member_entries`, `rep_bytes` |
| `save(path)`, `FlatCellIndex.load(path)` | persistence |

Set the configuration before adding data: scan mode, rerank precision and polysemy are fixed at
ingest. Vectors must be C-contiguous float32 and L2-normalised; everything is inner product on unit
vectors.

## 11. Cost model

Throughput follows bytes moved per query:

```text
routing = M * 64 B                         (flat 1-bit centroid scan)
          or depth * beam * fanout * 64 B  (tree)
        + nprobe * 2 * 384 B               (SQ8 rescore of the shortlist)
scan    = nprobe * 128 * (1 + replicas/cell) * 64 B
rerank  = k * rf * (1536 B f32 | 384 B SQ8)
```

Cell-major execution divides the scan's memory traffic by the number of queries sharing each cell,
which is why batch throughput rises faster than single-query latency falls.

On scaling: at a fixed cell size the scan per probed cell is O(1) in N, and the tree makes routing
logarithmic in the number of cells. But the number of cells a query has to probe to hold recall
still grows with N. On HotpotQA real questions, vectors scanned at R@10 0.99 grow as about N^0.58
with polysemy (N^0.76 without). End to end, throughput falls with N at about the same rate as
HNSW's on real questions up to 1M, and faster on held-out passages ([`RESULTS.md`](RESULTS.md),
section 3). It isn't O(log N) at fixed recall.

## 12. Build and test

```bash
# Linux
python -m venv .venv && . .venv/bin/activate
pip install maturin numpy faiss-cpu
maturin develop --release              # builds and installs aigvs_rust
python tests/test_index.py

# or without maturin
cargo build --release
cp target/release/libaigvs_rust.so aigvs_rust.so   # Windows: aigvs_rust.dll -> aigvs_rust.pyd
```

`.cargo/config.toml` sets `target-cpu=native`, and `Cargo.toml` sets `lto = "fat"` and
`codegen-units = 1`. Without them the AVX2 kernels don't inline. Binaries built with
`target-cpu=native` won't run on older CPUs.

On Windows, set `KMP_DUPLICATE_LIB_OK=TRUE` when faiss and torch are in the same process; they
ship clashing OpenMP runtimes.
