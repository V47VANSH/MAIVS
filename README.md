# MAIVS

Multi-Attractor Inverted Vector Search: a partitioned vector index in Rust with a Python binding.

On HotpotQA passages embedded with MiniLM (d = 384), measured on Linux against AVX2 faiss, ScaNN,
hnswlib and USearch in one harness, at 500k vectors:

* about 3x the batch throughput of HNSW at matched recall (2.6-3.9x from R@10 0.95 to 0.995),
  1.5-2.6x even without batching, with lower median latency and a build about 30x faster;
* level with Google's ScaNN + SOAR on held-out passages and 5-20% behind it on real questions,
  with a 4-5x faster build and ~10% less memory; ScaNN has the lower single-query latency;
* 25% of a 400k index replaced (100k deletes and 100k inserts) in about 2 s, cleanup included,
  ending at fresh-build quality.

```text
fixed 128-vector cells over one flat arena
  -> route:  1-bit centroid codes, exact SQ8 rescore of a shortlist
  -> scan:   each probed cell is one contiguous run of 1-bit codes,
             scored with an asymmetric FastScan estimator
  -> exact rerank of the top k * rerank_factor
+ orthogonal polysemy: every vector has a second home whose residual is
  orthogonal to the first, so the two homes miss independently
+ cell-major batches: each cell is read once for all the queries that probe it
```

## Results

HotpotQA 500k, WSL2 Ubuntu on a 16-thread laptop (Ryzen 7 6800H), exact ground truth. IND =
held-out passages, OOD = 1,000 real HotpotQA questions. Batch QPS (1,000 queries, 16 threads) at
matched R@10:

| system | IND 0.95 | IND 0.99 | IND 0.995 | OOD 0.95 | OOD 0.99 | OOD 0.995 | build | B/vector |
|---|---|---|---|---|---|---|---|---|
| MAIVS `quality` | 29,738 | 12,798 | 8,548 | 25,777 | 10,206 | 7,199 | 5.2 s | 1,761 |
| MAIVS `quality`, no batching | 16,814 | 6,513 | 3,961 | 17,134 | 4,984 | 3,191 | | |
| ScaNN + SOAR | 31,008 | 12,349 | 9,459 | 32,397 | 10,711 | 8,949 | 20-27 s | 1,940 |
| faiss IVF-PQ FastScan | 22,712 | 10,041 | - | 21,623 | 8,316 | - | 29-61 s | 1,665 |
| hnswlib HNSW32 | 11,472 | 3,302 | 1,589 | 6,152 | 2,797 | 1,981 | 148 s | 1,812 |
| faiss HNSW32 | 10,101 | 3,427 | 2,268 | 6,697 | 2,602 | 1,667 | 153 s | 1,808 |

"-" = not reached in that system's grid. Latency, 250k, the memory-matched and filtered runs and
everything else: [`docs/RESULTS.md`](docs/RESULTS.md). Raw output:
[`results/linux/`](results/linux/). Harness: [`bench/fair/`](bench/fair/).

Updates (25% turnover of a 400k index, 10 rounds):

| system | delete + insert | cleanup | result |
|---|---|---|---|
| MAIVS `quality` | 0.24 s + 1.41 s | `compact()` 0.39 s | same recall and speed as a fresh build |
| ScaNN (no SOAR) | 0.48 s + 1.60 s | `rebalance()` 19.1 s | fine, recall ceiling ~0.98 in this grid |
| ScaNN + SOAR | crashes (ScaNN 1.4.2) | | reproduced in `bench/fair/scann_soar_crash.py` |
| hnswlib | 0.10 s + 427 s | | inserts into deleted slots took longer than a rebuild |
| faiss HNSW | no delete | rebuild 122 s | |

What these numbers don't show:

* One corpus, one embedding model, one laptop, up to 500k on Linux. No SIFT/GIST/GloVe/Deep, no
  other dimensions, nothing past 5.2M.
* No comparison yet with Quake, SPFresh or DiskANN, which matters most for the update numbers.
* Filtered search holds up (R@10 >= 0.9987 down to 0.1% selectivity, where faiss HNSW falls to
  0.83) but never the fastest option: over-fetching ScaNN wins at 10-50%, brute force over the
  matching subset at <= 1%.
* The batch numbers use cell-major execution with 1,000-query batches; at 500k batching is worth
  about 2x.

Earlier Windows runs at 1M-5.2M (`results/windows/`) used a faiss build without AVX2. The HNSW
comparisons there hold within 10-20%; the faiss IVF numbers from those runs were wrong and aren't
used anywhere.

## How it works

| piece | what it does | where |
|---|---|---|
| fixed-size cells | 128 vectors per cell, so the scan per probed cell is O(1) in N and the growth moves into routing | `index.rs` |
| centroid tree | depth comes from N; routes queries and ingest | `tree.rs` |
| randomised Hadamard + 1-bit codes | 64 B/vector scan codes, POPCNT Hamming | `quantize.rs` |
| FastScan estimator | the query becomes a 16-entry u8 table per 4-bit group of the code; one `pshufb` scores 32 vectors | `quantize.rs` |
| orthogonal polysemy | the second home minimises `norm(r')^2 + lam*(r'.r)^2`; the misses become independent (measured joint miss 0.018 vs 0.019 predicted) | `index.rs` |
| polysemic runs | after `compact()` each cell is its primaries followed by its replicas, two sequential streams; every vector is scored exactly once | `index.rs` |
| cell-major batches | route all queries, invert to cell -> queries, scan each cell once with a shared atomic cut-off | `index.rs` |
| exact rerank | f32 (`quality`) or per-dimension SQ8 (`memory`) | `index.rs` |

Why it's fast: a partition scan is limited by memory bandwidth, so cutting bytes (1-bit codes,
contiguous cells, reading each cell once per batch) speeds it up almost proportionally, while a
graph search is limited by memory latency. 1-bit codes lose recall on their own; the exact rerank,
the independent second home and the FastScan estimator win it back. There's no graph to build and
no codebook to train, which is where the build time goes.

More in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The research behind polysemy, with the
theory and the negative results, is in [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md).

### Prior work

The pieces are known: the polysemy assignment loss is SOAR's (Sun, Guo & Kumar, NeurIPS 2023),
the 1-bit codes follow RaBitQ (Gao & Long, SIGMOD 2024), and the FastScan layout is from Andre et
al. (VLDB 2015). Quake (OSDI 2025) and SPFresh (SOSP 2023) are the closest work on updates. What
this repository adds is a direct measurement of why orthogonal homes miss independently while
next-nearest or independently trained homes don't, exactly-once scoring over contiguous replica
runs, a cell-major join, and an end-to-end comparison in one harness. See
[`docs/REFERENCES.md`](docs/REFERENCES.md).

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install maturin numpy faiss-cpu       # the tests use faiss
maturin develop --release
python tests/test_index.py
```

Without maturin: `cargo build --release`, then copy `target/release/libmaivs.so` to `maivs.so` (on
Windows, `maivs.dll` to `maivs.pyd`). `.cargo/config.toml` sets `target-cpu=native`; the kernels
need AVX2.

```python
import numpy as np, maivs

X = ...                                    # (N, 384) float32, L2-normalised
N, d = X.shape

ix = maivs.FlatCellIndex(d, "binary", 32, 0x5EED, 0)
ix.set_per_dim_sq(True)                    # `memory` profile; leave out for f32 rerank
ix.set_orthogonal_polysemy(2, 1.0)
ix.set_fastscan(True)
ix.set_two_stage(True, 2)
ix.set_tree_route(True, 2048, 16)
ix.set_adaptive(True, 0.02)

ix.train_centroids(X, N // 128)
ix.add_batch(list(range(N)), X)
ix.compact()                               # cluster-major layout + FastScan blocks

ix.set_rerank_factor(16)
# k=10, nprobe=256, target_confidence=1.0 (use the whole probe budget)
ids, scores = ix.search(X[0], 10, 256, 1.0)
ids, scores = ix.search_batch(X[:1000], 10, 256, 1.0)  # cell-major; flat, 1000*10
ix.remove(42)
```

Set the configuration before adding data. Scan mode, rerank precision and polysemy are fixed at
ingest. After a lot of deletes and inserts, call `compact()` again.

## Reproducing the benchmarks

The Linux comparison is in [`bench/fair/`](bench/fair/README.md):

```bash
bash bench/fair/setup.sh
python bench/fair/prep_data.py
bash bench/fair/run_all.sh
bash bench/fair/run_edges.sh
```

The corpus builders and the earlier single-machine harness are described in
[`bench/README.md`](bench/README.md).

## Known limitations

* AVX2 kernels only, no ARM/NEON. Python binding only: no server, write-ahead log or sharding.
* `compact()` has to be called after heavy churn and nothing schedules it; until it runs, a
  churned index is several times slower.
* In cell-major mode with early termination on, a replica whose lowest-id home was pruned can be
  skipped (overlap with the per-query path 0.9985-1.0000).
* Deletes leave the two-stage routing codes stale until the next refresh.
* `ConcurrentIndex` doesn't use cell-major execution, and save/load doesn't store the polysemy or
  FastScan settings.
* Many small `add_batch` calls cost more than one big one; the derived-state refresh is O(N*d) per
  call.

## Layout

```text
src/
  index.rs       FlatCellIndex: arena, cells, routing, scan, polysemy,
                 cell-major batches, rerank, delete, compaction, filters
  quantize.rs    Hadamard rotation, 1-bit codes, Hamming, AsymQuery, FastScan
  tree.rs        dynamic-depth centroid tree, hierarchical k-means
  concurrent.rs  ConcurrentIndex: lock-free sealed segments + mutable head
  persist.rs     save / load
  maintain.rs    online split / merge
  graph.rs       HNSW over centroids (slower, off)
  prof.rs        opt-in phase timers
bench/           benchmarks, data builders, research scripts; bench/fair/ for the Linux comparison
results/         raw benchmark output
tests/           invariant suite
docs/            architecture, results, research log, references, history/
```

## Background

MAIVS started in June 2026 as AIGVS (Attractor Indexed Graphical Vector Search): a Python engine
that routed queries through a recursive Hopfield attractor tree into graph leaves, with a small
Rust core. The goal was O(log N) search through a deep attractor tree without the cost of copying
boundary vectors. Measuring it properly in July showed the attractor router was really IVF probe
selection, and the flat-cell engine and the polysemy work here came out of that. The name changed
with the design: there's no graph any more, and each vector lives in several attractor cells. The
prototype is in the early git history, and [`docs/history/`](docs/history/) keeps its notes and
the v2 studies, negative results included.

## License

AGPL-3.0. See [LICENSE](LICENSE).
