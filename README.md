# AIGVS

A partitioned vector index in Rust with a Python binding.

On 1M-vector MiniLM corpora, at the same memory as faiss-HNSW, it gets 2.5-4.6x the batch
throughput at every matched recall I measured from R@10 0.95 to 0.995, lower single-query latency
at each of those points, and a 53-68x faster build.

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

HotpotQA, 1M passages, MiniLM-L6-v2 (d = 384), exact ground truth, one 16-thread laptop CPU. IND =
held-out passages, OOD = 1,000 real HotpotQA questions. Batch QPS / single-thread p50 in us at
matched R@10.

| R@10 | AIGVS, IND | faiss HNSW32, IND | AIGVS, OOD | faiss HNSW32, OOD |
|---|---|---|---|---|
| 0.95 | 23,543 / 285 | 8,485 / 636 | 18,616 / 431 | 5,639 / 1,044 |
| 0.98 | 11,890 / 783 | 4,740 / 1,168 | 11,481 / 1,066 | 3,176 / 1,928 |
| 0.99 | 10,300 / 937 | 3,155 / 1,734 | 7,600 / 1,337 | 1,978 / 2,923 |
| 0.995 | 6,351 / 1,674 | 1,890 / 2,702 | 4,792 / 2,311 | 1,159 / 4,470 |

| | AIGVS `quality` | AIGVS `memory` | faiss HNSW32 | faiss IVF-RaBitQ + refine |
|---|---|---|---|---|
| build, 1M | 6.0 s | 5.5 s | 324 s | 97 s |
| bytes / vector | 1,761 | 609 | 1,808 | 1,606 |
| QPS at IND R@10 0.95 | 23,543 | 26,176 | 8,485 | 3,201 |
| best R@10 (IND / OOD) | 0.9993 / 0.9978 | 0.992 / 0.989 | 0.9982 / 0.9968 | 0.962 / 0.940 |

On MS MARCO 1M the lead is 2.6x at R@10 0.98, 3.8x at 0.99 and 4.6x at 0.995. Full tables, the
other baselines (USearch, IVF-PQ FastScan), scaling from 250k to 5.2M and how everything was
measured are in [`docs/RESULTS.md`](docs/RESULTS.md); the raw output is in
[`results/windows/`](results/windows/).

What these numbers don't show:

* No comparison yet against ScaNN, DiskANN or CAGRA, or on the ann-benchmarks datasets.
* Everything ran on one Windows laptop.
* The batch numbers use 1,000-query batches.
* The lead doesn't grow with scale. On held-out passages it narrows from ~5x at 250k to ~2.5x at
  2M (R@10 0.98-0.99). On real questions it stays near 4x up to 1M at 0.99-0.995.

## How it works

| piece | what it does | where |
|---|---|---|
| fixed-size cells | 128 vectors per cell, so the scan per probed cell is O(1) in N and the growth moves into routing | `index.rs` |
| centroid tree | depth comes from N; routes queries and ingest | `tree.rs` |
| randomised Hadamard + 1-bit codes | 64 B/vector scan codes, POPCNT Hamming | `quantize.rs` |
| FastScan estimator | the query becomes a 16-entry u8 table per 4-bit group of the code; one `pshufb` scores 32 vectors | `quantize.rs` |
| orthogonal polysemy | the second home minimises `norm(r')^2 + lam*(r'.r)^2`; the misses become independent (measured joint miss 0.018 vs 0.019 predicted) | `index.rs` |
| polysemic runs | after `compact()` each cell is its primaries followed by its replicas, two sequential streams; every vector is scored exactly once | `index.rs` |
| cell-major batches | route all queries, invert to cell -> queries, scan each cell once with a shared atomic cut-off; same results as per-query search | `index.rs` |
| exact rerank | f32 (`quality`) or per-dimension SQ8 (`memory`) | `index.rs` |

Because it isn't a graph:

* delete is O(1), with no edges to repair;
* a filter checked inside the scan gets faster the more selective it is;
* sealed segments can be read without locks;
* the query cost is a byte budget you can work out in advance.

More in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The research behind polysemy, with the
theory and the negative results, is in [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md).

### Prior work

The individual pieces are known:

* the polysemy assignment loss is SOAR's (Sun, Guo & Kumar, NeurIPS 2023);
* the 1-bit codes follow RaBitQ (Gao & Long, SIGMOD 2024);
* the FastScan layout is from Andre et al. (VLDB 2015).

What's new here is putting them together and measuring them:

* a direct measurement that orthogonal homes miss independently, while next-nearest or
  independently trained homes don't;
* exactly-once scoring over contiguous replica runs;
* a cell-major join that gives exactly the same results as per-query search;
* an end-to-end comparison in one harness.

See [`docs/REFERENCES.md`](docs/REFERENCES.md).

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install maturin numpy faiss-cpu       # the tests use faiss
maturin develop --release
python tests/test_index.py
```

Without maturin: `cargo build --release`, then copy `target/release/libaigvs_rust.so` to `aigvs_rust.so`
(on Windows, `aigvs_rust.dll` to `aigvs_rust.pyd`). `.cargo/config.toml` sets `target-cpu=native`; the kernels
need AVX2.

```python
import numpy as np, aigvs_rust

X = ...                                    # (N, 384) float32, L2-normalised
N, d = X.shape

ix = aigvs_rust.FlatCellIndex(d, "binary", 32, 0x5EED, 0)
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
ingest.

## Reproducing the benchmarks

```bash
python bench/data/build_hotpot_corpus.py --out hotpot_5M.npy   # BEIR HotpotQA, MiniLM; a GPU helps
python bench/data/embed_hotpot_questions.py                    # the 1,000 real questions
python bench/build_hnsw_cache.py hotpot_5M.npy 1000000
python bench/bench_vs_faiss.py hotpot_5M.npy --n 1000000 \
    --questions bench/data/hotpot_questions_minilm.npy \
    --systems ours-quality+poly+fs,ours-memory+poly+fs,hnsw \
    --rfs 8,16,32 --probes 64,128,256,512,1024,2048,4096
```

See [`bench/README.md`](bench/README.md).

## Known limitations

* AVX2 kernels only, no ARM/NEON. All measurements so far are on Windows.
* Python binding only: no server, write-ahead log or sharding.
* After `compact()`, the first mutation drops back to the slower plain Hamming path until the
  next `compact()`, and nothing schedules compaction.
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
bench/           head-to-head harness, data builders, research scripts
results/         raw benchmark output
tests/           invariant suite
docs/            architecture, results, research log, references, history/
```

## Background

AIGVS (Attractor Indexed Graphical Vector Search) started in June 2026 as a Python engine that
routed queries through a recursive Hopfield attractor tree into graph leaves, with a small Rust
core. The goal was O(log N) search through a deep attractor tree without the cost of copying
boundary vectors, and the flat-cell engine and the polysemy work here were measured against that
goal. The design has changed completely since then (there's no graph any more); the name hasn't
caught up. The prototype is in the early git history, and [`docs/history/`](docs/history/) keeps
its notes and the v2 studies, negative results included.

## License

AGPL-3.0. See [LICENSE](LICENSE).
