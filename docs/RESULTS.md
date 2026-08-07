# A partitioned vector index at 5.2M: results so far

Results report. Every number comes from one machine against exact brute-force ground
truth. Most of the results are negative, and they're the more useful half.

## Summary

I evaluate a partitioned (IVF-style) index (fixed-size cells over one flat arena, tree routing
over the cell centroids, SIMD scan of 1-bit codes, exact rerank) against exhaustive ground truth
on two corpora (MS MARCO and HotpotQA embeddings, d=384), from 10k to 5.23M vectors.

**What works.** A higher recall ceiling than faiss-HNSW (R@10 0.9999 vs 0.9993 at 1M; HNSW doesn't
reach 0.9999 at any efSearch I tried), with 3.1x less memory, a 24x faster build, O(1) delete
(~1.5-2.0 us, flat in N) and filters that make selective queries cheaper. One free throughput win
is in (stage-2 re-order precision, 1.12-1.28x at |dR@10| <= 1e-4, with less memory), and two more
are measured (rerank depth, cell size).

**What doesn't.** At fixed recall, throughput falls as N^-0.78 against N^-0.21 for faiss-HNSW on
the same data. 3.6x faster at 50k, 2.15x slower at 2M, crossing between 250k and 1M. I designed and
measured ten fixes and all of them failed, and there's one geometric fact that explains them
together: a query's true top-10 are further from each other (mean cos 0.460) than from the query
(0.567). They form a shell in ~10 directions, not a cluster.

I also found six problems in my own measurement code, three of which had already produced
numbers that looked final and were off by 1.4-2.6x or had the wrong sign.

## 1. Setup

| | |
|---|---|
| corpora | MS MARCO (999k) and HotpotQA (5,233,329), d=384, L2-normalised |
| ground truth | exact top-100 by brute force, streamed on the GPU, cached |
| queries | 1,000 held-out vectors, the same at every scale |
| metric | recall@10 and MRR against exhaustive search; throughput best of 7 batch runs |
| hardware | 16-core CPU, 15.3 GB RAM, RTX 3050 (ground truth only) |

The HotpotQA corpus is shuffled on disk before slicing, so the first N rows are a uniform sample
at every scale. The source is in document order: adjacent rows have cosine 0.0885 in source order
and 0.0407 shuffled (random baseline 0.038). Without the shuffle, small scales would be a narrow
set of topics and the recall-vs-N curve would measure the ordering.

## Part I: what works

### 1.1 Recall ceiling (1M MS MARCO)

| system | max R@10 | qps there | build | memory | delete |
|---|---|---|---|---|---|
| this index, quality | 0.9999 | 333 | 66.8 s | 1,664 MB | 1.58 us |
| this index, memory | 0.9939 | 337 | 74.7 s | 562 MB | 1.77 us |
| faiss-HNSW | 0.9993 | 1,542 | 298.3 s | 1,742 MB | none |
| faiss-IVF | 0.9935 | 221 | 9.3 s | 1,520 MB | O(N) |
| faiss-IVF-SQ | 0.9935 | 56 | 7.9 s | 489 MB | O(N) |

faiss-HNSW doesn't reach 0.9999 at any efSearch up to 512. The quality profile passes HNSW's
ceiling at np=2048 (0.9994) and goes on to 0.9999.

### 1.2 The free one: stage-2 re-order precision

The router re-ordered `nprobe*mult` centroids exactly in f32, 1,536 B each: 1,536 KB per query at
1M / np=512, a quarter of the byte budget, spent just to order cells whose order already has ~0.05
of structured error.

Probe counts are the same at f32, f16, sq8, sq6 and sq4 on both corpora, and vectors scanned
differ by 3 in 66,555 (MS MARCO) and 4 in 200,497 (HotpotQA). It's in as `set_centroid_sq8`, on by
default. Measured on a live index by flipping the flag on one build:

| nprobe | R@10 f32 | R@10 sq8 | delta | speedup |
|---|---|---|---|---|
| 64 | 0.9301 | 0.9301 | 0 | 1.122x |
| 256 | 0.9760 | 0.9761 | +0.0001 | 1.277x |
| 1024 | 0.9877 | 0.9876 | -0.0001 | 1.125x |

1.12-1.28x throughput, |dR@10| <= 1e-4, and less memory (centroids 12 MB -> 3 MB at 1M). The same
sweep settles `shortlist_mult`: mult=1 needs 768 probes against 512 for mult=2, so the shortlist
matters.

### 1.3 Two more, measured but not the default yet

**Rerank depth.** With exhaustive probing every vector is scored by its 1-bit code and only the
top `k*rf` are rescored exactly, so a neighbour is lost when its Hamming rank is past `k*rf`.
Being in the top 320 of 10k is easy, of 5.2M it isn't, so a fixed `rerank_factor` has to lose
recall as N grows, and it does:

| N | rf=32 (old default) | rf=128 | rf=512 |
|---|---|---|---|
| 100,000 | 0.9840 | 0.9946 | 0.9950 |
| 1,000,000 | 0.9758 | 0.9902 | 0.9918 |

+0.0144 recall at 1M for 4% throughput at exhaustive probing. It saturates past 256, where the
1-bit code becomes the limit.

**Cell size.** `TARGET_CELL_SIZE = 128` was the best of {128, 256, 512, 1024}; I never tried 32 or
64. Vectors scanned at fixed recall, N=2M:

| target | c=16 | c=32 | c=128 (current) |
|---|---|---|---|
| R@10 0.90 | 6,544 | ~9,900 | 19,295 (2.95x) |
| R@10 0.95 | 23,197 | 31,447 | 54,613 (2.35x) |

### 1.4 Things HNSW doesn't have

* **O(1) delete**: 1.5-2.0 us, flat from 10k to 5.2M; recall moves -0.0046 over 100% turnover.
  HNSW has no delete.
* **Build**: 12.2 s at 1M (one `add_batch`), 94 s at 5.2M (~56k vectors/s), against 298 s for
  HNSW at 1M, 24x.
* **Filters**: pushdown makes a 1%-selective query 2x faster. Graph indexes get worse because
  selective filters break up the graph.
* **Predictable memory**: 464 B/vector (memory) or 1,616 B/vector (quality), linear.

## Part II: what doesn't work

### 2.1 Scaling

Batch qps at R@10 >= 0.95, same corpus, queries and ground truth, HNSW built up to 2M (802 s,
4.5 GB):

| N | this index | faiss-HNSW | ratio |
|---|---|---|---|
| 10,000 | 146,720 | 157,699 | 1.07x |
| 50,000 | 72,937 | 20,356 | 0.28x (3.6x faster) |
| 250,000 | 23,188 | 11,914 | 0.51x |
| 1,000,000 | 7,041 | 10,054 | 1.43x (HNSW faster) |
| 2,000,000 | 4,225 | 9,083 | 2.15x (HNSW faster) |
| alpha(qps), without 10k | -0.781 | -0.214 | |

Extrapolated to 1B at R@10 0.95: ~33 qps against ~2,400, about 70x. "3-5x behind faiss" was a
snapshot at 1M.

alpha also depends on the recall target: +0.19 at R@10 0.90 and +0.94 at 0.98. Most queries are
already nearly independent of N; only the tail scales.

On the target itself: nothing gets O(log N) at fixed recall. HNSW's O(log N) is hop count at a
fixed efSearch, and holding recall pushes efSearch up with N, hence -0.21 rather than 0. The real
goal is closing -0.78 -> -0.21.

### 2.2 Ten ideas that failed

| # | idea | result |
|---|---|---|
| 1 | mean + angular radius (cone) | loses to the plain mean; no gamma beats 0 |
| 2 | element-wise max / box bound | 8x worse; misses the target on HotpotQA |
| 3 | low-rank residual (rank 4) | same as rank 0; residuals are close to isotropic |
| 4 | sub-centroids / hull points | <= 1.25x once priced in their storage format |
| 5 | spill / replication (R=4) | even in bytes; alpha +0.579 -> +0.581 |
| 6 | L independent tables (LSH) | 3x worse at equal scan |
| 7 | attractor / Hopfield query refinement | worse, in both the blend and neighbour-probe versions |
| 8 | sample ladder (HNSW-like layers) | alpha +0.184 (flat) but never above R@10 0.90 |
| 9 | cell graph, greedy walk | alpha +0.515 vs +0.426 for flat; fixed entry +0.574 |
| 10 | precomputed query-side lists | coverage falls with N; breaks on real queries |

### 2.3 The geometry behind them

At 500k over 300 queries against exact ground truth:

| | mean cosine |
|---|---|
| q -> its own top-10 | 0.567 |
| top-10 -> each other | 0.460 |
| 1st -> 10th neighbour | 0.455 |
| random pair | 0.038 |

A query's nearest neighbours are further from each other than from the query. In 384 dimensions
there's room for ~10 nearly orthogonal directions all at cos ~0.57 from q, so the answer set is a
shell, not a cluster. That explains:

* why the floor stays at 3.5-5.2 cells and doesn't shrink with N;
* why hopping from a found neighbour fails: it moves away from the others;
* why nearest-b expansion can't cover the answer set at any depth;
* why replication fails: a vector's R nearest cells are all near it, so its boundary failures
  are correlated, and more correlated chances don't thin the tail.

### 2.4 Why the whole family is limited

Cells are chosen by ranking fixed-precision summaries globally. The number of cells that outrank
the right one is `M * P(error > gap)`. For that to stay constant as `M = N/c` grows, summary error
would have to shrink like 1/M, and a fixed-size summary can't do that. Every idea above is inside
this family, and the cell-graph result shows it directly: a walk can never reach a cell that
centroid ranking wouldn't already rank highly, since both read the same centroids.

### 2.5 A build-time cliff

Graph and tree off, 1M training points:

| cells | train |
|---|---|
| 31,250 | 4.7 s |
| 62,500 | 7.8 s |
| 90,000 | > 550 s |

A FLOP count says ~20 s at 90,000, so it's a ~25x implementation problem, not asymptotics. Likely
cause: in `hierarchical_kmeans` the group loop ran sequentially while `kmeans_spherical` only
parallelises within each ~3,300-point group. It blocks the cell-size win above ~62,500 cells, i.e.
c=32 above ~2M and c=16 above ~1M. A fix is in (Part VI) but not verified yet.

## Part III: problems in my own measurement code

Listed because three of them had already produced wrong numbers that looked final.

| problem | effect |
|---|---|
| `interp_at` returned the most expensive config when every grid point already cleared the target | HNSW's scaling exponents came out negative and meaningless; left alone it would have shown HNSW scaling worse than me, the opposite of the truth |
| summaries charged 72 B while scored in f32 | sub-group win reported as 1.70x; it's 1.25x |
| counted probes instead of vectors scanned | spill looked like "2.6x fewer probes"; in bytes it's even, because cells get 3.9x fatter |
| probe grid capped at 8,192 | "max recall" at >= 2M measured while probing only 20% of cells |
| benchmark never called `set_tree_route` | the routing path I use was never the one benchmarked; build fell back to the O(N*M) assignment |
| `add_batch` runs an O(N*d) `refresh_derived` per call | chunked ingest costs `chunks * N * d`: 74.7 s in 5 chunks vs 12.2 s in one |

The rules I took from these: price a summary in the format it'll be stored in, and compare bytes
moved, never probes.

## Part IV: where it stands

The results I'm confident in: the highest R@10 of anything tested (0.9999 vs 0.9993 for
faiss-HNSW); 3.1x less memory; 24x faster build; O(1) delete flat to 5.2M; filters that get faster
as they get more selective; stage-2 precision doesn't matter down to 4 bits (two corpora); cell
size 128 isn't optimal (2.35-2.95x available); and the shell geometry of the answer set.

What it isn't: the fastest. 2.15x behind HNSW at 2M and the gap is growing. Not O(log N) either:
alpha is +0.19 to +0.94 depending on the target, and a logarithmic tree doesn't make the query
logarithmic. And nothing here says anything about real (out-of-distribution) queries.

## Part V: threats to validity

1. **Only in-distribution queries.** Every query is a held-out corpus vector; real queries are
   questions. That doesn't affect comparisons (both sides saw the same queries), but it rules out
   trusting any design that relies on queries resembling the data. The precomputed index is the
   pure case of that, and I built it before applying this caveat to it. Getting question
   embeddings matters more right now than any algorithm.
2. Two corpora, both d=384 from the same embedder family. Nothing tested at other dimensions.
3. The alphas are power-law fits over 3-5 scales. A logarithm isn't a power law, so the thing to
   look for is a falling local alpha, not one number.
4. alpha from the pure-routing harness (exact in-cell scan, 100-300 queries) is consistently lower
   than from the end-to-end harness. Compare within a table, not across documents.
5. faiss-HNSW measured only up to 2M; the 1B projection is 500x past the last point.
6. Run-to-run spread can reach 35% on this harness; anything inside the spread is a tie.

## Part VI: status

### Code changes done, verification left

| # | change | state |
|---|---|---|
| 1 | `centroid_sq8`, SQ8 stage-2 re-order, on by default | done; 30/30 tests passed when it went in |
| 2 | `set_rerank_factor`, rerank depth at query time | done, builds |
| 3 | `hierarchical_kmeans` parallel over groups (`src/tree.rs`) | done, builds, not verified |
| 4 | `quality` profile rerank 32 -> 128 | done |
| 5 | harness fixes (`--tree`, probe grid, strided subsample, `interp_at`) | done |

Change 3 is the important one. The old loop carried a running `written` counter, which forced the
sub-clusterings to run one after another, while `kmeans_spherical` only parallelises over its own
points (~3,300 per group at k=90,000, too few for 16 cores). It's now two phases: the budget
arithmetic sequentially, then the clusterings in parallel into separate output slices. The output
is the same by construction (same arithmetic, same per-group seed). Peak memory goes up by
`threads * group_size * d * 4`, about 69 MB at 1M / 125k cells.

`TARGET_CELL_SIZE` stays at 128 for now. The measured optimum is 32 (2.35-2.95x fewer vectors
scanned), but that's vectors scanned, not end-to-end throughput, and c=32 quadruples the stage-1
centroid scan (M = N/32, 4 MB/query of Hamming at 2M if routed flat). It needs an end-to-end
throughput run first, plus the cliff fix confirmed. Changing it based on a scan-count proxy would
be exactly the kind of mistake in Part III.

### Blocked on the machine

None of the above is verified yet, and the reason is the machine, not the code. It has ~1.7 GB
free: the test suite dies in faiss with `std::bad_alloc`, numpy can't allocate 146 MiB, and a Rust
allocation of 2.7 MB failed. Nothing measured in that state is worth trusting.

On an idle machine, in order:

1. `KMP_DUPLICATE_LIB_OK=TRUE python tests/test_index.py`, all 30 checks, before anything else.
2. Check the cliff is gone: train on 1M points for k in {31250, 62500, 90000, 125000}. Before the
   fix it was 4.7 s / 7.8 s / > 550 s / > 900 s; 90k and 125k should now take tens of seconds.
3. Check change 3 didn't change the partitions: build once at 1M and compare recall with Part I.
4. Only then look at `TARGET_CELL_SIZE`, with an end-to-end throughput run.

## Part VII: open problems

1. **The k-means cliff**: 25x off its own FLOP count; it's what blocks smaller cells.
2. **Closing -0.78 -> -0.21.** Everything I tried ranks cells globally. The one direction outside
   that I haven't tried is a router learned from supervision (MoE-style gating trained on the
   true top-k) instead of read off the geometry, and that needs question embeddings to be worth
   evaluating.
3. **A ceiling above ~0.995**: limited by the 1-bit code, not the rerank, so it needs multi-bit
   codes.
4. **Tiered storage**: 464 B/vector x 1B = 464 GB. Neither this nor HNSW (1.8 TB) runs 1B on one
   machine; billion scale is a sharding problem for both.
