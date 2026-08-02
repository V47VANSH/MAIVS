# Making it scale: what I tried, what worked, what's next

All measured on HotpotQA (`hotpot_5M.npy`, 5,233,329 vectors, shuffled and normalised on disk)
with 1,000 held-out queries and exact ground truth. I didn't run faiss above 100k here; the
head-to-head numbers in `SCALING_STUDY.md` are enough.

## The problem, stated properly

`bench_scale_full.py` gave alpha ~0.5 for work at fixed recall. The refinement is that alpha
isn't one number, it depends on the recall target:

| target R@10 | alpha (vectors scanned) |
|---|---|
| 0.90 | +0.19, nearly O(1) |
| 0.95 | +0.68 |
| 0.98 | +0.94, nearly O(N) |

Most of the work is already close to independent of N; only the tail scales. Everything I tried
before this was aimed at the average case, which is why none of it moved anything. The tail is the
(query, neighbour) pairs where the neighbour's own cell centroid is a bad proxy for it. In 384
dimensions a vector at q.x = 0.85 can be in the 400th-nearest centroid.

## Tried and dropped

| idea | result | why it fails |
|---|---|---|
| richer per-cell summaries | <= 1.25x | `SUMMARISER_STUDY.md`, priced in the format they'd be stored in |
| spill / replication R=4 | even in bytes, alpha +0.579 -> +0.581 | probes drop 2.6x but cells get 3.9x fatter. The failures are correlated: a vector's R nearest cells are all near it, so it's a boundary case for all of them at once |
| L independent tables (LSH style) | 3x worse at equal scan | splitting the probe budget P/L across tables costs more than the decorrelation gives |

The replication run did show one useful thing: an extra cell membership costs 4 B (a u32 slot
id), not 464 B, because cells hold slot references into one arena. A textbook spill tree pays
~100x more for the same overlap. Overlap is nearly free here, it just doesn't help.

## Tried and kept

### 1. `rerank_factor` 32 -> 128 (recall, almost free)

The ceiling, measured with every cell probed and adaptive off, so routing is out of the picture
and only scan and rerank are being tested:

| N | rf=8 | rf=32 (old default) | rf=64 | rf=128 | rf=256 | rf=512 |
|---|---|---|---|---|---|---|
| 100,000 | 0.9208 | 0.9840 | 0.9920 | 0.9946 | 0.9948 | 0.9950 |
| 1,000,000 | 0.9154 | 0.9758 | 0.9864 | 0.9902 | 0.9916 | 0.9918 |

`rf` is only a query-time setting (`set_rerank_factor`, added for this), so one build covers the
whole sweep and it can be changed on a live index.

* +0.0144 recall at 1M for 4% throughput at exhaustive probing (240 -> 236 qps). The scan
  dominates, so the extra rerank hardly shows.
* The decay with N is the rerank shortlist: -0.0082 per decade at rf=32, -0.0044 at rf=128,
  -0.0032 at rf=512.
* It saturates past 256. Beyond that the 1-bit code is the limit, and more rerank can't help;
  that needs more code bits.

### 2. Cell size 128 -> 32 (throughput, ~2x)

`TARGET_CELL_SIZE = 128` was the best of {128, 256, 512, 1024} at 999k. I never tried 32 or 64.
Vectors scanned at fixed recall, N=2M:

| target | c=16 | c=32 | c=128 (current) | best gain |
|---|---|---|---|---|
| R@10 0.90 | 6,544 | ~9,900 | 19,295 | 2.95x |
| R@10 0.95 | 23,197 | 31,447 | 54,613 | 2.35x |
| R@10 0.98 | 120,149 | | 245,411 | 2.04x |

alpha at 0.90 improves too, +0.536 -> +0.388. The biggest single win in this whole investigation,
and it's one constant.

c=32 works today (2M builds in 55 s). c=16 is better still but runs into the training problem
below.

## The blocker: `hierarchical_kmeans` falls off a cliff

Smaller cells mean more centroids, and centroid training suddenly gets very slow. With the
centroid graph and tree both off and 1,000,000 training points, so this is `hierarchical_kmeans`
alone:

| cells k | train time |
|---|---|
| 31,250 | 4.7 s |
| 62,500 | 7.8 s |
| 90,000 | > 550 s (timed out) |
| 125,000 | > 900 s (timed out) |

Not the tree (k=62,500 takes 8.1 s without it, 15.9 s with it), and not the training sample (a
random gather of 1M rows from the 8 GB memmap takes 6.5 s, a strided one 1.5-3.1 s).

By FLOP count k=90,000 should take ~20 s: the two-level split clusters `k1 = ceil(sqrt(k))` groups
at the top and then ~sqrt(k) sub-clusters per group, ~2e12 FLOP in total. It's more than 25x
that, so it's an implementation problem, not an asymptotic one.

My guess, not verified yet: in `hierarchical_kmeans` the loop over groups is sequential, and
`kmeans_spherical` only parallelises over its own points. At k=90,000 each group has ~3,300
points, so 300 small sub-problems with poor parallel efficiency run one after another. The fix
would be to parallelise over groups (`par_iter` over the group loop) instead of inside them.

This is the most useful next task: it would make c=16 usable (another ~1.4x over c=32), and it's a small
change in `src/tree.rs`.

## Why "navigate without a graph" can't work

The next idea was to get away from global ranking the way HNSW does, by navigating. The reasoning:
the summary approach is floored (above), HNSW avoids that with local relative comparisons, and a
query's true top-10 are all neighbours of q and so, I assumed, of each other. A neighbour you
can't reach from q should then be easy to reach from a neighbour you've already found: probe the
cells next to the current best answers. No edges needed. It's also the modern Hopfield update
(`xi <- X softmax(beta X^T xi)`, Ramsauer et al. 2020), which is sparse for large beta and so
computable from candidates already scanned.

I implemented it two ways (blend `q + gamma*mean(found)`, and each found neighbour as its own
probe point) and both lost at matched work: at R@10 0.95 / N=500k, 2 rounds needed 44,312 vectors
against 24,726 for one.

The reason is worth stating by itself. At 500k over 300 queries with exact ground truth:

| | mean cosine |
|---|---|
| q -> its own top-10 | 0.567 |
| top-10 -> each other | 0.460 |
| 1st neighbour -> 10th neighbour | 0.455 |
| random pair | 0.038 |

A query's nearest neighbours are further from each other than from the query. In 384 dimensions
there's room for ten nearly orthogonal directions all at cosine ~0.57 from q, so the top-k is a
shell around the query, not a cluster. So:

* A found neighbour y is a worse starting point for a missing neighbour x than q was (0.46 vs
  0.57). Hopping between answers moves away from the remaining answers. The premise was wrong.
* It explains the flat floor: the shell points in ~10 directions, so it lands in several cells
  at every N. That's why the floor is 3.5-5.2 cells and doesn't grow.
* HNSW doesn't work by hopping between answers either. It comes in from far away towards q and
  collects the shell with its final ef-sized beam. The navigating happens on the way in.

This one rules out a whole class of fixes, not just one implementation.

## What the literature says

**Gottesbueren, Dhulipala, Jayaram, Lacki, "Unleashing Graph Partitioning for Large-Scale Nearest
Neighbor Search", PVLDB 18(6), 2025** (arXiv 2403.01797, code at
`github.com/larsgottesbueren/gp-ann`) is the closest published work to this, and reports the same
failure:

> "Large shards cannot be as easily represented with a single center... routing with a single
> center resulted in only 28% 10-recall in the top-ranked shard on the MS Turing dataset, as
> opposed to 81% with our best routing index."

Their design:

* **Partition** by balanced graph partitioning of an approximate k-NN graph, minimising cut k-NN
  edges, since the cut is the recall loss. Even a poor k-NN graph (graph recall ~0.3) still gives
  > 81% of the top-10 in one shard, so the input can be cheap.
* **Route** by representing each shard with several actual data points `R_i`, finding the query's
  approximate nearest points `Q` in the union of all `R_i` with an ANN index, and ranking shards
  by the closest point of `Q` in each. Two versions: `KRT` (hierarchical k-means tree over R,
  optionally HNSW-accelerated) and `HRT` (SortingLSH). Plus frequency-based routing (rank shards
  by how often they appear in `Q`), which is best for the first shard and worse after, so they
  combine them.
* Theorem 1: routes to a shard containing an approximate nearest neighbour with high
  probability.
* Result: 2.14x QPS at 90% recall@10 at billion scale against the best competitor.

**Why it doesn't port directly.** Their shards are ~10^7 points and not convex (graph
partitioning doesn't give convex regions), so a single mean is hopeless there. Mine are 128-point
convex k-means cells, where the mean is already a decent summary, which is why `hull`/`subc` only
bought <= 1.25%. Their result is about a different regime, not a fix I missed.

**What does carry over** is the principle: their routing hierarchy is built from real data points,
not summaries. HNSW's layers are the same thing, recursive samples, each level seeded by the one
above. That's the one structural property my centroid tree doesn't have, and centroid error
compounds at every level.

## Tried: a ladder of samples (`bench_ladder.py`)

Each level is a 1/f sample of the corpus (f=8). The top rung is scanned outright, then each rung
down only probes the `b` cells nearest each of `B` beam points carried from above. Work is
`log_f(N) * B * b * c`, O(log N) by construction, so the only question is whether B has to grow
with N.

At 100k and 500k, 3 rungs:

| beam B | scanned at 500k | R@10 |
|---|---|---|
| 4 | 11,904 | 0.093 |
| 16 | 24,192 | 0.463 |
| 64 | 73,344 | 0.825 |

alpha at R@10 0.80 is +0.184, properly flat, but the absolute cost is bad (50k-68k vectors for
0.80, where flat routing gets 0.90 in ~8-11k), and it never reaches 0.90 at any beam I tried.

Same geometry as before. Expanding a beam point to its `b` nearest cells explores one direction.
The top-10 sit in ~10 directions (pairwise cos 0.46 vs 0.57 to q), so nearest-b expansion can't
cover the shell however deep it goes. Only a bigger B covers it, and then B is doing the job of a
global ranking.

HNSW covers the shell because its edge selection diversifies: an edge x->y is kept only if y is
closer to x than to any neighbour of x already chosen (relative neighbourhood graph pruning).
That's an explicit anti-redundancy rule, and it's what nearest-b expansion lacks. So the next
thing to test is diversified cell expansion: from a beam point, pick cells that are near it and far
from each other, instead of the b nearest. Cheap over centroids and no edges needed.

## Tried and refuted: a diversified cell graph (`bench_cellgraph.py`)

Idea: put HNSW's anti-redundancy rule into a graph over cells (M = N/128 nodes, ~5 MB at 5.2M,
arena and O(1) delete untouched) and walk it greedily, so cell selection becomes local and
relative instead of a global ranking.

Vectors scanned at fixed recall, same partition for all three:

| R@10 >= 0.95 | alpha | 100k | 500k | 2M |
|---|---|---|---|---|
| flat (current) | +0.426 | 8,039 | 16,421 | 28,752 |
| nearest-b walk | +0.457 | 8,194 | 18,720 | 32,104 |
| diversified walk | +0.515 | 7,891 | 22,284 | 36,549 |

Both walks lose, and the diversified one is worst. Two conclusions, and the second matters more:

1. The walk starts from the top-4 cells by centroid score, so it has already paid for a global
   ranking and then explores locally, which tells you strictly less than just continuing down
   that ranking. Diversified edges make it worse because they point partly away from q on
   purpose. HNSW needs them to navigate in from a random entry point; I don't.
2. Flat ranking and the walk use the same centroid geometry. A walk can never reach a cell that
   centroid ranking wouldn't already have ranked highly. If the true neighbours' cells score
   badly, no router built on those centroids fixes it.

### So the problem is the partition, not the router

Every routing idea so far (richer summaries, spill, multiple tables, attractor hops, the sample
ladder, the cell graph) reads the same k-means geometry a different way, and all of them are
limited by it. k-means minimises distance to the centroid. What routing needs minimised is how
many cells a query's true neighbours are spread across. Those are different objectives and I've
only ever optimised the first.

That's the part of Gottesbueren et al. I underweighted above: they partition by balanced graph
partitioning of the approximate k-NN graph, minimising cut k-NN edges, and the cut is the recall
loss by definition. They get > 81% of a query's top-10 into a single shard, and a poor k-NN graph
(recall ~0.3) is enough, so the input is cheap.

My floor is 3.5-5.2 cells for 10 neighbours (~2 per cell). If a k-NN-graph partition put 8 of the
10 in one cell, routing would be looking for 1-2 cells instead of 5. That changes the problem
rather than the solver, and it's the first idea here that goes after the objective instead of the
estimator.

Next experiment: build the partition by k-NN-graph partitioning instead of k-means, keep the
router as it is, and measure the floor (cells holding the true top-10) and alpha. The floor is
the leading indicator and is cheap to compute offline. If it doesn't drop below ~2 cells, this is
dead too, and the conclusion is that the IVF family can't reach graph-index scaling on this data.

## Tried and refuted: k-NN-graph partitioning (`bench_knnpart.py`)

The last open objection was that the partition objective is wrong. I built the approximate k-NN
graph with my own index, then moved each point into the cell holding most of its k-NN, under a
balance cap. That's the local-search core of graph partitioning, seeded from k-means, directly
reducing cut k-NN edges.

| N=100k | floor (cells) | R@10 at np=32 | R@10 at np=128 |
|---|---|---|---|
| k-means | 4.91 | 0.9107 | 0.9767 |
| k-NN-graph refined | 4.84 (-1.4%) | 0.9017 | 0.9717 |

The local moves converged after ~15k moves in 100k points: k-means already puts almost every point
with most of its k-NN. There's almost nothing for a different objective to change.

### The floor is geometric, and that closes the investigation

A query's top-10 sit at pairwise cosine 0.46 while each is at 0.57 from q. A 128-vector cell
doesn't cover that much of the space, so those 10 points can't share a cell whatever the
partitioner optimises. The floor of 3.5-5.2 cells comes from cell size against the spread of the
answer set. Gottesbueren et al. get > 81% of a top-10 into one shard because their shards hold
~10^7 points and cover the whole neighbourhood; at 128 points per cell that can't happen.

That explains every negative in this document with one cause:

* the floor is flat in N (geometry, not scaling), confirmed at every scale;
* no summary of a cell helps, because the answers aren't in one cell;
* no router over those centroids helps, because it isn't a ranking failure;
* no partition objective helps, because k-means is already at a local optimum of the graph
  objective too;
* bigger cells would lower the floor but multiply the scan, which is a real trade-off, not a
  defect.

So the IVF family's scaling on this data is limited by the answer set being spread over many
small cells while the number of competing cells grows with N. Matching a graph index's exponent
means leaving the family, not writing another router. The constant-factor wins I already have
(`centroid_sq8` 1.22x, `rerank_factor` 32 -> 128 for +0.014 recall, cell size 128 -> 32 for ~2x)
are real and independent of all this. They move the crossover point but don't remove it.

## Where this leaves O(log N)

First, the goal needs restating. No measured system does O(log N) at fixed recall. My own
head-to-head put faiss-HNSW at alpha(qps) = -0.21, not 0: its O(log N) is hop count at a fixed
efSearch, and holding recall makes efSearch grow with N. So the target isn't alpha = 0, it's
closing -0.78 -> -0.21, and anything claiming much better than ~0.2 should be measured the same
way before I believe it.

Nothing so far moves the exponent. The wins are constant factors (~2x from cell size, 1.22x from
`centroid_sq8`), and against HNSW's -0.21 vs my -0.78 they move the crossover but don't remove it.

The exponent is a tail problem, so what's left are ideas that change what happens to the ~5% of
badly placed vectors rather than the average case:

1. **Two tiers.** Vectors far from their centroid can be found at build time; `r_bound`/`r_max`
   already compute exactly that. Route the bulk through the IVF and the badly placed minority
   through a separate structure. ~5% of N is small enough for a graph, and it doesn't fatten the
   main cells the way replication does.
2. **A local stopping rule instead of a global top-P.** The structural difference from HNSW is
   that I rank cells globally (gets harder as M grows) while it makes local, relative comparisons
   (doesn't). This is where a Hopfield formulation becomes interesting again: modern Hopfield
   networks have a separation criterion for when a pattern is retrieved unambiguously, which is a
   query-local answer to "is this cell certainly the right one", something global ranking can't
   give.
3. **More code bits.** Separately from routing, the recall ceiling saturates at rf >= 256 because
   the 1-bit code can't separate more candidates. That caps max recall at every scale.
