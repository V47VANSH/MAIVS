# Prototype notes: the recursive Hopfield tree

Written in June 2026 during the first prototype (Python orchestration, a recursive
Hopfield/attractor tree, HNSW graphs in the leaves), before the flat-cell engine. Kept because
it states the goal the later work was measured against: O(log N) routing through a deep
attractor tree without the duplication blow-up. The numbers are from the prototype at 100k
vectors and its recall is the MS MARCO qrel hit rate, so they aren't comparable with the
engine's ANN recall.

What I've learned so far from the MS MARCO 100k runs about build time, memory, latency and
recall: what worked, what didn't, and where I want to take it.

My notes after the last run:

> I'm not sure these results are what I want. There is a triangular trade-off between memory,
> latency and recall, and we can optimise one or two at the cost of the rest. To improve all
> three we need a better splitting algorithm. My issue is "Total chunks tracked in tree: 146732",
> that's 46% copying, and going down to 1.02 leaves/query dropped recall from 98 to 88, which is
> just the maths. What I expected from this run was less copying, a deeper tree, still fewer
> leaves per query, and high recall. That was the whole point of the ideas from those papers.

## What failed: deep trees

I expected a deep recursive Hopfield tree (4+ levels) to give O(log N) queries. With the current
splitting it doesn't.

* **Boundaries.** At every split some vectors sit between two clusters. To avoid losing them I
  duplicate them into both branches.
* **Duplication compounds.** It happens again at every level. With 4 levels on 100k vectors the
  tree ended up holding 462,460 vectors.
* **Latency.** To keep recall the router had to visit lots of small leaves, 166 per query. That
  many tiny graphs is mostly FFI and cache overhead: 4.16 ms against 0.71 ms for the flat tree.
* **Strict routing.** Cutting `target_confidence` from 0.95 to 0.80 to bring latency down made
  the router miss the right leaves, and recall fell from 98% to 31.5%.

## What worked: a flat tree

At 100k a flat tree is much better.

* Two levels only: root, macro trunks, leaves.
* Duplication happens once, so it stays at ~46%, which is acceptable for now.
* Routing at the top is one step, and the leaves are large Rust HNSW graphs.
* Result: 8 s build, 0.71 ms latency, 98.0% recall@10.

## Where I want to go

The flat tree is fine at 100k, but at 1M or 1B vectors it won't hold up, and that's the scale I
actually care about. That means deep trees, which means solving the boundary problem.

1. **Sharper basins.** Spherical k-means leaves fuzzy boundaries. I want a clustering where each
   vector clearly belongs to one cluster.
2. **No duplication.** With clean boundaries there's nothing to copy, and memory goes back to the
   minimum.
3. **Learned routing.** Instead of distance to centroids, a small model at each node (a logistic
   classifier or a small MLP) that predicts the right child, including for vectors near edges.
4. **Choosing depth.** Some way to work out the right depth for a given N (1M vs 10M vs 1B) so
   the tree doesn't over-nest.
5. **Cheap levels.** Each level has to be close to O(1), so going deeper doesn't cost much.

If the basins are clean, a query can drop straight to one small leaf without checking the
neighbours, and that's the O(log N) version: no copying, high recall, very low latency.
