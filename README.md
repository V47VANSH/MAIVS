# AIGVS: Attractor Indexed Graphical Vector Search

A retrieval index for RAG that routes a query with a Hopfield-style attractor step instead of a
global graph walk, then searches a small local graph inside the community it lands in.

Work in progress. The Python pipeline works end to end; the Rust core (`src/lib.rs`) handles the
leaf graphs.

## Why

The two usual options for RAG retrieval each have a problem:

* A vector database (HNSW) is fast, but it returns isolated chunks. Nothing about how the chunks
  relate to each other survives.
* A knowledge graph (GraphRAG) keeps the relations, but global queries need a lot of LLM work and
  adding documents can mean re-clustering the whole graph.

The idea here is to split the problem in two:

* **Routing.** A continuous Hopfield network over the community centroids picks which part of the
  index a query belongs to. That's one matrix product per level, not a walk over the whole graph.
* **Local search.** Inside a community, a small navigable small world (NSW) graph finds the
  nearest chunks, and their graph neighbours come back with them as extra context.

## Structure

* `AIGVSPipeline` (`src/aigvs/pipeline.py`): the top level. Owns storage (in-memory or JSON for
  now) and the thread pool.
* `DisjointFamily`: a router node. No text, just a matrix of child centroids that it routes over.
* `JointFamily`: a leaf. Holds the actual chunks, wired into an NSW adjacency list.

## Routing

With query `q` and the matrix of child centroids `X`, the Hopfield update is

```text
xi(t+1) = normalize( X^T softmax(beta * X xi(t)) )
```

I only evaluate it once (t = 1) and use the softmax activations directly. If a child is above the
threshold (`activation_threshold`, 0.25 by default) the query goes there. If nothing is, the query is treated as out of
distribution and a new branch gets created for it.

### Temperature

One fixed beta doesn't work for every level. Deeper nodes hold very similar children and need a
sharper softmax, so beta is set from the variance of the pairwise cosines between the children:

```text
beta(sigma^2) = beta_min + k / (sigma^2 + eps)
```

Gentle near the root, sharp near the leaves.

### Splitting a leaf

When a leaf is over capacity it's split in two. Instead of k-means I start two seeds at extreme
points of the leaf and run the same update on the leaf's own vectors until they settle:

```text
xi_i(t+1) = normalize( X_local^T softmax(beta * X_local xi_i(t)) )
```

Each seed falls into a different minimum, and the leaf's vectors are divided between them.

## Read, write, delete

**Query:** embed, route down through the `DisjointFamily` levels (if two children are both above
threshold, both are searched in parallel), greedy NSW search inside the leaf, then return the hits
plus their 1-hop neighbours.

**Insert:** route the vector to a leaf, add it to the leaf's graph, update the leaf centroid as a
running mean

```text
mu_new = (mu_old * n + x) / (n + 1)
```

and write the new centroid into the parent's matrix. Nothing else gets re-indexed.

**Delete:** add the id to the leaf's `deleted_indices` and leave the edges in place, subtract the
vector from the centroid, and rebuild the leaf's adjacency list once more than 20% of it is
deleted.

## Complexity (expected)

With N chunks, M communities and branching factor b:

* HNSW query: O(log N) hops over the whole graph
* AIGVS query: O(log_b M) routing steps, then O(log(N/M)) hops inside one leaf
* GraphRAG insert: re-clustering, O(V log V)
* AIGVS insert: route plus a local insert, no global work

If routing really is that cheap, query cost should scale like a normal vector index while keeping
the local graph structure around each hit. That's the part I want to measure next.

## Running it

```bash
pip install maturin
maturin develop --release
python main.py
python benchmark.py
```
