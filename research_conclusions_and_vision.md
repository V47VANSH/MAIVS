# Research notes: deep vs flat trees

The goal is to split the embedding space into stable attractor basins and route to them in
O(log N). The MS MARCO 100k runs make the trade-off clear: memory, latency and recall, and so far
I can only get two of the three.

## Flat tree (macro trunks only)

* 4 trunks of ~25,000 vectors each
* boundary duplication: 4.8%
* recall@10: 86.5%
* latency: 0.79 ms
* build: 9.68 s

Fast to build and fast to query, because there's only one routing level. But each leaf graph is
big and has to do most of the work, and recall stays around 86-88% unless I duplicate a lot more
across the boundaries.

## Deep tree (recursive top-k splitting)

* 4 trunks split recursively into 60 leaves
* boundary duplication: 4.8% (gated at insert)
* recall@10: 97.5%
* latency: 1.75 ms
* build: 80.25 s

With a deeper tree and a wide search (`max_beam=8`, `target_confidence=0.999`) recall gets to
97.5%. The cost is build time (the recursive Hopfield splitting is slow) and about 1 ms more
latency, since the router evaluates several levels and boundaries.

## What this says

Every extra level adds a chance of routing to the wrong branch at a boundary. To keep recall I
have to either:

1. duplicate more boundary vectors so the basins overlap (memory), or
2. keep the tree flat or search a wider beam (latency).

The deep run is the best so far: 97.5% at 1.75 ms with only 4.8% duplication, by gating which
branches a vector is inserted into (top-k) and using the adaptive softmax at query time.

## Next

* Grow depth only where it's needed: split a leaf when its spread passes a threshold (like growing
  neural gas), instead of fixing the depth up front.
* Sharper basins, so vectors fall clearly into one branch and a deep tree can be searched without
  a wide beam.
