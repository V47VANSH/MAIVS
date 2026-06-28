# What made the 100k benchmark fast

Notes on the changes that got the prototype to roughly faiss-HNSW recall (98%) and latency
(under 1 ms) on MS MARCO 100k, with a build about 10x faster (8 s vs 84 s). All numbers are from
`benchmark_msmarco.py` on my laptop.

## 1. Flat trunks instead of one big graph

faiss builds one HNSW graph, and every insert has to search the whole thing for neighbours.

Here a few rounds of spherical k-means split the data into a handful of macro buckets first (4
for 100k), and each bucket gets its own graph. So instead of one graph over 100,000 points it's
four graphs over ~25,000, which is where most of the build time goes away.

## 2. Boundary duplication

Points near the edge of a bucket are the ones the router sends to the wrong place. During ingest,
if a vector's similarity to its second-best bucket is within 0.08 of its best, it's added to both.

* Cost: memory goes up by about 46% (146k stored vectors for 100k inputs).
* Benefit: the router can pick just the top bucket and the right answer is almost always in it.

## 3. int8 vectors

384-d f32 vectors are 1,536 bytes each. The Rust side stores them as int8 (384 bytes), so 4x
more of them fit in cache, and each bucket's vectors sit in one contiguous block, so the scan
reads memory in order.

## 4. SIMD dot products

With int8 vectors and `target-cpu=native`, the dot product compiles to AVX2 instructions that
handle many dimensions per instruction instead of one multiply-add at a time.

## 5. Parallel build with rayon

Python can't build several graphs at once because of the GIL. Since the buckets are independent,
all of them are passed to Rust together and rayon builds one per thread. They share nothing, so
there are no locks.

## 6. Softmax routing with a confidence cut-off

The router turns centroid similarities into probabilities with a softmax, and two settings
control how much gets searched:

* `beta`: temperature, how sharply good branches are separated from bad ones.
* `target_confidence`: branches are taken in order of probability until their total reaches
  this (e.g. 0.95). If the top branch alone is at 0.96, only that one is searched.

That's what keeps latency at 0.71 ms: most queries only touch one bucket.
