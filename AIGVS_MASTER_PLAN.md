# AIGVS plan

Started 2026-07-26 after going through the literature properly for the first time. Updated
2026-07-27 with the first real measurements (section 1a) and a section on positioning (11).

## Summary

Most of what I had planned has already been published in the last three years:

* **Quake** (OSDI 2025): in-memory partitioned index with online split/merge driven by a latency
  cost model, per-query adaptive nprobe from a recall model (APS), O(1) delete with compaction,
  NUMA-aware scan. Beats HNSW, DiskANN, SVS, ScaNN, faiss-IVF, DeDrift and SPFresh on dynamic
  workloads.
* **SPFresh** (SOSP 2023): incremental in-place updates for billion-scale partitioned indexes
  (LIRE, local rebalancing).
* **SOAR** (NeurIPS 2023) and **RAIRS** (2026): redundant assignment done well.
* **RaBitQ** (SIGMOD 2024) and **Extended RaBitQ** (SIGMOD 2025): quantization with error
  bounds.
* **DARTH** (SIGMOD 2026): per-query recall targets through early termination.
* The **NeurIPS'23 Big-ANN streaming track** is already the churn benchmark I was going to
  invent.

Better to find this out now than in six months. Two consequences:

1. The "new Hopfield architecture" framing goes. The router is a softmax over centroid cosines,
   which is IVF probe selection.
2. There's still a narrow gap that I think is real (section 3). It's smaller than a new index
   family and bigger than polishing.

## 1. Where the code is

Things I found reading through the current code:

| Finding | Where | Effect |
|---|---|---|
| SQ8 is `x * 127.0` on unit vectors, so only 66 of 255 levels are used (~6 bits) | `src/lib.rs:482` | -3.6 recall points vs exact, free to fix |
| No full-precision rerank, top-k is decided on int8 dots | `src/lib.rs:692` | rerank of top-50 recovers 100% of exact top-10 |
| Centroids are EMAs (0.9/0.1, 0.7/0.3), not means | `pipeline.py:235`, `:459` | centroids drift across ingest batches |
| OOD gate disabled (`activation_threshold = NEG_INFINITY`) | `src/lib.rs:665` | the OOD branch in the README never runs |
| Dynamic beta only fires when `beta == 0.0` | `src/lib.rs:185` | never benchmarked |
| No `target-cpu=native`, no LTO | `Cargo.toml` | the AVX2 numbers in my notes weren't AVX2 |
| `dot_i8` widens per element, won't vectorise to VPDPBUSD | `src/lib.rs:116` | 4-8x left on the table |
| Delete is an O(n) string scan; tombstones are permanent and still traversed | `src/lib.rs:612`, `:275` | the delete story isn't implemented |
| Compaction from the README doesn't exist | | same |
| NSW adjacency at M=64 is 1.42x the size of the vectors | `src/lib.rs:81` | the graph costs more than the data |
| Benchmarks measure MS MARCO qrels, never ANN recall vs brute force | `benchmark_aigvs.py:229` | no real ANN number yet |

Quantization, 200 queries x 5,000 docs (MS MARCO, MiniLM), top-10 overlap with exact:

| Config | Overlap |
|---|---|
| current SQ8 (`x * 127`) | 94.85% |
| global scale `127 / max|x|` | 98.45% |
| plus f32 query (asymmetric) | 98.90% |
| plus f32 rerank of top-50 | 100.00% |

Leaf structure at d=384, flat int8 scan vs NSW (ef=100):

```text
n=   500   flat  2.7 us   NSW 59 us
n=  1000   flat  6.2 us   NSW 59 us
n=  5000   flat 29.5 us   NSW 59 us
n= 10000   flat 56.9 us   NSW 59 us
```

A graph inside a leaf only pays off around 10k vectors, and my leaves are 3-5k. The leaf graphs
should go.

## 1a. First measurements (26-27 July)

### Baseline: real ANN recall for the old engine

MS MARCO, 99,000 base vectors, 1,000 held-out corpus vectors as queries, k=100, exact ground
truth. Harness: `bench/groundtruth.py` and `bench/run_ann_recall.py`, raw output in
`bench/results/`.

| System | Best R@10 | QPS at best | Build | Memory |
|---|---|---|---|---|
| AIGVS (old engine) | 0.9648 | 78 | 9.6 s | 361 MB |
| faiss-hnsw | 1.0000 | 1,738 | 20.0 s | 171 MB |
| faiss-ivf (nlist=314) | 0.9942 | 281 | 0.9 s | 237 MB |

The old engine loses on every axis:

* faiss-hnsw at efSearch=16 gets R@10 0.9655 at 6,486 QPS, which is better than my best recall at
  83x the throughput.
* faiss-ivf at nprobe=32 gets 0.9661 at 1,073 QPS (13x) and builds in 0.9 s instead of 9.6 s.

So three things I wrote in June were wrong:

1. "10x faster build than faiss": faiss IVFFlat builds this 10x faster than I do. I'd only
   compared against HNSW, and even there it's 2x, not 10x.
2. "Same or better memory": it's 2.1x HNSW (361 vs 171 MB). `stats()` didn't show the boundary
   copies because `chunk_location_map` is keyed by chunk id and only keeps the last leaf, but the
   copies and their edges are all in memory.
3. "98% recall": that was the MS MARCO qrel hit rate, not ANN recall. ANN recall is 0.9648.

Recall stops at 0.9648: conf=0.98/beam=32 and conf=0.995/beam=64 give exactly the same number.
More search doesn't help, so the limit is in the scoring, not the traversal, which matches the SQ8
problem above. That's the good news here, since phase 1 is exactly that fix.

### v2: the flat-cell core

New core in `src/index_v2.rs`: one global arena, u32 cell offsets, flat AVX2 scan, exact
`(sum, count)` centroids, `swap_remove` delete, built with `target-cpu=native` and fat LTO. Same
corpus and ground truth.

| System | Best R@10 | QPS | Build | Memory | Delete |
|---|---|---|---|---|---|
| AIGVS (old) | 0.9648 | 78 | 9.6 s | 361 MB | O(n) + tombstones |
| v2-i8 | 0.9952 | 390 | 2.2 s | 186 MB | 1.34 us, O(1) |
| v2-f32 | 0.9952 | 242 | 2.2 s | 165 MB | 1.87 us, O(1) |
| faiss-hnsw | 1.0000 | 1,738 | 20.0 s | 171 MB | not supported |
| faiss-ivf | 0.9942 | 281 | 0.9 s | 237 MB | O(N) `remove_ids` |

Better than the old engine on all four at once: recall 0.9648 -> 0.9952, QPS 78 -> 390, build
9.6 -> 2.2 s, memory 361 -> 186 MB.

Against faiss IVFFlat at the same nlist=314:

| nprobe | faiss-ivf | v2-i8 |
|---|---|---|
| 32 | 0.9661 at 1,073 qps | 0.9653 at 967 qps |
| 128 | 0.9942 at 281 qps | 0.9952 at 390 qps |

Level at moderate recall, a bit ahead at high recall.

i8 and f32 give exactly the same recall at every nprobe (0.9952 at 128, 0.9653 at 32). With a
proper global scale and an f32 rerank, the int8 scan costs nothing in recall. The old 0.9648
ceiling was a missing rerank and a 6-bit scale, not a memory/recall trade-off (D6 in practice,
not just from the papers).

Delete is 1.3-1.9 us per call and doesn't depend on index size. faiss IVFFlat needs an O(N)
`remove_ids`, and faiss HNSW can't delete.

#### v2 at 1M (999,000 base, 999 cells, same 1,000 queries)

| nprobe | v2-i8 R@10 | QPS | p99 | v2-f32 QPS |
|---|---|---|---|---|
| 32 | 0.9694 | 690 | 1.83 ms | 351 |
| 64 | 0.9854 | 438 | 2.86 ms | 205 |
| 128 | 0.9940 | 240 | 5.28 ms | 110 |

Build 13.8 s, memory 1,894 MB, delete 1.14 us.

* Delete stays flat: 1.34 us at 100k, 1.14 us at 1M.
* int8 helps more as N grows: 1.3x faster than f32 at 100k, 2.2x at 1M, because the scan
  dominates once cells are bigger.
* The sqrt(N) wall is already visible. With nlist = sqrt(N), cells grew from 315 to 1000 vectors
  and QPS at nprobe=128 fell from 390 to 240. At 100M that's 10k cells of 10k vectors and 1.28M
  vectors scanned per query. Fixed-size cells and an index over the centroids are the fix, and at
  10M a flat centroid scan (10k x 384 floats, ~500 us) is already over budget, so it's needed at
  10M, not just at 1B.

p99 at 1M is fine (5.28 ms max), so the 52 ms outlier at 100k was probably warm-up. Still need a
proper latency histogram.

Known problems at this point:

1. i8 mode uses more memory than f32 (1,920 vs 1,536 B/vector), because it keeps the f32 rerank
   copy and the int8 scan copy. An f16 rerank store would make it 1,152.
2. i8 was only 1.3x faster than f32 at 100k, not the ~4x the byte ratio suggests. I suspected
   the per-cell `Vec` allocation in the `flat_map` scan.
3. p99 spikes to 52 ms against a 2-7 ms median, probably the same cause.

### Parallelising across queries

Search was parallel within a query: rayon was handing out 32 tasks of ~10 us each, with 2-5 us of
overhead per task, and `flat_map` allocated a `Vec` per cell. Parallelising across queries instead
raised batch throughput about 4x (this fixed problem 2 above).

Batch throughput, all cores, 100k:

| Recall | v2-i8 | faiss-ivf | faiss-hnsw |
|---|---|---|---|
| R@10 ~0.966 | 6,618 | 2,871 | 37,912 |
| R@10 ~0.994 | 1,845 | 740 | 7,073 |

2.3-2.5x faster than faiss IVFFlat at every recall level, still 3.8-5.7x behind HNSW.

The scan is now limited by memory bandwidth:

| nprobe | v2-f32 | v2-i8 | speedup | byte ratio |
|---|---|---|---|---|
| 32 | 1,896 | 6,424 | 3.4x | 4x |
| 128 | 484 | 1,910 | 3.9x | 4x |

Throughput follows bytes per vector almost exactly: at nprobe=32 each query moves 3.87 MB, times
6,618 qps is ~25.6 GB/s, which is about what this machine can do. So making it faster is now a
question of bytes per query, and that's predictable.

Bytes per query at nprobe=32:

| Part | Bytes | Could be |
|---|---|---|
| cell scan (10,080 x 384 B int8) | 3.87 MB | 0.48 MB with 1-bit codes |
| f32 rerank (400 x 1,536 B) | 0.61 MB | 0.15 MB with int8 |
| f32 centroid scan (314 x 1,536 B) | 0.48 MB | 0.12 MB with int8 |
| total | 4.96 MB | 0.75 MB (6.6x) |

1-bit codes alone don't give 8x, because rerank and centroids are already 22% of the bytes;
quantizing only the scan gives ~3.1x (~20,000 qps, still below HNSW). Quantizing all three
gives ~6.6x, around 42,000 qps, which would pass faiss-hnsw's 37,912 at the same recall, with
about 8x less memory, O(1) delete and a 14x faster build. Adaptive probing would come on top.

So RaBitQ is the throughput lever, not just a memory saving, and everything a query reads has to
shrink, not only the scan. That moves phase 4 ahead of the f16 rerank fix.

### Binary codes

`src/rabitq.rs`: random rotation (FHT with random sign flips, 2 rounds), then 1 bit per
dimension, ranked by Hamming distance (POPCNT), then exact rerank. 64 B/vector at d=384 (padded to
512 bits), vs 384 B for int8.

The rotation is an FHT rather than a dense random matrix: a d x d rotation is 147k FLOPs per query
(20-40 us), which would cost more than the scan it speeds up, while the FHT is d*log2(d) ~ 4.6k
ops (1-2 us). The rotation is what makes the sign bits useful; plain sign(x) is a poor quantizer
because a few high-variance coordinates dominate.

100k, batch throughput at matched recall:

| Recall | v2-binary | v2-i8 | faiss-ivf | faiss-hnsw |
|---|---|---|---|---|
| R@10 ~0.966 | 24,851 | 5,873 | 2,871 | 37,912 |
| R@10 ~0.994 | 6,706 (rf=8) | 1,779 | 740 | 7,073 |

* 4.3x faster than int8 at the same recall, close to the 6x byte ratio.
* 8.7x faster than faiss IVFFlat.
* The gap to faiss-hnsw went from 5.7x to 1.5x.
* Memory is level with HNSW (172 MB), build 11x faster, delete 0.93 us.

Recall with enough rerank:

| rerank_factor | R@10 at nprobe=128 | batch qps |
|---|---|---|
| 4 | 0.9925 | 10,645 |
| 8 | 0.9942 | 6,706 |
| 16 | 0.9943 | 5,201 |

rf=8 gets back to the int8/f32 recall. With a big enough candidate pool, 1-bit codes cost no
recall either.

Why it's still 1.5x behind HNSW: at rf=8 / nprobe=128 most of the bytes are now in the parts that
aren't quantized yet.

| Part | Bytes/query | After |
|---|---|---|
| binary scan (128 x 315 x 64 B) | 2.58 MB | already minimal |
| f32 rerank (800 x 1,536 B) | 1.23 MB | 0.31 MB with int8 |
| f32 centroid scan (314 x 1,536 B) | 0.48 MB | 0.12 MB with int8 |
| total | 4.29 MB | 3.01 MB (1.4x) |

Rerank is 29% of the bytes now. Next: int8 rerank store, int8 centroids, then adaptive probing.
~1.4x from quantization and 1.5-2x from APS would put it past HNSW, and the int8 rerank store
also brings memory to ~448 B/vector against ~1,700 for HNSW.

### Storage bug found during phase 0

`InMemoryStorage.write_chunk` drops the embedding, so `read_chunk` returns `embedding=None`
(`storage/in_memory.py:17`). The Hopfield split reads chunks back from storage to get their
vectors (`nodes.py:271`), so the online split can't run with in-memory storage at all; it crashed
the first time it was called.

The underlying problem: the pipeline frees `chunk.embedding` after handing it to Rust
(`pipeline.py:225`), and the Rust side has no way to read a vector back. The only copy is the int8
`vector_cache` in `RustNSWGraphInternal`. Any online maintenance needs the vectors, so the global
arena in phase 2 needs a read-back API. One more reason for flat cells.

For now `TextChunk` accepts `embedding=None` instead of crashing.

## 2. The papers

| Paper | Venue | What it does | What it leaves |
|---|---|---|---|
| SPANN | NeurIPS 2021 | fine postings, graph over centroids, bounded replication, SSD | static |
| SPFresh | SOSP 2023 | in-place split/merge/reassign, only boundary vectors move | SSD-resident, fixed probe count |
| SOAR | NeurIPS 2023 | orthogonality-amplified residuals for multiple assignment | static, no maintenance |
| RaBitQ | SIGMOD 2024 | 1-bit codes with an error bound, 32x compression | needs rerank at 1 bit |
| Extended RaBitQ | SIGMOD 2025 | 2-6 bit codes, high recall without rerank | |
| Quake | OSDI 2025 | adaptive partitioned index: cost-model split/merge, APS, O(1) delete, NUMA scan | no quantization, no replication, maintenance triggered by hand, assumes uniform density |
| DARTH | SIGMOD 2026 | per-query recall target by early termination (HNSW and IVF) | |
| RAIRS | 2026 | better redundant assignment and list layout, ~50% fewer replicas | static |
| Big-ANN streaming | NeurIPS 2023 | 10M MSTuring, 4:4:1 insert:delete:search, recall@10 at checkpoints; best entry 0.9849 | |

Quake's numbers, which are the bar:

* Wikipedia-12M: 1.53 h for the whole workload vs 165.8 h for faiss-IVF (108x)
* OpenImages-13M: 0.03 h vs 0.22 h for DiskANN
* update latency 18-126x lower than graph indexes
* SIFT1M: 0.48 ms at 91.2% recall
* 100M on a 4-socket, 500 GB machine

Gaps Quake states itself:

1. Quantization is disabled for everything, so it's f32 only. 100M x 768d f32 is ~300 GB, which
   is why it needs that machine.
2. No redundant assignment.
3. APS assumes uniform density inside a partition, which they note may not hold for skewed
   embeddings. Embeddings are very skewed.
4. Maintenance is called by hand.
5. Its thresholds are hand-set.
6. SVS still beats it on static search (0.33 h vs 0.63 h on MSTuring10M).
7. Distributed, filtered and high-concurrency search are left as future work.

## 3. What I think is left

Adaptive partitioned indexing under heavy quantization.

Quake showed that cost-model maintenance plus adaptive probing beats graph indexes on dynamic
workloads, in f32. Quantization isn't something you just bolt onto that: it changes the cost model
and the recall model the method is built on.

When scan cost per vector drops 10-30x with RaBitQ:

1. **The cost model changes.** Quake balances scan cost against centroid overhead. If scan cost
   per vector shrinks 20x, the best partition size grows and the split/merge thresholds are wrong.
2. **The recall model changes.** APS estimates the chance a partition holds a true neighbour from
   a hypersphere of radius rho, the current k-th NN distance. Under quantization rho is itself
   noisy, and every distance has an error term with a known distribution (that's what RaBitQ's
   bound gives). APS needs to account for that or it will under-probe.
3. **Rerank becomes a cost term** of its own, tied to nprobe.

Point 2 is the strongest: RaBitQ gives a distributional error bound and APS is a probabilistic
recall model, so they should compose. I haven't found anyone who's done it.

A secondary idea if there's time: replication-aware maintenance. Quake has no replication and
SOAR/RAIRS have no maintenance. When a partition splits, every vector whose secondary assignment
pointed into it is stale. Local repair of redundant assignments doesn't seem to exist yet.

## 4. Claims and how to test them

| # | Claim | Metric | Baseline | Target |
|---|---|---|---|---|
| C1 | quantized adaptive index matches f32 on recall | R@10 at fixed latency, 10M and 100M | Quake | within 1 point |
| C2 | with 4-6x less memory | bytes/vector at recall 0.90 | Quake (3072 B at d=768) | <= 600 B |
| C3 | quantization-aware APS beats plain APS on quantized data | probes at target recall | APS assuming exact distances | >= 25% fewer |
| C4 | recall holds under churn | R@10 over the Big-ANN streaming runbook | hnswlib, DiskANN, Quake | >= 0.98, no decay |
| C5 | O(1) delete | us/delete vs index size | DiskANN consolidation | flat |
| C6 | build time | seconds for 10M | faiss HNSW | 10-50x faster |
| C7 | filtered search | recall at 1% selectivity | HNSW + post-filter | clearly better |

Not claiming:

* faster than graph indexes on static data (SVS and DiskANN win there)
* better than HNSW at low recall (0.80-0.90), where a small ef wins
* billion scale; single node up to 100M
* adaptive maintenance or adaptive probing themselves, which are Quake's

## 5. Weak points and which ones the papers fix

| # | Problem | Status | How |
|---|---|---|---|
| D1 | fine cells need lots of replication (R~8) | fixed | Quake uses no replication and gets recall from adaptive probing instead. If I want replication later, RAIRS halves it and SOAR makes each copy count more. |
| D2 | queries are questions, documents are passages | mostly fixed | APS adapts per query, so an out-of-distribution query just probes more. Its uniform-density assumption is weakest here, which is why C3 matters. |
| D3 | no advantage below ~10M | real | Quake's wins show at 12M+. Benchmarks have to move to 10M; the 100k runs can't test any of this. |
| D4 | adaptive probing adds p99 variance | fixed | DARTH treats it as a per-query recall target. Add a hard probe cap and report the recall lost at the cap. |
| D5 | splits cause latency sawtooth | fixed | Quake only acts when the cost improves by more than tau. Use the same guard. |
| D6 | the rerank store is a memory floor (~864 B/vector) | fixed | Extended RaBitQ at 4-6 bits gets high recall without rerank. At d=768, 4-bit is 384 B and no rerank store. |
| D7 | lose to graphs at low recall / static data | real | say so up front |
| D8 | distributed / resharding | real, later | single node, up to 100M |
| D9 | prior art is dense, contribution is narrow | real | be precise about the claim |

## 6. Build plan

Each phase ends with a measurement that can stop it.

### Phase 0: measure

- [ ] Read Quake, SPFresh, RaBitQ + Extended RaBitQ, SOAR and DARTH in full.
- [ ] Clone Quake and reproduce one SIFT1M number.
- [ ] Set up the Big-ANN streaming harness with the official runbook.
- [x] Brute-force ground truth and real ANN recall@k (`bench/groundtruth.py`).
- [ ] Get a 10M corpus (MSTuring-10M is the standard). The 1M MS MARCO cache is on disk, use
  that in the meantime.
- [ ] Check the workload assumption: measure the real delete/update rate of an agent-memory or
  changing RAG corpus. If it's append-mostly, point 5 in section 11 applies to me as well.
  Cheap to check, and it could change the whole pitch.
- [ ] Measure D2: R@10 with held-out corpus vectors as queries vs real query embeddings. The
  difference is the query/document mismatch.

Gate: Quake reproduces and I can draw a recall/QPS curve against brute force.

### Phase 1: fix the basics

- [ ] global SQ8 scale `127 / max|x|` (+3.6 points, one line)
- [ ] rerank store, exact rerank of the top 4k
- [ ] f32 query against int8 data
- [ ] exact centroids from `(sum, count)` instead of the EMA
- [ ] `.cargo/config.toml` with `target-cpu=native`; fat LTO, one codegen unit
- [ ] a real AVX2 `dot_i8` (`_mm256_maddubs_epi16` + `_mm256_madd_epi16`)
- [ ] u32 ids inside, strings only for the final top-k
- [ ] reusable visited markers instead of `vec![false; n]` per query

Gate: R@10 vs exact >= 0.99 at 100k, latency at least 2x better.

### Phase 2: flat leaves, real deletes

- [ ] flat scan instead of the leaf graph
- [ ] one global vector arena; cells hold u32 offsets
- [ ] `swap_remove` delete: O(1), no tombstones, nothing to compact
- [ ] delete through an `id -> (cell, slot)` map
- [ ] remove the `deleted_indices` path

Gate: 30% less memory, 5x faster build, recall no worse, delete time flat in index size.

### Phase 3: Quake's mechanisms

Implement as described in the paper and cite it.

- [ ] cost model with per-partition access counts
- [ ] split / merge / refine, only when the cost improves by more than tau
- [ ] APS for per-query nprobe, with a hard cap
- [ ] NUMA-aware parallel scan

Gate: within 1.5x of Quake's SIFT1M point (0.48 ms at 91.2%). If I can't match it in f32,
quantization won't help.

### Phase 4: quantization-aware adaptation

The actual research.

- [ ] RaBitQ with a fast Hadamard rotation (~1 us, not a ~40 us dense matmul)
- [ ] Extended RaBitQ at 2/4/6 bits; check the no-rerank claim on my data
- [ ] re-fit the cost model with quantized scan cost; record how the best partition size moves
- [ ] fold RaBitQ's error distribution into APS so rho is treated as noisy
- [ ] rerank as an explicit cost term tied to nprobe
- [ ] ablation: plain APS on quantized data vs quantization-aware APS (this is C3)

Gate: C1, C2 and C3. If C3 fails there's no paper, only a well-built system.

### Phase 5: scale and streaming

- [ ] 10M, then 100M
- [ ] full Big-ANN streaming runbook; aim above 0.9849 average R@10
- [ ] ann-benchmarks static protocol (I expect to lose to SVS there)
- [ ] filtered search (C7)

### Phase 6: write it up

VLDB / SIGMOD / OSDI. Lead with the quantization-aware adaptation, with Quake as the closest
baseline.

## 7. When to stop

| If | Then |
|---|---|
| 4-bit Extended RaBitQ already gives >= 0.95 without rerank and plain APS handles it | nothing to research; build the product |
| can't get within 1.5x of Quake on SIFT1M | it's an engineering gap; fix that first |
| quantization-aware APS is < 15% better than plain | stop the research track, ship the system |
| > 2x behind faiss HNSW QPS at recall 0.95 | a niche update-heavy index; smaller claim |
| someone publishes quantization-aware adaptive partitioning | switch to replication-aware maintenance |

## 8. Product path

If the research dies at a gate, this is still worth doing. Quake, SPFresh, RAIRS and SOAR are
research code. Qdrant, Milvus and Weaviate use HNSW with segment rebuilds and get recall decay and
compaction spikes. A solid Rust partitioned index with quantization, O(1) deletes, no rebuilds,
filter pushdown and a recall target doesn't exist yet, and phases 1-3 build it anyway.

Use cases: long-term memory for agents (constant writes, corrections, expiry), right to be
forgotten (a tombstoned HNSW node still holds the vector), live RAG over changing data, removing a
tenant.

## 9. Stop doing

* **Hopfield framing.** There's no xi iteration in the query path (`lib.rs:150`); it's a softmax
  over centroid cosines. If Hopfield shows up at all it's as theory for single-step routing
  (Ramsauer's separation bound).
* **Deep trees.** Routing error compounds as p^d and depth doesn't reduce scan volume. My own
  numbers showed it (0.71 ms -> 4.16 ms, 98% -> 31.5%). One partition level, with a graph over
  centroids once there are more than ~4096 of them.
* **Graphs in leaves.** They lose below 10k vectors, cost 1.42x the data, and make O(1) delete
  impossible.
* **Benchmarking at 100k.**
* **qrels as the accuracy metric.** That measures MiniLM, not the index.

## 10. Next

1. Run Quake on SIFT1M and reproduce one number.
2. Read Extended RaBitQ; try the no-rerank claim on `msmarco_embeddings_cache_v2_100000.npy`.
3. Start phase 1, which helps whatever happens with the rest.
4. Get MSTuring-10M.

## 11. Positioning

### Quake is a baseline

Quake is a research prototype (`marius-team/quake`): f32 only, L2/IP only, int64 ids, in memory,
single node, no persistence, no concurrency, no filters. It's the number to beat in a paper, not
something people will deploy. Its limit is memory:

```text
100M x 768d f32 = 300 GB   (they used a 4-socket, 500 GB, 80-core machine)
  1B x 768d f32 = 3.0 TB   (not possible on one node)
```

The paper lists compression, filters, distribution and concurrency as extensions, so the same
group will probably do the follow-up. "We added quantization" isn't a contribution; the re-derived
models in section 3 would be.

### Who people actually use

| System | Index | Updates | Problem |
|---|---|---|---|
| Qdrant | HNSW per immutable segment | seal + background merge | 2-5x memory, compaction spikes |
| Milvus | HNSW, hash-sharded segments | flush at a size threshold, then index | same |
| Weaviate / Pinecone | HNSW | segment rebuild | same |
| pgvector | HNSW or IVFFlat | HNSW incremental, IVF degrades | pick one |

The usual advice is "HNSW for most workloads under ~10M vectors with active writes, IVFFlat for
very large, mostly static datasets". Nothing is recommended for large and actively written, which
is what agent memory looks like.

VectorChord is also arguing that HNSW isn't the answer and shipping IVF-based search in Postgres,
so I'm not alone in thinking this.

### Why partitioned indexes aren't the default

1. HNSW needs no training: no nlist, no nprobe, no centroid upkeep. Simple wins in
   infrastructure.
2. The standard objection to IVF (new points go into old clusters and quality drifts) was true
   until SPFresh (2023) and Quake (2025) fixed it.
3. Benchmarks are static. ann-benchmarks is build once, query forever, where HNSW does well.
   The streaming track only started in 2023.
4. Segments already made any index updatable at the system level (the LSM approach), so nobody
   had to change the index itself.
5. Most workloads are append-mostly, and then HNSW's weakness never shows. I need to check this
   doesn't apply to the workloads I'm targeting.
6. Index internals change slowly.

Where scale forces it, partitioning is already standard: Google uses ScaNN, Meta runs faiss IVF.
HNSW mainly dominates open-source vector databases.

### What I can claim

Something like "the most memory-efficient index that keeps its recall under continuous writes and
deletes" is measurable and fair.

"The best RAG system" isn't. An ANN index isn't a RAG system: answer quality comes from the
embedding model, chunking, hybrid BM25 + dense, cross-encoder reranking and query rewriting, and
above ~0.95 ANN recall the index barely matters. My own HotpotQA run
(`hotpot_code/full_aigvs_metrics.json`) has recall@10 75.5% and nDCG 0.447, which is limited by the
embedding model; a perfect index would get ~76%.

So the opportunity is that adoption lags the research, not an undiscovered algorithm. The
technical bet (quantization plus adaptive partitioning) looks sound and open.

## References

* SPFresh (SOSP 2023): https://www.microsoft.com/en-us/research/publication/spfresh-incremental-in-place-update-for-billion-scale-vector-search/
* Quake (OSDI 2025): https://www.usenix.org/system/files/osdi25-mohoney.pdf, https://arxiv.org/html/2506.03437v2
* RaBitQ (SIGMOD 2024): https://dl.acm.org/doi/pdf/10.1145/3654970, code https://github.com/gaoj0017/RaBitQ
* Extended RaBitQ (SIGMOD 2025): https://github.com/VectorDB-NTU/Extended-RaBitQ
* SOAR (NeurIPS 2023): https://arxiv.org/abs/2404.00774
* DARTH (SIGMOD 2026): https://helios2.mi.parisdescartes.fr/~themisp/publications/sigmod26-darth.pdf
* RAIRS (2026): https://arxiv.org/pdf/2601.07183
* Big-ANN streaming track: https://github.com/harsha-simhadri/big-ann-benchmarks/tree/main/neurips23/streaming, results https://arxiv.org/pdf/2409.17424
* Distribution-aware exploration for adaptive HNSW search (2025): https://arxiv.org/pdf/2512.06636
