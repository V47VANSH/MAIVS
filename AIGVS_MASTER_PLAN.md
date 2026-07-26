# AIGVS plan

Draft, 2026-07-26. Written after going through the literature properly for the first time.

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
- [ ] Brute-force ground truth and real ANN recall@k (right now I only measure qrels).
- [ ] Get a 10M corpus (MSTuring-10M is the standard).

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
