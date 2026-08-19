# Polysemic attractor DAG: research log

The goal, restated: an attractor-tree index (the original AIGVS idea) that gets O(log N) query
cost at fixed recall while keeping the partitioned index's build, memory and O(1) delete. This log
has the theory, every experiment, and where things stand. Projections are marked as projections.

**Status.** Three mechanisms came out of the theory and were measured.

1. **Orthogonal polysemy**: a second home for each vector whose residual is orthogonal to the
   first. The two homes' misses become statistically independent (measured), the vectors needed
   for R@10 0.99 drop 1.8-2.7x, and the gain grows with N on both corpora and on real questions.
2. **A multi-parent attractor DAG** with decorrelated parents: routing cost in a deep tree grows
   sub-linearly (centroids scored 781 -> 1,349 -> 1,580 for 100k -> 250k -> 500k) for at most
   0.004 recall.
3. **A residual attractor channel** (2 bytes/vector): another 1.4-2.1x in vectors examined, but
   that gain doesn't grow with N.

Orthogonal polysemy is in the engine and tested. End to end at 250k it's level on throughput, not
a win yet, for reasons measured in section 5. O(log N) at fixed recall on the hard corpus isn't
reached: on HotpotQA with real questions the exponent of vectors scanned at R@10 0.99 is +0.58
with orthogonal polysemy, from +0.76 without.

## 1. Going back over the code

Before starting I went back through the v2 code, the earlier docs and the old prototype. What I
found and what I did about it:

1. **"No question embeddings exist" (RESULTS.md, Part V) wasn't really true.** The 1,000 HotpotQA
   questions exported by the June prototype's harness were still on disk, and MiniLM is cached
   locally. They're now embedded (`bench/embed_hotpot_questions.py` ->
   `bench/_lab_cache/hotpot_questions_minilm.npy`), and every result below is reported on these
   real questions as well as on held-out corpus vectors.
2. **The spill/polysemy negative tested the wrong rule.** Every spill I'd tried, and the June
   prototype's polysemy (`max - 0.08`), puts the extra copy in the next-nearest cell. Section 3
   shows why that has to fail and section 4 shows a rule that doesn't.
3. **Batch search rebuilt its scratch buffers for every query, and the multi-home dedup buffer was
   N x 4 bytes.** With any replication every query allocated and zeroed 1 MB at 250k, which capped
   batch throughput at ~8k QPS whatever the nprobe. Fixed: dedup is per cell now (M stamps; a
   vector is scored in the first of its homes to be scanned) and `search_batch` reuses one buffer
   per worker (`map_init`). The earlier spill results were in vectors scanned, so they weren't
   affected, but any throughput number with replication would have been.
4. **Replication had no tests.** Added section `[10]` to `tests/test_index.py`: it's actually on,
   no duplicate ids at any nprobe, exhaustive search equals brute force, adaptive is still exact,
   deletes unlink every home, for R=2 and R=3 in f32 and binary. Plus a real-data check that
   polysemy raises recall. All pass.
5. **Chunked ingest costs 6x (known, still open).** `add_batch` ends in an O(N*d)
   `refresh_derived`; 1M in 5 chunks is 74.7 s against 12.2 s in one call. Every build time here
   is a single `add_batch` and says so.
6. `import sentence_transformers` segfaults on this machine (exit 139), so the embedding uses
   `transformers` directly with MiniLM's pooling config.

## 2. How things are measured here

* **Build time = train + ingest + compact**, wall clock, one process; only ground truth is left
  out. Anything a method needs at build time (e.g. a kNN graph) is counted in the build.
* **Vectors scanned / examined is a cost model**, used to compare routing ideas cheaply and
  labelled as such. Only the engine measurements in section 5 are throughput.
* Same partition, queries and exact ground truth for every arm in a table. Recall is R@10 against
  brute force. Batch QPS is best of 3 on all 16 threads.
* Machine: Ryzen 7 6800H (8C/16T), 15 GB RAM, RTX 3050 4 GB (ground truth and kNN only).

## 3. Theory

**Miss model.** A query q probing the top P cells by attractor score `q.mu` misses neighbour x
when x's home cell ranks below P. With residual `r = x - mu`, `q.mu = q.x - q.r`, so the miss is
driven by the gap `g = q.r`. In high intrinsic dimension the number of cells that outrank x's cell
grows roughly like `exp(kappa*g)`, so high recall is about the tail of g. That's why alpha depends
so much on the recall target.

**Splitting the residual.** Write `r = s + c`, where `s = x - m(x)` is x's deviation from its local
mean and `c = m(x) - mu` is the partition-specific part. `s` is the same in every partition. So:

* the next-nearest cell has a residual nearly parallel to the first, and misses are correlated
  (which is why spill was a wash);
* an independently trained second partition still shares `s`, so its misses are correlated too
  (which is why multiple tables lost);
* a home whose residual is orthogonal to r cancels both parts, and its misses are independent.

Prediction: with independent homes `P(miss) = prod(p_i)`. Section 4.2 measures it: it holds
exactly for orthogonal homes and is off by 5.7x for the other two.

**How many homes.** If one home's miss rate at beam B is `p(B) ~ B^(-a)` and there are R
independent, equally good homes, reaching miss rate m costs `R * m^(-1/(aR))`. The best R is
`R* = ln(1/m)/a`, at cost `~ e*ln(1/m)/a`: logarithmic in 1/m instead of polynomial. That's the
lever for high recall. The catch is the open problem: in one partition each further orthogonal
home is a worse cell (marginal miss 0.06 -> 0.32 -> higher), so R=3 adds little (measured).

**Depth without the memory blow-up.** In a pointer arena, extra parents above the leaves only cost
routing bytes (node counts shrink by the fanout at each level). A vector reachable through R homes
and R parents per level has `R^L` paths for `~ N*R` entries of memory, independent of depth. A
spill tree pays `(1+s)^L` in data for `(1+s)^L` paths, which is the 462% at depth 4 from June.
Decorrelated parents turn a per-level miss p into `~ p^R`, which is what lets a deep tree keep a
small beam.

**A guard against silent recall loss (proposed, not built).** A per-query stopping rule: keep
probing until the estimated neighbour mass left in unprobed cells (from a calibrated per-cell tail
model, with misses multiplied across a vector's independent homes) is below eps. Then a harder
dataset costs more time instead of recall. Independence (4.2) is what makes multiplying the misses
valid.

## 4. Measurements (cost model: vectors scanned, exact centroid ranking)

### 4.1 Orthogonal polysemy: MS MARCO, held-out queries

Vectors scanned to reach R@10 (faiss spherical k-means, 128 per cell, partitions retrained per N;
1M uses the cached lab partition):

| arm (entries/vector) | 124k -> 499k at 0.98 | alpha at 0.98 | 124k -> 499k at 0.99 | alpha at 0.99 | 1M at 0.99 |
|---|---|---|---|---|---|
| none (1) | 10,302 -> 19,673 | +0.46 | 18,707 -> 37,345 | +0.50 | 39,285 |
| next-nearest spill (2) | 9,044 -> 15,284 | +0.38 | 14,734 -> 31,267 | +0.54 | 27,325 |
| orthogonal, lambda=1 (2) | 6,292 -> 9,050 | +0.26 | 10,747 -> 17,109 | +0.33 | 17,210 |
| orthogonal, lambda=4 (2) | 6,768 -> 8,415 | +0.16 | 11,190 -> 13,794 | +0.15 | 17,106 |

Gain over none at 499k: 1.6x at 0.95, 2.3x at 0.98, 2.7x at 0.99.

### 4.2 Independence, measured directly (HotpotQA 100k, real questions)

| design (2 homes) | P(both miss at depth 64) | product of marginals | |
|---|---|---|---|
| orthogonal home | 0.018 | 0.019 | independent |
| two independently trained partitions | 0.074 | 0.013 | 5.7x correlated |
| next-nearest home (250k) | 0.051 | 0.016 | 3.2x correlated |

Rank correlation between the two homes: 0.36-0.40 for next-nearest, -0.13 to 0.06 for orthogonal.
Four independent partitions (4 entries/vector) do worse than one partition with an orthogonal home
(2 entries/vector).

### 4.3 HotpotQA: the hard corpus, held-out and real questions

| | alpha at 0.99, none | alpha at 0.99, orthogonal | gain at 100k / 250k / 500k (0.99) |
|---|---|---|---|
| held-out | +0.73 | +0.65 | 1.54 / 1.57 / 1.78x |
| real questions | +0.76 | +0.58 | 1.40 / 1.67 / 1.86x |

Next-nearest spill on real questions is worse than nothing at 100k (41,471 vs 34,659 vectors for
0.99). The gain grows with N in every row, but on this corpus the exponent drops by ~0.1-0.2, not
by half: two homes aren't enough here.

### 4.4 Making the second home better (HotpotQA 100k)

Candidate pool 16 vs 64 and lambda in {1, 4, 16}: all within noise. Polysemic attractor training
(re-centre each attractor on its whole basin, secondary members included, 3 rounds) is best on
held-out (17,018 vs 19,389 vectors for 0.99) and neutral on real questions. Two homes are close to
saturated at this scale.

### 4.5 A multi-parent DAG over the attractors (HotpotQA, real questions)

Plain beam descent, fanout 32, two orthogonal leaf homes. Recall at 64 leaf cells / centroids
scored per query:

| N (depth) | flat ranking | DAG, beam 16 | DAG, beam 32 |
|---|---|---|---|
| 100k (1) | 0.9786 / 781 | 0.9782 / 644 | 0.9786 / 781 |
| 250k (1) | 0.9648 / 1,953 | 0.9594 / 779 | 0.9644 / 1,349 |
| 500k (2) | 0.9464 / 3,906 | 0.9396 / 890 | 0.9426 / 1,580 |

Routing cost grows sub-linearly; at depth 2 the DAG is 0.002-0.004 below flat at beam 32. A
single-parent descent at beam 16 loses 1.7-2.4 points at 500k, so the extra parents are what make
the tree work. What the DAG doesn't fix: recall at a fixed number of leaf cells still falls with N
even with perfect routing (0.979 -> 0.965 -> 0.946). The remaining exponent is in the leaf
decision.

### 4.6 Navigating at the leaf level (point -> cell links)

Each point stores the cells holding its own kNN, and the best points found so far vote (by rank
or softmax) for the cells to scan next. 1.1-1.4x over flat ranking at equal vectors scanned with
one home, at most 1.2x with two, and the gain grows only slowly with N. Every vote costs a whole
cell scan. Parked.

### 4.7 Residual attractor channel (2 B/vector)

Two residual codebooks (256 x 256) on `x - mu`; members ordered by code, groups enumerated by
`q.mu + q.rho1 + q.rho2`. Vectors examined (not counting the 2 B/vector pre-pass, which a real
implementation has to pay for):

| | 100k at 0.95 | 100k at 0.99 | 250k at 0.95 | 250k at 0.99 |
|---|---|---|---|---|
| real questions | 10,171 -> 6,145 (1.66x) | 31,100 -> 20,042 (1.55x) | 19,453 -> 13,596 (1.43x) | 76,749 -> 44,966 (1.71x) |
| held-out | 8,116 -> 3,992 (2.03x) | 30,859 -> 14,647 (2.11x) | 15,160 -> 9,117 (1.66x) | 67,262 -> 36,437 (1.85x) |

A constant-factor cascade (the residual variance explained falls from 18% to 15% with N), not a
change in exponent.

## 5. In the engine: orthogonal polysemy end to end

`set_orthogonal_polysemy(homes, lam)` in `src/index.rs`, tests passing. `bench/bench_polysemy.py`,
HotpotQA 250k, one `add_batch`, everything the same except the extra home:

| memory profile (rf=64) | none | next-nearest | orthogonal lambda=1 |
|---|---|---|---|
| build (train + ingest + compact) | 5.5 s | 6.9 s | 5.5 s |
| bytes/vector | 464 | 472 | 472 |
| delete | 1.62 us | 1.74 us | 1.95 us |
| QPS at R@10 0.95, held-out | 20,311 | 16,781 | 18,803 |
| QPS at R@10 0.98, held-out | 5,815 | 4,684 | 6,097 |
| QPS at R@10 0.95, real questions | 16,772 | 14,032 | 16,041 |
| max R@10 (np=1024), real questions | 0.9775 | 0.9786 | 0.9793 |

Quality profile (f32 rerank, rf=128): QPS at 0.99 on real questions 5,032 vs 5,211; only the
orthogonal arm reaches 0.995 (2,895 QPS); max R@10 0.9942 -> 0.9963.

Why the routing gain doesn't turn into throughput yet: at equal entries scanned the engine matches
the model (real questions: 0.9651 at nprobe 128 with polysemy vs 0.9573 at nprobe 256 without),
but a replica entry costs ~25% more to scan than a primary. After `compact()` the primaries are
contiguous and the replicas are scattered single-cache-line reads, and adaptive termination fires
later. In the quality profile a fixed 1,280 x 1.5 KB f32 rerank dominates the per-query cost. Both
are implementation costs, not limits of the idea, and both are next (section 7).

## 6. The architecture

```text
level L..1   attractor DAG: every node has R decorrelated parents
             (orthogonal-residual rule on centroids); beam descent, cost L*B*f
level 0      cells of c vectors in one pointer arena
             every vector: nearest attractor + orthogonal home(s)
             (+ 2-byte residual code; members ordered by it)
query        DAG descent -> leaf cells -> residual-ordered scan -> exact rerank
             stop rule: unprobed neighbour mass < eps (multiplies across homes)
delete       O(R): unlink R member entries, free one slot
insert       O(log N) descent + R exact candidate scores
```

| part | theory | lab | engine |
|---|---|---|---|
| orthogonal polysemy | independence, so misses multiply | 1.8-2.7x at 0.99, grows with N | built and tested; level end to end at 250k |
| multi-parent DAG | R^L paths for O(R) memory | routing sub-linear, <= 0.004 loss | not yet (the engine's tree is a shortlist, not a descent) |
| residual channel | a second, anti-correlated score | 1.4-2.1x, constant | not yet |
| certified stopping | recall fixed, cost absorbs difficulty | not yet | not yet |

## 7. Next, by expected effect

1. **Contiguous replicas.** Copy the 64-byte scan code of each extra home into that cell's
   contiguous run (+64 B/vector, counted; rerank data stays single). That removes the 25% replica
   penalty from section 5. Then re-measure at 1M against faiss-HNSW with the same threads and full
   build time.
2. **Residual channel in the engine** as a 2-byte pre-pass over a wider probe set, then the 64-byte
   code on the survivors.
3. **DAG descent in the engine** instead of the shortlist tree.
4. **The stopping rule** from section 3.
5. **The leaf exponent** (still +0.58 on real questions). Ideas with a reason to bend it: give
   extra homes (R ~ ln(1/m)/a) only to the vectors with a large residual, since the tail is a
   minority (polysemy on the worst half alone gets most of the gain); and attractors trained with a
   ranking loss on real questions (the HotpotQA train split, 90k questions, is cached locally).

## 8. Reproduce

```bash
python bench/embed_hotpot_questions.py                        # once
KMP_DUPLICATE_LIB_OK=TRUE python tests/test_index.py          # all checks
# model-level studies
python bench/research/polysemy/polysemy_lab.py msmarco_embeddings_cache_v2_1000000.npy 500000 "" <gt n499000>
python bench/research/polysemy/polysemy_ood.py hotpot_5M.npy 250000 bench/_lab_cache/hotpot_questions_minilm.npy
python bench/research/polysemy/multipart.py    hotpot_5M.npy 100000 bench/_lab_cache/hotpot_questions_minilm.npy
python bench/research/polysemy/tree_lab.py hotpot hotpot_5M.npy 250000 bench/_lab_cache/hotpot_questions_minilm.npy
python bench/research/polysemy/residual_lab.py hotpot_5M.npy 250000 bench/_lab_cache/hotpot_questions_minilm.npy
# engine, end to end
python bench/bench_polysemy.py hotpot_5M.npy --n 250000 --questions bench/_lab_cache/hotpot_questions_minilm.npy --arms none,near,ortho1
```

Raw output goes to `bench/results/research/*.json` and `bench/results/polysemy_*.json`
(`bench/results/` is git-ignored).
