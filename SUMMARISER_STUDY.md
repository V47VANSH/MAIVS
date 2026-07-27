# Cell summaries: can a better summary than the centroid fix routing?

All runs on this machine against exact ground truth, at 100k and 999k, 1,000 held-out queries,
k=10. Most of the findings here are negative.

**Result:** the 75x routing headroom from `notes/2026-07-27.md` is real, but a per-cell summary
can't collect it. Priced in the format it would actually be stored in, the best summary is worth
about 1.25x in query bytes at 1M (section 6e). All three candidates from the notes fail, each for a
different reason that I could measure.

(An earlier version of this said 1.70x. That charged sub-summaries 72 B, the size of a 1-bit code,
while scoring them in f32. Section 6e has the corrected number. Rule I took from it: price a
summary in the format it will be stored in.)

## 0. Status

**Shipped** (`index_v2.rs`, on by default):

* `set_centroid_sq8`: the stage-2 centroid re-order reads SQ8 instead of f32. 1.12-1.28x
  throughput with |dR@10| <= 1e-4, 9 MB less memory at 1M, all 30 invariant checks pass. Section
  6b. The only change here that cost nothing anywhere.

**Two fixes to the benchmark harness** (`run_ann_recall.py`). Both had been skewing every
comparison in yesterday's notes:

* it never called `set_tree_route`, so it wasn't benchmarking the routing path I actually use,
  and ingest fell back to the O(N*M) assignment;
* the adaptive probe grid jumped from 256 straight to n_cells, so R@10 0.98-0.994, where the
  faiss comparison is decided, was never measured.

**Measured, not built:**

* sub-group scanning: 1.25x, ~1% memory, but touches a lot of code (6e)
* a scan cascade: 1.57x at -0.0009 recall, or 1.15x with no loss, for +24 B/vector (~5% memory),
  and much less invasive (6f)

I took the free win first; the rest all cost something.

**Not tested:** every query here is a held-out corpus vector. Real queries are questions, not
passages, and I don't have question embeddings yet (6c).

## 1. Harness

`bench/summariser_lab.py`. Every candidate summary is an estimator of

```text
    s*(q, C) = max over x in C of q.x
```

so comparing them is a ranking problem over one `(n_queries x n_cells)` matrix, with no Rust and
no rebuild. The lab computes the exact `s*` once on the GPU, caches it, and scores a candidate in
seconds. I got through about forty configurations in the time one rebuild and benchmark would
take.

**Metric:** routing recall@k, the fraction of a query's true top-k that lives in the chosen cells.
It's the most the scan could find, so it separates routing from scan and rerank losses.

**Check:** the lab trains its own partition in torch instead of reading it out of the engine, so it
has to match. At 999k / 7,804 cells the lab's centroid router needs 384 probes for 0.99 and 48 for
0.95; the Rust engine needs 404 and 60. Close enough that results should carry over.

## 2. The floor, and how good a summary has to be

| N | cells | cells holding the true top-10 | oracle probes for 0.99 | oracle vectors |
|---|---|---|---|---|
| 99k | 773 | 3.78 | 8 | 1,163 |
| 999k | 7,804 | 3.49 | 8 | 1,242 |

The centroid router scans 50,181 vectors for 0.99 at 1M, so there's 40x of headroom in vectors
scanned.

The most useful experiment was to add Gaussian noise to the oracle and see how accurate a summary
would need to be. Not a candidate, a target:

| error eps | probes for 0.99 | vectors scanned |
|---|---|---|
| 0.02 | 12 | 1,839 |
| 0.04 | 32 | 4,779 |
| 0.08 | 256 | 36,203 |
| centroid router | 384 | 50,181 |

A summary has to estimate `s*` to about +/-0.04 to get the headroom. Having that number first made
the rest quick, since every candidate could be checked against it.

## 3. Candidates

| Candidate | Result at 1M | Why |
|---|---|---|
| mean + angular radius (`cone`, `gamma`), candidate 1 from the notes | no gamma beats gamma=0 (eps 0.0566 vs 0.0523 for the centroid) | not the outlier problem: even a fitted radius fails. A scalar radius doesn't depend on q, so it can't re-rank cells for a given query |
| element-wise max / box, candidate 3 | 8x worse than the mean (256 probes vs 32 for 0.95, at 100k) | a box around 128 points in 384-d holds far more than their hull, and a loose bound ranks badly |
| low-rank residual cone (`rank`, `var`) | `var4` ~ `var0` (16,265 vs 16,394 vectors) | four principal directions add nothing; a 128-point residual cloud in 384-d is close to isotropic. Rules out the anisotropic variants |
| hull vertices (greedy support-function reps) | worse than the mean, `hull8` 64 probes vs 32 for 0.95 | represents 8 members exactly and 120 not at all, and the query usually needs one of the 120 |
| learned routing vector (`lrv`, ridge, same bytes) | worse than the mean, 48 vs 32 | bounds what any single vector per cell can do: `s*` is a support function, convex and piecewise linear, and no linear functional matches it |
| several representatives (`subc`), candidate 2 | the only one that helps: eps 0.0436 at m=32, 1.63x bytes | covers every member and takes a max of m linear functionals, which is the right shape |
| sub-group order statistic (`subvar`, mean + beta*sigma per group) | no better than `subc` | the correction is real but the scatter is as big as the correction |

### The check that settled it

As m goes to the cell size, sub-centroids should converge to the oracle. They don't:

| | groups | vectors per group | probes for 0.99 |
|---|---|---|---|
| `subc32` | 32 | 4 | 96 |
| oracle | 128 | 1 | 8 |

Averaging just four vectors already costs 12x. A group with one match at 0.85 and three at 0.45
reports 0.55. Every summary built on averages hits this, whatever the corpus.

## 4. Why the rest isn't reachable

Effective eps (after an affine calibration, over the cells in contention) against the probes
actually needed, at 1M:

| variant | eps | probes for 0.99 | probes if that eps were random noise |
|---|---|---|---|
| centroid | 0.0523 | 384 | ~60 |
| `subc32` | 0.0436 | 128 | ~35 |

Structured error costs about 4x more than random noise of the same size, and it isn't an accident.
A cell matters exactly when q is close to one of its members, which is the case a mean-plus-spread
summary underestimates. Knowing whether a cell has a near neighbour without reading its members is
the search problem itself, so a small per-cell statistic can't answer it. That one point explains
every negative above.

## 5. The one positive

A sub-summary doesn't only re-rank cells, it also says which part of a cell is worth scanning.
With m sub-groups only one or two can hold the match, and after `compact()` a sub-group's members
are contiguous, so skipping the rest is easy. Using the sub-group as the scan unit:

| config at 1M | groups | vectors scanned | stage 2 | scan | KB/query | vs centroid |
|---|---|---|---|---|---|---|
| centroid, whole cells | 384 | 50,181 | 26 KB | 3,136 KB | 3,770 | 1.00x |
| `subvar8` beta=1.5 | 1,218 | 22,347 | 216 KB | 1,397 KB | 2,220 | 1.70x |
| `subvar16` beta=2.0 | 2,048 | 20,308 | 432 KB | 1,269 KB | 2,309 | 1.63x |
| `subvar32` beta=2.0 | 2,435 | 12,734 | 864 KB | 796 KB | 2,268 | 1.66x |

Bytes: `M*64` (stage 1) + `L*m*72` (sub-summaries for the shortlist) + `scanned*64` +
`k*rf*384` (rerank). These 1.70x numbers use f32 scoring for the summaries; see 6e for the real
figure.

It plateaus at every m: more sub-groups save scan and cost the same in stage 2. Nothing to tune.

Two accounting mistakes this turned up, both of which had been making earlier estimates look
better:

* The shortlist `L` is set by how far the stage-1 router reaches (384 cells for 0.99 at 1M), not
  by the final probe count. A cell the Hamming scan never shortlists can't be recovered later.
* At rf=32 the rerank is a fixed cost. Once the scan shrinks it's no longer small, and it limits
  what any routing work can give.

So the floor is flat in N (3.78 cells at 99k, 3.49 at 999k), a summary needs eps ~0.04, and
sub-groups help some. None of this closes the 3-5x gap, and none of it is an end-to-end throughput
number yet; it's the byte model.

## 6b. The free one: precision of the stage-2 re-order

The sections above compare summaries in f32, which isolates the estimator but isn't what the
engine runs. With `set_two_stage(True, 2)` the router re-orders `nprobe*mult` centroids exactly in
f32, 1,536 B each. At 1M with nprobe=512 that's 1,536 KB per query, a quarter of the whole budget,
spent at full precision on a step that only orders cells and whose order already has ~0.05 of
structured error.

999k, target R@10 0.99, same partition and queries:

| re-order | B/cell | np for 0.99 | scan | stage 2 | KB/query | vs f32 |
|---|---|---|---|---|---|---|
| f32 (before) | 1536 | 512 | 66,555 | 1,536 KB | 6,303 | 1.00x |
| f16 | 768 | 512 | 66,555 | 768 KB | 5,535 | 1.14x |
| sq8 | 384 | 512 | 66,555 | 384 KB | 5,151 | 1.22x |
| sq6 | 288 | 512 | 66,556 | 288 KB | 5,056 | 1.25x |
| sq4 | 192 | 512 | 66,552 | 192 KB | 4,959 | 1.27x |

The probe count is the same at every precision down to sq4, and the scan differs by 3 vectors in
66,555. Nothing gets worse, latency drops with the bytes, and memory goes down (the f32 centroid
array is 12 MB at 1M, 3 MB as SQ8).

The same sweep answers `shortlist_mult`: mult=1 needs 768 probes against 512 for mult=2, so the
shortlist is doing real work and stays at 2.

The coarse stage in the lab is SimHash (random rotation + signs), not RaBitQ, which is why it
needs 512 probes here vs 384 for exact centroid ranking. RaBitQ is better, so this understates the
real router; the conclusion about precision holds across an 8x range on the same coarse stage.

### On the engine

`cent_sq8` + `set_centroid_sq8(bool)` in `index_v2.rs`, default on. It's derived state, rebuilt in
`finish_load`, so the file format doesn't change.

`bench/ab_centroid_sq8.py` builds one index at 999k and flips the flag on the live index, so the
partition, layout and queries are identical between the two arms:

| nprobe | R@10 f32 | R@10 sq8 | delta | qps f32 | qps sq8 | speedup | row overlap |
|---|---|---|---|---|---|---|---|
| 64 | 0.9301 | 0.9301 | 0 | 32,132 | 36,051 | 1.122x | 0.9995 |
| 128 | 0.9580 | 0.9580 | 0 | 17,530 | 20,330 | 1.160x | 0.9996 |
| 256 | 0.9760 | 0.9761 | +0.0001 | 8,180 | 10,447 | 1.277x | 0.9998 |
| 512 | 0.9832 | 0.9832 | 0 | 3,409 | 3,938 | 1.155x | 0.9996 |
| 1024 | 0.9877 | 0.9876 | -0.0001 | 1,584 | 1,782 | 1.125x | 0.9999 |

1.12-1.28x across the range with |recall delta| <= 1e-4. The lab predicted 1.22x in bytes and the
engine gave 1.155x in throughput at the same nprobe, about what you'd expect when bytes are the
main cost but not the only one.

Row overlap is 0.9995-0.9999, not 1.0000, because quantization sometimes swaps two cells with
nearly equal scores, which changes probe order and can swap ties in the final list. Recall doesn't
move because the rerank decides the answer. Check [7] in the tests ("two-stage routing must not
change results") still passes at 1.0000, and all 30 pass.

### What it adds up to

| step | scan | total | cumulative | cost |
|---|---|---|---|---|
| before | 4,160 KB | 6,303 KB | 1.00x | |
| + sq8 re-order | 4,160 KB | 5,151 KB | 1.22x | none, 9 MB less memory |
| + tree as stage 1 (already built) | 4,160 KB | 4,794 KB | 1.31x | none |
| + sub-group scan | ~2,450 KB | ~3,082 KB | ~2.0x | < 1% memory |

(The last row used the f32-scored sub-groups; with the correction in 6e it's smaller.)

Sub-group summaries at m=8 are 8 x (64+8) = 576 B per cell against 128 x 464 B = 59,392 B of data,
0.97%. The recall ceiling doesn't change, since probing every sub-group scans the same as probing
the cell.

## 6c. Does it hold on another corpus?

Everything above is MS MARCO. I re-ran it all on HotpotQA at 1M (`hotpot_1M.npy`, built from
`hotpot_bench_data/`), which is harder: floor 5.21 cells against 3.49, and the centroid router
needs 1,536 probes for 0.99 against 512.

| | MS MARCO 999k | HotpotQA 999k |
|---|---|---|
| floor | 3.49 | 5.21 |
| centroid router, np for 0.99 | 512 | 1,536 |
| effective eps, centroid | 0.0523 | 0.0494 |
| effective eps, `subc32` | 0.0436 | 0.0436 |
| `gamma` (scalar radius) | fails, 1.03x | fails, 1.03x |
| `box` (element-wise max) | fails, 8x worse | fails, misses the target |
| sq8 re-order | 1.22x | 1.24x |
| sub-group scan (f32-scored) | 1.70x | 2.27x |

All the negatives come back, the eps ordering comes back, and both positives come back, the
sub-group one bigger on the harder corpus because the scan is a larger share there. The
structured-error penalty also grows: on HotpotQA eps=0.049 needs 1,536 probes where random noise
of that size would need ~100, about 15x against ~4x on MS MARCO.

Still untested: both corpora are 384-d from the same embedder family, and both use held-out
corpus vectors as queries. Real queries are questions, and routing is where that would show. I
don't have question embeddings in this repo, so it stays open. Also untested: other dimensions,
k other than 10.

## 6d. Against faiss, 999k / 1,000 queries / k=100

Batch qps, 16 threads, best of 7, exact ground truth. Tree routing on, the extended probe grid,
`centroid_sq8` on. faiss numbers from the same machine, data and day.

### Recall ceiling

| system | max R@10 | batch qps there | build | memory | delete |
|---|---|---|---|---|---|
| v2, accurate (binary + f32 rerank) | 0.9999 | 333 | 66.8 s | 1,664 MB | 1.58 us |
| v2, small (binary + SQ8 rerank) | 0.9939 | 337 | 74.7 s | 562 MB | 1.77 us |
| faiss-hnsw | 0.9993 | 1,542 | 298.3 s | 1,742 MB | none |
| faiss-ivf | 0.9935 | 221 | 9.3 s | 1,520 MB | O(N) |
| faiss-ivf-sq | 0.9935 | 56 | 7.9 s | 489 MB | O(N) |

faiss-hnsw doesn't reach 0.9999 at any efSearch I tried (up to 512). The accurate config passes
HNSW's ceiling at np=2048 (0.9994) and goes on to 0.9999.

### Throughput at matched recall

| R@10 | v2 accurate | faiss-hnsw | gap |
|---|---|---|---|
| ~0.955 | 5,123 (np=64) | 29,345 (ef=16) | 5.7x |
| ~0.983 | 3,626 (np=256) | 18,320 (ef=32) | 5.1x |
| ~0.993 | 2,774 (np=512) | 10,882 (ef=64) | 3.9x |
| ~0.9976 | 1,900 (np=1024) | 5,898 (ef=128) | 3.1x |
| ~0.9993 | 1,138 (np=2048) | 1,542 (ef=512) | 1.35x |

The small config is faster below its 0.9939 ceiling (10,495 qps at R@10 0.951, 2.8x behind HNSW)
because it scans 464 B/vector instead of 1,616.

The gap depends a lot on recall: 5.7x at 0.955, 1.35x at 0.9993. HNSW slows down faster as recall
goes up, because reaching the tail makes its beam wide, while my cost stays linear in probes. "3-5x
behind" is right in the middle of the range and misleading at both ends.

From np=512 to 1024 costs 1.66x throughput for +0.0044 recall (small config), and 1024 to 7804
costs 6.8x for +0.0022. Past ~512 the small config gains almost nothing, which is where its 0.9939
ceiling comes from: the SQ8 rerank is the limit there, not routing.

## 6e. Correction: sub-groups are worth 1.25x, not 1.70x

The sub-group numbers in 5 and 6c (1.70x and 2.27x) used an inconsistent byte model: the
summaries were charged 72 B, the price of a 1-bit code, but scored in f32. Scoring them from the
codes they'd actually be stored as:

| sub-centroid scoring | scan | KB/query | gain |
|---|---|---|---|
| f32 (1,536 B, not affordable) | 6,960 | 1,619 | 1.70x |
| 1-bit (64 B, what I'd build) | 16,302 | 2,203 | 1.25x |

Storing them as SQ8 doesn't help: stage 2 becomes `L*m*388 B` = 3.2 MB at L=512, m=16, and the
total goes above the baseline. Sub-group pruning is a real but small win, and it touches
`compact()`, delete and persistence.

## 6f. A better lever: a cheaper first pass

After the SQ8 change the scan is ~77% of query bytes at 64 B/vector. Sub-groups go after it by
scanning fewer vectors; a cascade scans the same vectors with fewer bytes and promotes a
shortlist to the full code:

```text
  b-bit prefilter over every probed vector  -> top T
  64 B binary over those T                  -> top k*rf
  SQ8 rerank                                -> top k
```

999k, nprobe=256, 1,000 queries (`bench/scan_cascade.py`). Recall if every probed vector were
scored exactly is 0.9887, so the losses below are against that:

| first-pass code | B/vector | T | recall | KB/query | vs now |
|---|---|---|---|---|---|
| now: 64 B over everything probed | 64 | | 0.9887 | 2,228 | 1.00x |
| 128-bit | 16 | 4096 | 0.9762 | 903 | 2.47x |
| 192-bit | 24 | 8192 | 0.9878 (-0.0009) | 1,423 | 1.57x |
| 192-bit | 24 | 16384 | 0.9887 (0) | 1,935 | 1.15x |
| 256-bit | 32 | 2048 | 0.9865 (-0.0022) | 1,302 | 1.71x |
| 256-bit | 32 | 4096 | 0.9878 (-0.0009) | 1,430 | 1.56x |

1.57x at -0.0009, or 1.15x with no loss at all, against 1.25x at -0.004 for sub-groups. And it's
much less invasive: one extra arena read in the same loop order, no change to the partition, so
`compact()`, delete, split/merge and save/load are untouched. It costs +24 B/vector, ~5% of 464 B,
so it's the first lever that isn't free on memory.

The prefilter here is SimHash while the real 64 B code is RaBitQ, which is better at the same bit
count, so the 64 B row is simulated at b=384 to compare like with like. A RaBitQ prefilter should
do better than this.

## 7. Where the bytes are

With the summary doing its best at 1M: stage 1 488 KB, stage 2 216 KB, scan 1,397 KB, rerank
120 KB. The scan still dominates, and now it's bytes per vector that matter, not vectors scanned.
So the next target is compressing the scan, not routing, and unlike routing it doesn't have to
solve the search problem to work.

The other untested idea is replication. The floor of 3.49 cells exists because neighbours
straddle boundaries, and putting boundary vectors in two cells lowers the floor itself instead of
estimating around it. `Partition` in the lab already supports multiple assignment; I haven't run
it yet.

## 8. Reproducing

```bash
# the target table and the candidate table with effective eps
KMP_DUPLICATE_LIB_OK=TRUE python bench/summariser_lab.py \
    msmarco_embeddings_cache_v2_1000000.npy --queries 1000 --cell-size 128 \
    --variants mean,oracle,noise,gamma,subc,subvar --subs 8,32 --eps 0.02,0.04,0.08

# sub-groups as the scan unit
KMP_DUPLICATE_LIB_OK=TRUE python bench/summariser_lab.py \
    msmarco_embeddings_cache_v2_1000000.npy --queries 1000 --cell-size 128 \
    --hier --subs 8,16,32 --betas 1.5,2.0,2.5 --target 0.99
```

Partition, oracle and ground truth are cached in `bench/_lab_cache/` and `bench/_gt_cache/`. The
first 1M run trains k-means (~40 s on a 3050), the rest take seconds.
