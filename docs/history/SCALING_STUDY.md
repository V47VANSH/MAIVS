# Scaling study: HotpotQA, 10k to 5.23M

All figures from `bench/bench_scale_full.py` on this machine against exact ground truth, and
reproducible from `bench/results/scaling_full.json`. The plot is `bench/results/scaling.png`
(`bench/plot_scaling.py`).

**Result: O(log N) doesn't hold up at 5M, and the gap to HNSW widens with N.** Work at fixed recall
grows roughly like N^0.5. The centroid tree is logarithmic (its depth goes 0 -> 1 -> 2 over this
range), but a logarithmic router doesn't make a logarithmic query, and this is the first run that
separates the two.

On the same data, throughput at R@10 0.95 falls with alpha = -0.78 for this index and -0.21 for
faiss-HNSW. I'm 3.6x faster at 50k and 2.15x slower at 2M; they cross between 250k and 1M. So the
"3-5x behind" from the 1M runs is a trend, not a constant.

The cause: the oracle floor is flat in N, so the routing loss grows from ~6x to ~200x over this
range (see "Against faiss-HNSW").

## Setup

| | |
|---|---|
| corpus | HotpotQA, 5,233,329 x 384, shuffled and L2-normalised on disk |
| scales | 10k, 25k, 50k, 100k, 250k, 500k, 1M, 2M, 3.5M, 5.23M |
| queries | the same 1,000 held-out vectors at every scale (end of the file, never in a base) |
| ground truth | exact top-100 by brute force, streamed on the GPU, cached per scale |
| profiles | `memory` (464 B/vector), `quality` (1,616 B/vector) |
| policy | cell size 128, tree routing, `centroid_sq8`, adaptive on |
| threads | all cores; batch is best of 7, serial is 300 queries |

The shuffle matters. The source chunks are in document order: adjacent rows have mean cosine
0.0885 against -0.035 for random pairs. Without shuffling, the small scales would be a narrow slice
of topics and the curve would mostly measure the ordering. After shuffling, adjacent rows are at
0.0407.

## What scales

| N | cells | depth | build s | index MB | max R@10 | delete us |
|---|---|---|---|---|---|---|
| 10,000 | 78 | 0 | 0.1 | 5 | 0.9902 | 0.98 |
| 100,000 | 781 | 0 | 1.5 | 46 | 0.9820 | 1.32 |
| 500,000 | 3,906 | 1 | 6.6 | 232 | 0.9766 | 1.76 |
| 1,000,000 | 7,812 | 2 | 12.6 | 464 | 0.9770 | 1.78 |
| 2,000,000 | 15,625 | 2 | 27.0 | 928 | 0.9770 | 2.04 |
| 3,500,000 | 27,343 | 2 | 52.8 | 1,624 | 0.9716 | 1.91 |
| 5,232,329 | 40,877 | 2 | 94.0 | 2,428 | 0.9727 | 2.00 |

(Memory profile. The quality profile has the same structure at 1,616 B/vector and 3.5x the size;
see the JSON.)

These hold:

* **Memory is exactly linear**: alpha = +1.00, 464 B/vector at every scale.
* **Delete is flat**: 0.98 us at 10k, 2.00 us at 5.2M. 523x the data for 2x the time, and that's
  cache, not the algorithm.
* **Build is linear**: alpha = +1.05, 94 s for 5.2M including training, ingest and compaction.
  Training is only 8.5 s; ingest is 83 s, mostly the O(N*d) `refresh_derived` at the end of every
  `add_batch` (this run used 500k chunks; the default 200k would be ~2.5x slower).
* **Tree depth is logarithmic**: 0 below 2,048 cells, 1 at 3,906, 2 from 7,812 to 40,877.

## O(log N), as I'd stated it, is wrong

Fitting `nprobe ~ N^alpha` at fixed recall over 10k -> 5.23M (2.7 decades):

| target | alpha (memory) | local alpha, low N -> high N | nprobe, 10k -> 5.2M |
|---|---|---|---|
| R@10 >= 0.70 | +0.29 | +0.30 -> +0.19 | 3 -> 15 |
| R@10 >= 0.80 | +0.41 | +0.39 -> +0.37 | 5 -> 58 |
| R@10 >= 0.85 | +0.46 | +0.46 -> +0.40 | 7 -> 113 |
| R@10 >= 0.90 | +0.52 | +0.54 -> +0.43 | 12 -> 261 |
| R@10 >= 0.95 | +0.66 | +0.68 -> +0.56 | 23 -> 1,060 |

0 is O(1), 0.5 is O(sqrt(N)), 1 is linear. At every useful target the exponent is 0.5 or more.
End-to-end p50 at R@10 >= 0.90 fits alpha = +0.50, batch throughput -0.55.

Three separate statements:

1. **"Query cost is O(log N)" is false** at these scales on this corpus: about O(sqrt(N)) at
   R@10 0.90 and worse at 0.95.
2. **"Routing is O(log N)" can still be true**: depth 2, constant beam. But routing isn't the
   part that grows. What grows is the number of cells that have to be probed to find the true
   top-10, and that comes from the data, not the router.
3. **alpha depends on the target**, from 0.29 to 0.66, so a single alpha without its target means
   nothing. The 0.213 from July was at R@10 >= 0.95 on MS MARCO over 124k -> 999k; the same target
   here over a 3x wider range gives 0.66. Treat the July number as specific to that corpus and
   range.

One good sign: the local exponent falls as N grows at every target (0.54 -> 0.43 at 0.90, 0.68 ->
0.56 at 0.95). A power law has a constant local exponent and a logarithm's falls, so the growth is
slower than polynomial, just nowhere near log N over this range.

### A confound that doesn't change the answer

Max R@10 drops with N (0.9902 -> 0.9727 memory, 0.9947 -> 0.9781 quality), because
`rerank_factor` is fixed: k*rf = 320 candidates get rescored whatever the corpus size, while the
number of competing near neighbours grows. A fixed target therefore sits closer to the ceiling at
large N, which pushes alpha up.

That's why the table sweeps the target. At R@10 >= 0.70, well below the ceiling everywhere, alpha
is still +0.29. The confound inflates alpha at high targets but isn't the cause.

It's a finding in itself, though: at a fixed rerank depth the recall ceiling falls with N.
`rerank_factor` should grow with N and right now nothing does that.

## Against faiss-HNSW

`bench/bench_scale_hnsw.py` runs HNSW (M=32, efConstruction=200) on the same scales, queries and
ground truth, and fits alpha the same way. Batch qps at R@10 >= 0.95:

| N | this index (memory) | faiss-hnsw | ratio |
|---|---|---|---|
| 10,000 | 146,720 | 157,699 | 1.07x |
| 50,000 | 72,937 | 20,356 | 0.28x (I'm 3.6x faster) |
| 250,000 | 23,188 | 11,914 | 0.51x (2x faster) |
| 1,000,000 | 7,041 | 10,054 | 1.43x (HNSW faster) |
| 2,000,000 | 4,225 | 9,083 | 2.15x (HNSW faster) |
| alpha(qps) without 10k | -0.781 | -0.214 | |

(10k is left out of the fits: at that size both indexes are in cache and the drop to 50k is the
memory hierarchy, not scaling. HNSW's 10k point is 7x its 50k point for 5x the data.)

My throughput falls about 3.6x faster in the exponent, and the crossover is already behind me,
between 250k and 1M. Extrapolated to 1B at R@10 0.95 that's ~33 qps against ~2,400 for HNSW, a
~70x gap.

### Why

The oracle floor (cells the true top-10 actually sit in) is flat in N: 3.5-5.2 cells at every
scale, since cell size is fixed at 128. But the probes needed for R@10 0.95 go 23 -> 1,060 from 10k
to 5.2M. So the routing loss grows from ~6x to ~200x, and HNSW has nothing like that.

This changes how to read `SUMMARISER_STUDY.md`, which put summaries at ~1.25x. That was at a
fixed 1M. The term it goes after is exactly the one whose growth is the scaling problem. A router
that kept probes near the floor would make alpha ~0 in cells probed and would beat HNSW at every
scale. The 1.25x is a limit on per-cell summaries as a family of fixes, not on how much there is
to gain.

### Caveats

* HNSW only measured to 2M (802 s build, 4.5 GB RSS); I didn't try 5.2M. Extrapolating 500x past
  the last point is a big jump either way.
* At 1B neither fits on one machine: 464 GB for this index, ~1.8 TB for HNSW. Billion scale is a
  sharding/tiering problem anyway, and this comparison is per shard.
* HNSW takes 802 s to build 2M against 27 s here (30x). Build, memory, delete and filtering are
  unaffected by any of this.

## Memory: the quality profile doesn't reach 5.2M

The quality profile ran out of memory at 5,232,329 on a 15.3 GB machine. It asked for
12,884,901,888 bytes, exactly 12 GiB, for an f32 arena that ends up at 7.48 GiB. That's `Vec`
doubling: the arena grows by pushes without a `reserve`, so peak memory during a build can be
nearly 2x the finished index.

So quality tops out around 3.5M here (5,656 MB, 57 s build), and a `Vec::reserve` for the known
final size in `add_batch` would probably make 5.2M fit. Also: a failed allocation in Rust calls
`abort()`, which kills the process instead of raising in Python. The harness's `try/except` around
the build can't catch it, and only the JSON checkpoints saved the other 19 configurations.

## Reproducing

```bash
python bench/make_hotpot_corpus.py --out hotpot_5M.npy      # once
python bench/bench_scale_full.py hotpot_5M.npy
python bench/plot_scaling.py --target 0.90
```
