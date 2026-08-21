# Engine v3: what changed and what tests cover it

Every change here either keeps results exactly the same (and says so, with an equivalence test) or
changes them on purpose (and says so, measured against exact ground truth). `tests/test_index.py`
has to pass in full.

## 1. Build

| change | before -> after (MS MARCO 1M, memory profile) | results |
|---|---|---|
| bulk ingest encodes in parallel straight into the arena (`append_parallel`); placement and bookkeeping stay serial, in input order | encode + place 6.2 s -> 0.56 s | identical index (same slots, codes, member order, centroid sums) |
| per-dimension SQ8 range: parallel min/max | 302 ms -> 42 ms | identical (min/max are exact) |
| radii: parallel over cells, SIMD dots, `select_nth` for the p99 | 510 ms -> 23 ms | identical |
| Hadamard rotation: levels 1/2/4 in registers, h >= 8 as contiguous halves | | bit-identical (same butterflies, same order) |
| fixed-width 512-bit Hamming | | identical |
| full build | 12.06 s -> 4.8 s | |

## 2. Query path

| change | effect | results |
|---|---|---|
| top-k with an integer bucket queue over Hamming distances (histogram + falling threshold); ties at the cut-off are all kept and the exact rerank decides | the scan no longer sorts | deterministic, never worse |
| adaptive bounds O(nprobe^2) -> O(nprobe) | 289 us -> 23 us at nprobe 1024 | identical |
| stage-2 centroid rescore through the AVX2 u8 x f32 kernel | | same arithmetic, SIMD summation order |
| compacted cells stream their contiguous code run (`compacted` flag, cleared by any mutation) | | identical |
| tree selection by integer counting instead of a float `select_nth` | | same top set, ties in encounter order |
| software prefetch for rescore rows, rerank rows (SQ8 and f32) and the next cell's run | rerank 46 -> 28 us | identical |
| shortlist cut-off keeps every tied cell (both routers) | | deterministic |
| batch scratch reused per worker (`map_init`) instead of per query | | identical |
| multi-home dedup per cell (M stamps) instead of per slot (N x 4 B zeroed per query) | removed a memset wall at ~8k QPS | identical |

## 3. Cell-major batch execution (`search_batch(..., cell_major=True)`, the default)

1. **Route** every query. For batches the router is `route_batch`: exact flat Hamming ranking of
   all queries against all cell codes in cache-blocked tiles (32 queries' codes stay in L1 while
   the centroid codes stream through once per tile), counting selection for the shortlist, exact
   SQ8 rescore, and the same nucleus rule as `route_query`.
2. **Seed** each query by scanning its 8 best cells, so it has a valid cut-off before the join.
3. **Join**: invert the probe lists to cell -> queries. Each worker streams a contiguous range of
   cells once and scores every query that wants that cell while its codes are in L1. Workers share
   a per-query `AtomicU32` cut-off: any worker's k-th best is an upper bound on the global k-th
   best, so rejecting above it can't drop a winner.
4. **Merge** per query, exact cut-off (ties kept), exact rerank.

Test `[11]`: identical results to the per-query path (overlap 1.0000 on real MS MARCO, with and
without two-stage routing).

## 4. Orthogonal polysemy (`set_orthogonal_polysemy(homes, lam)`)

A vector's extra homes minimise `||r'||^2 + lam * sum_h (r'.r_h)^2` over normalised residuals
`r_h` of the homes already chosen: close to the vector, with a residual orthogonal to those homes.
That makes the homes' misses independent (measured: the joint miss equals the product of the
marginals; next-nearest homes are 3-6x correlated).

Fast path (polysemic runs): after `compact()` every cell is `[primary run | replica run]`, and the
replicas' 64-byte codes are copied into `rep_bits` so both runs stream. Each vector is scored once:

* per query: at its first home in probe order (valid with early termination, since cells are
  scanned in probe order);
* cell-major: at its lowest-id probed home (every probed cell is scanned in full), using a
  per-query probe bitset (M bits).

Tests `[10]`, `[12]`: actually on, no duplicate ids at any nprobe, exhaustive equals brute force,
adaptive still exact, deletes unlink every home; streaming and cell-major match the member-list
path exactly (R = 2, 3).

## 5. Asymmetric estimation

* `quantize::AsymQuery`: 4-bit query bit-planes against 1-bit codes. Inside the same probed cells
  it lifts R@10 after rerank from 0.934 (symmetric Hamming) to 0.994. As a cascade after Hamming it
  costs more than it saves (`set_asym_cascade`, off by default, not used in any reported config).
* `quantize::FsQuery` + FastScan blocks (`set_fastscan(True)`): the asymmetric estimate as the
  first stage, at table-lookup speed. Codes are interleaved in 32-vector blocks by nibble, and one
  `pshufb` per half scores 32 vectors at two positions. Test `[14]`.

## 6. Measurement

* `profile(on)` / `profile_read()` (module level): opt-in phase timers, one relaxed atomic load
  per phase when off.
* `stats()["total_bytes"]`: every allocation by capacity (arena, member lists, id map, per-cell
  vectors, routing structures, replica and FastScan blocks). The old `arena_bytes_per_vec` missed
  ~25-60 B/vector.
* `bench/bench_vs_faiss.py`: same data, queries and exact ground truth for both sides; full build
  time; Pareto frontier over each system's own parameters; batch QPS best of 3 on all threads;
  single-thread p50/p99 including Python overhead for both; matched-recall interpolation, never
  extrapolated.
