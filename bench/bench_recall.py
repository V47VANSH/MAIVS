"""run_ann_recall.py: recall-vs-QPS for this index against faiss HNSW and faiss IVFFlat.

Recall is measured against exact ground truth (bench/groundtruth.py), so it has a hard ceiling of
1.0 and doesn't depend on the embedding model. That's the number to report for an index, not a
retrieval metric like qrel hits.

Usage:
    python bench/run_ann_recall.py msmarco_embeddings_cache_v2_100000.npy
    python bench/run_ann_recall.py <corpus.npy> --limit 1000000 --queries 1000

Prints a table and writes bench/results/<tag>.json with every (system, params, recall@10,
recall@100, qps, build_s, mem_mb) point.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import psutil

# Same thread count for both sides. faiss uses OpenMP and we use rayon; they happened to agree on
# this machine, but that shouldn't be left to luck. rayon reads RAYON_NUM_THREADS when its pool is
# first built, so this has to run before the extension is imported.
_THREADS = int(os.environ.get("BENCH_THREADS", os.cpu_count() or 8))
os.environ.setdefault("RAYON_NUM_THREADS", str(_THREADS))

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))
# Repo root first on sys.path. Running the file by path leaves it off, and the import would then
# pick up an old copy from site-packages instead of the fresh build.
sys.path.insert(0, _ROOT)

from groundtruth import load_or_build, recall_at_k, mean_reciprocal_rank  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def _timed_queries(fn, queries: np.ndarray, k: int):
    """Run fn(q) per query, returning (retrieved array, latencies ms)."""
    out = np.full((len(queries), k), -1, dtype=np.int64)
    lat = np.empty(len(queries), dtype=np.float64)
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        ids = fn(q)
        lat[i] = (time.perf_counter() - t0) * 1000.0
        n = min(len(ids), k)
        if n:
            out[i, :n] = ids[:n]
    return out, lat


def _timed_batch(fn, queries: np.ndarray, k: int, repeats: int = 7):
    """Batch throughput, best of N, with the run-to-run spread.

    At repeats=3 I saw up to 35% movement on an idle machine, enough to flip the order between
    systems, so the spread is printed too. Treat gaps smaller than the spread as ties.
    """
    times = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(queries)
        times.append(time.perf_counter() - t0)
    times.sort()
    best = times[0]
    med = times[len(times) // 2]
    spread = (times[-1] - times[0]) / med * 100.0 if med > 0 else 0.0
    return out, len(queries) / best, spread


def _record(rows: List[Dict], system: str, params: dict, retrieved,
            lat, gt_ids, build_s: float, mem_mb: float,
            batch_qps: float = 0.0, exact_mb: float = 0.0,
            spread: float = 0.0) -> None:
    r10 = recall_at_k(retrieved, gt_ids, 10)
    r100 = recall_at_k(retrieved, gt_ids, min(100, gt_ids.shape[1]))
    mrr = mean_reciprocal_rank(retrieved, gt_ids)
    row = {
        "system": system, "params": params,
        "recall@10": round(r10, 4), "recall@100": round(r100, 4),
        "mrr": round(mrr, 4),
        "qps": round(1000.0 / float(np.mean(lat)), 1),
        "batch_qps": round(batch_qps, 1),
        "lat_mean_ms": round(float(np.mean(lat)), 3),
        "lat_p99_ms": round(float(np.percentile(lat, 99)), 3),
        "build_s": round(build_s, 2), "mem_mb": round(mem_mb, 1),
        "exact_mb": round(exact_mb, 1), "threads": _THREADS,
        "batch_spread_pct": round(spread, 1),
    }
    rows.append(row)
    pstr = ",".join(f"{k}={v}" for k, v in params.items())
    print(f"  {system:14s} {pstr:32s} R@10={r10:.4f}  "
          f"serial={row['qps']:>7.0f}qps  batch={row['batch_qps']:>8.0f}qps  "
          f"p99={row['lat_p99_ms']:.2f}ms  +-{row['batch_spread_pct']:.0f}%")


# --- systems ---

def bench_faiss_hnsw(base, queries, gt_ids, k, rows) -> None:
    import faiss
    faiss.omp_set_num_threads(_THREADS)
    gc.collect()
    m0 = _rss_mb()
    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(base.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(base)
    build_s = time.perf_counter() - t0
    mem = _rss_mb() - m0
    print(f"\n[faiss-hnsw] build {build_s:.1f}s  mem {mem:.0f}MB")
    for ef in (16, 32, 64, 128, 256, 512):
        index.hnsw.efSearch = ef
        ret, lat = _timed_queries(
            lambda q: index.search(q.reshape(1, -1), k)[1][0], queries, k)
        bret, bqps, spr = _timed_batch(lambda qq: index.search(qq, k)[1], queries, k)
        _record(rows, "faiss-hnsw", {"efSearch": ef}, bret, lat, gt_ids,
                build_s, mem, batch_qps=bqps, spread=spr)
    del index
    gc.collect()


def bench_faiss_ivf(base, queries, gt_ids, k, rows) -> None:
    import faiss
    faiss.omp_set_num_threads(_THREADS)
    gc.collect()
    m0 = _rss_mb()
    nlist = max(16, int(np.sqrt(len(base))))
    t0 = time.perf_counter()
    quant = faiss.IndexFlatIP(base.shape[1])
    index = faiss.IndexIVFFlat(quant, base.shape[1], nlist,
                               faiss.METRIC_INNER_PRODUCT)
    index.train(base)
    index.add(base)
    build_s = time.perf_counter() - t0
    mem = _rss_mb() - m0
    print(f"\n[faiss-ivf] nlist={nlist} build {build_s:.1f}s  mem {mem:.0f}MB")
    for nprobe in (1, 4, 8, 16, 32, 64, 128):
        index.nprobe = nprobe
        ret, lat = _timed_queries(
            lambda q: index.search(q.reshape(1, -1), k)[1][0], queries, k)
        bret, bqps, spr = _timed_batch(lambda qq: index.search(qq, k)[1], queries, k)
        _record(rows, "faiss-ivf", {"nlist": nlist, "nprobe": nprobe},
                bret, lat, gt_ids, build_s, mem, batch_qps=bqps, spread=spr)
    del index
    gc.collect()


def bench_faiss_hnsw_sq(base, queries, gt_ids, k, rows) -> None:
    """HNSW + scalar quantization, i.e. faiss's compressed graph.

    The question is how much compression buys each design. A flat scan is bandwidth bound, so fewer
    bytes per vector means roughly proportionally more throughput. HNSW is latency bound: each hop
    is a dependent load, so smaller nodes don't reduce the number of round trips.
    """
    import faiss
    faiss.omp_set_num_threads(_THREADS)
    gc.collect(); m0 = _rss_mb(); t0 = time.perf_counter()
    index = faiss.IndexHNSWSQ(base.shape[1], faiss.ScalarQuantizer.QT_8bit, 32,
                              faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.train(base)
    index.add(base)
    build_s = time.perf_counter() - t0
    mem = _rss_mb() - m0
    print(f"\n[faiss-hnsw-sq] build {build_s:.1f}s  mem {mem:.0f}MB")
    for ef in (16, 32, 64, 128, 256):
        index.hnsw.efSearch = ef
        ret, lat = _timed_queries(
            lambda q: index.search(q.reshape(1, -1), k)[1][0], queries, k)
        bret, bqps, spr = _timed_batch(lambda qq: index.search(qq, k)[1], queries, k)
        _record(rows, "faiss-hnsw-sq", {"efSearch": ef}, bret, lat, gt_ids,
                build_s, mem, batch_qps=bqps, spread=spr)
    del index; gc.collect()


def bench_faiss_ivf_sq(base, queries, gt_ids, k, rows) -> None:
    """IVF + 8-bit scalar quantization (matched-precision IVF baseline)."""
    import faiss
    faiss.omp_set_num_threads(_THREADS)
    d = base.shape[1]
    nlist = max(16, int(np.sqrt(len(base))))
    gc.collect(); m0 = _rss_mb(); t0 = time.perf_counter()
    quant = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFScalarQuantizer(
        quant, d, nlist, faiss.ScalarQuantizer.QT_8bit,
        faiss.METRIC_INNER_PRODUCT)
    index.train(base)
    index.add(base)
    build_s = time.perf_counter() - t0
    mem = _rss_mb() - m0
    print("")
    print(f"[faiss-ivf-sq] nlist={nlist} build {build_s:.1f}s mem {mem:.0f}MB")
    for nprobe in (8, 16, 32, 64, 128, 256, 314):
        index.nprobe = min(nprobe, nlist)
        ret, lat = _timed_queries(
            lambda q: index.search(q.reshape(1, -1), k)[1][0], queries, k)
        bret, bqps, spr = _timed_batch(lambda qq: index.search(qq, k)[1], queries, k)
        _record(rows, "faiss-ivf-sq", {"nlist": nlist, "nprobe": nprobe},
                bret, lat, gt_ids, build_s, mem, batch_qps=bqps, spread=spr)
    del index; gc.collect()


def bench_faiss_ivfpq(base, queries, gt_ids, k, rows) -> None:
    """IVFPQ with exact refinement, faiss's closest equivalent to our quantized scan + rerank.
    """
    import faiss
    faiss.omp_set_num_threads(_THREADS)
    d = base.shape[1]
    m = 48 if d % 48 == 0 else (32 if d % 32 == 0 else 16)
    nlist = max(16, int(np.sqrt(len(base))))
    gc.collect(); m0 = _rss_mb(); t0 = time.perf_counter()
    quant = faiss.IndexFlatIP(d)
    ivfpq = faiss.IndexIVFPQ(quant, d, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
    ivfpq.train(base)
    # Build the refine wrapper before adding vectors: it checks that the base and refine indexes
    # have the same ntotal, so vectors must go through the wrapper.
    index = faiss.IndexRefineFlat(ivfpq)
    index.k_factor = 8          # same candidate multiplier as our rerank_factor
    index.add(base)
    build_s = time.perf_counter() - t0
    mem = _rss_mb() - m0
    print(f"\n[faiss-ivfpq+refine] nlist={nlist} m={m} "
          f"build {build_s:.1f}s mem {mem:.0f}MB")
    for nprobe in (8, 16, 32, 64, 128):
        ivfpq.nprobe = nprobe
        ret, lat = _timed_queries(
            lambda q: index.search(q.reshape(1, -1), k)[1][0], queries, k)
        bret, bqps, spr = _timed_batch(lambda qq: index.search(qq, k)[1], queries, k)
        _record(rows, "faiss-ivfpq", {"nlist": nlist, "m": m, "nprobe": nprobe},
                bret, lat, gt_ids, build_s, mem, batch_qps=bqps, spread=spr)
    del index; gc.collect()


def bench_v2(base, queries, gt_ids, k, rows, scan_mode: str,
             n_cells: int = 0, rerank_factor: int = 4,
             max_assign: int = 1, spill_margin: float = 0.05,
             i8_rerank: bool = False, f16_rerank: bool = False,
             per_dim_sq: bool = False,
             adaptive: bool = False,
             at_margin: float = 0.02, code_bits: int = 0, graph_min: int = 0,
             two_stage: bool = False, compact: bool = True,
             shortlist_mult: int = 4,
             tree_route: bool = False, tree_min: int = 2048,
             tree_fanout: int = 16) -> None:
    """The flat-cell index.

    scan_mode="f32" is exact inside each scanned cell, so any recall loss is routing loss.
    scan_mode="i8" adds a global-scale int8 scan plus an f32 rerank of the top k*rerank_factor.
    Running both separates the cost of quantization from the cost of partitioning.
    """
    import faiss
    import aigvs_rust
    faiss.omp_set_num_threads(_THREADS)

    d = base.shape[1]
    if n_cells <= 0:
        n_cells = max(16, int(np.sqrt(len(base))))

    gc.collect()
    m0 = _rss_mb()
    t0 = time.perf_counter()

    # Centroids are trained in-process with two-level hierarchical k-means (src/tree.rs). Flat
    # k-means in faiss took 54.5 s of a 67.9 s build at 1M; this takes 2.5 s, and it keeps faiss out
    # of our build time.
    ix = aigvs_rust.FlatCellIndex(d, scan_mode, rerank_factor, 0x5EED, code_bits)
    if max_assign > 1:
        ix.set_replication(max_assign, spill_margin)
    if f16_rerank:
        ix.set_f16_rerank(True)
    elif per_dim_sq:
        ix.set_per_dim_sq(True)
    elif i8_rerank:
        ix.set_i8_rerank(True)
    if adaptive:
        ix.set_adaptive(True, at_margin)
    if two_stage:
        ix.set_two_stage(True, shortlist_mult)
    if graph_min:
        ix.set_graph_route(True, graph_min)
    ix.train_centroids(np.ascontiguousarray(base), n_cells)
    ids = np.arange(len(base), dtype=np.uint64)
    chunk = 200_000
    for s in range(0, len(base), chunk):
        e = min(s + chunk, len(base))
        ix.add_batch(ids[s:e].tolist(), np.ascontiguousarray(base[s:e]))
    # Cluster-major layout: slots are handed out in insertion order, so a cell's members end up
    # scattered and the scan becomes a random gather. compact() makes each cell one contiguous run.
    # Counted as part of the build.
    if compact:
        ix.compact()
    build_s = time.perf_counter() - t0
    mem = _rss_mb() - m0

    st = ix.stats()
    name = f"v2-{scan_mode}"
    print(f"\n[{name}] cells={n_cells} build {build_s:.1f}s mem {mem:.0f}MB  "
          f"cell_size mean/min/max={st['cell_size_mean']:.0f}/"
          f"{st['cell_size_min']:.0f}/{st['cell_size_max']:.0f}  "
          f"arena={st['arena_bytes_per_vec']:.0f} B/vec")

    qc = np.ascontiguousarray(queries)
    if adaptive:
        # The old grid jumped from 256 straight to n_cells, which skipped the whole high-recall
        # range (at 1M, everything between R@10 0.98 and the exhaustive 0.994). That's the range
        # where the faiss comparison is decided.
        probe_grid = tuple(p for p in
                           (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
                           if p < n_cells) + (n_cells,)
    elif n_cells > 2000:
        # Smaller cells need more probes for the same recall; compare vectors scanned (nprobe *
        # cell_size), not nprobe.
        probe_grid = (8, 16, 32, 64, 128, 256, 512)
    else:
        probe_grid = (1, 4, 8, 16, 32, 64, 128)
    for nprobe in probe_grid:
        ret, lat = _timed_queries(
            lambda q: ix.search(q, k, nprobe, 1.0, 8.0)[0], queries, k)

        def _b(qq, _np=nprobe):
            ids, _ = ix.search_batch(qq, k, _np, 1.0, 8.0)
            # Fewer than k results is fine: with small cells a low nprobe may not reach k
            # candidates. Missing slots come back as u64::MAX, which doesn't fit in int64, so map
            # them to -1.
            a = np.asarray(ids, dtype=np.uint64).reshape(len(qq), k)
            out = np.full(a.shape, -1, dtype=np.int64)
            live = a != np.uint64(2**64-1)
            out[live] = a[live].astype(np.int64)
            return out
        bret, bqps, spr = _timed_batch(_b, qc, k)
        _record(rows, name, {"cells": n_cells, "nprobe": nprobe},
                bret, lat, gt_ids, build_s, mem, batch_qps=bqps, spread=spr,
                exact_mb=st["arena_bytes_per_vec"] * len(base) / 1e6)

    # delete cost check
    n_del = min(2000, len(base) // 10)
    t0 = time.perf_counter()
    for i in range(n_del):
        ix.remove(i)
    del_us = (time.perf_counter() - t0) / n_del * 1e6
    print(f"[{name}] delete: {del_us:.2f} us/op over {n_del} ops "
          f"(n_live {len(ix):,}, free_slots {ix.stats()['free_slots']:.0f})")

    del ix
    gc.collect()


def bench_aigvs(base, queries, gt_ids, k, rows) -> None:
    from aigvs.pipeline import AIGVSPipeline
    from aigvs.storage import InMemoryStorage

    gc.collect()
    m0 = _rss_mb()
    pipe = AIGVSPipeline(embedding_dim=base.shape[1],
                         ef_construction=32, max_connections=64)
    pipe.storage = InMemoryStorage()

    ids = [str(i) for i in range(len(base))]
    texts = [""] * len(base)
    t0 = time.perf_counter()
    pipe.bulk_ingest_arrays(ids, texts, base)
    build_s = time.perf_counter() - t0
    mem = _rss_mb() - m0
    st = pipe.stats()
    print(f"\n[aigvs] build {build_s:.1f}s  mem {mem:.0f}MB  "
          f"branches={st['branches']} leaves={st['leaves']} "
          f"indexed={st['indexed_chunks']:,}  "
          f"(duplication {st['indexed_chunks']/len(base):.2f}x)")

    def run(q, ef, conf, beam, beta):
        res = pipe.query(q, top_k=k, ef_search=ef, target_confidence=conf,
                         max_beam=beam, beta=beta)
        return [int(r["id"]) for r in res]

    grid = [
        # (ef_search, target_confidence, max_beam, beta)
        (64, 0.80, 4, 10.0),
        (128, 0.90, 8, 10.0),
        (256, 0.95, 16, 10.0),
        (512, 0.98, 32, 5.0),
        (512, 0.995, 64, 2.0),
    ]
    for ef, conf, beam, beta in grid:
        ret, lat = _timed_queries(
            lambda q: run(q, ef, conf, beam, beta), queries, k)
        _record(rows, "aigvs",
                {"ef": ef, "conf": conf, "beam": beam, "beta": beta},
                ret, lat, gt_ids, build_s, mem)
    del pipe
    gc.collect()



def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("corpus")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--query-path", default=None)
    p.add_argument("--systems", default="v2-f32,v2-i8,aigvs,faiss-hnsw,faiss-ivf")
    p.add_argument("--graph-min", type=int, default=0,
                   help="force centroid-graph routing at this cell count")
    p.add_argument("--code-bits", type=int, default=0,
                   help="binary code length; 0 = full rotated length")
    p.add_argument("--two-stage", action="store_true",
                   help="binary centroid shortlist then exact re-order")
    p.add_argument("--adaptive", action="store_true",
                   help="adaptive probing: nprobe becomes an upper bound")
    p.add_argument("--at-margin", type=float, default=0.02,
                   help="safety margin on the early-stop bound")
    p.add_argument("--per-dim-sq", action="store_true",
                   help="per-dimension affine SQ8 rerank (faiss QT_8bit style)")
    p.add_argument("--f16-rerank", action="store_true",
                   help="rerank against an f16 copy (2 B/dim)")
    p.add_argument("--i8-rerank", action="store_true",
                   help="rerank against the int8 copy instead of f32")
    p.add_argument("--assign", type=int, default=1,
                   help="max cells per vector (redundant assignment)")
    p.add_argument("--margin", type=float, default=0.05,
                   help="spill margin: replicate only within this cosine gap")
    p.add_argument("--rerank", type=int, default=4,
                   help="candidate multiplier for quantized scan modes")
    p.add_argument("--no-compact", action="store_true",
                   help="skip cluster-major arena compaction (A/B the layout)")
    p.add_argument("--tree", action="store_true",
                   help="hierarchical routing; also routes ingestion, which "
                        "is what takes build off the O(N*M) assignment path")
    p.add_argument("--tree-min", type=int, default=2048)
    p.add_argument("--tree-fanout", type=int, default=16)
    p.add_argument("--cells", type=int, default=0,
                   help="partition count for v2; 0 = sqrt(N)")
    args = p.parse_args()

    base, queries, gt_ids, gt_sims, tag = load_or_build(
        args.corpus, k=args.k, n_queries=args.queries,
        query_path=args.query_path, limit=args.limit)

    print(f"\n{'='*78}\nANN RECALL BENCHMARK - {tag}")
    print(f"base={base.shape}  queries={queries.shape}  k={args.k}\n{'='*78}")

    rows: List[Dict] = []
    want = {s.strip() for s in args.systems.split(",")}
    if "v2-f32" in want:
        bench_v2(base, queries, gt_ids, args.k, rows, "f32", args.cells)
    if "v2-i8" in want:
        bench_v2(base, queries, gt_ids, args.k, rows, "i8", args.cells,
                 rerank_factor=args.rerank)
    if "v2-binary" in want:
        bench_v2(base, queries, gt_ids, args.k, rows, "binary", args.cells,
                 rerank_factor=args.rerank, max_assign=args.assign,
                 spill_margin=args.margin, i8_rerank=args.i8_rerank,
                 adaptive=args.adaptive, at_margin=args.at_margin,
                 code_bits=args.code_bits, two_stage=args.two_stage,
                 graph_min=args.graph_min, compact=not args.no_compact,
                 f16_rerank=args.f16_rerank, per_dim_sq=args.per_dim_sq,
                 tree_route=args.tree, tree_min=args.tree_min,
                 tree_fanout=args.tree_fanout)
    if "aigvs" in want:
        bench_aigvs(base, queries, gt_ids, args.k, rows)
    if "faiss-hnsw" in want:
        bench_faiss_hnsw(base, queries, gt_ids, args.k, rows)
    if "faiss-ivf" in want:
        bench_faiss_ivf(base, queries, gt_ids, args.k, rows)
    if "faiss-hnsw-sq" in want:
        bench_faiss_hnsw_sq(base, queries, gt_ids, args.k, rows)
    if "faiss-ivf-sq" in want:
        bench_faiss_ivf_sq(base, queries, gt_ids, args.k, rows)
    if "faiss-ivfpq" in want:
        bench_faiss_ivfpq(base, queries, gt_ids, args.k, rows)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"{tag}.json")
    with open(out, "w") as f:
        json.dump({"tag": tag, "base": list(base.shape),
                   "n_queries": len(queries), "k": args.k, "rows": rows},
                  f, indent=2)

    print(f"\n{'='*78}\nBEST RECALL@10 PER SYSTEM")
    for s in sorted({r["system"] for r in rows}):
        sub = [r for r in rows if r["system"] == s]
        b = max(sub, key=lambda r: r["recall@10"])
        print(f"  {s:14s} R@10={b['recall@10']:.4f}  {b['qps']:>8.1f} qps  "
              f"build={b['build_s']}s  mem={b['mem_mb']}MB  {b['params']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
