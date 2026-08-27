"""common.py: shared helpers for the benchmarks in this directory.

Keeps three things consistent across scripts:

  * how an index is built: named profiles, so "memory profile" means the same thing everywhere
  * how time is measured: serial latency and batch throughput are separate passes
  * thread count: faiss uses OpenMP and we use rayon; both are pinned to the same number here,
    before the extension is imported, because rayon reads RAYON_NUM_THREADS once

Centroids are trained in-process by train_centroids (hierarchical k-means in src/tree.rs), so faiss
is not part of our build time.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, Dict, List

import numpy as np

# Must happen before maivs is imported anywhere.
THREADS = int(os.environ.get("BENCH_THREADS", os.cpu_count() or 8))
os.environ.setdefault("RAYON_NUM_THREADS", str(THREADS))

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Vectors per cell.  This is the policy that sets the complexity class:
# sqrt(N) cells means sqrt(N) vectors per cell and an O(sqrt N) scan.
TARGET_CELL_SIZE = 128


# --- profiles --- Each published configuration, defined once. `rerank` is the rerank depth (k *
# rerank candidates rescored); `per_dim_sq` replaces the f32 rerank arena with per-dimension SQ8,
# giving up ~0.6 recall points for 3.5x less memory.
PROFILES: Dict[str, dict] = {
    # 464 B/vec.  The configuration to quote for memory and throughput.
    "memory": dict(scan="binary", rerank=32, per_dim_sq=True, adaptive=True,
                   parallel=False),
    # 1,616 B/vec. Highest recall: R@10 0.9999 at 1M.
    #
    # rerank=128 instead of 32. With exhaustive probing every vector is scored by its 1-bit code and
    # only the top k*rf are rescored exactly, so a neighbour is lost once its Hamming rank exceeds
    # k*rf. That's easy to avoid in the top 320 of 10k, not of 5M, so a fixed rf loses recall as N
    # grows: -0.0082 per decade at rf=32 vs -0.0044 at rf=128. At 1M, rf=128 gives +0.0144 R@10 for
    # 4% throughput at exhaustive probing (RESULTS_v2.md 1.3).
    #
    # The memory profile keeps rf=32 because it runs at low nprobe, where the rerank is a big part
    # of the cost: at nprobe=64 the scan is 524 KB and rf=128 would add 492 KB. Better long term to
    # scale rf with the number of vectors scanned; set_rerank_factor makes that a query-time choice.
    "quality": dict(scan="binary", rerank=128, per_dim_sq=False, adaptive=True,
                    parallel=False),
    # Latency-optimised: intra-query parallel scan, adaptive off (it is
    # sequential by nature and costs serial latency).
    "latency": dict(scan="binary", rerank=32, per_dim_sq=False, adaptive=False,
                    parallel=True),
}


def build_index(base: np.ndarray, profile: str = "memory", cells: int = 0,
                seal_at: int = 0, verbose: bool = True,
                train_cap: int = 0, ingest_chunk: int = 200_000):
    """Build an index from `base` with a named profile.

    Returns (index, timings), with training and ingestion timed separately since they scale
    differently.

    `base` can be memory-mapped. Training reads at most `train_cap` rows and ingestion copies one
    `ingest_chunk` at a time, which matters above ~2M vectors where the f32 corpus and a quality
    index don't fit in RAM together.

    `train_cap` caps the k-means training set (0 = use everything, the default for all published
    numbers). faiss also trains on a subsample, but it changes build time, so any run that sets it
    should say so.
    """
    import maivs

    cfg = PROFILES[profile]
    d = base.shape[1]
    n_cells = cells or max(16, len(base) // TARGET_CELL_SIZE)

    cls = maivs.ConcurrentIndex if seal_at else maivs.FlatCellIndex
    kwargs = {"seal_at": seal_at} if seal_at else {}
    ix = cls(d, cfg["scan"], cfg["rerank"], 0x5EED, 0, **kwargs)

    # Configuration must precede data: scan mode and rerank precision are
    # baked in at ingestion time.
    if cfg["per_dim_sq"]:
        ix.set_per_dim_sq(True)
    ix.set_two_stage(True, 2)
    ix.set_graph_route(False, 10 ** 9)          # graph routing was slower; keep it off
    ix.set_tree_route(True, 2048, 16)
    if cfg["adaptive"]:
        ix.set_adaptive(True, 0.02)
    if cfg.get("parallel"):
        ix.set_parallel_scan(True)

    if train_cap and len(base) > train_cap:
        # Strided, not a random gather. `base` is memory-mapped, and fancy-indexing it with a
        # million scattered rows reads a million random pages from an 8 GB file, far slower than
        # a stride; it was the whole reason a 2M "build" was dominated by I/O. The file is
        # shuffled on disk, so a stride is a uniform sample anyway.
        step = max(1, len(base) // train_cap)
        train = np.ascontiguousarray(base[::step][:train_cap], dtype=np.float32)
    else:
        train = np.ascontiguousarray(base, dtype=np.float32)
    t0 = time.perf_counter()
    ix.train_centroids(train, n_cells)
    t_train = time.perf_counter() - t0
    del train                                    # 8 GB at the top scale

    ids = np.arange(len(base), dtype=np.uint64)
    t0 = time.perf_counter()
    for s in range(0, len(base), ingest_chunk):
        e = min(s + ingest_chunk, len(base))
        ix.add_batch(ids[s:e].tolist(),
                     np.ascontiguousarray(base[s:e], dtype=np.float32))
    t_ingest = time.perf_counter() - t0

    t0 = time.perf_counter()
    ix.compact()                                 # cluster-major layout: 3x
    t_compact = time.perf_counter() - t0

    timings = {"train_s": t_train, "ingest_s": t_ingest,
               "compact_s": t_compact, "total_s": t_train + t_ingest + t_compact,
               "cells": n_cells}
    if verbose:
        print(f"  [build {profile}] cells={n_cells:,}  train {t_train:.1f}s + "
              f"ingest {t_ingest:.1f}s + compact {t_compact:.1f}s "
              f"= {timings['total_s']:.1f}s", flush=True)
    return ix, timings


# --- timing ---

def timed_serial(fn: Callable, queries: np.ndarray, k: int):
    """One query at a time, single thread. This is the latency number.

    It includes the Python -> pyo3 call, which is a real part of the total at sub-millisecond scale,
    so treat it as an upper bound.
    """
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


def timed_batch(fn: Callable, queries: np.ndarray, repeats: int = 7):
    """Whole batch at once, all threads, best of N. This is the throughput number.

    Also returns the spread, since I've seen up to 35% run-to-run movement on an idle machine. Gaps
    smaller than the spread are ties.
    """
    times, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(queries)
        times.append(time.perf_counter() - t0)
    times.sort()
    med = times[len(times) // 2]
    spread = (times[-1] - times[0]) / med * 100.0 if med > 0 else 0.0
    return out, len(queries) / times[0], spread


def batch_ids(ix, queries: np.ndarray, k: int, nprobe: int) -> np.ndarray:
    """Batch search returning an (nq, k) int64 array; short rows become -1."""
    ids, _ = ix.search_batch(np.ascontiguousarray(queries), k, nprobe, 1.0, 8.0)
    a = np.asarray(ids, dtype=np.uint64).reshape(len(queries), k)
    out = np.full(a.shape, -1, dtype=np.int64)
    live = a != np.uint64(2 ** 64-1)
    out[live] = a[live].astype(np.int64)
    return out


def sweep(ix, queries, gt_ids, k: int, probes, label: str = "") -> List[dict]:
    """Recall / MRR / latency / throughput across a probe grid."""
    from groundtruth import recall_at_k, mean_reciprocal_rank

    rows = []
    for nprobe in probes:
        _, lat = timed_serial(lambda q, n=nprobe: ix.search(q, k, n, 1.0, 8.0)[0],
                              queries, k)
        bret, bqps, spread = timed_batch(
            lambda qq, n=nprobe: batch_ids(ix, qq, k, n), queries)
        rows.append({
            "nprobe": nprobe,
            "recall@10": round(recall_at_k(bret, gt_ids, 10), 4),
            "recall@100": round(recall_at_k(bret, gt_ids, min(100, gt_ids.shape[1])), 4),
            "mrr": round(mean_reciprocal_rank(bret, gt_ids), 4),
            "serial_qps": round(1000.0 / float(np.mean(lat)), 1),
            "batch_qps": round(bqps, 1),
            "p50_ms": round(float(np.percentile(lat, 50)), 3),
            "p99_ms": round(float(np.percentile(lat, 99)), 3),
            "spread_pct": round(spread, 1),
        })
        r = rows[-1]
        print(f"    {label:<10} np={nprobe:<5} R@10={r['recall@10']:.4f} "
              f"MRR={r['mrr']:.4f}  serial={r['serial_qps']:>7.0f} "
              f"batch={r['batch_qps']:>8.0f} p99={r['p99_ms']:.2f}ms "
              f"+-{r['spread_pct']:.0f}%", flush=True)
    return rows


def best_at(rows: List[dict], target: float, key: str = "batch_qps"):
    """Fastest configuration reaching at least `target` R@10."""
    c = [r for r in rows if r["recall@10"] >= target]
    return max(c, key=lambda r: r[key]) if c else None


def rss_mb() -> float:
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def load_corpus(path: str, k: int = 100, n_queries: int = 1000, limit=None):
    """Corpus + held-out queries + exact ground truth (cached)."""
    from groundtruth import load_or_build
    base, queries, gt_ids, _sims, tag = load_or_build(
        path, k=k, n_queries=n_queries, query_path=None, limit=limit)
    return base, np.ascontiguousarray(queries), gt_ids, tag
