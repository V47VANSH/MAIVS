"""A/B test of the stage-2 centroid re-order precision on the real index.

The summariser lab suggested that re-ordering the shortlist from an SQ8 copy of the centroids
instead of f32 leaves probe counts and the scan unchanged, while saving nprobe * mult * 1152 B per
query. This checks it on the engine itself: one build, then the flag is flipped on the same index,
so the only difference between the two runs is which array the re-order reads.

    KMP_DUPLICATE_LIB_OK=TRUE python bench/ab_centroid_sq8.py msmarco_embeddings_cache_v2_1000000.npy --queries 1000 --nprobe 512
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Put the repo root first on sys.path, otherwise a stale build of the extension can shadow the fresh
# one (same issue as in run_ann_recall.py).
sys.path.insert(0, _ROOT)
# rayon reads this when its pool is first built, so it must be set before the
# extension is imported, not before it is used.
os.environ.setdefault("RAYON_NUM_THREADS", str(os.cpu_count()))
import groundtruth as GT  # noqa: E402
import maivs  # noqa: E402


def timed_batch(ix, Q, k, nprobe, reps=7):
    """Best-of-`reps`, matching `run_ann_recall.py`'s trustworthy number."""
    best, out = 0.0, None
    for _ in range(reps):
        t0 = time.perf_counter()
        ids, _sc = ix.search_batch(Q, k, nprobe, 1.0, 8.0)
        dt = time.perf_counter() - t0
        if dt > 0 and len(Q) / dt > best:
            best, out = len(Q) / dt, ids
    # Returning fewer than k is legitimate; missing slots come back as
    # u64::MAX, which does not fit int64, so mask them to -1.
    a = np.asarray(out, dtype=np.uint64).reshape(len(Q), k)
    res = np.full(a.shape, -1, dtype=np.int64)
    live = a != np.uint64(2 ** 64-1)
    res[live] = a[live].astype(np.int64)
    return best, res


def recall_at(ids, gt, k):
    hit = 0
    for r, g in zip(ids, gt):
        hit += len(set(r[:k]) & set(g[:k]))
    return hit / (len(gt) * k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nprobe", default="512")
    ap.add_argument("--cell-size", type=int, default=128)
    ap.add_argument("--rerank", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    base, queries, gt_ids, _s, tag = GT.load_or_build(
        args.corpus, k=100, n_queries=args.queries, limit=args.limit)
    base = np.ascontiguousarray(base, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    n, d = base.shape
    n_cells = max(1, n // args.cell_size)
    print(f"[ab] {tag}  n={n:,} d={d} cells={n_cells}")

    ix = maivs.FlatCellIndex(d, "binary", args.rerank, 0x5EED, 0)
    ix.set_per_dim_sq(True)
    ix.set_two_stage(True, 2)
    ix.set_tree_route(True, 2048, 16)
    t0 = time.time()
    ix.train_centroids(base, n_cells)
    ix.add_batch(list(range(n)), base)
    ix.compact()
    print(f"[ab] built in {time.time() - t0:.1f}s")

    nps = [int(x) for x in str(args.nprobe).split(",")]
    print()
    print(f"{'nprobe':>7} {'R@k f32':>9} {'R@k sq8':>9} {'delta':>9} "
          f"{'qps f32':>9} {'qps sq8':>9} {'speedup':>8} {'overlap':>8} "
          f"{'KB saved':>9}")
    print("-" * 86)
    for npb in nps:
        arm = {}
        for on in (False, True):
            ix.set_centroid_sq8(on)
            qps, ids = timed_batch(ix, queries, args.k, npb)
            arm[on] = (qps, recall_at(ids, gt_ids, args.k), ids)
        (qf, rf, idf), (qs, rs, ids8) = arm[False], arm[True]
        overlap = float(np.mean([
            len(set(a[:args.k]) & set(b[:args.k])) / args.k
            for a, b in zip(idf, ids8)]))
        saved = npb * 2 * (1536-384) / 1024
        print(f"{npb:>7} {rf:>9.4f} {rs:>9.4f} {rs - rf:>+9.5f} "
              f"{qf:>9,.0f} {qs:>9,.0f} {qs / qf:>7.3f}x {overlap:>8.4f} "
              f"{saved:>9,.0f}")
    print("\nSame index, same queries; the only difference is which array the "
          "stage-2 re-order reads.")


if __name__ == "__main__":
    main()
