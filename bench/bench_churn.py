"""bench_churn.py: replace the whole index over N cycles and track what degrades.

Exact ground truth is recomputed for the live set at every checkpoint.

  recall       should stay flat: deletes are physical (swap_remove + free list), no tombstones
  delete cost  should stay flat: O(1), no edges to repair
  locality     does decay: slot reuse undoes the cluster-major layout, and throughput drops
               until compact() runs

I originally assumed nothing would ever need compacting. That stopped being true once the arena
became cluster-major, so compact() is timed here as well.
"""
import argparse
import json
import os
import time

import numpy as np
import faiss

# import common first: it sets sys.path and RAYON_NUM_THREADS, which rayon reads once when its pool
# starts
from common import PROFILES, RESULTS_DIR, TARGET_CELL_SIZE, THREADS
from groundtruth import normalize
import maivs


def build_with_ids(X, ids, profile):
    """Build keyed by the caller's ids, so churn can address vectors directly."""
    cfg = PROFILES[profile]
    ix = maivs.FlatCellIndex(X.shape[1], cfg["scan"], cfg["rerank"], 0x5EED, 0)
    if cfg["per_dim_sq"]:
        ix.set_per_dim_sq(True)
    ix.set_two_stage(True, 2)
    ix.set_graph_route(False, 10 ** 9)
    ix.set_tree_route(True, 2048, 16)
    if cfg["adaptive"]:
        ix.set_adaptive(True, 0.02)
    cells = max(16, len(ids) // TARGET_CELL_SIZE)
    t0 = time.perf_counter()
    rows = np.ascontiguousarray(X[ids])
    ix.train_centroids(rows, cells)
    ix.add_batch([int(i) for i in ids], rows)
    ix.compact()
    print(f"  [build] {time.perf_counter()-t0:.1f}s  cells={cells:,}  "
          f"tree_depth={ix.stats()['tree_depth']:.0f}")
    return ix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--churn", type=float, default=0.10)
    ap.add_argument("--nprobe", type=int, default=64)
    ap.add_argument("--profile", default="memory", choices=list(PROFILES))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    faiss.omp_set_num_threads(THREADS)
    X = normalize(np.array(np.load(a.corpus, mmap_mode="r"), dtype=np.float32),
                  inplace=True)
    if a.limit:
        X = np.ascontiguousarray(X[: a.limit])

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    q_idx, rest = perm[: a.queries], perm[a.queries:]
    queries = np.ascontiguousarray(X[np.sort(q_idx)])
    half = len(rest) // 2
    live = list(rest[:half])
    reserve = list(rest[half:])
    print(f"live={len(live):,} reserve={len(reserve):,} queries={a.queries} "
          f"churn={a.churn:.0%}/cycle x {a.cycles} "
          f"(= {a.churn*a.cycles:.0%} turnover)")

    ix = build_with_ids(X, np.sort(np.array(live)), a.profile)

    def measure():
        act = np.sort(np.array(live))
        _d, gi = faiss.knn(queries, np.ascontiguousarray(X[act]), a.k,
                           metric=faiss.METRIC_INNER_PRODUCT)
        gt = act[gi.astype(np.int64)]
        t0 = time.perf_counter()
        ids, _ = ix.search_batch(queries, a.k, a.nprobe, 1.0, 8.0)
        qps = len(queries) / (time.perf_counter() - t0)
        got = np.asarray(ids, dtype=np.uint64).reshape(len(queries), a.k)
        r = float(np.mean([len(set(got[i].tolist()) & set(gt[i].tolist())) / a.k
                           for i in range(len(queries))]))
        return r, qps

    rows = []
    r, qps = measure()
    print(f"\n{'cycle':>6} {'live':>9} {'R@10':>8} {'batch qps':>10} "
          f"{'del us':>8} {'ins us':>8} {'max cell':>9}")
    print(f"{0:>6} {len(live):>9,} {r:>8.4f} {qps:>10.0f} {'-':>8} {'-':>8} "
          f"{ix.stats()['cell_size_max']:>9.0f}")
    rows.append(dict(cycle=0, recall=round(r, 4), qps=round(qps)))

    for c in range(1, a.cycles + 1):
        n = int(len(live) * a.churn)
        drop = rng.choice(len(live), n, replace=False)
        victims = [int(live[i]) for i in drop]

        t0 = time.perf_counter()
        for i in victims:
            ix.remove(i)
        del_us = (time.perf_counter() - t0) / n * 1e6
        for i in sorted(drop, reverse=True):
            live.pop(i)

        take = [int(reserve.pop()) for _ in range(n)]
        arr = np.sort(np.array(take))
        t0 = time.perf_counter()
        ix.add_batch([int(i) for i in arr], np.ascontiguousarray(X[arr]))
        ins_us = (time.perf_counter() - t0) / n * 1e6
        live.extend(take)

        r, qps = measure()
        st = ix.stats()
        print(f"{c:>6} {len(live):>9,} {r:>8.4f} {qps:>10.0f} {del_us:>8.2f} "
              f"{ins_us:>8.2f} {st['cell_size_max']:>9.0f}")
        rows.append(dict(cycle=c, recall=round(r, 4), qps=round(qps),
                         delete_us=round(del_us, 2), insert_us=round(ins_us, 2),
                         cell_size_max=st["cell_size_max"]))

    drift = rows[-1]["recall"] - rows[0]["recall"]
    retained = rows[-1]["qps"] / max(rows[0]["qps"], 1)
    print(f"\n  recall drift over {a.churn*a.cycles:.0%} turnover: {drift:+.4f}")
    print(f"  throughput retained: {retained:.2f}x  "
          f"(locality decays; recall does not)")

    t0 = time.perf_counter()
    ix.compact()
    ix.rebuild_tree()
    mt = time.perf_counter() - t0
    r2, q2 = measure()
    print(f"  compact() + rebuild_tree() in {mt:.2f}s -> R@10={r2:.4f} "
          f"qps={q2:.0f}  ({q2/max(rows[-1]['qps'],1):.2f}x recovered)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "churn.json"), "w") as f:
        json.dump({"rows": rows, "drift": round(drift, 4),
                   "qps_retained": round(retained, 3),
                   "maintenance_s": round(mt, 2),
                   "qps_after_maintenance": round(q2)}, f, indent=2)


if __name__ == "__main__":
    main()
