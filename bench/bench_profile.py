"""bench_profile.py: where build and query time go.

Builds one index (single add_batch, train + ingest + compact) with the phase profiler on, then
profiles serial queries at a few nprobe values. Prints per-phase microseconds per query next to
batch QPS and recall.

    python bench/bench_profile.py msmarco_embeddings_cache_v2_1000000.npy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from common import PROFILES, RESULTS_DIR, TARGET_CELL_SIZE, batch_ids, timed_batch  # noqa: E402
from groundtruth import load_or_build  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--profile", default="memory", choices=list(PROFILES))
    ap.add_argument("--rf", type=int, default=0, help="0 = profile default")
    ap.add_argument("--probes", default="32,64,128,256,512,1024")
    ap.add_argument("--serial", type=int, default=300)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import aigvs_rust
    base, queries, gt, _, tag = load_or_build(a.corpus, k=100, n_queries=1000,
                                              limit=a.limit or None, verbose=False)
    gt10 = gt[:, :10]
    cfg = PROFILES[a.profile]
    rf = a.rf or cfg["rerank"]
    ix = aigvs_rust.FlatCellIndex(base.shape[1], cfg["scan"], rf, 0x5EED, 0)
    if cfg["per_dim_sq"]:
        ix.set_per_dim_sq(True)
    ix.set_two_stage(True, 2)
    ix.set_graph_route(False, 10 ** 9)
    ix.set_tree_route(True, 2048, 16)
    if cfg["adaptive"]:
        ix.set_adaptive(True, 0.02)

    aigvs_rust.profile(True)
    aigvs_rust.profile_read(True)
    t0 = time.perf_counter()
    ix.train_centroids(base, max(16, len(base) // TARGET_CELL_SIZE))
    t1 = time.perf_counter()
    ix.add_batch(list(range(len(base))), base)
    t2 = time.perf_counter()
    ix.compact()
    t3 = time.perf_counter()
    bp = dict(aigvs_rust.profile_read(True))
    build = {"train_s": t1 - t0, "ingest_s": t2 - t1, "compact_s": t3 - t2, "total_s": t3 - t0}
    print(f"N={len(base):,} profile={a.profile} rf={rf}  BUILD total {build['total_s']:.2f}s "
          f"(train {build['train_s']:.2f} + ingest {build['ingest_s']:.2f} + compact {build['compact_s']:.2f})")
    print("  build phases (ms): " + "  ".join(f"{k[2:]}={v:.0f}" for k, v in bp.items() if k.startswith("b.") and v > 0))

    out = {"tag": tag, "n": len(base), "profile": a.profile, "rf": rf, "build": build,
           "build_phases_ms": {k: v for k, v in bp.items() if k.startswith("b.")}, "probes": {}}
    qs = np.ascontiguousarray(queries[: a.serial])
    for p in [int(x) for x in a.probes.split(",")]:
        aigvs_rust.profile(True); aigvs_rust.profile_read(True)
        for q in qs:
            ix.search(q, 10, p, 1.0, 8.0)
        ph = dict(aigvs_rust.profile_read(True))
        aigvs_rust.profile(False)
        n = max(ph["q.count"], 1)
        us = {k[2:]: ph[k] * 1e3 / n for k in ph if k.startswith("q.") and k != "q.count"}
        def run(cm):
            def f(qq):
                ids_, _ = ix.search_batch(np.ascontiguousarray(qq), 10, p, 1.0, 8.0, cm)
                return np.asarray(ids_, dtype=np.int64).reshape(len(qq), 10)
            ids_, qps_, spread_ = timed_batch(f, queries, 3)
            rec_ = float(np.mean([len(set(x) & set(y)) / 10 for x, y in zip(ids_, gt10)]))
            return rec_, qps_, spread_
        rec, qps, spread = run(False)
        rec_cm, qps_cm, spread_cm = run(True)
        tot = sum(us.values())
        print(f"  np={p:5d}  per-query: R@10={rec:.4f} {qps:8.0f} qps | cell-major: R@10={rec_cm:.4f} "
              f"{qps_cm:8.0f} qps | serial {tot:7.1f} us/q = " +
              "  ".join(f"{k} {v:6.1f}" for k, v in us.items()))
        out["probes"][p] = {"recall": rec, "batch_qps": qps, "spread_pct": spread,
                            "recall_cm": rec_cm, "batch_qps_cm": qps_cm, "spread_cm": spread_cm,
                            "us_per_query": us}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"profile_{a.profile}_n{len(base)}{a.tag}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
