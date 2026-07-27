"""bench_scale_hnsw.py: does faiss HNSW actually scale better than we do?

bench_scale_full.py gave us alpha ~0.5 for work at fixed recall. The usual answer is "HNSW is O(log
N)", but that's hop count at fixed efSearch. What decides throughput is cost at fixed recall, and
efSearch has to grow with N to hold recall, same as our nprobe.

So this runs HNSW over the same scales, queries and ground truth as bench_scale_full.py and fits
alpha the same way.

    python bench/bench_scale_hnsw.py hotpot_5M.npy --sizes 10000,50000,250000,1000000,2000000
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import numpy as np

from bench_scale_full import exact_gt, fit_alpha, interp_at
from common import RESULTS_DIR, rss_mb, timed_batch, timed_serial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--sizes", default="10000,50000,250000,1000000,2000000")
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--lat-queries", type=int, default=300)
    ap.add_argument("--efs", default="16,32,64,128,256,512,1024")
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--targets", default="0.70,0.80,0.90,0.95")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import faiss
    from groundtruth import recall_at_k, mean_reciprocal_rank
    faiss.omp_set_num_threads(int(os.environ.get("RAYON_NUM_THREADS",
                                                 os.cpu_count() or 8)))

    out_path = a.out or os.path.join(RESULTS_DIR, "scaling_hnsw.json")
    sizes = [int(x) for x in a.sizes.split(",")]
    efs = [int(x) for x in a.efs.split(",")]

    mm = np.load(a.corpus, mmap_mode="r")
    n_total, d = mm.shape
    queries = np.ascontiguousarray(mm[n_total - a.queries:], dtype=np.float32)
    sizes = [min(s, n_total - a.queries) for s in sizes]

    res = {"meta": {"system": "faiss-hnsw", "M": a.M, "d": d,
                    "n_queries": len(queries)}, "runs": []}
    if os.path.exists(out_path):
        try:
            res["runs"] = json.load(open(out_path))["runs"]
        except Exception:
            pass
    done = {r["n"] for r in res["runs"]}

    for n in sizes:
        if n in done:
            print(f"[hnsw] skip N={n:,}")
            continue
        gt = exact_gt(mm, n, queries, a.k)
        print(f"\n{'=' * 70}\nfaiss-HNSW  N={n:,}\n{'=' * 70}", flush=True)
        gc.collect()
        rss0 = rss_mb()
        # Build from the memmap in blocks so an 8 GB corpus is never resident.
        ix = faiss.IndexHNSWFlat(d, a.M, faiss.METRIC_INNER_PRODUCT)
        ix.hnsw.efConstruction = 200
        t0 = time.perf_counter()
        for s in range(0, n, 500_000):
            e = min(s + 500_000, n)
            ix.add(np.ascontiguousarray(mm[s:e], dtype=np.float32))
        build_s = time.perf_counter() - t0
        mem = rss_mb() - rss0
        print(f"  build {build_s:.1f}s  rss +{mem:,.0f} MB", flush=True)

        rows = []
        qlat = queries[:min(a.lat_queries, len(queries))]
        for ef in efs:
            ix.hnsw.efSearch = ef
            _, lat = timed_serial(
                lambda q: ix.search(q.reshape(1, -1), 10)[1][0], qlat, 10)
            bret, bqps, spread = timed_batch(
                lambda qq: ix.search(qq, 10)[1], queries)
            rows.append({
                "ef": ef,
                "recall@10": round(recall_at_k(bret, gt, 10), 5),
                "mrr": round(mean_reciprocal_rank(bret, gt), 5),
                "serial_qps": round(1000.0 / float(np.mean(lat)), 1),
                "batch_qps": round(bqps, 1),
                "p50_ms": round(float(np.percentile(lat, 50)), 4),
                "p99_ms": round(float(np.percentile(lat, 99)), 4),
                "spread_pct": round(spread, 1),
            })
            r = rows[-1]
            print(f"    ef={ef:<5} R@10={r['recall@10']:.4f} "
                  f"serial={r['serial_qps']:>7.0f} batch={r['batch_qps']:>8.0f} "
                  f"p50={r['p50_ms']:.3f}ms +-{r['spread_pct']:.0f}%", flush=True)

        res["runs"].append({"n": n, "build_s": round(build_s, 2),
                            "rss_mb": round(mem, 1),
                            "max_recall": max(r["recall@10"] for r in rows),
                            "sweep": rows})
        json.dump(res, open(out_path, "w"), indent=1)
        del ix
        gc.collect()

    rs = sorted([r for r in res["runs"] if r.get("sweep")], key=lambda r: r["n"])
    print(f"\n{'=' * 70}\nfaiss-HNSW SCALING\n{'=' * 70}")
    print(f"{'N':>10} {'build s':>9} {'RSS MB':>9} {'maxR@10':>9}")
    for r in rs:
        print(f"{r['n']:>10,} {r['build_s']:>9.1f} {r['rss_mb']:>9,.0f} "
              f"{r['max_recall']:>9.4f}")
    print("\nwork at fixed recall ~ N^alpha  (ef, and its latency consequence)")
    for tgt in [float(t) for t in a.targets.split(",")]:
        ns, efv, lat, qps = [], [], [], []
        for r in rs:
            v = interp_at(r["sweep"], tgt, "ef")
            if v is None:
                continue
            ns.append(r["n"])
            efv.append(v)
            lat.append(interp_at(r["sweep"], tgt, "p50_ms"))
            qps.append(interp_at(r["sweep"], tgt, "batch_qps"))
        if len(ns) < 3:
            continue
        h = len(ns) // 2
        print(f"  R@10>={tgt}: alpha(ef)={fit_alpha(ns, efv):+.3f} "
              f" alpha(p50)={fit_alpha(ns, lat):+.3f} "
              f" alpha(qps)={fit_alpha(ns, qps):+.3f}"
              f"   local p50 {fit_alpha(ns[:h + 1], lat[:h + 1]):+.3f} ->"
              f" {fit_alpha(ns[h:], lat[h:]):+.3f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
