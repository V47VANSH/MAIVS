"""bench_latency.py: single-query latency (p50 / p99) vs faiss HNSW.

One query at a time, timed individually. Throughput is measured separately elsewhere.

Also tests the one thing a partitioned index can do that a graph can't: an HNSW search is a chain of
dependent hops, so one query can't use more cores, while probed cells are independent
(set_parallel_scan). That comparison uses 16 cores against faiss's 1 and is labelled that way. It
matters for latency-sensitive serving with spare cores; for batch throughput leave it off.
"""
import argparse
import json
import os

import numpy as np
import faiss

from common import (build_index, load_corpus, timed_serial, RESULTS_DIR,
                    THREADS, PROFILES)
from groundtruth import recall_at_k


def report(rows, label, param, ret, lat, gt, k):
    r = recall_at_k(ret, gt, 10)
    row = dict(system=label, param=param, recall10=round(r, 4),
               mean_ms=round(float(np.mean(lat)), 3),
               p50_ms=round(float(np.percentile(lat, 50)), 3),
               p99_ms=round(float(np.percentile(lat, 99)), 3))
    rows.append(row)
    print(f"  {label:>24} {param:>10} R@10={r:.4f} "
          f"mean={row['mean_ms']:>7.3f} p50={row['p50_ms']:>7.3f} "
          f"p99={row['p99_ms']:>7.3f}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--probes", default="128,256,512")
    ap.add_argument("--efs", default="32,64,128,256")
    a = ap.parse_args()

    base, q, gt, tag = load_corpus(a.corpus, k=a.k, limit=a.limit)
    faiss.omp_set_num_threads(THREADS)

    # "quality" keeps the exact f32 rerank, which is what reaches the top of
    # the recall curve where the comparison against HNSW is meaningful.
    ix, t = build_index(base, "quality")
    print(f"\n  build: ours {t['total_s']:.1f}s")

    rows = []
    print(f"\n  {'system':>24} {'param':>10}")
    for par in (False, True):
        ix.set_parallel_scan(par)
        label = f"ours ({THREADS} cores)" if par else "ours (1 core)"
        for nprobe in [int(x) for x in a.probes.split(",")]:
            ret, lat = timed_serial(
                lambda x, n=nprobe: ix.search(x, a.k, n, 1.0, 8.0)[0], q, a.k)
            report(rows, label, f"np={nprobe}", ret, lat, gt, a.k)
    ix.set_parallel_scan(False)

    h = faiss.IndexHNSWFlat(base.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    h.hnsw.efConstruction = 200
    h.add(base)
    for ef in [int(x) for x in a.efs.split(",")]:
        h.hnsw.efSearch = ef
        ret, lat = timed_serial(
            lambda x: h.search(x.reshape(1, -1), a.k)[1][0], q, a.k)
        report(rows, "faiss-hnsw (1 core)", f"ef={ef}", ret, lat, gt, a.k)

    print(f"\n  NOTE: the parallel row spends {THREADS} cores on one query; "
          f"faiss-hnsw spends 1 and\n        structurally cannot use more "
          f"(each hop depends on the previous).\n        Not resource-fair - "
          f"it is the latency-with-spare-cores number.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"latency_{tag}.json"), "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
