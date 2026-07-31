"""bench_ceiling.py: what limits max recall, and does it get worse with N?

Every cell is probed, so routing is out of the picture and only scan + rerank is tested.

With exhaustive probing every vector is scored by its 1-bit code and only the top k * rerank_factor
get rescored exactly. A true neighbour is lost if its Hamming rank is above k*rf, which gets more
likely as N grows. That matches what I see with a fixed rf: 0.9902 at 10k, 0.9770 at 1M.

If that's the cause, scaling rerank_factor with N should recover it, at a cost of k * rf * 384 B of
rerank reads. rf is a query-time setting, so one build covers the sweep.

    python bench/bench_ceiling.py hotpot_5M.npy
    python bench/bench_ceiling.py hotpot_5M.npy --sizes 100000,1000000 --rfs 32,128
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import numpy as np

from bench_scale_full import exact_gt
from common import RESULTS_DIR, build_index, batch_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--sizes", default="100000,250000,1000000,2000000,5232329")
    ap.add_argument("--profiles", default="memory")
    ap.add_argument("--rfs", default="8,32,64,128,256,512")
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--gt-queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--train-cap", type=int, default=1_000_000)
    ap.add_argument("--ingest-chunk", type=int, default=500_000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from groundtruth import recall_at_k
    out_path = a.out or os.path.join(RESULTS_DIR, "ceiling.json")
    sizes = [int(x) for x in a.sizes.split(",")]
    rfs = [int(x) for x in a.rfs.split(",")]

    mm = np.load(a.corpus, mmap_mode="r")
    n_total, d = mm.shape
    gt_q = np.ascontiguousarray(mm[n_total - a.gt_queries:], dtype=np.float32)
    Q = gt_q[:a.queries]

    res = {"meta": {"queries": a.queries, "rfs": rfs}, "runs": []}
    if os.path.exists(out_path):
        try:
            res["runs"] = json.load(open(out_path))["runs"]
        except Exception:
            pass
    done = {(r["n"], r["profile"]) for r in res["runs"]}

    for n in sizes:
        n = min(n, n_total - a.gt_queries)
        gt = exact_gt(mm, n, gt_q, a.k)[:a.queries]
        for prof in [p.strip() for p in a.profiles.split(",")]:
            if (n, prof) in done:
                print(f"[ceil] skip N={n:,} {prof}")
                continue
            gc.collect()
            print(f"\n{'=' * 72}\nN={n:,}  profile={prof}\n{'=' * 72}", flush=True)
            ix, tim = build_index(mm[:n], prof, train_cap=a.train_cap,
                                  ingest_chunk=a.ingest_chunk)
            st = ix.stats()
            cells = int(st["n_cells"])
            # Adaptive termination off here, otherwise this measures the stopping rule, not the
            # ceiling.
            ix.set_adaptive(False, 0.0)

            rows = []
            for rf in rfs:
                ix.set_rerank_factor(rf)
                t0 = time.perf_counter()
                ids = batch_ids(ix, Q, 10, cells)      # every cell probed
                dt = time.perf_counter() - t0
                r10 = recall_at_k(ids, gt, 10)
                rows.append({"rf": rf, "recall@10": round(r10, 5),
                             "qps": round(len(Q) / dt, 1),
                             "rerank_kb": round(10 * rf * 384 / 1024, 1)})
                print(f"    rf={rf:<5} ceiling R@10={r10:.5f}  "
                      f"({rows[-1]['qps']:,.0f} qps exhaustive, "
                      f"rerank {rows[-1]['rerank_kb']:.0f} KB/query)",
                      flush=True)

            res["runs"].append({"n": n, "profile": prof, "cells": cells,
                                "build_s": round(tim["total_s"], 2),
                                "sweep": rows})
            json.dump(res, open(out_path, "w"), indent=1)
            del ix
            gc.collect()

    print(f"\n{'=' * 72}\nRECALL CEILING vs N  (every cell probed, adaptive off)"
          f"\n{'=' * 72}")
    hdr = "".join(f"{('rf=' + str(r)):>10}" for r in rfs)
    print(f"{'N':>10} {'cells':>8}{hdr}")
    for r in sorted([x for x in res["runs"] if x.get("sweep")],
                    key=lambda x: x["n"]):
        got = {w["rf"]: w["recall@10"] for w in r["sweep"]}
        cells = "".join(f"{got.get(rf, float('nan')):>10.4f}" for rf in rfs)
        print(f"{r['n']:>10,} {r['cells']:>8,}{cells}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
