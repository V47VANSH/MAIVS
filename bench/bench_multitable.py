"""bench_multitable.py: do L independent partitions change the exponent?

Two fixes so far gave constant factors at best:

  * richer per-cell summaries: 1.25x ceiling (SUMMARISER_STUDY.md)
  * overlap / spill (R=4): no gain in bytes, alpha unchanged (bench_spill)

Spill probably fails because a vector's R nearest cells are all near it, so if it's a boundary case
for one it tends to be one for all of them. The misses are correlated.

L independent partitions (different k-means seeds) is the LSH answer: more tables lower the exponent
instead of just shifting a constant. A vector that sits badly in partition 1 should sit differently
in partition 2.

Compared at equal scan: L tables probing P/L cells each scan the same P*128 vectors as one table
probing P.

This uses L separate indexes for simplicity. A real version would keep one arena and L membership
lists: an extra membership is a 4 B slot id (bench_spill), so L=4 would cost under 3% extra memory.

    python bench/bench_multitable.py hotpot_5M.npy --sizes 100000,500000
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import numpy as np

from bench_scale_full import exact_gt, fit_alpha, interp_at
from common import PROFILES, RESULTS_DIR, TARGET_CELL_SIZE, batch_ids


def build_one(base, seed, profile, rerank, train_cap, ingest_chunk):
    import maivs
    cfg = PROFILES[profile]
    n_cells = max(16, len(base) // TARGET_CELL_SIZE)
    ix = maivs.FlatCellIndex(base.shape[1], cfg["scan"], cfg["rerank"],
                                  seed, 0)
    if cfg["per_dim_sq"]:
        ix.set_per_dim_sq(True)
    ix.set_two_stage(True, 2)
    ix.set_graph_route(False, 10 ** 9)
    ix.set_tree_route(True, 2048, 16)
    t0 = time.perf_counter()
    if train_cap and len(base) > train_cap:
        # Strided, not a random gather: see common.build_index.
        step = max(1, len(base) // train_cap)
        tr = np.ascontiguousarray(base[::step][:train_cap], dtype=np.float32)
    else:
        tr = np.ascontiguousarray(base, dtype=np.float32)
    ix.train_centroids(tr, n_cells)
    del tr
    ids = np.arange(len(base), dtype=np.uint64)
    for s in range(0, len(base), ingest_chunk):
        e = min(s + ingest_chunk, len(base))
        ix.add_batch(ids[s:e].tolist(),
                     np.ascontiguousarray(base[s:e], dtype=np.float32))
    ix.compact()
    ix.set_rerank_factor(rerank)
    ix.set_adaptive(False, 0.0)
    return ix, time.perf_counter() - t0, n_cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--sizes", default="100000,500000,2000000")
    ap.add_argument("--tables", default="1,2,4")
    ap.add_argument("--profile", default="memory")
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--gt-queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--rerank", type=int, default=128)
    ap.add_argument("--scans", default="64,128,256,512,1024,2048,4096",
                    help="TOTAL cells probed across all tables")
    ap.add_argument("--targets", default="0.90,0.95,0.98")
    ap.add_argument("--train-cap", type=int, default=1_000_000)
    ap.add_argument("--ingest-chunk", type=int, default=500_000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out_path = a.out or os.path.join(RESULTS_DIR, "multitable.json")
    sizes = [int(x) for x in a.sizes.split(",")]
    tables = [int(x) for x in a.tables.split(",")]
    totals = [int(x) for x in a.scans.split(",")]
    targets = [float(t) for t in a.targets.split(",")]

    mm = np.load(a.corpus, mmap_mode="r")
    n_total, d = mm.shape
    gt_q = np.ascontiguousarray(mm[n_total - a.gt_queries:], dtype=np.float32)
    Q = gt_q[:a.queries]

    res = {"meta": {"rerank": a.rerank, "queries": a.queries}, "runs": []}
    if os.path.exists(out_path):
        try:
            res["runs"] = json.load(open(out_path))["runs"]
        except Exception:
            pass
    done = {(r["n"], r["tables"]) for r in res["runs"]}

    for n in sizes:
        n = min(n, n_total - a.gt_queries)
        gt = exact_gt(mm, n, gt_q, a.k)[:a.queries]
        gt10 = [set(g[:10].tolist()) for g in gt]
        for L in tables:
            if (n, L) in done:
                print(f"[mt] skip N={n:,} L={L}")
                continue
            gc.collect()
            print(f"\n{'=' * 72}\nN={n:,}  tables L={L}\n{'=' * 72}", flush=True)
            ixs, build_s, cells = [], 0.0, 0
            for t in range(L):
                ix, bs, cells = build_one(mm[:n], 0x5EED + 7919 * t, a.profile,
                                          a.rerank, a.train_cap, a.ingest_chunk)
                ixs.append(ix)
                build_s += bs
            print(f"  built {L} table(s) in {build_s:.1f}s, "
                  f"{cells:,} cells each", flush=True)

            rows = []
            for tot in totals:
                per = max(1, tot // L)
                if per > cells:
                    continue
                # union of each table's top-10; a real system merges the
                # candidate lists and reranks once, which can only do better.
                got = [set() for _ in range(len(Q))]
                for ix in ixs:
                    ids = batch_ids(ix, Q, 10, per)
                    for i, row in enumerate(ids):
                        got[i].update(int(v) for v in row if v >= 0)
                hit = sum(len(g & t) for g, t in zip(got, gt10))
                r10 = hit / (len(Q) * 10)
                rows.append({"total_probes": per * L, "per_table": per,
                             "recall@10": round(r10, 5),
                             "scanned": int(per * L * 128)})
                print(f"    total={per * L:<6} ({per}/table) "
                      f"scanned~{per * L * 128:>9,}  R@10={r10:.4f}",
                      flush=True)
                if r10 >= 0.995:
                    break

            res["runs"].append({"n": n, "tables": L, "cells": cells,
                                "build_s": round(build_s, 2),
                                "max_recall": max(r["recall@10"] for r in rows),
                                "sweep": rows})
            json.dump(res, open(out_path, "w"), indent=1)
            for ix in ixs:
                del ix
            del ixs
            gc.collect()

    print(f"\n{'=' * 72}\nDO INDEPENDENT TABLES MOVE THE EXPONENT?"
          f"\n{'=' * 72}")
    for tgt in targets:
        print(f"\n  target R@10 >= {tgt}   (vectors scanned)")
        print(f"    {'L':>3} {'alpha':>8} " +
              " ".join(f"{('N=' + f'{x:,}'):>14}" for x in sizes))
        for L in tables:
            rs = sorted([r for r in res["runs"] if r["tables"] == L],
                        key=lambda r: r["n"])
            ns, vs = [], []
            for r in rs:
                v = interp_at(r["sweep"], tgt, "scanned")
                if v is None:
                    continue
                ns.append(r["n"])
                vs.append(v)
            if len(ns) < 2:
                continue
            print(f"    {L:>3} {fit_alpha(ns, vs):>+8.3f} " +
                  " ".join(f"{v:>14,.0f}" for v in vs))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
