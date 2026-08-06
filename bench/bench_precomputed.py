"""bench_precomputed.py: index the query side instead of the data side.

Everything so far searches at query time and is bounded by ranking M cell summaries or walking a
coarse graph (alpha +0.57). Three things I've measured suggest skipping the search:

  * queries here are held-out corpus vectors, so in-distribution;
  * the true top-10 sit in ~640 vectors;
  * that floor doesn't grow with N.

So: for each cell, precompute the union of the top-k lists of its own members. At query time find
the cell (one centroid lookup) and scan its candidate list.

    query cost = |candidate list| ~ cell_size * k ids, independent of N

The lists are ~25 B/vector (ids only), built with our own index.

The question is whether a new query's true top-10 lies inside that union for the cell it lands in.
That's `coverage`, an upper bound on recall for this design. If it's high and flat in N this would
be an O(1) index; if it decays with N, it doesn't work. Also reported: `own_cell` (just scanning the
landing cell) and the list size, since coverage bought with a list that grows with N isn't a win.

    python bench/bench_precomputed.py hotpot_5M.npy --sizes 100000,500000,2000000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "research"))
from bench_scale_full import exact_gt  # noqa: E402
from bench_knnpart import approx_knn  # noqa: E402
from common import RESULTS_DIR  # noqa: E402
from summariser_lab import Partition, spherical_kmeans, _dev  # noqa: E402

import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--sizes", default="100000,500000,2000000")
    ap.add_argument("--cell-size", type=int, default=128)
    ap.add_argument("--kg", default="10,25,50", help="k of the stored lists")
    ap.add_argument("--nprobe", default="1,2,4", help="cells whose lists we merge")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--gt-queries", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out_path = a.out or os.path.join(RESULTS_DIR, "precomputed.json")
    sizes = [int(x) for x in a.sizes.split(",")]
    kgs = [int(x) for x in a.kg.split(",")]
    nps = [int(x) for x in a.nprobe.split(",")]

    mm = np.load(a.corpus, mmap_mode="r")
    n_total = mm.shape[0]
    gt_q = np.ascontiguousarray(mm[n_total - a.gt_queries:], dtype=np.float32)
    Q = gt_q[:a.queries]

    res = {"meta": {"cell_size": a.cell_size}, "runs": []}
    for n in sizes:
        n = min(n, n_total - a.gt_queries)
        gt = exact_gt(mm, n, gt_q, 100)[:a.queries]
        gt10 = [set(g[:10].tolist()) for g in gt]
        X = np.ascontiguousarray(mm[:n], dtype=np.float32)
        M = max(16, n // a.cell_size)
        print(f"\n{'=' * 72}\nN={n:,}  cells={M:,}\n{'=' * 72}", flush=True)

        C, asg = spherical_kmeans(X, M, iters=12, seed=0, verbose=False)
        part = Partition(C, asg, M)
        t0 = time.time()
        knn = approx_knn(X, max(kgs), a.cell_size)
        print(f"  k-NN lists ({max(kgs)}) in {time.time() - t0:.0f}s", flush=True)

        # which cell each query lands in, best-first
        Cd = torch.from_numpy(C).to(_dev())
        land = (torch.from_numpy(Q).to(_dev()) @ Cd.T
                ).topk(max(nps), dim=1).indices.cpu().numpy()

        row = {"n": n, "cells": M, "results": []}
        for kg in kgs:
            for p in nps:
                cov, own, size = 0, 0, 0
                for i in range(len(Q)):
                    cand = set()
                    for c in land[i, :p]:
                        mem = part.members(int(c))
                        cand.update(int(v) for v in knn[mem, :kg].ravel()
                                    if v >= 0)
                        cand.update(int(v) for v in mem)
                    size += len(cand)
                    cov += len(gt10[i] & cand)
                    if p == 1:
                        own += len(gt10[i] & set(int(v) for v in
                                                 part.members(int(land[i, 0]))))
                covr = cov / (len(Q) * 10)
                row["results"].append(
                    {"kg": kg, "nprobe": p, "coverage": round(covr, 5),
                     "cand_size": int(size / len(Q)),
                     "own_cell": round(own / (len(Q) * 10), 5) if p == 1 else None})
                print(f"    k={kg:<4} probe={p}  coverage={covr:.4f}  "
                      f"candidates~{int(size / len(Q)):,}"
                      + (f"   (own cell alone: {own / (len(Q) * 10):.4f})"
                         if p == 1 else ""), flush=True)
        res["runs"].append(row)
        json.dump(res, open(out_path, "w"), indent=1)
        del X, part, knn
        torch.cuda.empty_cache()

    print(f"\n{'=' * 72}\nIS COVERAGE FLAT IN N?   (it must be, for alpha=0)"
          f"\n{'=' * 72}")
    for kg in kgs:
        for p in nps:
            cells = []
            for r in res["runs"]:
                w = [x for x in r["results"] if x["kg"] == kg and x["nprobe"] == p]
                if w:
                    cells.append((r["n"], w[0]["coverage"], w[0]["cand_size"]))
            if len(cells) < 2:
                continue
            print(f"  k={kg:<4} probe={p}: " +
                  "  ".join(f"N={x:,}:{c:.4f}({s:,})" for x, c, s in cells))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
