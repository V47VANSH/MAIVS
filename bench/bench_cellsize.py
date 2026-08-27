"""bench_cellsize.py: is 128 vectors per cell still right at 5M?

128 was the best of {128, 256, 512, 1024} at 999k; 32 and 64 were never tried. Since then the
centroid tree has made routing over many cells cheaper, and bench_multitable showed the scaling
exponent comes mostly from the tail (alpha +0.19 at R@10 0.90, +0.94 at 0.98).

If the tail comes from members sitting far from their centroid, smaller cells should help. If it
comes from the number of competing cells, smaller cells should hurt. Either answer is useful.

Compared in vectors scanned at fixed recall, since nprobe means different amounts of work at
different cell sizes.

    python bench/bench_cellsize.py hotpot_5M.npy --sizes 100000,500000,2000000
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import numpy as np

from bench_scale_full import exact_gt, fit_alpha, interp_at
from common import PROFILES, RESULTS_DIR, batch_ids


def build(base, cell_size, profile, rerank, train_cap, ingest_chunk):
    import maivs
    cfg = PROFILES[profile]
    n_cells = max(16, len(base) // cell_size)
    ix = maivs.FlatCellIndex(base.shape[1], cfg["scan"], cfg["rerank"],
                                  0x5EED, 0)
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
    ap.add_argument("--cells", default="16,32,64,128")
    ap.add_argument("--profile", default="memory")
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--gt-queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--rerank", type=int, default=128)
    ap.add_argument("--targets", default="0.90,0.95,0.98")
    ap.add_argument("--train-cap", type=int, default=1_000_000)
    ap.add_argument("--ingest-chunk", type=int, default=500_000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from groundtruth import recall_at_k
    out_path = a.out or os.path.join(RESULTS_DIR, "cellsize.json")
    sizes = [int(x) for x in a.sizes.split(",")]
    csizes = [int(x) for x in a.cells.split(",")]
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
    done = {(r["n"], r["cell_size"]) for r in res["runs"]}

    for n in sizes:
        n = min(n, n_total - a.gt_queries)
        gt = exact_gt(mm, n, gt_q, a.k)[:a.queries]
        for cs in csizes:
            if (n, cs) in done:
                print(f"[cs] skip N={n:,} c={cs}")
                continue
            gc.collect()
            print(f"\n{'=' * 72}\nN={n:,}  cell_size={cs}\n{'=' * 72}",
                  flush=True)
            ix, bs, cells = build(mm[:n], cs, a.profile, a.rerank,
                                  a.train_cap, a.ingest_chunk)
            st = ix.stats()
            actual = float(st["cell_size_mean"])
            print(f"  build {bs:.1f}s  cells={cells:,}  mean size {actual:.1f} "
                  f"tree_depth={int(st['tree_depth'])}", flush=True)
            rows = []
            # Probe grid in units of VECTORS, so every cell size is compared at
            # matched work rather than matched probe count.
            for target_scan in (2048, 4096, 8192, 16384, 32768, 65536, 131072,
                                262144, 524288):
                npb = max(1, int(round(target_scan / max(actual, 1))))
                if npb > cells:
                    continue
                ids = batch_ids(ix, Q, 10, npb)
                r10 = recall_at_k(ids, gt, 10)
                rows.append({"nprobe": npb, "recall@10": round(r10, 5),
                             "scanned": int(npb * actual)})
                print(f"    np={npb:<6} scanned~{int(npb * actual):>9,}  "
                      f"R@10={r10:.4f}", flush=True)
                if r10 >= 0.995:
                    break
            res["runs"].append({"n": n, "cell_size": cs, "cells": cells,
                                "mean_size": round(actual, 1),
                                "tree_depth": int(st["tree_depth"]),
                                "arena_b_per_vec": round(
                                    float(st["arena_bytes_per_vec"]), 1),
                                "build_s": round(bs, 2),
                                "max_recall": max(r["recall@10"] for r in rows),
                                "sweep": rows})
            json.dump(res, open(out_path, "w"), indent=1)
            del ix
            gc.collect()

    print(f"\n{'=' * 72}\nCELL SIZE vs SCALING  (vectors scanned at fixed "
          f"recall)\n{'=' * 72}")
    for tgt in targets:
        print(f"\n  target R@10 >= {tgt}")
        print(f"    {'c':>5} {'alpha':>8} " +
              " ".join(f"{('N=' + f'{x:,}'):>14}" for x in sizes))
        for cs in csizes:
            rs = sorted([r for r in res["runs"] if r["cell_size"] == cs],
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
            print(f"    {cs:>5} {fit_alpha(ns, vs):>+8.3f} " +
                  " ".join(f"{v:>14,.0f}" for v in vs))
    print(f"\n{'N':>10} {'c':>5} {'cells':>9} {'depth':>6} {'B/vec':>7} "
          f"{'build s':>8} {'maxR@10':>8}")
    for r in sorted(res["runs"], key=lambda r: (r["n"], r["cell_size"])):
        print(f"{r['n']:>10,} {r['cell_size']:>5} {r['cells']:>9,} "
              f"{r['tree_depth']:>6} {r['arena_b_per_vec']:>7.0f} "
              f"{r['build_s']:>8.1f} {r['max_recall']:>8.4f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
