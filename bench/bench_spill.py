"""bench_spill.py: can overlapping cells fix the scaling exponent?

bench_scale_full.py shows probes at fixed recall growing as ~N^0.66, while the floor (how many cells
the true top-10 occupy) stays at 3.5-5.2 cells at every scale. So the routing overhead goes from ~6x
at 10k to ~200x at 5.2M.

My reading: cells are ranked globally and we take the top P. A neighbour at the edge of its cell has
a poor centroid score, and the number of unrelated cells that outrank it grows with M = N/128. HNSW
never does that global ranking; it only compares a few neighbours at a time.

The classic fix is spill trees (Liu, Moore, Gray, Yang, NIPS 2004): overlap the partitions so
boundary points live in both neighbouring cells, then descend to a fixed number of leaves without
backtracking. set_replication(max_assign, spill_margin) already does the ingestion side: a vector
goes into every cell whose centroid is within spill_margin of the best.

For each (N, replication) this records probes needed at fixed recall and fits nprobe ~ N^alpha.
Replication obviously raises recall at fixed probes; the question is whether alpha changes or just
shifts by a constant. Memory and build cost are recorded too.

    python bench/bench_spill.py hotpot_5M.npy
"""
from __future__ import annotations

import argparse
import gc
import json
import os

import numpy as np

from bench_scale_full import exact_gt, fit_alpha, interp_at, probe_grid
from common import (PROFILES, RESULTS_DIR, TARGET_CELL_SIZE, batch_ids,
                    build_index, rss_mb)


def build_spilled(base, profile, max_assign, spill, train_cap, ingest_chunk):
    """build_index, but with replication configured before ingestion."""
    import maivs
    import time
    cfg = PROFILES[profile]
    d = base.shape[1]
    n_cells = max(16, len(base) // TARGET_CELL_SIZE)
    ix = maivs.FlatCellIndex(d, cfg["scan"], cfg["rerank"], 0x5EED, 0)
    if cfg["per_dim_sq"]:
        ix.set_per_dim_sq(True)
    ix.set_two_stage(True, 2)
    ix.set_graph_route(False, 10 ** 9)
    ix.set_tree_route(True, 2048, 16)
    # Replication must precede ingestion: it decides how many cells each
    # vector is written into.
    if max_assign > 1:
        ix.set_replication(max_assign, spill)

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
    return ix, time.perf_counter() - t0, n_cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--sizes", default="100000,500000,2000000")
    ap.add_argument("--assigns", default="1,2,3")
    ap.add_argument("--spills", default="0.10")
    ap.add_argument("--profile", default="memory")
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--gt-queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--rerank", type=int, default=128)
    ap.add_argument("--targets", default="0.90,0.95")
    ap.add_argument("--train-cap", type=int, default=1_000_000)
    ap.add_argument("--ingest-chunk", type=int, default=500_000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from groundtruth import recall_at_k
    out_path = a.out or os.path.join(RESULTS_DIR, "spill.json")
    sizes = [int(x) for x in a.sizes.split(",")]
    assigns = [int(x) for x in a.assigns.split(",")]
    targets = [float(t) for t in a.targets.split(",")]

    mm = np.load(a.corpus, mmap_mode="r")
    n_total, d = mm.shape
    gt_q = np.ascontiguousarray(mm[n_total - a.gt_queries:], dtype=np.float32)
    Q = gt_q[:a.queries]

    spills = [float(x) for x in a.spills.split(",")]
    res = {"meta": {"spills": spills, "rerank": a.rerank,
                    "queries": a.queries}, "runs": []}
    if os.path.exists(out_path):
        try:
            res["runs"] = json.load(open(out_path))["runs"]
        except Exception:
            pass
    done = {(r["n"], r["max_assign"], r["spill"]) for r in res["runs"]}

    for n in sizes:
        n = min(n, n_total - a.gt_queries)
        gt = exact_gt(mm, n, gt_q, a.k)[:a.queries]
        for ma, spill in [(m, sp) for m in assigns for sp in spills]:
            if (n, ma, spill) in done:
                print(f"[spill] skip N={n:,} R={ma} s={spill}")
                continue
            gc.collect()
            r0 = rss_mb()
            print(f"\n{'=' * 72}\nN={n:,}  max_assign={ma}  "
                  f"spill={spill}\n{'=' * 72}", flush=True)
            ix, build_s, cells = build_spilled(
                mm[:n], a.profile, ma, spill, a.train_cap, a.ingest_chunk)
            ix.set_rerank_factor(a.rerank)
            ix.set_adaptive(False, 0.0)
            st = ix.stats()
            slots = int(st["n_live"])
            repl = float(st.get("replication_factor", 1.0))
            print(f"  build {build_s:.1f}s  cells={cells:,}  slots={slots:,} "
                  f"(x{repl:.2f})  arena={st['arena_bytes_per_vec']:.0f} B/vec "
                  f"rss+{rss_mb() - r0:,.0f} MB", flush=True)

            rows = []
            for npb in probe_grid(cells, cap=4096):
                ids = batch_ids(ix, Q, 10, npb)
                r10 = recall_at_k(ids, gt, 10)
                rows.append({"nprobe": npb, "recall@10": round(r10, 5),
                             "scanned": int(npb * float(st["cell_size_mean"]))})
                print(f"    np={npb:<6} scanned={rows[-1]['scanned']:>9,} "
                      f"R@10={r10:.4f}", flush=True)
                if r10 >= 0.995:
                    break
            res["runs"].append({
                "n": n, "max_assign": ma, "spill": spill, "cells": cells,
                "slots": slots, "repl_factor": round(repl, 3),
                "arena_b_per_vec": round(float(st["arena_bytes_per_vec"]), 1),
                "index_mb": round(float(st["arena_bytes_per_vec"]) * slots / 1e6, 1),
                "build_s": round(build_s, 2),
                "max_recall": max(r["recall@10"] for r in rows),
                "sweep": rows})
            json.dump(res, open(out_path, "w"), indent=1)
            del ix
            gc.collect()

    print(f"\n{'=' * 72}\nDOES OVERLAP CHANGE THE EXPONENT?\n{'=' * 72}")
    for tgt in targets:
        print(f"\n  target R@10 >= {tgt}")
        print(f"    {'R':>3} {'alpha(scanned)':>14} " +
              " ".join(f"{('N=' + f'{n:,}'):>16}" for n in sizes))
        for ma in assigns:
            rs = sorted([r for r in res["runs"] if r["max_assign"] == ma
                         and r["spill"] == spills[-1]], key=lambda r: r["n"])
            ns, ps = [], []
            for r in rs:
                v = interp_at(r["sweep"], tgt, "scanned")
                if v is None:
                    continue
                ns.append(r["n"])
                ps.append(v)
            if len(ns) < 2:
                continue
            al = fit_alpha(ns, ps)
            cellstr = " ".join(f"{p:>16,.0f}" for p in ps)
            print(f"    {ma:>3} {al:>+14.3f} {cellstr}")
    print(f"\n{'N':>10} {'R':>3} {'repl':>6} {'idx MB':>9} {'build s':>8} "
          f"{'maxR@10':>8}")
    for r in sorted(res["runs"], key=lambda r: (r["n"], r["max_assign"])):
        print(f"{r['n']:>10,} {r['max_assign']:>3} {r['repl_factor']:>6.2f} "
              f"{r['index_mb']:>9,.0f} {r['build_s']:>8.1f} "
              f"{r['max_recall']:>8.4f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
