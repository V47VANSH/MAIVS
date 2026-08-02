"""bench_ladder.py: a hierarchy of sampled points instead of summaries.

So far richer summaries (<=1.25x), spill (no gain), independent multi-table (3x worse) and attractor
hops (worse) all pick cells by ranking summaries globally, and that cost grows with M = N/c.

Gottesbueren, Dhulipala, Jayaram and Lacki (VLDB 2025) see the same thing: routing by a single
center gives 28% top-shard recall on MS Turing vs 81% with their routing index, which represents
each shard by real data points. Their shards are huge and non-convex and ours are 128 convex points,
so the method doesn't transfer as-is. The part that might transfer is building the routing hierarchy
from real points, like HNSW's layers, instead of from centroids whose error adds up at every level.

    level L    a 1/f^L sample of the corpus, cells of size c (small, scan all of it)
    ...
    level 1    a 1/f sample
    level 0    everything

Search scans the top level fully and keeps a beam of B points, then at each level only probes the b
cells nearest each beam point. Work per level is B * b * c, so total work is log_f(N) * B * b * c.

It only works if a fixed B is enough. The top-10 are spread out (pairwise cos ~0.46, ~0.57 to q), so
B has to cover several directions. If B has to grow with N to hold recall, this is just global
ranking again. Measured as alpha of vectors scanned at fixed recall, like the others.

Samples point at vectors already in the arena, so a level costs memberships and centroids, not
copies.

    python bench/bench_ladder.py hotpot_5M.npy --sizes 100000,500000,2000000
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


def build_level(vecs, ids_global, cell_size, rerank):
    """One rung: an index over `vecs`, whose ids are global row numbers."""
    import aigvs_rust
    cfg = PROFILES["memory"]
    n_cells = max(16, len(vecs) // cell_size)
    ix = aigvs_rust.FlatCellIndex(vecs.shape[1], cfg["scan"], cfg["rerank"],
                                  0x5EED, 0)
    ix.set_per_dim_sq(True)
    ix.set_two_stage(True, 2)
    ix.set_graph_route(False, 10 ** 9)
    if n_cells >= 2048:
        ix.set_tree_route(True, 2048, 16)
    ix.train_centroids(np.ascontiguousarray(vecs), n_cells)
    ix.add_batch(ids_global.tolist(), np.ascontiguousarray(vecs))
    ix.compact()
    ix.set_rerank_factor(rerank)
    ix.set_adaptive(False, 0.0)
    return ix, n_cells


def ladder_search(levels, mm, Q, beam, b, cell_size, k=10):
    """Descend the ladder. Returns (found dict per query, cells probed per query).

    `levels` is coarsest first. At each level the probe set comes only from the previous level's
    answers, never from a global ranking.
    """
    nq = len(Q)
    found = [dict() for _ in range(nq)]
    probed = 0
    entry = None                                    # [nq, beam] global ids

    for li, (ix, ncells) in enumerate(levels):
        if entry is None:
            # top rung: small enough to scan outright
            ids = batch_ids(ix, Q, beam, ncells)
            probed += ncells
        else:
            # local expansion: each beam point contributes b cells
            parts = []
            for e in range(entry.shape[1]):
                pts = np.ascontiguousarray(mm[entry[:, e]], dtype=np.float32)
                parts.append(batch_ids(ix, pts, beam, b))
            ids = np.concatenate(parts, axis=1)
            probed += entry.shape[1] * b

        flat = np.unique(ids[ids >= 0])
        if flat.size:
            vecs = np.ascontiguousarray(mm[flat], dtype=np.float32)
            pos = {int(v): i for i, v in enumerate(flat)}
            sc = Q @ vecs.T
            for i in range(nq):
                for v in ids[i]:
                    if v >= 0:
                        found[i][int(v)] = float(sc[i, pos[int(v)]])
        # beam for the next rung: the best points seen SO FAR, which is what
        # makes each descent step local rather than a fresh global choice
        nxt = np.zeros((nq, beam), dtype=np.int64)
        for i in range(nq):
            top = sorted(found[i].items(), key=lambda kv: -kv[1])[:beam]
            if not top:
                top = [(0, 0.0)]
            for j in range(beam):
                nxt[i, j] = top[min(j, len(top) - 1)][0]
        entry = nxt
    return found, probed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--sizes", default="100000,500000,2000000")
    ap.add_argument("--fanout", type=int, default=8, help="sampling factor f")
    ap.add_argument("--cell-size", type=int, default=128)
    ap.add_argument("--beams", default="4,8,16,32")
    ap.add_argument("--b", type=int, default=4, help="cells per beam point")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--gt-queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--rerank", type=int, default=128)
    ap.add_argument("--targets", default="0.80,0.90,0.95")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out_path = a.out or os.path.join(RESULTS_DIR, "ladder.json")
    sizes = [int(x) for x in a.sizes.split(",")]
    beams = [int(x) for x in a.beams.split(",")]
    targets = [float(t) for t in a.targets.split(",")]

    mm = np.load(a.corpus, mmap_mode="r")
    n_total, d = mm.shape
    gt_q = np.ascontiguousarray(mm[n_total - a.gt_queries:], dtype=np.float32)
    Q = gt_q[:a.queries]

    res = {"meta": {"fanout": a.fanout, "b": a.b, "cell_size": a.cell_size,
                    "queries": a.queries}, "runs": []}
    for n in sizes:
        n = min(n, n_total - a.gt_queries)
        gt = exact_gt(mm, n, gt_q, a.k)[:a.queries]
        gt10 = [set(g[:10].tolist()) for g in gt]
        print(f"\n{'=' * 72}\nN={n:,}\n{'=' * 72}", flush=True)

        # build the ladder, coarsest last then reversed
        levels, sizes_l = [], []
        t0 = time.perf_counter()
        cur = np.arange(n, dtype=np.int64)
        while True:
            vecs = np.ascontiguousarray(mm[cur], dtype=np.float32)
            ix, nc = build_level(vecs, cur.astype(np.uint64), a.cell_size,
                                 a.rerank)
            levels.append((ix, nc))
            sizes_l.append(len(cur))
            del vecs
            if len(cur) <= a.cell_size * 64:
                break
            cur = cur[::a.fanout]
        levels.reverse()
        sizes_l.reverse()
        bt = time.perf_counter() - t0
        print(f"  ladder: {len(levels)} rungs {sizes_l} built in {bt:.1f}s",
              flush=True)

        rows = []
        for B in beams:
            found, probed = ladder_search(levels, mm[:n], Q, B, a.b,
                                          a.cell_size)
            hit = 0
            for i in range(len(Q)):
                top = sorted(found[i].items(), key=lambda kv: -kv[1])[:10]
                hit += len(gt10[i] & {x for x, _ in top})
            r10 = hit / (len(Q) * 10)
            scanned = probed * a.cell_size
            rows.append({"beam": B, "recall@10": round(r10, 5),
                         "cells_probed": probed, "scanned": int(scanned)})
            print(f"    beam={B:<4} cells={probed:<7,} scanned~{scanned:>9,}  "
                  f"R@10={r10:.4f}", flush=True)
        res["runs"].append({"n": n, "rungs": len(levels), "level_sizes": sizes_l,
                            "build_s": round(bt, 2), "sweep": rows})
        json.dump(res, open(out_path, "w"), indent=1)
        for ix, _ in levels:
            del ix
        del levels
        gc.collect()

    print(f"\n{'=' * 72}\nDOES A SAMPLE LADDER MOVE THE EXPONENT?\n{'=' * 72}")
    print(f"\n  rungs grow as log_f(N): " +
          " ".join(f"N={r['n']:,}->{r['rungs']}" for r in res["runs"]))
    for tgt in targets:
        ns, vs = [], []
        for r in sorted(res["runs"], key=lambda r: r["n"]):
            v = interp_at(r["sweep"], tgt, "scanned")
            if v is None:
                continue
            ns.append(r["n"])
            vs.append(v)
        if len(ns) < 2:
            print(f"  R@10>={tgt}: not reached at these beams")
            continue
        print(f"  R@10>={tgt}: alpha={fit_alpha(ns, vs):+.3f}   " +
              " ".join(f"N={x:,}:{v:,.0f}" for x, v in zip(ns, vs)))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
