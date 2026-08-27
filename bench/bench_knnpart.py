"""bench_knnpart.py: partition by the k-NN graph instead of by k-means.

All the routing ideas so far (richer summaries, spill, multi-table, attractor hops, the sample
ladder, the cell graph) read the same k-means geometry, and they all hit the same limit.
bench_cellgraph.py showed it directly: a walk can't reach a cell that centroid ranking wouldn't
already rank highly, because both look at the same centroids.

k-means minimises distance to the centroid. Routing cares about how many cells a query's true
neighbours are split across. Gottesbueren et al. (PVLDB 2025) optimise the second one: balanced
partitioning of an approximate k-NN graph, minimising cut k-NN edges. They report >81% of a query's
top-10 in one shard, and show that a rough k-NN graph (recall ~0.3) is good enough.

What this does:
1. Build an approximate k-NN graph with our own index (one build plus N self-queries).
2. Refine the k-means partition by local moves: move a point into the cell that holds most of its
   k-NN, with a balance cap.
3. Report the floor (how many cells a query's true top-10 occupy) for both partitions, plus routing
   recall vs cells probed.

The floor is the number to watch, since it doesn't depend on the router. It's currently 3.5-5.2
cells for 10 neighbours. If refinement doesn't push it toward ~2, changing the objective doesn't
help on this data.

    python bench/bench_knnpart.py hotpot_5M.npy --sizes 100000,500000
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
from bench_scale_full import exact_gt, fit_alpha, interp_at  # noqa: E402
from common import PROFILES, RESULTS_DIR, batch_ids  # noqa: E402
from summariser_lab import Partition, spherical_kmeans, _dev  # noqa: E402

import torch  # noqa: E402


def approx_knn(X, kg, cell_size, rerank=64, chunk=200_000):
    """Approximate k-NN graph, built with this engine (one build + N queries)."""
    import maivs
    n, d = X.shape
    cfg = PROFILES["memory"]
    ix = maivs.FlatCellIndex(d, cfg["scan"], cfg["rerank"], 0x5EED, 0)
    ix.set_per_dim_sq(True)
    ix.set_two_stage(True, 2)
    ix.set_graph_route(False, 10 ** 9)
    ix.set_tree_route(True, 2048, 16)
    ix.train_centroids(X, max(16, n // cell_size))
    ix.add_batch(np.arange(n, dtype=np.uint64).tolist(), X)
    ix.compact()
    ix.set_rerank_factor(rerank)
    ix.set_adaptive(False, 0.0)
    out = np.zeros((n, kg), dtype=np.int32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s:e] = batch_ids(ix, X[s:e], kg, 64)[:, :kg]
    del ix
    return out


def local_moving(assign, knn, n_cells, rounds=6, cap_mult=1.6, cell_size=128):
    """Move each point into the cell that holds most of its k-NN.

    Each such move reduces the number of cut k-NN edges. The balance cap keeps cells a scannable
    size; without it cells would grow and trade routing quality for scan cost.
    """
    assign = assign.copy().astype(np.int32)
    cap = int(cell_size * cap_mult)
    counts = np.bincount(assign, minlength=n_cells).astype(np.int32)
    n = len(assign)
    for r in range(rounds):
        moved = 0
        order = np.random.default_rng(r).permutation(n)
        nbr_cells = assign[knn]                       # [n, kg]
        for i in order:
            cur = assign[i]
            row = nbr_cells[i]
            vals, cnt = np.unique(row, return_counts=True)
            best = vals[np.argmax(cnt)]
            if best == cur or counts[best] >= cap or counts[cur] <= 2:
                continue
            here = cnt[vals == cur]
            if cnt.max() <= (here[0] if here.size else 0):
                continue
            counts[cur] -= 1
            counts[best] += 1
            assign[i] = best
            moved += 1
        print(f"    local-moving round {r}: {moved:,} moves", flush=True)
        if moved < n // 500:
            break
    return assign


def floor_of(part, gt10):
    """Cells occupied by the true top-10, and vectors in them."""
    occ, vec = [], []
    for g in gt10:
        c = np.unique(part.assign[g])
        occ.append(len(c))
        vec.append(part.sizes[c].sum())
    return float(np.mean(occ)), float(np.mean(vec))


def route_curve(part, C, Q, gt10, probes):
    dev = _dev()
    S = torch.from_numpy(Q).to(dev) @ torch.from_numpy(C).to(dev).T
    order = S.topk(min(max(probes), part.n_cells), dim=1).indices.cpu().numpy()
    out = []
    for p in probes:
        if p > part.n_cells:
            continue
        hit, sc = 0, 0
        for i, g in enumerate(gt10):
            cells = set(int(c) for c in order[i, :p])
            hit += sum(1 for x in g if int(part.assign[x]) in cells)
            sc += part.sizes[order[i, :p]].sum()
        out.append({"nprobe": p, "recall@10": hit / (len(gt10) * 10),
                    "scanned": int(sc / len(gt10))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--sizes", default="100000,500000")
    ap.add_argument("--cell-size", type=int, default=128)
    ap.add_argument("--kg", type=int, default=10, help="k for the k-NN graph")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--cap-mult", type=float, default=1.6)
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--gt-queries", type=int, default=1000)
    ap.add_argument("--targets", default="0.90,0.95")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out_path = a.out or os.path.join(RESULTS_DIR, "knnpart.json")
    sizes = [int(x) for x in a.sizes.split(",")]
    probes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    res = {"meta": {"kg": a.kg, "cap_mult": a.cap_mult}, "runs": []}

    mm = np.load(a.corpus, mmap_mode="r")
    n_total = mm.shape[0]
    gt_q = np.ascontiguousarray(mm[n_total - a.gt_queries:], dtype=np.float32)
    Q = gt_q[:a.queries]

    for n in sizes:
        n = min(n, n_total - a.gt_queries)
        gt = exact_gt(mm, n, gt_q, 100)[:a.queries]
        gt10 = [g[:10] for g in gt]
        X = np.ascontiguousarray(mm[:n], dtype=np.float32)
        M = max(16, n // a.cell_size)
        print(f"\n{'=' * 72}\nN={n:,}  cells={M:,}\n{'=' * 72}", flush=True)

        t0 = time.time()
        C0, a0 = spherical_kmeans(X, M, iters=12, seed=0, verbose=False)
        p0 = Partition(C0, a0, M)
        print(f"  k-means {time.time() - t0:.0f}s", flush=True)

        t0 = time.time()
        knn = approx_knn(X, a.kg, a.cell_size)
        print(f"  approx k-NN graph {time.time() - t0:.0f}s", flush=True)

        t0 = time.time()
        a1 = local_moving(a0, knn, M, rounds=a.rounds,
                          cap_mult=a.cap_mult, cell_size=a.cell_size)
        # centroids follow the new assignment
        C1 = np.zeros_like(C0)
        np.add.at(C1, a1, X)
        nrm = np.linalg.norm(C1, axis=1, keepdims=True)
        C1 = np.where(nrm > 1e-9, C1 / np.maximum(nrm, 1e-9), C0)
        p1 = Partition(C1, a1, M)
        print(f"  refine {time.time() - t0:.0f}s", flush=True)

        row = {"n": n, "cells": M}
        for tag, P, C in (("kmeans", p0, C0), ("knnpart", p1, C1)):
            f, v = floor_of(P, gt10)
            cur = route_curve(P, C, Q, gt10, probes)
            row[tag] = {"floor_cells": round(f, 3), "floor_vectors": round(v, 1),
                        "max_cell": int(P.sizes.max()),
                        "mean_cell": round(float(P.sizes.mean()), 1),
                        "sweep": cur}
            print(f"  [{tag:<8}] floor={f:.2f} cells ({v:,.0f} vec)  "
                  f"max_cell={P.sizes.max()}", flush=True)
            for w in cur:
                if w["nprobe"] in (8, 32, 128, 512):
                    print(f"      np={w['nprobe']:<5} scanned~{w['scanned']:>8,}"
                          f"  R@10={w['recall@10']:.4f}", flush=True)
        res["runs"].append(row)
        json.dump(res, open(out_path, "w"), indent=1)
        del X, p0, p1
        torch.cuda.empty_cache()

    print(f"\n{'=' * 72}\nDOES THE OBJECTIVE SWAP HELP?\n{'=' * 72}")
    print(f"{'N':>10} {'kmeans floor':>14} {'knnpart floor':>15} {'change':>9}")
    for r in res["runs"]:
        f0, f1 = r["kmeans"]["floor_cells"], r["knnpart"]["floor_cells"]
        print(f"{r['n']:>10,} {f0:>14.2f} {f1:>15.2f} {f1 / f0-1:>+8.1%}")
    for tgt in [float(t) for t in a.targets.split(",")]:
        print(f"\n  vectors scanned @ R@10 >= {tgt}")
        for tag in ("kmeans", "knnpart"):
            ns, vs = [], []
            for r in res["runs"]:
                v = interp_at(r[tag]["sweep"], tgt, "scanned")
                if v is None:
                    continue
                ns.append(r["n"])
                vs.append(v)
            if len(ns) < 2:
                continue
            print(f"    {tag:<8} alpha={fit_alpha(ns, vs):+.3f}  " +
                  " ".join(f"N={x:,}:{v:,.0f}" for x, v in zip(ns, vs)))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
