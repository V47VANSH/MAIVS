"""bench_cellgraph.py: greedy walk over a diversified cell graph.

Ranking all cell summaries globally seems to have a floor (see bench_attractor.py), so this tries a
local selection like HNSW's. bench_ladder.py expanded to the b nearest cells at each step, which
only explores one direction; a query's top-10 are spread over ~10 directions (pairwise cos ~0.46, vs
~0.57 to q). HNSW handles that with its pruning rule: keep y as a neighbour of x only if y is closer
to x than to any neighbour already kept.

So build that pruned graph over cells and walk it. It has M = N/128 nodes: ~41k nodes and ~5 MB of
edges at 5.2M. Each hop is 32 centroid dots plus one contiguous cell scan. Deletes never touch
edges, since edges join cells, not vectors.

Measured as recall vs vectors scanned, fitted as N^alpha, against two controls on the same
partition:

  flat      rank all cells by centroid, probe the top P (current behaviour)
  nearest   same walk, but edges are just the b nearest cells (no pruning)

If graph beats nearest, the pruning is what helps. If they tie, it isn't.

    python bench/bench_cellgraph.py hotpot_5M.npy --sizes 100000,500000,2000000
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'research'))
from bench_scale_full import exact_gt, fit_alpha, interp_at  # noqa: E402
from common import RESULTS_DIR  # noqa: E402
from summariser_lab import Partition, spherical_kmeans, _dev, _log  # noqa: E402

import torch  # noqa: E402


def build_adjacency(C, deg=32, cand=64, diversify=True, chunk=2048):
    """Cell adjacency: the `cand` nearest cells by centroid, pruned to `deg`.

    With `diversify`, uses the HNSW relative-neighbourhood rule: accept y as a neighbour of x only
    if y is closer to x than to every neighbour already accepted. Stops all the edges from pointing
    the same way.
    """
    dev = _dev()
    Cd = torch.from_numpy(C).to(dev)
    M = C.shape[0]
    adj = np.zeros((M, deg), dtype=np.int32)
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        sim = Cd[s:e] @ Cd.T
        sim[torch.arange(e - s, device=dev), torch.arange(s, e, device=dev)] = -2
        top = sim.topk(min(cand, M - 1), dim=1)
        ti, tv = top.indices.cpu().numpy(), top.values.cpu().numpy()
        if not diversify:
            adj[s:e] = ti[:, :deg]
        else:
            sub = Cd[torch.from_numpy(ti).to(dev).reshape(-1)].reshape(
                e - s, -1, C.shape[1])
            cross = torch.bmm(sub, sub.transpose(1, 2)).cpu().numpy()
            for r in range(e - s):
                keep = []
                for j in range(ti.shape[1]):
                    # keep j only if it is closer to the source than to every neighbour already kept
                    # (RNG-style pruning)
                    if all(cross[r, j, kk] < tv[r, j] for kk in keep):
                        keep.append(j)
                        if len(keep) == deg:
                            break
                while len(keep) < deg:
                    keep.append(keep[-1] if keep else 0)
                adj[s + r] = ti[r, keep]
            del sub, cross
        del sim
    return adj


def walk(q, C, part, X, adj, entries, ef, budget):
    """Greedy best-first search over the cell graph. Returns (top-10 ids, cells scanned).

    Same shape as an HNSW search with cells as nodes: pop the best unvisited cell, scan it exactly,
    push its neighbours.
    """
    visited = set()
    cand = []                                   # max-heap by centroid score
    for c in entries:
        s = float(q @ C[c])
        heapq.heappush(cand, (-s, int(c)))
        visited.add(int(c))
    best = []                                   # min-heap of (score, id)
    scanned = 0
    while cand and scanned < budget:
        negs, c = heapq.heappop(cand)
        mem = part.members(c)
        if mem.size:
            sc = X[mem] @ q
            for i in range(mem.size):
                v = float(sc[i])
                if len(best) < 10:
                    heapq.heappush(best, (v, int(mem[i])))
                elif v > best[0][0]:
                    heapq.heapreplace(best, (v, int(mem[i])))
        scanned += 1
        # push unvisited neighbours; ef caps the frontier so per-query work doesn't grow with N
        if len(visited) < ef * 8:
            for nb in adj[c]:
                nb = int(nb)
                if nb not in visited:
                    visited.add(nb)
                    heapq.heappush(cand, (-float(q @ C[nb]), nb))
        if len(cand) > ef:
            cand = heapq.nsmallest(ef, cand)
            heapq.heapify(cand)
    return [i for _, i in sorted(best, reverse=True)], scanned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--sizes", default="100000,500000,2000000")
    ap.add_argument("--cell-size", type=int, default=128)
    ap.add_argument("--deg", type=int, default=32)
    ap.add_argument("--efs", default="8,16,32,64,128,256,512")
    ap.add_argument("--entries", type=int, default=4)
    ap.add_argument("--seed-mode", default="rank",
                    choices=["rank", "fixed"],
                    help="rank = top-entries cells by centroid (a global "
                         "ranking, so the walk is redundant with flat); "
                         "fixed = one fixed entry cell, navigate in")
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--gt-queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--modes", default="flat,nearest,graph")
    ap.add_argument("--targets", default="0.85,0.90,0.95")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out_path = a.out or os.path.join(RESULTS_DIR, "cellgraph.json")
    sizes = [int(x) for x in a.sizes.split(",")]
    efs = [int(x) for x in a.efs.split(",")]
    modes = [m.strip() for m in a.modes.split(",")]
    targets = [float(t) for t in a.targets.split(",")]

    mm = np.load(a.corpus, mmap_mode="r")
    n_total = mm.shape[0]
    gt_q = np.ascontiguousarray(mm[n_total - a.gt_queries:], dtype=np.float32)
    Q = gt_q[:a.queries]

    res = {"meta": {"deg": a.deg, "cell_size": a.cell_size,
                    "queries": a.queries}, "runs": []}
    for n in sizes:
        n = min(n, n_total - a.gt_queries)
        gt = exact_gt(mm, n, gt_q, a.k)[:a.queries]
        gt10 = [set(g[:10].tolist()) for g in gt]
        X = np.ascontiguousarray(mm[:n], dtype=np.float32)
        M = max(16, n // a.cell_size)
        print(f"\n{'=' * 72}\nN={n:,}  cells={M:,}\n{'=' * 72}", flush=True)
        t0 = time.time()
        C, assign = spherical_kmeans(X, M, iters=12, seed=a.seed, verbose=False)
        part = Partition(C, assign, M)
        _log(f"partition {time.time() - t0:.0f}s")

        adjs = {}
        for mode in modes:
            if mode == "flat":
                continue
            t0 = time.time()
            adjs[mode] = build_adjacency(C, deg=a.deg,
                                         diversify=(mode == "graph"))
            _log(f"adjacency[{mode}] {time.time() - t0:.0f}s")

        Cd = torch.from_numpy(C).to(_dev())
        Qd = torch.from_numpy(Q).to(_dev())
        cent_rank = (Qd @ Cd.T).topk(min(4096, M), dim=1).indices.cpu().numpy()
        sizes_arr = part.sizes.astype(np.float64)
        # medoid cell: the single best-connected entry point, the
        # role HNSW's top layer plays.
        fixed_entry = np.array([int(np.argmax(C @ C.mean(0)))])

        rows = []
        for mode in modes:
            for ef in efs:
                if mode == "flat":
                    P = ef
                    if P > M:
                        continue
                    hit, sc = 0, 0.0
                    for i in range(len(Q)):
                        cells = cent_rank[i, :P]
                        mem = np.concatenate([part.members(c) for c in cells])
                        sc += mem.size
                        s = X[mem] @ Q[i]
                        top = mem[np.argpartition(-s, min(10, s.size - 1))[:10]]
                        hit += len(gt10[i] & set(int(v) for v in top))
                    r10, scanned = hit / (len(Q) * 10), sc / len(Q)
                else:
                    hit, sc = 0, 0
                    for i in range(len(Q)):
                        ent = (cent_rank[i, :a.entries] if a.seed_mode == "rank"
                               else fixed_entry)
                        ids, cells = walk(Q[i], C, part, X, adjs[mode],
                                          ent, ef, budget=ef)
                        hit += len(gt10[i] & set(ids))
                        sc += cells
                    r10 = hit / (len(Q) * 10)
                    scanned = (sc / len(Q)) * float(sizes_arr.mean())
                rows.append({"mode": mode, "ef": ef,
                             "recall@10": round(r10, 5),
                             "scanned": int(scanned)})
                print(f"    {mode:<8} ef/P={ef:<5} scanned~{int(scanned):>9,}  "
                      f"R@10={r10:.4f}", flush=True)
                if r10 >= 0.995:
                    break
        res["runs"].append({"n": n, "cells": M, "sweep": rows})
        json.dump(res, open(out_path, "w"), indent=1)
        del X, C, part, adjs
        torch.cuda.empty_cache()

    print(f"\n{'=' * 72}\nDIVERSIFIED CELL GRAPH vs FLAT RANKING\n{'=' * 72}")
    for tgt in targets:
        print(f"\n  R@10 >= {tgt}  (vectors scanned)")
        for mode in modes:
            ns, vs = [], []
            for r in sorted(res["runs"], key=lambda r: r["n"]):
                sub = [w for w in r["sweep"] if w["mode"] == mode]
                v = interp_at(sub, tgt, "scanned")
                if v is None:
                    continue
                ns.append(r["n"])
                vs.append(v)
            if len(ns) < 2:
                continue
            print(f"    {mode:<8} alpha={fit_alpha(ns, vs):+.3f}   " +
                  " ".join(f"N={x:,}:{v:,.0f}" for x, v in zip(ns, vs)))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
