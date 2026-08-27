"""bench_attractor.py: iterative attractor routing, i.e. graph-like navigation without a graph.

Everything tried so far (richer summaries, spilling, multiple tables) picks cells by ranking
fixed-size summaries globally. That has a floor: the number of cells that outrank the true one is
roughly M * P(summary error > gap), and as M = N/c grows a fixed-size summary can't keep that
constant. HNSW avoids it because each step only compares a handful of neighbours.

Idea: a query's true top-10 are close to each other, so a neighbour that is hard to reach from q may
be easy to reach from one we already found. The probe point moves with the modern Hopfield update
(Ramsauer et al. 2020), computed only over candidates already scanned:

    xi_{t+1} = X softmax(beta X^T xi_t)

    round 0:  probe p cells around q          -> candidates S
    round t:  xi <- normalise(q + gamma * sum softmax(beta q.x) x over top-m(S))
              probe p cells around xi         -> add to S
    answer:   top-k of S by exact q.x

Compared at equal scan work (T rounds of p probes vs one round of T*p). It only counts if the gap
grows with N; a constant-factor win doesn't change the scaling.

    python bench/bench_attractor.py hotpot_5M.npy --sizes 100000,500000,2000000
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


def attractor_search(ix, mm, Q, per_round, rounds, k=10, cand=64,
                     beta=24.0, gamma=1.0, expand=8):
    """T rounds of `per_round` probes, moving the probe point with a Hopfield step each round.

    Returns the candidate ids per query. The caller ranks them on exact q.x, so this only changes
    which cells get scanned.
    """
    nq = len(Q)
    found = [dict() for _ in range(nq)]        # id -> exact score
    xi = np.ascontiguousarray(Q, dtype=np.float32)
    xi_multi = None

    for t in range(rounds):
        if xi_multi is None:
            ids = batch_ids(ix, xi, cand, per_round)
        else:
            # Each of the `expand` best-so-far vectors is its own probe point,
            # splitting this round's cell budget between them.
            sub = max(1, per_round // expand)
            parts = []
            for e in range(expand):
                pk = np.ascontiguousarray(
                    [(p[e] if getattr(p, "ndim", 1) == 2 and len(p) > e
                      else (p if getattr(p, "ndim", 1) == 1 else p[-1]))
                     for p in xi_multi], dtype=np.float32)
                parts.append(batch_ids(ix, pk, cand, sub))
            ids = np.concatenate(parts, axis=1)
        # exact scores for everything seen this round
        flat = np.unique(ids[ids >= 0])
        if flat.size:
            vecs = np.ascontiguousarray(mm[flat], dtype=np.float32)
            pos = {int(v): i for i, v in enumerate(flat)}
            sc_all = Q @ vecs.T                       # [nq, |flat|]
            for i in range(nq):
                for v in ids[i]:
                    if v >= 0:
                        found[i][int(v)] = float(sc_all[i, pos[int(v)]])
        if t == rounds - 1:
            break
        # Hop from the neighbours we already found, not from a blend with q. A hard neighbour is
        # usually rank 1 for its own cell; it's only far from q in centroid terms. Blending toward q
        # just re-probes cells we already scanned (tried it, worse).
        pts = []
        for i in range(nq):
            if not found[i]:
                pts.append(xi[i])
                continue
            best = sorted(found[i].items(), key=lambda kv: -kv[1])[:expand]
            pts.append(np.ascontiguousarray(mm[[b for b, _ in best]],
                                            dtype=np.float32))
        xi_multi = pts
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--sizes", default="100000,500000,2000000")
    ap.add_argument("--rounds", default="1,2,3")
    ap.add_argument("--totals", default="16,32,64,128,256,512,1024,2048")
    ap.add_argument("--cell-size", type=int, default=128)
    ap.add_argument("--profile", default="memory")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--gt-queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--rerank", type=int, default=128)
    ap.add_argument("--beta", type=float, default=24.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--expand", type=int, default=8,
                    help="best-so-far vectors used as probe points per hop")
    ap.add_argument("--targets", default="0.90,0.95,0.98")
    ap.add_argument("--train-cap", type=int, default=1_000_000)
    ap.add_argument("--ingest-chunk", type=int, default=500_000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out_path = a.out or os.path.join(RESULTS_DIR, "attractor.json")
    sizes = [int(x) for x in a.sizes.split(",")]
    rounds = [int(x) for x in a.rounds.split(",")]
    totals = [int(x) for x in a.totals.split(",")]
    targets = [float(t) for t in a.targets.split(",")]

    mm = np.load(a.corpus, mmap_mode="r")
    n_total, d = mm.shape
    gt_q = np.ascontiguousarray(mm[n_total - a.gt_queries:], dtype=np.float32)
    Q = gt_q[:a.queries]

    res = {"meta": {"beta": a.beta, "gamma": a.gamma, "cell_size": a.cell_size,
                    "queries": a.queries}, "runs": []}
    if os.path.exists(out_path):
        try:
            res["runs"] = json.load(open(out_path))["runs"]
        except Exception:
            pass
    done = {(r["n"], r["rounds"]) for r in res["runs"]}

    for n in sizes:
        n = min(n, n_total - a.gt_queries)
        gt = exact_gt(mm, n, gt_q, a.k)[:a.queries]
        gt10 = [set(g[:10].tolist()) for g in gt]
        ix = None
        for T in rounds:
            if (n, T) in done:
                print(f"[att] skip N={n:,} T={T}")
                continue
            if ix is None:
                gc.collect()
                print(f"\n{'=' * 72}\nN={n:,}\n{'=' * 72}", flush=True)
                ix, bs, cells = build(mm[:n], a.cell_size, a.profile, a.rerank,
                                      a.train_cap, a.ingest_chunk)
                print(f"  build {bs:.1f}s  cells={cells:,}", flush=True)
            rows = []
            for tot in totals:
                per = max(1, tot // T)
                if per > cells:
                    continue
                found = attractor_search(ix, mm[:n], Q, per, T, k=10,
                                         beta=a.beta, gamma=a.gamma,
                                         expand=a.expand)
                hit = 0
                for i in range(len(Q)):
                    top = sorted(found[i].items(), key=lambda kv: -kv[1])[:10]
                    hit += len(gt10[i] & {a_ for a_, _ in top})
                r10 = hit / (len(Q) * 10)
                rows.append({"total_probes": per * T, "per_round": per,
                             "recall@10": round(r10, 5),
                             "scanned": int(per * T * a.cell_size)})
                print(f"    T={T} total={per * T:<5} ({per}/round) "
                      f"scanned~{per * T * a.cell_size:>9,}  R@10={r10:.4f}",
                      flush=True)
                if r10 >= 0.995:
                    break
            res["runs"].append({"n": n, "rounds": T, "cells": cells,
                                "max_recall": max(r["recall@10"] for r in rows),
                                "sweep": rows})
            json.dump(res, open(out_path, "w"), indent=1)
        if ix is not None:
            del ix
        gc.collect()

    print(f"\n{'=' * 72}\nDOES ATTRACTOR REFINEMENT MOVE THE EXPONENT?"
          f"\n{'=' * 72}")
    for tgt in targets:
        print(f"\n  target R@10 >= {tgt}   (vectors scanned, matched work)")
        print(f"    {'T':>3} {'alpha':>8} " +
              " ".join(f"{('N=' + f'{x:,}'):>14}" for x in sizes))
        for T in rounds:
            rs = sorted([r for r in res["runs"] if r["rounds"] == T],
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
            print(f"    {T:>3} {fit_alpha(ns, vs):>+8.3f} " +
                  " ".join(f"{v:>14,.0f}" for v in vs))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
