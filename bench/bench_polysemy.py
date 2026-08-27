"""bench_polysemy.py: orthogonal polysemy end to end in the engine.

Same partition, profile, queries and ground truth for every arm. Only the extra homes differ:

  none     one home
  near     + next-nearest home (the old spill rule)
  ortho    + orthogonal-residual home (set_orthogonal_polysemy)

Reports build time, memory, delete cost and batch QPS at matched R@10 on held-out passages (IND) and
on real HotpotQA questions (OOD).

    python bench/bench_polysemy.py hotpot_5M.npy --n 250000 --questions bench/data/hotpot_questions_minilm.npy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from common import PROFILES, RESULTS_DIR, TARGET_CELL_SIZE, batch_ids, timed_batch  # noqa: E402

K = 10


def build(base, arm, rf, lam, profile="memory"):
    import maivs
    cfg = PROFILES[profile]
    ix = maivs.FlatCellIndex(base.shape[1], cfg["scan"], rf, 0x5EED, 0)
    if cfg["per_dim_sq"]:
        ix.set_per_dim_sq(True)
    ix.set_two_stage(True, 2)
    ix.set_graph_route(False, 10 ** 9)
    ix.set_tree_route(True, 2048, 16)
    ix.set_adaptive(True, 0.02)
    if arm == "near":
        ix.set_replication(2, 1.0)                 # margin 1.0 = always take 2nd nearest
    elif arm.startswith("ortho"):
        ix.set_orthogonal_polysemy(2, lam)
    n_cells = max(16, len(base) // TARGET_CELL_SIZE)
    t0 = time.perf_counter()
    ix.train_centroids(base, n_cells)
    t_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    ix.add_batch(list(range(len(base))), base)    # one call: refresh_derived once
    t_ingest = time.perf_counter() - t0
    t0 = time.perf_counter()
    ix.compact()
    t_compact = time.perf_counter() - t0
    return ix, {"train_s": t_train, "ingest_s": t_ingest, "compact_s": t_compact,
                "total_s": t_train + t_ingest + t_compact}


def recall(ids, gt):
    return float(np.mean([len(set(a[:K]) & set(b[:K])) / K for a, b in zip(ids, gt)]))


def at_recall(curve, target):
    """QPS at a recall target, log-interpolated between sweep points."""
    pts = sorted(curve, key=lambda r: r["nprobe"])
    for a, b in zip(pts, pts[1:]):
        if a["recall"] <= target <= b["recall"]:
            t = (target - a["recall"]) / max(b["recall"] - a["recall"], 1e-9)
            return float(np.exp(np.log(a["qps"]) + t * (np.log(b["qps"]) - np.log(a["qps"]))))
    if pts and pts[0]["recall"] >= target:
        return pts[0]["qps"]
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--n", type=int, default=250_000)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--arms", default="none,near,ortho1")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--rf", type=int, default=64)
    ap.add_argument("--profile", default="memory", choices=list(PROFILES))
    ap.add_argument("--probes", default="16,32,64,128,256,512,1024")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()

    import faiss
    mm = np.load(a.corpus, mmap_mode="r")
    base = np.ascontiguousarray(mm[:a.n], dtype=np.float32)
    qsets = {"IND": np.ascontiguousarray(mm[-1000:], dtype=np.float32),
             "OOD": np.ascontiguousarray(np.load(a.questions)[:1000], dtype=np.float32)}
    fi = faiss.IndexFlatIP(base.shape[1]); fi.add(base)
    gts = {k: fi.search(v, K)[1] for k, v in qsets.items()}
    del fi
    probes = [int(p) for p in a.probes.split(",")]
    targets = (0.90, 0.95, 0.98, 0.99, 0.995)

    out = {"n": a.n, "rf": a.rf, "lam": a.lam, "profile": a.profile, "arms": {}}
    for arm in a.arms.split(","):
        lam = float(arm[5:]) if arm.startswith("ortho") and len(arm) > 5 else a.lam
        ix, tm = build(base, arm, a.rf, lam, a.profile)
        st = ix.stats()
        homes = 2 if arm != "none" else 1
        mem = st["arena_bytes_per_vec"] + 4 * (homes - 1)   # + member-list entry per extra home
        res = {"build": tm, "bytes_per_vec": mem, "cells": st["n_cells"],
               "tree_depth": st.get("tree_depth"), "curves": {}}
        print(f"\n== {arm}  build {tm['total_s']:.1f}s (train {tm['train_s']:.1f} + ingest "
              f"{tm['ingest_s']:.1f} + compact {tm['compact_s']:.1f})  {mem:.0f} B/vec", flush=True)
        for qn, Q in qsets.items():
            curve = []
            for p in probes:
                if p > st["n_cells"]:
                    continue
                ids, qps, spread = timed_batch(lambda qq: batch_ids(ix, qq, K, p), Q, a.repeats)
                r = recall(ids, gts[qn])
                curve.append({"nprobe": p, "recall": r, "qps": qps, "spread_pct": spread})
            res["curves"][qn] = curve
            print(f"  {qn}: " + "  ".join(f"np{c['nprobe']}:{c['recall']:.4f}@{c['qps']:.0f}" for c in curve), flush=True)
            print(f"  {qn} QPS at R@10 " + "  ".join(f"{t}:{at_recall(curve, t):.0f}" for t in targets), flush=True)
        # delete: O(1) and must unlink EVERY home
        rng = np.random.default_rng(1)
        victims = rng.choice(a.n, 2000, replace=False)
        t0 = time.perf_counter()
        for v in victims:
            ix.remove(int(v))
        res["delete_us"] = (time.perf_counter() - t0) / len(victims) * 1e6
        ids = batch_ids(ix, qsets["IND"], K, probes[-1])
        leaked = int(np.isin(ids, victims).sum())
        res["delete_leaked"] = leaked
        print(f"  delete {res['delete_us']:.2f} us/op, leaked after delete = {leaked}", flush=True)
        out["arms"][arm] = res
        del ix

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"polysemy_{a.profile}_rf{a.rf}_n{a.n}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
