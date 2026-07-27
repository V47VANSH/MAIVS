"""bench_scale_full.py: the scaling study, 10k to 5.2M, both profiles.

Records everything needed to see how the index behaves as the corpus grows:

  build      train / ingest / compact seconds
  structure  cells, cell size, tree depth, tree bytes, arena bytes/vector
  accuracy   R@10, R@100, MRR against exact ground truth
  speed      serial qps, batch qps, p50 / p95 / p99, run-to-run spread
  memory     index bytes and process RSS delta
  mutation   delete latency

For the O(log N) question I look at three things: tree depth vs N, nprobe at fixed recall (should be
roughly flat), and p50 at fixed recall. Each is fitted as y ~ N^alpha. A log curve isn't a power
law, so what matters is alpha falling as N grows, not one value.

Notes:
* The corpus is shuffled and normalised on disk (make_hotpot_corpus.py), so the first N rows are a
  uniform sample at every scale.
* The same 1,000 held-out queries are used at every scale.
* The base stays memory-mapped. At 5.2M the f32 corpus is 8 GB and the quality index 8.5 GB, which
  don't both fit in 15 GB.

Results are written to JSON after every (scale, profile), so a crash at the top scale keeps
everything below it.

    python bench/bench_scale_full.py hotpot_5M.npy
    python bench/bench_scale_full.py hotpot_5M.npy --sizes 10000,100000 --profiles memory
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time

import numpy as np

from common import (PROFILES, RESULTS_DIR, TARGET_CELL_SIZE, build_index,
                    batch_ids, rss_mb, timed_batch, timed_serial)

GT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_scale_gt")


# --- ground truth: exact, streamed, GPU if available ---

def exact_gt(mm, n: int, queries: np.ndarray, k: int, chunk: int = 65536):
    """Exact top-k of `queries` over `mm[:n]`, computed in blocks so peak memory is one block.

    Cached per (n, k, n_queries); brute force at 5.2M is ~4e12 FLOP.
    """
    os.makedirs(GT_DIR, exist_ok=True)
    path = os.path.join(GT_DIR, f"gt_n{n}_q{len(queries)}_k{k}.npz")
    if os.path.exists(path):
        return np.load(path)["ids"]

    t0 = time.time()
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        torch, dev = None, "cpu"

    if torch is not None:
        torch.backends.cuda.matmul.allow_tf32 = False
        Q = torch.from_numpy(queries).to(dev)
        bv = torch.full((len(queries), k), -2.0, device=dev)
        bi = torch.zeros(len(queries), k, dtype=torch.long, device=dev)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            xb = torch.from_numpy(np.ascontiguousarray(mm[s:e])).to(dev)
            sc = Q @ xb.T
            idx = torch.arange(s, e, device=dev).unsqueeze(0).expand(len(queries), -1)
            cv = torch.cat([bv, sc], 1)
            ci = torch.cat([bi, idx], 1)
            tv, ti = cv.topk(k, dim=1)
            bv, bi = tv, ci.gather(1, ti)
            del xb, sc, idx, cv, ci
        ids = bi.cpu().numpy().astype(np.int64)
        del Q, bv, bi
        torch.cuda.empty_cache() if dev == "cuda" else None
    else:
        bv = np.full((len(queries), k), -2.0, dtype=np.float32)
        bi = np.zeros((len(queries), k), dtype=np.int64)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            sc = queries @ np.ascontiguousarray(mm[s:e]).T
            idx = np.arange(s, e)[None, :].repeat(len(queries), 0)
            cv = np.concatenate([bv, sc], 1)
            ci = np.concatenate([bi, idx], 1)
            part = np.argpartition(-cv, k, axis=1)[:, :k]
            bv = np.take_along_axis(cv, part, 1)
            bi = np.take_along_axis(ci, part, 1)
        ids = bi
    # recall@k only needs the sets, but MRR needs the order, so sort each row by score
    np.savez_compressed(path, ids=ids)
    print(f"  [gt] n={n:,} in {time.time() - t0:.0f}s ({dev})", flush=True)
    return ids


def probe_grid(cells: int, cap: int = 8192):
    g = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    out = [p for p in g if p < min(cells, cap)]
    return out + [min(cells, cap)]


def fit_alpha(xs, ys):
    """Least-squares slope of log y on log x; None if degenerate."""
    pts = [(x, y) for x, y in zip(xs, ys) if x and y and x > 0 and y > 0]
    if len(pts) < 2:
        return None
    lx = np.log(np.array([p[0] for p in pts], dtype=float))
    ly = np.log(np.array([p[1] for p in pts], dtype=float))
    return float(np.polyfit(lx, ly, 1)[0])


def interp_at(rows, target, field):
    """Value of `field` where recall@10 first reaches `target`, interpolated."""
    pts = sorted((r["recall@10"], r[field]) for r in rows)
    for i in range(1, len(pts)):
        (r0, v0), (r1, v1) = pts[i - 1], pts[i]
        if r0 <= target <= r1 and r1 > r0:
            f = (target - r0) / (r1 - r0)
            return v0 + f * (v1 - v0)
    # If every point already clears the target, the cost of the target is the cheapest config, i.e.
    # the lowest-recall point. Returning pts[-1] reported the most expensive setting instead, and
    # gave negative scaling exponents for systems whose grid starts above the target (faiss HNSW at
    # ef=16).
    if pts and pts[0][0] >= target:
        return pts[0][1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--sizes", default="10000,25000,50000,100000,250000,"
                                       "500000,1000000,2000000,3500000,5232329")
    ap.add_argument("--profiles", default="memory,quality")
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--lat-queries", type=int, default=300,
                    help="queries for the serial latency pass; the batch pass "
                         "always uses all of them")
    ap.add_argument("--train-cap", type=int, default=1_000_000)
    ap.add_argument("--ingest-chunk", type=int, default=500_000,
                    help="every add_batch call ends in refresh_derived, which "
                         "is an O(N*d) pass over the arena, so ingestion costs "
                         "chunks*N*d -- use few, large chunks")
    ap.add_argument("--targets", default="0.90,0.95,0.99")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = a.out or os.path.join(RESULTS_DIR, "scaling_full.json")
    sizes = [int(x) for x in a.sizes.split(",")]
    profiles = [p.strip() for p in a.profiles.split(",")]
    targets = [float(t) for t in a.targets.split(",")]

    mm = np.load(a.corpus, mmap_mode="r")
    n_total, d = mm.shape
    # Held-out queries: the tail of the file, never part of any base, the same at every scale. The
    # file is shuffled on disk, so this is a random slice.
    queries = np.ascontiguousarray(mm[n_total - a.queries:], dtype=np.float32)
    max_base = n_total - a.queries
    sizes = [min(s, max_base) for s in sizes]
    print(f"[scale] corpus {n_total:,} x {d}  base<= {max_base:,}  "
          f"queries {len(queries):,}  profiles {profiles}")

    results = {"meta": {"corpus": os.path.basename(a.corpus), "d": d,
                        "n_total": int(n_total), "n_queries": len(queries),
                        "k": a.k, "train_cap": a.train_cap,
                        "target_cell_size": TARGET_CELL_SIZE,
                        "profiles": {p: PROFILES[p] for p in profiles},
                        "threads": int(os.environ.get("RAYON_NUM_THREADS", 0))},
               "runs": []}
    if os.path.exists(out_path):
        try:
            prev = json.load(open(out_path))
            done = {(r["n"], r["profile"]) for r in prev.get("runs", [])}
            results["runs"] = prev["runs"]
            print(f"[scale] resuming; {len(done)} configs already done")
        except Exception:
            done = set()
    else:
        done = set()

    for n in sizes:
        gt = exact_gt(mm, n, queries, a.k)
        for prof in profiles:
            if (n, prof) in done:
                print(f"[scale] skip N={n:,} {prof} (already recorded)")
                continue
            gc.collect()
            rss0 = rss_mb()
            print(f"\n{'=' * 78}\nN={n:,}  profile={prof}\n{'=' * 78}", flush=True)
            base = mm[:n]
            try:
                ix, tim = build_index(base, prof, train_cap=a.train_cap,
                                      ingest_chunk=a.ingest_chunk)
            except Exception as exc:                     # OOM at the top scale
                print(f"  !! build failed: {type(exc).__name__}: {exc}")
                results["runs"].append({"n": n, "profile": prof,
                                        "error": f"{type(exc).__name__}: {exc}"})
                json.dump(results, open(out_path, "w"), indent=1)
                gc.collect()
                continue
            st = ix.stats()
            rss_peak = rss_mb()

            cells = int(st["n_cells"])
            grid = probe_grid(cells)
            qlat = queries[:min(a.lat_queries, len(queries))]
            rows = []
            for npb in grid:
                from groundtruth import recall_at_k, mean_reciprocal_rank
                _, lat = timed_serial(
                    lambda q, m=npb: ix.search(q, 10, m, 1.0, 8.0)[0], qlat, 10)
                bret, bqps, spread = timed_batch(
                    lambda qq, m=npb: batch_ids(ix, qq, 10, m), queries)
                rows.append({
                    "nprobe": npb,
                    "recall@10": round(recall_at_k(bret, gt, 10), 5),
                    "mrr": round(mean_reciprocal_rank(bret, gt), 5),
                    "serial_qps": round(1000.0 / float(np.mean(lat)), 1),
                    "batch_qps": round(bqps, 1),
                    "p50_ms": round(float(np.percentile(lat, 50)), 4),
                    "p95_ms": round(float(np.percentile(lat, 95)), 4),
                    "p99_ms": round(float(np.percentile(lat, 99)), 4),
                    "spread_pct": round(spread, 1),
                    "scanned_est": int(npb * st["cell_size_mean"]),
                })
                r = rows[-1]
                print(f"    np={npb:<6} R@10={r['recall@10']:.4f} "
                      f"MRR={r['mrr']:.4f} serial={r['serial_qps']:>7.0f} "
                      f"batch={r['batch_qps']:>8.0f} p50={r['p50_ms']:.3f}ms "
                      f"p99={r['p99_ms']:.3f}ms +-{r['spread_pct']:.0f}%",
                      flush=True)

            # delete latency, on live data, then reinsert nothing (index is
            # discarded straight after)
            nd = min(2000, n // 10)
            t0 = time.perf_counter()
            for i in range(nd):
                ix.remove(int(i))
            del_us = (time.perf_counter() - t0) / max(nd, 1) * 1e6

            run = {
                "n": n, "profile": prof,
                "cells": cells,
                "cell_size_mean": round(float(st["cell_size_mean"]), 2),
                "cell_size_max": int(st["cell_size_max"]),
                "tree_depth": int(st.get("tree_depth", 0)),
                "tree_bytes": int(st.get("tree_bytes", 0)),
                "tree_top_width": int(st.get("tree_top_width", 0)),
                "arena_b_per_vec": round(float(st["arena_bytes_per_vec"]), 1),
                "scan_b_per_vec": round(float(st["scan_bytes_per_vec"]), 1),
                "index_mb": round(float(st["arena_bytes_per_vec"]) * n / 1e6, 1),
                "rss_delta_mb": round(rss_peak - rss0, 1),
                "build": {k: round(v, 3) for k, v in tim.items()},
                "delete_us": round(del_us, 3),
                "max_recall": max(r["recall@10"] for r in rows),
                "sweep": rows,
            }
            results["runs"].append(run)
            json.dump(results, open(out_path, "w"), indent=1)
            print(f"  built {tim['total_s']:.1f}s  tree_depth={run['tree_depth']}  "
                  f"arena={run['arena_b_per_vec']:.0f} B/vec  "
                  f"index={run['index_mb']:,.0f} MB  delete={del_us:.2f}us  "
                  f"maxR@10={run['max_recall']:.4f}")
            del ix
            gc.collect()

    # --- report ---
    print(f"\n{'=' * 78}\nSCALING SUMMARY\n{'=' * 78}")
    for prof in profiles:
        rs = [r for r in results["runs"] if r["profile"] == prof and "sweep" in r]
        rs.sort(key=lambda r: r["n"])
        if not rs:
            continue
        print(f"\nprofile = {prof}")
        print(f"{'N':>10} {'cells':>8} {'depth':>6} {'build s':>8} "
              f"{'idx MB':>8} {'maxR@10':>8} {'del us':>7}")
        for r in rs:
            print(f"{r['n']:>10,} {r['cells']:>8,} {r['tree_depth']:>6} "
                  f"{r['build']['total_s']:>8.1f} {r['index_mb']:>8,.0f} "
                  f"{r['max_recall']:>8.4f} {r['delete_us']:>7.2f}")
        for tgt in targets:
            ns, nps, lats, scans = [], [], [], []
            for r in rs:
                v = interp_at(r["sweep"], tgt, "nprobe")
                if v is None:
                    continue
                ns.append(r["n"])
                nps.append(v)
                lats.append(interp_at(r["sweep"], tgt, "p50_ms"))
                scans.append(interp_at(r["sweep"], tgt, "scanned_est"))
            if len(ns) < 2:
                continue
            print(f"  R@10>={tgt}:  alpha(nprobe)={fit_alpha(ns, nps)!s:>7.7} "
                  f" alpha(scanned)={fit_alpha(ns, scans)!s:>7.7} "
                  f" alpha(p50)={fit_alpha(ns, lats)!s:>7.7}"
                  f"   over N={ns[0]:,}..{ns[-1]:,}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
