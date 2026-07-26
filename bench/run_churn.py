"""run_churn.py: recall and throughput under insert/delete churn.

Everything else in bench/ builds once and queries forever, which is where graph indexes do best.
What I want to know here is whether recall and latency stay flat under continuous inserts and
deletes, without rebuilds.

Loosely follows the NeurIPS'23 Big-ANN streaming track: start with half the corpus indexed, then run
cycles of (delete a random slice, insert a fresh slice from the reserve). The live set stays the
same size, so any recall change comes from churn, not size. Exact ground truth is recomputed for the
current live set at every checkpoint; stale ground truth would hide the degradation we're looking
for.

Baselines:
  faiss-hnsw   IndexHNSWFlat has no remove_ids, so a delete means a full rebuild; that cost is
               measured
  faiss-ivf    IVFFlat remove_ids, an O(N) scan of every list
  ours         swap_remove from the owning cell + arena free list, O(1)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, _ROOT)

from groundtruth import compute_groundtruth, normalize, recall_at_k  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def exact_gt(active_vecs, active_ids, queries, k):
    """Exact top-k over the CURRENT active set, mapped back to external ids."""
    ids, _ = compute_groundtruth(active_vecs, queries, k=k, verbose=False)
    return active_ids[ids]


def measure(name, search_fn, queries, gt_ids, k):
    t0 = time.perf_counter()
    got = search_fn(queries)
    dt = time.perf_counter() - t0
    return recall_at_k(got, gt_ids, k), len(queries) / dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--churn", type=float, default=0.10,
                    help="fraction of the active set replaced per cycle")
    ap.add_argument("--cells", type=int, default=314)
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--rerank", type=int, default=8)
    args = ap.parse_args()

    import faiss
    import aigvs_rust

    # np.load(mmap) is read-only; materialise a writable copy for in-place norm.
    X = normalize(np.array(np.load(args.corpus, mmap_mode="r"), dtype=np.float32),
                  inplace=True)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    q_idx, rest = perm[:args.queries], perm[args.queries:]
    queries = np.ascontiguousarray(X[np.sort(q_idx)])

    half = len(rest) // 2
    live = list(rest[:half])          # currently indexed external ids
    reserve = list(rest[half:])       # not yet inserted
    d = X.shape[1]
    print(f"active={len(live):,}  reserve={len(reserve):,}  queries={len(queries)}  "
          f"churn={args.churn:.0%}/cycle x {args.cycles}")

    # --- build our index ---
    n_train = min(len(live), max(args.cells * 256, 50_000), 262_144)
    tr = np.ascontiguousarray(X[np.sort(rng.choice(live, n_train, replace=False))])
    km = faiss.Kmeans(d, args.cells, niter=15, spherical=True, seed=0)
    km.train(tr)
    cent = np.ascontiguousarray(km.centroids); cent /= np.linalg.norm(cent, axis=1, keepdims=True)

    ix = aigvs_rust.FlatCellIndex(d, "binary", args.rerank)
    ix.set_i8_rerank(True)
    ix.set_centroids(cent)
    live0 = np.sort(np.array(live))
    ix.add_batch([int(i) for i in live0], np.ascontiguousarray(X[live0]))

    # --- build our index ---
    quant = faiss.IndexFlatIP(d)
    ivf = faiss.IndexIVFFlat(quant, d, args.cells, faiss.METRIC_INNER_PRODUCT)
    ivf.train(tr)
    # IndexIVFFlat supports external ids directly. Wrapping it in IndexIDMap2 trips 'j ==
    # index->ntotal' in remove_ids (faiss IndexIDMap.cpp:265), so use the IVF directly.
    ivf.add_with_ids(np.ascontiguousarray(X[live0]), live0.astype(np.int64))
    ivf.nprobe = args.nprobe

    def v2_search(qq):
        ids, _ = ix.search_batch(np.ascontiguousarray(qq), args.k, args.nprobe, 1.0, 8.0)
        return np.asarray(ids, dtype=np.int64).reshape(len(qq), args.k)

    def ivf_search(qq):
        return ivf.search(np.ascontiguousarray(qq), args.k)[1]

    rows = []
    hnsw_rebuild_total = 0.0
    for cycle in range(args.cycles + 1):
        act = np.sort(np.array(live))
        gt = exact_gt(np.ascontiguousarray(X[act]), act, queries, args.k)

        r_v2, q_v2 = measure("v2", v2_search, queries, gt, args.k)
        r_iv, q_iv = measure("ivf", ivf_search, queries, gt, args.k)

        # HNSW cannot delete: the only correct option is a full rebuild.
        t0 = time.perf_counter()
        h = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        h.hnsw.efConstruction = 200
        h.add(np.ascontiguousarray(X[act]))
        rebuild_s = time.perf_counter() - t0
        hnsw_rebuild_total += rebuild_s
        h.hnsw.efSearch = 32
        got = act[h.search(np.ascontiguousarray(queries), args.k)[1]]
        r_hn = recall_at_k(got, gt, args.k)
        del h; gc.collect()

        rows.append(dict(cycle=cycle, n=len(live), v2_recall=round(r_v2, 4),
                         ivf_recall=round(r_iv, 4), hnsw_recall=round(r_hn, 4),
                         v2_qps=round(q_v2), ivf_qps=round(q_iv),
                         hnsw_rebuild_s=round(rebuild_s, 2)))
        print(f"  cycle {cycle:>2}  n={len(live):,}   "
              f"v2 R@{args.k}={r_v2:.4f} ({q_v2:>7.0f} qps)   "
              f"ivf={r_iv:.4f} ({q_iv:>6.0f} qps)   "
              f"hnsw={r_hn:.4f} (rebuilt in {rebuild_s:.1f}s)")

        if cycle == args.cycles:
            break

        # --- churn: delete a random slice, insert fresh vectors ---
        nch = int(len(live) * args.churn)
        nch = min(nch, len(reserve))
        victims = rng.choice(len(live), nch, replace=False)
        vic_ids = [live[i] for i in victims]

        t0 = time.perf_counter()
        for vid in vic_ids:
            ix.remove(int(vid))
        v2_del_us = (time.perf_counter() - t0) / max(nch, 1) * 1e6
        t0 = time.perf_counter()
        ivf.remove_ids(faiss.IDSelectorBatch(
            np.array(vic_ids, dtype=np.int64)))
        ivf_del_us = (time.perf_counter() - t0) / max(nch, 1) * 1e6

        for i in sorted(victims, reverse=True):
            live.pop(i)
        newcomers = [reserve.pop() for _ in range(nch)]

        add = np.sort(np.array(newcomers))
        t0 = time.perf_counter()
        ix.add_batch([int(i) for i in add], np.ascontiguousarray(X[add]))
        v2_add_us = (time.perf_counter() - t0) / max(nch, 1) * 1e6
        ivf.add_with_ids(np.ascontiguousarray(X[add]), add.astype(np.int64))
        live.extend(newcomers)

        print(f"            churn {nch:,}: v2 del {v2_del_us:.2f}us/op add "
              f"{v2_add_us:.2f}us/op | ivf del {ivf_del_us:.2f}us/op")

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "churn.json")
    with open(out, "w") as f:
        json.dump({"config": vars(args), "rows": rows,
                   "hnsw_rebuild_total_s": round(hnsw_rebuild_total, 1)}, f, indent=2)

    v2r = [r["v2_recall"] for r in rows]
    ivr = [r["ivf_recall"] for r in rows]
    print(f"\n{'='*70}\nRECALL DRIFT over {args.cycles} cycles "
          f"({args.cycles*args.churn:.0%} of the index replaced)")
    print(f"  v2        {v2r[0]:.4f} -> {v2r[-1]:.4f}   (drift {v2r[-1]-v2r[0]:+.4f})")
    print(f"  faiss-ivf {ivr[0]:.4f} -> {ivr[-1]:.4f}   (drift {ivr[-1]-ivr[0]:+.4f})")
    print(f"  faiss-hnsw held at {rows[-1]['hnsw_recall']:.4f} ONLY by rebuilding "
          f"every cycle: {hnsw_rebuild_total:.0f}s total")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
