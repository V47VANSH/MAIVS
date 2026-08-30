"""Update workload: 25% turnover of a live index, in 10 rounds.

    python bench/fair/churn.py SYSTEM

SYSTEM is maivs, maivs-memory, scann, scann-soar, hnswlib or faiss-hnsw.

  S0 = HotpotQA rows [0, 400k)                      the initial index
  churn: 10 rounds of (delete 10k random S0 ids, insert 10k new rows from [400k, 500k))
  S1 = (S0 minus the 100k deleted) + 100k new       400k live; exact ground truth over S1

For each system: the build on S0; delete and insert wall time (ops/s); the recall/QPS grid on the
churned index; the same after the system's own maintenance step (MAIVS compact(), ScaNN
rebalance()); and a fresh build on S1 for reference. faiss HNSW has no delete, so its only update
path is a full rebuild, and that's what it reports. Batch QPS = 1,000 queries, 16 threads, best
of 3.

scann-soar crashes inside ScaNN 1.4.2 (see scann_soar_crash.py), so the results use scann.
"""
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fairbench import THREADS, K, DATA, OUTD, recall, best_of, rss_mb  # noqa: E402
import numpy as np  # noqa: E402

N0, NNEW, ROUNDS = 400_000, 100_000, 10


def exact_gt(b, Q, chunk=131072):
    best_s = np.full((len(Q), K), -9.0, np.float32)
    best_i = np.zeros((len(Q), K), np.int64)
    for s in range(0, len(b), chunk):
        S = Q @ b[s:s + chunk].T
        idx = np.argpartition(-S, K, axis=1)[:, :K]
        sc = np.take_along_axis(S, idx, 1)
        cs = np.concatenate([best_s, sc], 1)
        ci = np.concatenate([best_i, idx + s], 1)
        o = np.argsort(-cs, axis=1)[:, :K]
        best_s = np.take_along_axis(cs, o, 1)
        best_i = np.take_along_axis(ci, o, 1)
    return best_i


def load():
    base = np.load(os.path.join(DATA, "base_500k.npy"), mmap_mode="r")
    base = np.ascontiguousarray(base[:N0 + NNEW], np.float32)
    rng = np.random.default_rng(1)
    dele = rng.choice(N0, NNEW, replace=False)
    keep = np.ones(N0 + NNEW, bool)
    keep[dele] = False
    s1 = np.nonzero(keep)[0]
    qs = {"IND": np.load(os.path.join(DATA, "IND.npy")), "OOD": np.load(os.path.join(DATA, "OOD.npy"))}
    gp = os.path.join(DATA, "gt_churn.npz")
    if not os.path.exists(gp):
        b1 = base[s1]
        np.savez(gp, IND=s1[exact_gt(b1, qs["IND"])], OOD=s1[exact_gt(b1, qs["OOD"])])
        del b1
    g = np.load(gp)
    return base, dele, s1, qs, {"IND": g["IND"], "OOD": g["OOD"]}


def grid(search, params, qs, gts, label):
    out = {}
    for qn, Q in qs.items():
        pts = []
        for p in params:
            ids, dt = best_of(lambda: search(Q, p))
            r = {"param": str(p), "recall": recall(ids, gts[qn]), "qps": 1000.0 / dt}
            pts.append(r)
            print(f"    [{label}] {qn} {str(p):<14} R@10 {r['recall']:.4f} {r['qps']:9.0f} qps", flush=True)
        out[qn] = pts
    return out


# ---------------------------------------------------------------- systems
MAIVS_PROFILE = "quality"
SCANN_SOAR, SCANN_INT8 = True, False


def maivs_build(X, ids):
    import maivs
    ix = maivs.FlatCellIndex(X.shape[1], "binary", 32, 0x5EED, 0)
    if MAIVS_PROFILE == "memory":
        ix.set_per_dim_sq(True)
    ix.set_orthogonal_polysemy(2, 1.0)
    ix.set_fastscan(True)
    ix.set_two_stage(True, 2)
    ix.set_graph_route(False, 10 ** 9)
    ix.set_tree_route(True, 2048, 16)
    ix.set_adaptive(True, 0.02)
    t0 = time.perf_counter()
    ix.train_centroids(X, max(16, len(X) // 128))
    ix.add_batch([int(i) for i in ids], X)
    ix.compact()
    return ix, time.perf_counter() - t0


MAIVS_GRID = [(p, rf) for rf in (16, 32) for p in (64, 128, 256, 512, 1024, 2048)]


def maivs_search(ix):
    def f(Q, p):
        ix.set_rerank_factor(p[1])
        ids, _ = ix.search_batch(Q, K, p[0], 1.0, 8.0, True)
        return ids
    return f


def run_maivs(base, dele, s1, qs, gts):
    res = {}
    ix, b = maivs_build(base[:N0], np.arange(N0))
    res["build_s0_s"] = b
    td = ti = 0.0
    for r in range(ROUNDS):
        d = dele[r * 10_000:(r + 1) * 10_000]
        new = np.arange(N0 + r * 10_000, N0 + (r + 1) * 10_000)
        t0 = time.perf_counter()
        for i in d:
            ix.remove(int(i))
        td += time.perf_counter() - t0
        t0 = time.perf_counter()
        ix.add_batch([int(i) for i in new], np.ascontiguousarray(base[new]))
        ti += time.perf_counter() - t0
    res.update(delete_s=td, insert_s=ti, live=len(ix))
    print(f"  churn: {NNEW:,} deletes {td:.2f}s, {NNEW:,} inserts {ti:.2f}s, live {len(ix):,}", flush=True)
    res["churned"] = grid(maivs_search(ix), MAIVS_GRID, qs, gts, "churned")
    t0 = time.perf_counter()
    ix.compact()
    res["maintain_s"] = time.perf_counter() - t0
    res["maintain"] = "compact()"
    print(f"  compact {res['maintain_s']:.2f}s", flush=True)
    res["maintained"] = grid(maivs_search(ix), MAIVS_GRID, qs, gts, "compacted")
    del ix
    gc.collect()
    ix, b = maivs_build(np.ascontiguousarray(base[s1]), s1)
    res["fresh_build_s"] = b
    res["fresh"] = grid(maivs_search(ix), MAIVS_GRID, qs, gts, "fresh")
    return res


SCANN_GRID = [(lv, rr) for rr in (100, 250) for lv in (8, 16, 32, 64, 128, 256)]


def scann_build(X, ids):
    import scann
    from scann.scann_ops.py.scann_builder import ReorderType
    t0 = time.perf_counter()
    b = scann.scann_ops_pybind.builder(X, K, "dot_product")
    if hasattr(b, "set_n_training_threads"):
        b = b.set_n_training_threads(THREADS)
    kw = dict(num_leaves=len(X) // 250, num_leaves_to_search=32, training_sample_size=min(len(X), 250_000))
    if SCANN_SOAR:
        kw.update(soar_lambda=1.5, overretrieve_factor=2.0)
    rq = ReorderType.INT8 if SCANN_INT8 else ReorderType.FLOAT32
    s = (b.tree(**kw).score_ah(2, anisotropic_quantization_threshold=0.2).reorder(100, quantize=rq)
         .build(docids=[int(i) for i in ids]))
    return s, time.perf_counter() - t0


def scann_search(s):
    def f(Q, p):
        ids, _ = s.search_batched_parallel(Q, final_num_neighbors=K, leaves_to_search=p[0],
                                           pre_reorder_num_neighbors=p[1], batch_size=128)
        return ids
    return f


def run_scann(base, dele, s1, qs, gts):
    res = {}
    s, b = scann_build(base[:N0], np.arange(N0))
    res["build_s0_s"] = b
    td = ti = 0.0
    for r in range(ROUNDS):
        d = dele[r * 10_000:(r + 1) * 10_000]
        new = np.arange(N0 + r * 10_000, N0 + (r + 1) * 10_000)
        t0 = time.perf_counter()
        s.delete([int(i) for i in d])
        td += time.perf_counter() - t0
        t0 = time.perf_counter()
        s.upsert([int(i) for i in new], np.ascontiguousarray(base[new]), batch_size=256)
        ti += time.perf_counter() - t0
    res.update(delete_s=td, insert_s=ti, live=int(s.size()))
    print(f"  churn: {NNEW:,} deletes {td:.2f}s, {NNEW:,} upserts {ti:.2f}s, live {s.size():,}", flush=True)
    res["churned"] = grid(scann_search(s), SCANN_GRID, qs, gts, "churned")
    t0 = time.perf_counter()
    s.rebalance()
    res["maintain_s"] = time.perf_counter() - t0
    res["maintain"] = "rebalance() (full retrain)"
    print(f"  rebalance {res['maintain_s']:.2f}s", flush=True)
    res["maintained"] = grid(scann_search(s), SCANN_GRID, qs, gts, "rebalanced")
    del s
    gc.collect()
    s, b = scann_build(np.ascontiguousarray(base[s1]), s1)
    res["fresh_build_s"] = b
    res["fresh"] = grid(scann_search(s), SCANN_GRID, qs, gts, "fresh")
    return res


EF_GRID = (32, 64, 128, 256, 512, 1024)


def hnswlib_build(X, ids, cap):
    import hnswlib
    t0 = time.perf_counter()
    h = hnswlib.Index(space="ip", dim=X.shape[1])
    h.init_index(max_elements=cap, ef_construction=200, M=32, allow_replace_deleted=True)
    h.set_num_threads(THREADS)
    h.add_items(X, ids)
    return h, time.perf_counter() - t0


def hnswlib_search(h):
    def f(Q, ef):
        h.set_ef(ef)
        ids, _ = h.knn_query(Q, k=K, num_threads=THREADS)
        return ids
    return f


def run_hnswlib(base, dele, s1, qs, gts):
    res = {}
    h, b = hnswlib_build(base[:N0], np.arange(N0), N0)
    res["build_s0_s"] = b
    td = ti = 0.0
    for r in range(ROUNDS):
        d = dele[r * 10_000:(r + 1) * 10_000]
        new = np.arange(N0 + r * 10_000, N0 + (r + 1) * 10_000)
        t0 = time.perf_counter()
        for i in d:
            h.mark_deleted(int(i))
        td += time.perf_counter() - t0
        t0 = time.perf_counter()
        h.add_items(np.ascontiguousarray(base[new]), new, replace_deleted=True, num_threads=THREADS)
        ti += time.perf_counter() - t0
    res.update(delete_s=td, insert_s=ti, live=N0)
    print(f"  churn: {NNEW:,} mark_deleted {td:.2f}s, {NNEW:,} inserts (replace_deleted) {ti:.2f}s", flush=True)
    res["churned"] = grid(hnswlib_search(h), EF_GRID, qs, gts, "churned")
    res["maintain"] = "none available"
    del h
    gc.collect()
    h, b = hnswlib_build(np.ascontiguousarray(base[s1]), s1, len(s1))
    res["fresh_build_s"] = b
    res["fresh"] = grid(hnswlib_search(h), EF_GRID, qs, gts, "fresh")
    return res


def run_faiss_hnsw(base, dele, s1, qs, gts):
    import faiss
    faiss.omp_set_num_threads(THREADS)
    res = {"maintain": "no delete support: update = full rebuild"}
    ix = faiss.IndexIDMap(faiss.IndexHNSWFlat(base.shape[1], 32, faiss.METRIC_INNER_PRODUCT))
    faiss.downcast_index(ix.index).hnsw.efConstruction = 200
    t0 = time.perf_counter()
    ix.add_with_ids(np.ascontiguousarray(base[s1]), s1.astype(np.int64))
    res["fresh_build_s"] = time.perf_counter() - t0
    hn = faiss.downcast_index(ix.index)

    def f(Q, ef):
        hn.hnsw.efSearch = ef
        return ix.search(Q, K)[1]
    res["fresh"] = grid(f, EF_GRID, qs, gts, "rebuild")
    return res


def main():
    global MAIVS_PROFILE, SCANN_SOAR, SCANN_INT8
    system = sys.argv[1]
    MAIVS_PROFILE = "memory" if system == "maivs-memory" else "quality"
    SCANN_SOAR = system in ("scann-soar", "scann-soar-int8")
    SCANN_INT8 = system.endswith("int8")
    base, dele, s1, qs, gts = load()
    print(f"== churn {system}: S0 {N0:,} -> S1 {len(s1):,}  rss {rss_mb():.0f} MB", flush=True)
    t0 = time.time()
    res = {"maivs": run_maivs, "maivs-memory": run_maivs, "scann-soar": run_scann, "scann": run_scann,
           "scann-soar-int8": run_scann, "hnswlib": run_hnswlib,
           "faiss-hnsw": run_faiss_hnsw}[system](base, dele, s1, qs, gts)
    res.update(system=system, wall_s=time.time() - t0, n0=N0, churn=NNEW, rounds=ROUNDS)
    os.makedirs(OUTD, exist_ok=True)
    out = os.path.join(OUTD, f"churn_{system}.json")
    json.dump(res, open(out, "w"), indent=1)
    import resource
    print(f"== done churn {system} in {res['wall_s']:.0f}s, peak RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB -> {out}", flush=True)
    os._exit(0)   # skip interpreter teardown (faiss/scann destructors can abort at exit)


if __name__ == "__main__":
    main()
