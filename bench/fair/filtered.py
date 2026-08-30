"""k-NN restricted to the vectors whose attribute passes a filter.

    python bench/fair/filtered.py SYSTEM       SYSTEM in maivs | faiss-hnsw | scann-soar | bruteforce

HotpotQA 500k. Every vector gets a uniform random tag in [0, 1000) and the filter keeps tag < t,
for selectivity 50%, 10%, 1% and 0.1%. Ground truth is the exact top-10 among the vectors that
pass.
  maivs       built in: u32 tag checked inside the scan (set_tags / set_filter)
  faiss-hnsw  built in: IDSelectorBitmap through SearchParametersHNSW
  scann-soar  no filter API: over-fetch k' = mult * k / selectivity, then drop the ids that
              don't match (timed inside the search call)
  bruteforce  exact BLAS scan of the matching subset (gathered once, timed separately)
Batch QPS = 1,000 queries, 16 threads, best of 3.
"""
import gc
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fairbench import THREADS, K, DATA, OUTD, recall, best_of, rss_mb  # noqa: E402
from churn import exact_gt  # noqa: E402
import numpy as np  # noqa: E402

N = 500_000
SELS = {0.5: 500, 0.1: 100, 0.01: 10, 0.001: 1}       # selectivity -> tag threshold


def load():
    base = np.ascontiguousarray(np.load(os.path.join(DATA, "base_500k.npy"), mmap_mode="r")[:N], np.float32)
    tags = np.random.default_rng(2).integers(0, 1000, N).astype(np.uint32)
    qs = {"IND": np.load(os.path.join(DATA, "IND.npy")), "OOD": np.load(os.path.join(DATA, "OOD.npy"))}
    gp = os.path.join(DATA, "gt_filtered.npz")
    if not os.path.exists(gp):
        g = {}
        for s, t in SELS.items():
            sub = np.nonzero(tags < t)[0]
            bs = base[sub]
            for qn, Q in qs.items():
                g[f"{qn}_{t}"] = sub[exact_gt(bs, Q)]
            del bs
        np.savez(gp, **g)
    g = np.load(gp)
    return base, tags, qs, g


def sweep(search, params, qs, g, t, label):
    out = {}
    for qn, Q in qs.items():
        pts = []
        for p in params:
            ids, dt = best_of(lambda: search(Q, p))
            r = {"param": str(p), "recall": recall(ids, g[f"{qn}_{t}"]), "qps": 1000.0 / dt}
            pts.append(r)
            print(f"    [{label}] {qn} {str(p):<14} R@10 {r['recall']:.4f} {r['qps']:9.0f} qps", flush=True)
        out[qn] = pts
    return out


def run_maivs(base, tags, qs, g):
    import maivs
    ix = maivs.FlatCellIndex(base.shape[1], "binary", 32, 0x5EED, 0)
    ix.set_orthogonal_polysemy(2, 1.0)
    ix.set_fastscan(True)
    ix.set_two_stage(True, 2)
    ix.set_graph_route(False, 10 ** 9)
    ix.set_tree_route(True, 2048, 16)
    ix.set_adaptive(True, 0.02)
    t0 = time.perf_counter()
    ix.train_centroids(base, N // 128)
    ix.add_batch(list(range(N)), base)
    ix.compact()
    res = {"build_s": time.perf_counter() - t0, "sel": {}}
    ix.set_tags(list(range(N)), tags.tolist())
    cells = int(ix.stats()["n_cells"])
    params = [(p, rf) for rf in (16, 32) for p in (64, 256, 1024, 2048, cells)]

    def f(Q, p):
        ix.set_rerank_factor(p[1])
        return ix.search_batch(Q, K, p[0], 1.0, 8.0, True)[0]
    for s, t in SELS.items():
        ix.set_filter(0, t - 1)
        res["sel"][str(s)] = sweep(f, params, qs, g, t, f"sel {s}")
    return res


def run_faiss_hnsw(base, tags, qs, g):
    import faiss
    faiss.omp_set_num_threads(THREADS)
    ix = faiss.IndexHNSWFlat(base.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    ix.hnsw.efConstruction = 200
    t0 = time.perf_counter()
    ix.add(base)
    res = {"build_s": time.perf_counter() - t0, "sel": {}}
    for s, t in SELS.items():
        bm = np.packbits((tags < t).astype(np.uint8), bitorder="little")
        sel = faiss.IDSelectorBitmap(N, faiss.swig_ptr(bm))

        def f(Q, ef, sel=sel):
            return ix.search(Q, K, params=faiss.SearchParametersHNSW(sel=sel, efSearch=ef))[1]
        res["sel"][str(s)] = sweep(f, (16, 32, 64, 128, 256, 512, 1024, 2048), qs, g, t, f"sel {s}")
        del sel, bm
    return res


def postfilter(I, mask, k):
    I = np.asarray(I, np.int64)
    M = mask[np.clip(I, 0, len(mask) - 1)] & (I >= 0)
    C = np.cumsum(M, axis=1)
    keep = M & (C <= k)
    out = np.full((I.shape[0], k), -1, np.int64)
    r, c = np.nonzero(keep)
    out[r, C[r, c] - 1] = I[r, c]
    return out


def run_scann(base, tags, qs, g):
    import scann
    t0 = time.perf_counter()
    b = scann.scann_ops_pybind.builder(base, K, "dot_product")
    if hasattr(b, "set_n_training_threads"):
        b = b.set_n_training_threads(THREADS)
    s = (b.tree(num_leaves=N // 250, num_leaves_to_search=32, training_sample_size=250_000,
                soar_lambda=1.5, overretrieve_factor=2.0)
         .score_ah(2, anisotropic_quantization_threshold=0.2).reorder(100).build())
    res = {"build_s": time.perf_counter() - t0, "sel": {}}
    for sv, t in SELS.items():
        mask = tags < t
        params = []
        for lv in (32, 128, 512, N // 250):
            for mult in (1, 2, 4):
                kp = min(int(math.ceil(K / sv)) * mult, 20_000)
                if (lv, kp) not in params:
                    params.append((lv, kp))

        def f(Q, p, mask=mask):
            I, _ = s.search_batched_parallel(Q, final_num_neighbors=p[1], leaves_to_search=p[0],
                                             pre_reorder_num_neighbors=max(p[1], 100), batch_size=128)
            return postfilter(I, mask, K)
        res["sel"][str(sv)] = sweep(f, params, qs, g, t, f"sel {sv}")
    return res


def run_bruteforce(base, tags, qs, g):
    res = {"build_s": 0.0, "sel": {}, "gather_s": {}}
    for s, t in SELS.items():
        t0 = time.perf_counter()
        sub = np.nonzero(tags < t)[0]
        bs = np.ascontiguousarray(base[sub])
        res["gather_s"][str(s)] = time.perf_counter() - t0

        def f(Q, _p, sub=sub, bs=bs):
            return sub[exact_gt(bs, Q)]
        res["sel"][str(s)] = sweep(f, ("exact",), qs, g, t, f"sel {s}")
        del bs
        gc.collect()
    return res


def main():
    system = sys.argv[1]
    base, tags, qs, g = load()
    print(f"== filtered {system}  rss {rss_mb():.0f} MB", flush=True)
    t0 = time.time()
    res = {"maivs": run_maivs, "faiss-hnsw": run_faiss_hnsw, "scann-soar": run_scann,
           "bruteforce": run_bruteforce}[system](base, tags, qs, g)
    res.update(system=system, wall_s=time.time() - t0, n=N)
    os.makedirs(OUTD, exist_ok=True)
    out = os.path.join(OUTD, f"filtered_{system}.json")
    json.dump(res, open(out, "w"), indent=1)
    import resource
    print(f"== done filtered {system} in {res['wall_s']:.0f}s, peak RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB -> {out}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
