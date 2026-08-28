"""Run one system at one corpus size, in its own process.

    python bench/fair/fairbench.py SYSTEM N [TAG]

Every system is measured the same way as in bench/bench_vs_faiss.py:
  * build:   wall clock from raw vectors to a searchable index, training included, 16 threads
  * batch:   all 1,000 queries in one call, 16 threads, best of 3
  * latency: one query per call on one thread, p50/p99 over 200 queries, Python call included
  * recall:  R@10 against exact brute force
Each system gets its own parameter grid; summarise.py takes the frontiers and interpolates at
matched recall.

Memory: MAIVS stats()["total_bytes"], USearch memory_usage, everything else the serialized
index size.

Input comes from prep_data.py. FAIR_DIR (default ~/maivs-fair) holds data/ and results/.
"""
from __future__ import annotations

import gc
import json
import os
import resource
import sys
import time

THREADS = 16
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("RAYON_NUM_THREADS", str(THREADS))
import numpy as np  # noqa: E402

ROOT = os.environ.get("FAIR_DIR", os.path.expanduser("~/maivs-fair"))
DATA = os.path.join(ROOT, "data")
OUTD = os.path.join(ROOT, "results")
K = 10
NSER = 200


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def recall(ids, gt):
    ids = np.asarray(ids).reshape(len(gt), -1)[:, :K].astype(np.int64).tolist()
    return float(np.mean([len(set(a) & set(b)) / K for a, b in zip(ids, gt.tolist())]))


def best_of(fn, reps=3):
    best, out = float("inf"), None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def lat(fn1, Q):
    xs = []
    for q in Q[:NSER]:
        t0 = time.perf_counter()
        fn1(q)
        xs.append((time.perf_counter() - t0) * 1e6)
    return float(np.percentile(xs, 50)), float(np.percentile(xs, 99))


def point(param, ids, qn, gts, dt, p50p99, **extra):
    r = {"param": param, "recall": recall(ids, gts[qn]), "qps": 1000.0 / dt,
         "p50_us": p50p99[0], "p99_us": p50p99[1]}
    r.update(extra)
    print(f"    {qn} {param:<22} R@10 {r['recall']:.4f}  {r['qps']:9.0f} qps  p50 {r['p50_us']:7.0f} us"
          + "".join(f"  {k} {v:.4f}" if k.startswith("recall") else f"  {k} {v:.0f}"
                    for k, v in extra.items()),
          flush=True)
    return r


def shm_size(save):
    """Serialized size of an index. Written to /dev/shm and removed again, so it never
    touches the disk."""
    import shutil
    d = f"/dev/shm/fairbench_{os.getpid()}"
    os.makedirs(d, exist_ok=True)
    try:
        save(d)
        return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- MAIVS
def run_maivs(base, qs, gts, profile):
    import maivs
    n = len(base)
    ix = maivs.FlatCellIndex(base.shape[1], "binary", 32, 0x5EED, 0)
    if profile == "memory":
        ix.set_per_dim_sq(True)
    ix.set_orthogonal_polysemy(2, 1.0)
    ix.set_fastscan(True)
    ix.set_two_stage(True, 2)
    ix.set_graph_route(False, 10 ** 9)
    ix.set_tree_route(True, 2048, 16)
    ix.set_adaptive(True, 0.02)
    t0 = time.perf_counter()
    ix.train_centroids(base, max(16, n // 128))
    ix.add_batch(list(range(n)), base)
    ix.compact()
    build = time.perf_counter() - t0
    st = ix.stats()
    res = {"build_s": build, "mem_bytes": st["total_bytes"], "mem_method": "stats.total_bytes",
           "config": f"{profile}+poly+fs, {int(st['n_cells'])} cells", "grid": {}}
    print(f"  build {build:.1f}s  {st['total_bytes'] / n:.0f} B/vec", flush=True)
    for qn, Q in qs.items():
        pts = []
        for rf in (8, 16, 32):
            ix.set_rerank_factor(rf)
            for p in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
                if p > st["n_cells"]:
                    continue
                (ids, _), dt = best_of(lambda: ix.search_batch(Q, K, p, 1.0, 8.0, True))
                (ids2, _), dt2 = best_of(lambda: ix.search_batch(Q, K, p, 1.0, 8.0, False))
                pl = lat(lambda q: ix.search(q, K, p, 1.0, 8.0), Q)
                pts.append(point(f"np{p}/rf{rf}", ids, qn, gts, dt, pl,
                                 qps_perquery=1000.0 / dt2, recall_perquery=recall(ids2, gts[qn])))
        res["grid"][qn] = pts
    return res


# ---------------------------------------------------------------- HNSW family
EFS = (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)


def run_faiss_hnsw(base, qs, gts, sq8=False):
    import faiss
    faiss.omp_set_num_threads(THREADS)
    if sq8:
        ix = faiss.index_factory(base.shape[1], "HNSW32,SQ8", faiss.METRIC_INNER_PRODUCT)
        hn = faiss.downcast_index(ix)
    else:
        ix = hn = faiss.IndexHNSWFlat(base.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    hn.hnsw.efConstruction = 200
    t0 = time.perf_counter()
    if sq8:
        ix.train(base)
    ix.add(base)
    build = time.perf_counter() - t0
    res = {"build_s": build, "mem_bytes": len(faiss.serialize_index(ix)), "mem_method": "serialized",
           "config": "HNSW32,SQ8 efC=200" if sq8 else "IndexHNSWFlat M=32 efC=200",
           "faiss_compile": faiss.get_compile_options(), "grid": {}}
    print(f"  build {build:.1f}s  faiss {faiss.get_compile_options()}", flush=True)
    for qn, Q in qs.items():
        pts = []
        for ef in EFS:
            hn.hnsw.efSearch = ef
            faiss.omp_set_num_threads(THREADS)
            (_, I), dt = best_of(lambda: ix.search(Q, K))
            faiss.omp_set_num_threads(1)
            pl = lat(lambda q: ix.search(q[None, :], K), Q)
            faiss.omp_set_num_threads(THREADS)
            pts.append(point(f"ef{ef}", I, qn, gts, dt, pl))
        res["grid"][qn] = pts
    return res


def run_hnswlib(base, qs, gts):
    import hnswlib
    n, d = base.shape
    t0 = time.perf_counter()
    ix = hnswlib.Index(space="ip", dim=d)
    ix.init_index(max_elements=n, ef_construction=200, M=32)
    ix.set_num_threads(THREADS)
    ix.add_items(base, np.arange(n))
    build = time.perf_counter() - t0
    mem = shm_size(lambda dd: ix.save_index(os.path.join(dd, "h.bin")))
    res = {"build_s": build, "mem_bytes": mem, "mem_method": "serialized",
           "config": f"hnswlib {hnswlib.__version__ if hasattr(hnswlib, '__version__') else '0.8.0'} M=32 efC=200",
           "grid": {}}
    print(f"  build {build:.1f}s", flush=True)
    for qn, Q in qs.items():
        pts = []
        for ef in EFS:
            ix.set_ef(ef)
            (I, _), dt = best_of(lambda: ix.knn_query(Q, k=K, num_threads=THREADS))
            pl = lat(lambda q: ix.knn_query(q[None, :], k=K, num_threads=1), Q)
            pts.append(point(f"ef{ef}", I, qn, gts, dt, pl))
        res["grid"][qn] = pts
    return res


def run_usearch(base, qs, gts):
    from usearch.index import Index
    import usearch
    n, d = base.shape
    t0 = time.perf_counter()
    ix = Index(ndim=d, metric="ip", dtype="f32", connectivity=32, expansion_add=200)
    ix.add(np.arange(n, dtype=np.uint64), base, threads=THREADS)
    build = time.perf_counter() - t0
    res = {"build_s": build, "mem_bytes": int(ix.memory_usage), "mem_method": "memory_usage",
           "config": f"USearch {usearch.__version__} M=32 efC=200 f32", "grid": {}}
    print(f"  build {build:.1f}s", flush=True)
    for qn, Q in qs.items():
        pts = []
        for ef in EFS:
            ix.expansion_search = ef
            m, dt = best_of(lambda: ix.search(Q, K, threads=THREADS))
            pl = lat(lambda q: ix.search(q, K, threads=1), Q)
            pts.append(point(f"ef{ef}", np.asarray(m.keys, np.int64), qn, gts, dt, pl))
        res["grid"][qn] = pts
    return res


# ---------------------------------------------------------------- ScaNN
def run_scann(base, qs, gts, soar, int8=False):
    import scann
    from scann.scann_ops.py.scann_builder import ReorderType
    rq = ReorderType.INT8 if int8 else ReorderType.FLOAT32
    n = len(base)
    leaves_opts = (n // 500, n // 250)
    res = {"builds": [], "grid": {qn: [] for qn in qs}}
    for nl in leaves_opts:
        t0 = time.perf_counter()
        b = scann.scann_ops_pybind.builder(base, K, "dot_product")
        if hasattr(b, "set_n_training_threads"):
            b = b.set_n_training_threads(THREADS)
        kw = dict(num_leaves=nl, num_leaves_to_search=32, training_sample_size=min(n, 250_000))
        if soar:
            kw.update(soar_lambda=1.5, overretrieve_factor=2.0)
        s = b.tree(**kw).score_ah(2, anisotropic_quantization_threshold=0.2).reorder(100, quantize=rq).build()
        build = time.perf_counter() - t0
        mem = shm_size(lambda dd: s.serialize(dd))
        # search_batched_parallel splits the batch into chunks of batch_size queries, and the
        # default of 256 only keeps 4 threads busy with 1,000 queries. Pick the best per build.
        Qc = qs["IND"]
        bs_t = {}
        for bs in (4, 8, 16, 32, 64, 128, 256):
            _, dt = best_of(lambda: s.search_batched_parallel(Qc, final_num_neighbors=K, leaves_to_search=32,
                                                              pre_reorder_num_neighbors=100, batch_size=bs))
            bs_t[bs] = dt
        bs = min(bs_t, key=bs_t.get)
        tag = f"L{nl}"
        res["builds"].append({"tag": tag, "num_leaves": nl, "build_s": build, "mem_bytes": mem,
                              "batch_size": bs, "batch_size_qps": {k: 1000 / v for k, v in bs_t.items()}})
        print(f"  [{tag}] build {build:.1f}s  ~{mem / n:.0f} B/vec  batch_size {bs} "
              + " ".join(f"{k}:{1000 / v:.0f}" for k, v in bs_t.items()), flush=True)
        for qn, Q in qs.items():
            for lv in (4, 8, 16, 32, 64, 128, 256):
                if lv > nl:
                    continue
                for rr in (40, 100, 250, 500):
                    (I, _), dt = best_of(lambda: s.search_batched_parallel(
                        Q, final_num_neighbors=K, leaves_to_search=lv, pre_reorder_num_neighbors=rr, batch_size=bs))
                    pl = lat(lambda q: s.search(q, final_num_neighbors=K, leaves_to_search=lv,
                                                pre_reorder_num_neighbors=rr), Q)
                    res["grid"][qn].append(point(f"{tag}/l{lv}/r{rr}", I, qn, gts, dt, pl))
        del s, b
        gc.collect()
    res["build_s"] = min(x["build_s"] for x in res["builds"])
    res["build_s_range"] = [x["build_s"] for x in res["builds"]]
    res["mem_bytes"] = max(x["mem_bytes"] for x in res["builds"])
    res["mem_method"] = "serialized, largest build"
    res["config"] = ("tree+AH(2 dims/block, t=0.2)+exact reorder, leaves "
                     + "/".join(str(x) for x in leaves_opts) + (", SOAR lambda 1.5" if soar else "")
                     + (", INT8 reorder" if int8 else ""))
    return res


# ---------------------------------------------------------------- faiss IVF
IVF_SPECS = {
    "faiss-ivf-pqfs": "IVF{nl},PQ192x4fs,RFlat",
    "faiss-ivf-rabitqfs": "IVF{nl},RaBitQfs,RFlat",
    "faiss-ivf-rabitqfs4": "IVF{nl},RaBitQfs4,RFlat",
    "faiss-ivf-pqfs-sq8": "IVF{nl},PQ192x4fs,Refine(SQ8)",
}


def run_faiss_ivf(base, qs, gts, name):
    import faiss
    n = len(base)
    nls = (1024, 2048) if n <= 250_000 else (2048, 4096)
    res = {"builds": [], "grid": {qn: [] for qn in qs}, "faiss_compile": faiss.get_compile_options()}
    for nl in nls:
        spec = IVF_SPECS[name].format(nl=nl)
        faiss.omp_set_num_threads(THREADS)
        ix = faiss.index_factory(base.shape[1], spec, faiss.METRIC_INNER_PRODUCT)
        t0 = time.perf_counter()
        ix.train(base)
        ix.add(base)
        build = time.perf_counter() - t0
        mem = len(faiss.serialize_index(ix))
        res["builds"].append({"tag": f"nl{nl}", "spec": spec, "build_s": build, "mem_bytes": mem})
        print(f"  [{spec}] build {build:.1f}s  {mem / n:.0f} B/vec", flush=True)
        ref = faiss.downcast_index(ix)
        ivf = faiss.downcast_index(ref.base_index)
        for qn, Q in qs.items():
            for kf in (4, 16, 64):
                for npb in (4, 8, 16, 32, 64, 128, 256, 512):
                    if npb > nl // 2:
                        continue
                    ref.k_factor = kf
                    ivf.nprobe = npb
                    faiss.omp_set_num_threads(THREADS)
                    (_, I), dt = best_of(lambda: ix.search(Q, K))
                    faiss.omp_set_num_threads(1)
                    pl = lat(lambda q: ix.search(q[None, :], K), Q)
                    faiss.omp_set_num_threads(THREADS)
                    res["grid"][qn].append(point(f"nl{nl}/np{npb}/kf{kf}", I, qn, gts, dt, pl))
        del ix, ref, ivf
        gc.collect()
    res["build_s"] = min(x["build_s"] for x in res["builds"])
    res["build_s_range"] = [x["build_s"] for x in res["builds"]]
    res["mem_bytes"] = max(x["mem_bytes"] for x in res["builds"])
    res["mem_method"] = "serialized, largest build"
    res["config"] = " | ".join(x["spec"] for x in res["builds"])
    return res


def main():
    system, n = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else ""
    base = np.ascontiguousarray(np.load(os.path.join(DATA, "base_500k.npy"), mmap_mode="r")[:n], np.float32)
    qs = {"IND": np.load(os.path.join(DATA, "IND.npy")), "OOD": np.load(os.path.join(DATA, "OOD.npy"))}
    g = np.load(os.path.join(DATA, f"gt_{n}.npz"))
    gts = {"IND": g["IND"], "OOD": g["OOD"]}
    print(f"== {system} n={n:,} {tag}  rss {rss_mb():.0f} MB", flush=True)
    t0 = time.time()
    if system.startswith("maivs-"):
        res = run_maivs(base, qs, gts, system.split("-", 1)[1])
    elif system in ("faiss-hnsw", "faiss-hnsw-sq8"):
        res = run_faiss_hnsw(base, qs, gts, sq8=system.endswith("sq8"))
    elif system == "hnswlib":
        res = run_hnswlib(base, qs, gts)
    elif system == "usearch":
        res = run_usearch(base, qs, gts)
    elif system in ("scann", "scann-soar", "scann-int8", "scann-soar-int8"):
        res = run_scann(base, qs, gts, soar="soar" in system, int8=system.endswith("int8"))
    elif system in IVF_SPECS:
        res = run_faiss_ivf(base, qs, gts, system)
    else:
        raise SystemExit(f"unknown system {system}")
    res.update(system=system, n=n, wall_s=time.time() - t0,
               peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    os.makedirs(OUTD, exist_ok=True)
    out = os.path.join(OUTD, f"{system}_{n}{('_' + tag) if tag else ''}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"== done {system} n={n:,} in {res['wall_s']:.0f}s, peak RSS {res['peak_rss_mb']:.0f} MB -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
