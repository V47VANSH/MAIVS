"""bench_vs_faiss.py: head-to-head against faiss on the same data, queries and exact ground truth.

Same rules for every system:
  * build time  = the whole build, wall clock (ours: train + ingest + compact; HNSW: graph
                  construction), 16 threads
  * memory      = index bytes (faiss: serialized size; ours: stats()["total_bytes"])
  * batch QPS   = the whole query set in one call, all threads, best of 3
  * latency     = one thread, one query at a time, p50 / p99 over 200 queries, Python call
                  included for both
  * comparisons = at matched R@10, log-interpolated between measured points, never
                  extrapolated (NaN means the target wasn't reached)

    python bench/bench_vs_faiss.py hotpot_5M.npy --n 1000000 --questions bench/data/hotpot_questions_minilm.npy
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

from common import RESULTS_DIR, TARGET_CELL_SIZE  # noqa: E402

K = 10
TARGETS = (0.90, 0.95, 0.98, 0.99, 0.995)


def exact_gt(base, Q, chunk=131072):
    best_s = np.full((len(Q), K), -9.0, np.float32)
    best_i = np.zeros((len(Q), K), np.int64)
    for s in range(0, len(base), chunk):
        S = Q @ base[s:s + chunk].T
        idx = np.argpartition(-S, K, axis=1)[:, :K]
        sc = np.take_along_axis(S, idx, 1)
        cs = np.concatenate([best_s, sc], 1)
        ci = np.concatenate([best_i, idx + s], 1)
        o = np.argsort(-cs, axis=1)[:, :K]
        best_s = np.take_along_axis(cs, o, 1)
        best_i = np.take_along_axis(ci, o, 1)
    return best_i


def recall(ids, gt):
    return float(np.mean([len(set(a[:K]) & set(b[:K])) / K for a, b in zip(ids, gt)]))


def interp(curve, key, target):
    pts = sorted(curve, key=lambda r: r["recall"])
    for a, b in zip(pts, pts[1:]):
        if a["recall"] <= target <= b["recall"]:
            t = (target - a["recall"]) / max(b["recall"] - a["recall"], 1e-12)
            return float(np.exp(np.log(a[key]) + t * (np.log(b[key]) - np.log(a[key]))))
    return float("nan")


def best_of(fn, reps=3):
    best, out = float("inf"), None
    for _ in range(reps):
        t0 = time.perf_counter(); out = fn(); dt = time.perf_counter() - t0
        best = min(best, dt)
    return out, best


def serial_lat(fn1, Q):
    lat = []
    for q in Q:
        t0 = time.perf_counter(); fn1(q); lat.append((time.perf_counter() - t0) * 1e6)
    lat = np.array(lat)
    return float(np.percentile(lat, 50)), float(np.percentile(lat, 99))


# --- ours ---
def pareto(curve):
    """Keep only settings not dominated in (recall, qps), i.e. the ann-benchmarks frontier over each
    system's own grid.
    """
    pts = sorted(curve, key=lambda r: (-r["recall"], -r["qps"]))
    out, best = [], -1.0
    for r in pts:
        if r["qps"] > best:
            out.append(r)
            best = r["qps"]
    return sorted(out, key=lambda r: r["recall"])


def run_ours(base, qsets, gts, variant, probes, nser, rfs, cell=TARGET_CELL_SIZE):
    """variant: memory | quality, optionally '+poly' (orthogonal polysemy, 2 homes)
    and/or '+asym' (asymmetric cascade, factor 4)."""
    import aigvs_rust
    profile = variant.split("+")[0]
    ix = aigvs_rust.FlatCellIndex(base.shape[1], "binary", 32, 0x5EED, 0)
    if profile == "memory":
        ix.set_per_dim_sq(True)
    if "+poly" in variant:
        ix.set_orthogonal_polysemy(2, 1.0)
    if "+asym" in variant:
        ix.set_asym_cascade(4)
    if "+fs" in variant:
        ix.set_fastscan(True)                  # takes effect at compact()
    ix.set_two_stage(True, 2)
    ix.set_graph_route(False, 10 ** 9)
    ix.set_tree_route(True, 2048, 16)
    ix.set_adaptive(True, 0.02)
    t0 = time.perf_counter()
    ix.train_centroids(base, max(16, len(base) // cell))
    ix.add_batch(list(range(len(base))), base)
    ix.compact()
    build = time.perf_counter() - t0
    st = ix.stats()
    mem = st["total_bytes"]                    # every allocation, by capacity
    res = {"build_s": build, "mem_bytes": mem, "curves": {}, "raw": {}}
    for qn, Q in qsets.items():
        curve = []
        for rf in rfs:
            ix.set_rerank_factor(rf)
            for p in probes:
                if p > st["n_cells"]:
                    continue
                (ids, _), dt = best_of(lambda: ix.search_batch(Q, K, p, 1.0, 8.0, True))
                ids = np.asarray(ids, np.int64).reshape(len(Q), K)
                p50, p99 = serial_lat(lambda q: ix.search(q, K, p, 1.0, 8.0), Q[:nser])
                curve.append({"param": f"np{p}/rf{rf}", "recall": recall(ids, gts[qn]),
                              "qps": len(Q) / dt, "p50_us": p50, "p99_us": p99})
        res["raw"][qn] = curve
        res["curves"][qn] = pareto(curve)
    del ix
    return res


# --- ours ---
def run_hnsw(base, qsets, gts, corpus, efs, nser):
    import faiss
    n = len(base)
    path = os.path.join(HERE, "_index_cache",
                        f"hnsw32_{os.path.basename(corpus).replace('.npy', '')}_n{n}.faissindex")
    if os.path.exists(path):
        ix = faiss.read_index(path)
        build = json.load(open(path + ".json"))["build_s"]
    else:
        ix = faiss.IndexHNSWFlat(base.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        ix.hnsw.efConstruction = 200
        t0 = time.perf_counter(); ix.add(base); build = time.perf_counter() - t0
    mem = len(faiss.serialize_index(ix))
    res = {"build_s": build, "mem_bytes": mem, "curves": {}}
    nthreads = faiss.omp_get_max_threads()
    for qn, Q in qsets.items():
        curve = []
        for ef in efs:
            ix.hnsw.efSearch = ef
            faiss.omp_set_num_threads(nthreads)
            (D, I), dt = best_of(lambda: ix.search(Q, K))
            faiss.omp_set_num_threads(1)
            p50, p99 = serial_lat(lambda q: ix.search(q[None, :], K), Q[:nser])
            faiss.omp_set_num_threads(nthreads)
            curve.append({"param": ef, "recall": recall(I, gts[qn]), "qps": len(Q) / dt,
                          "p50_us": p50, "p99_us": p99})
        res["curves"][qn] = curve
    del ix
    return res


# --- ours ---
def run_usearch(base, qsets, gts, corpus, efs, nser):
    """USearch (a separate HNSW implementation) with the same graph settings as the faiss baseline:
    connectivity 32, expansion_add 200, f32, inner product.
    """
    from usearch.index import Index
    n, d = base.shape
    path = os.path.join(HERE, "_index_cache",
                        f"usearch32_{os.path.basename(corpus).replace('.npy', '')}_n{n}.usearch")
    nthreads = os.cpu_count()
    ix = Index(ndim=d, metric="ip", dtype="f32", connectivity=32, expansion_add=200)
    if os.path.exists(path):
        ix.load(path)
        build = json.load(open(path + ".json"))["build_s"]
    else:
        t0 = time.perf_counter()
        ix.add(np.arange(n, dtype=np.uint64), base, threads=nthreads)
        build = time.perf_counter() - t0
        ix.save(path)
        json.dump({"build_s": build}, open(path + ".json", "w"))
    mem = int(ix.memory_usage)
    res = {"build_s": build, "mem_bytes": mem, "curves": {}}
    for qn, Q in qsets.items():
        curve = []
        for ef in efs:
            ix.expansion_search = ef
            m, dt = best_of(lambda: ix.search(Q, K, threads=nthreads))
            I = np.asarray(m.keys, dtype=np.int64).reshape(len(Q), K)
            p50, p99 = serial_lat(lambda q: ix.search(q, K, threads=1), Q[:nser])
            curve.append({"param": ef, "recall": recall(I, gts[qn]), "qps": len(Q) / dt,
                          "p50_us": p50, "p99_us": p99})
        res["curves"][qn] = curve
    return res


# --- ours ---
FAISS_SPECS = {
    # closest public analogue: IVF + 1-bit RaBitQ codes + exact f32 refine
    "ivf-rabitq": "IVF{nlist},RaBitQ,RFlat",
    # faiss's fastest IVF scan: 4-bit PQ FastScan + exact f32 refine
    "ivf-pqfs": "IVF{nlist},PQ192x4fs,RFlat",
}


def run_faiss_ivf(base, qsets, gts, name, corpus, nser, nprobes, kfs):
    import faiss
    n = len(base)
    nlist = 1 << int(round(np.log2(4 * np.sqrt(n))))       # ~4*sqrt(N), power of 2
    spec = FAISS_SPECS[name].format(nlist=nlist)
    path = os.path.join(HERE, "_index_cache",
                        f"{name}_{os.path.basename(corpus).replace('.npy', '')}_n{n}.faissindex")
    if os.path.exists(path):
        ix = faiss.read_index(path)
        build = json.load(open(path + ".json"))["build_s"]
    else:
        ix = faiss.index_factory(base.shape[1], spec, faiss.METRIC_INNER_PRODUCT)
        t0 = time.perf_counter()
        ix.train(base)
        ix.add(base)
        build = time.perf_counter() - t0
        faiss.write_index(ix, path)
        json.dump({"build_s": build, "spec": spec}, open(path + ".json", "w"))
    mem = len(faiss.serialize_index(ix))
    ps = faiss.ParameterSpace()
    res = {"build_s": build, "mem_bytes": mem, "spec": spec, "curves": {}, "raw": {}}
    nthreads = faiss.omp_get_max_threads()
    for qn, Q in qsets.items():
        curve = []
        for kf in kfs:
            for npb in nprobes:
                if npb > nlist:
                    continue
                # IndexRefineFlat(k_factor) around IndexIVF*(nprobe). ParameterSpace in this faiss
                # build can't set k_factor, so set both attributes directly.
                ref = faiss.downcast_index(ix)
                ref.k_factor = kf
                faiss.downcast_index(ref.base_index).nprobe = npb
                faiss.omp_set_num_threads(nthreads)
                (D, I), dt = best_of(lambda: ix.search(Q, K))
                faiss.omp_set_num_threads(1)
                p50, p99 = serial_lat(lambda q: ix.search(q[None, :], K), Q[:nser])
                faiss.omp_set_num_threads(nthreads)
                curve.append({"param": f"np{npb}/kf{kf}", "recall": recall(I, gts[qn]),
                              "qps": len(Q) / dt, "p50_us": p50, "p99_us": p99})
        res["raw"][qn] = curve
        res["curves"][qn] = pareto(curve)
    del ix
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--questions", default="")
    ap.add_argument("--systems", default="ours-memory,ours-quality,hnsw")
    ap.add_argument("--probes", default="16,32,64,128,256,512,1024,2048")
    ap.add_argument("--efs", default="16,32,64,128,256,512")
    ap.add_argument("--serial", type=int, default=200)
    ap.add_argument("--rfs", default="32,64,128")
    ap.add_argument("--cell", type=int, default=TARGET_CELL_SIZE, help="vectors per cell (ours)")
    ap.add_argument("--fnprobes", default="16,32,64,128,256", help="faiss IVF nprobe grid")
    ap.add_argument("--fkfs", default="4,8,16", help="faiss refine k_factor grid")
    ap.add_argument("--reuse", default="", help="results JSON to take already-measured systems from")
    a = ap.parse_args()

    mm = np.load(a.corpus, mmap_mode="r")
    base = np.ascontiguousarray(mm[:a.n], dtype=np.float32)
    qsets = {"IND": np.ascontiguousarray(mm[-1000:], dtype=np.float32)}
    if a.questions:
        qsets["OOD"] = np.ascontiguousarray(np.load(a.questions)[:1000], dtype=np.float32)
    gts = {k: exact_gt(base, v) for k, v in qsets.items()}
    probes = [int(x) for x in a.probes.split(",")]
    efs = [int(x) for x in a.efs.split(",")]

    out = {"corpus": os.path.basename(a.corpus), "n": a.n, "systems": {}}
    prev = json.load(open(a.reuse))["systems"] if a.reuse else {}
    rfs = [int(x) for x in a.rfs.split(",")]
    for sysname in a.systems.split(","):
        if sysname in prev:
            r = prev[sysname]
            print(f"[reuse] {sysname} from {a.reuse}")
        elif sysname.startswith("ours"):
            r = run_ours(base, qsets, gts, sysname.split("-", 1)[1], probes, a.serial, rfs, a.cell)
        elif sysname == "hnsw":
            r = run_hnsw(base, qsets, gts, a.corpus, efs, a.serial)
        elif sysname == "usearch":
            r = run_usearch(base, qsets, gts, a.corpus, efs, a.serial)
        elif sysname in FAISS_SPECS:
            r = run_faiss_ivf(base, qsets, gts, sysname, a.corpus, a.serial,
                              [int(x) for x in a.fnprobes.split(",")],
                              [int(x) for x in a.fkfs.split(",")])
        else:
            raise SystemExit(f"unknown system {sysname}")
        out["systems"][sysname] = r
        print(f"\n== {sysname}  n={a.n:,}  build {r['build_s']:.1f}s  memory "
              f"{r['mem_bytes'] / 2 ** 20:.0f} MiB ({r['mem_bytes'] / a.n:.0f} B/vec)", flush=True)
        for qn, c in r["curves"].items():
            print(f"  {qn}: " + "  ".join(f"[{x['param']}] {x['recall']:.4f} {x['qps']:.0f}qps "
                                       f"p50 {x['p50_us']:.0f}us" for x in c), flush=True)

    print(f"\n==== matched recall, n={a.n:,} ====")
    for qn in qsets:
        print(f"-- {qn} queries")
        print(f"{'R@10':>6s} " + "".join(f"| {s:>13s} qps   p50us   p99us " for s in out["systems"]))
        for t in TARGETS:
            row = f"{t:6.3f} "
            for s, r in out["systems"].items():
                c = r["curves"][qn]
                row += (f"| {interp(c, 'qps', t):17.0f} {interp(c, 'p50_us', t):7.0f} "
                        f"{interp(c, 'p99_us', t):7.0f} ")
            print(row)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tagx = "_".join(x.replace("+", "") for x in a.systems.split(",")) + f"_c{a.cell}"
    path = os.path.join(RESULTS_DIR, f"vs_faiss_{os.path.basename(a.corpus).replace('.npy', '')}_n{a.n}_{tagx}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
