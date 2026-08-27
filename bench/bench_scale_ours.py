"""bench_scale_ours.py: our index at sizes where the base doesn't fit in RAM next to the index.

  * train on a strided sample (the file is shuffled, so a stride is a uniform sample) and
    ingest in chunks straight from the memory-mapped file
  * build time = train + ingest + compact, wall clock, including reading the chunks
  * IND ground truth from bench/_scale_gt (cached by bench_scale_full); OOD ground truth is
    computed exactly by streaming the corpus

Writes the same JSON format as bench_vs_faiss.py, so results_tables.py can render it.

    python bench/bench_scale_ours.py hotpot_5M.npy --n 2000000 --variant memory+poly+fs --questions bench/data/hotpot_questions_minilm.npy
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
from bench_vs_faiss import best_of, pareto, recall, serial_lat, K  # noqa: E402


def streamed_gt(mm, n, Q, chunk=131072):
    best_s = np.full((len(Q), K), -9.0, np.float32)
    best_i = np.zeros((len(Q), K), np.int64)
    for s in range(0, n, chunk):
        blk = np.asarray(mm[s:min(s + chunk, n)], dtype=np.float32)
        S = Q @ blk.T
        idx = np.argpartition(-S, K, axis=1)[:, :K]
        sc = np.take_along_axis(S, idx, 1)
        cs = np.concatenate([best_s, sc], 1)
        ci = np.concatenate([best_i, idx + s], 1)
        o = np.argsort(-cs, axis=1)[:, :K]
        best_s = np.take_along_axis(cs, o, 1)
        best_i = np.take_along_axis(ci, o, 1)
    return best_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--variant", default="memory+poly+fs")
    ap.add_argument("--questions", default="")
    ap.add_argument("--probes", default="64,128,256,512,1024,2048,4096")
    ap.add_argument("--rfs", default="16,32")
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--serial", type=int, default=30)
    a = ap.parse_args()
    import maivs

    mm = np.load(a.corpus, mmap_mode="r")
    n, d = a.n, mm.shape[1]
    qsets = {"IND": np.ascontiguousarray(mm[-1000:], dtype=np.float32)}
    gts = {}
    gtp = os.path.join(HERE, "_scale_gt", f"gt_n{n}_q1000_k100.npz")
    gts["IND"] = np.load(gtp)["ids"][:, :K] if os.path.exists(gtp) else streamed_gt(mm, n, qsets["IND"])
    if a.questions:
        qsets["OOD"] = np.ascontiguousarray(np.load(a.questions)[:1000], dtype=np.float32)
        gts["OOD"] = streamed_gt(mm, n, qsets["OOD"])
    print(f"[gt] ready", flush=True)

    profile = a.variant.split("+")[0]
    ix = maivs.FlatCellIndex(d, "binary", 32, 0x5EED, 0)
    if profile == "memory":
        ix.set_per_dim_sq(True)
    if "+poly" in a.variant:
        ix.set_orthogonal_polysemy(2, 1.0)
    if "+fs" in a.variant:
        ix.set_fastscan(True)
    ix.set_two_stage(True, 2)
    ix.set_graph_route(False, 10 ** 9)
    ix.set_tree_route(True, 2048, 16)
    ix.set_adaptive(True, 0.02)

    t0 = time.perf_counter()
    step = max(1, n // 262_144)
    train = np.ascontiguousarray(mm[:n:step][:262_144], dtype=np.float32)
    ix.train_centroids(train, max(16, n // TARGET_CELL_SIZE))
    del train
    for s in range(0, n, a.chunk):
        blk = np.ascontiguousarray(mm[s:min(s + a.chunk, n)], dtype=np.float32)
        ix.add_batch(list(range(s, s + len(blk))), blk)
        del blk
    ix.compact()
    build = time.perf_counter() - t0
    st = ix.stats()
    res = {"build_s": build, "mem_bytes": st["total_bytes"], "curves": {}, "raw": {}}
    print(f"[build] {a.variant} n={n:,} {build:.1f}s {st['total_bytes'] / n:.0f} B/vec", flush=True)

    rfs = [int(x) for x in a.rfs.split(",")]
    for qn, Q in qsets.items():
        curve = []
        for rf in rfs:
            ix.set_rerank_factor(rf)
            for p in [int(x) for x in a.probes.split(",")]:
                if p > st["n_cells"]:
                    continue
                (ids, _), dt = best_of(lambda: ix.search_batch(Q, K, p, 1.0, 8.0, True))
                ids = np.asarray(ids, np.int64).reshape(len(Q), K)
                p50, p99 = serial_lat(lambda q: ix.search(q, K, p, 1.0, 8.0), Q[:a.serial])
                curve.append({"param": f"np{p}/rf{rf}", "recall": recall(ids, gts[qn]),
                              "qps": len(Q) / dt, "p50_us": p50, "p99_us": p99})
                print(f"  {qn} np{p}/rf{rf}: {curve[-1]['recall']:.4f} {curve[-1]['qps']:.0f} qps", flush=True)
        res["raw"][qn] = curve
        res["curves"][qn] = pareto(curve)
    out = {"corpus": os.path.basename(a.corpus), "n": n,
           "systems": {f"ours-{a.variant}": res}}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"scale_ours_{a.variant.replace('+', '')}_n{n}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
