"""bench_concurrency.py: can the index be written while it's being searched?

1. Does a write during in-flight searches succeed? FlatCellIndex mutations take &mut self, so pyo3
   raises a borrow error and the write silently doesn't happen. ConcurrentIndex (sealed segments +
   one mutable head) accepts it.
2. Do concurrent readers scale, i.e. is the GIL released?

Readers run until a fixed deadline and the writer starts partway through, so they overlap. A fixed
iteration count let the readers finish before the writer even started. If you change the timings,
check that `during` isn't empty.
"""
import argparse
import json
import os
import threading
import time

import numpy as np

# import common first (sys.path, rayon thread count)
from common import PROFILES, RESULTS_DIR, TARGET_CELL_SIZE, load_corpus
import aigvs_rust


def build(cls, base, profile, **kw):
    cfg = PROFILES[profile]
    ix = cls(base.shape[1], cfg["scan"], cfg["rerank"], 0x5EED, 0, **kw)
    if cfg["per_dim_sq"]:
        ix.set_per_dim_sq(True)
    ix.set_two_stage(True, 2)
    ix.set_graph_route(False, 10 ** 9)
    ix.set_tree_route(True, 2048, 16)
    cells = max(16, len(base) // TARGET_CELL_SIZE)
    ix.train_centroids(np.ascontiguousarray(base), cells)
    ix.add_batch(list(range(len(base))), np.ascontiguousarray(base))
    ix.compact()
    return ix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--readers", type=int, default=4)
    ap.add_argument("--duration", type=float, default=2.5)
    ap.add_argument("--write-at", type=float, default=0.8)
    ap.add_argument("--profile", default="memory")
    a = ap.parse_args()

    base, q, _gt, tag = load_corpus(a.corpus, limit=a.limit)
    N = len(base)
    hold = min(5000, N // 10)
    head, tail = base[: N - hold], base[N - hold:]

    out = {}
    print("=" * 78)
    print("[1] write issued while searches are in flight")
    print("=" * 78)
    for name, cls, kw in (("FlatCellIndex", aigvs_rust.FlatCellIndex, {}),
                          ("ConcurrentIndex", aigvs_rust.ConcurrentIndex,
                           {"seal_at": 10 ** 9})):
        ix = build(cls, head, a.profile, **kw)
        err, stamps = [], [[] for _ in range(a.readers)]
        t_start = time.perf_counter()
        deadline = t_start + a.duration            # absolute: cannot run away

        def reader(i):
            c = 0
            try:
                while time.perf_counter() < deadline:
                    ix.search(q[c % len(q)], 10, 64, 1.0, 8.0)
                    c += 1
                    stamps[i].append(time.perf_counter() - t_start)
            except Exception as e:                  # noqa: BLE001
                err.append(type(e).__name__ + ": " + str(e)[:60])

        ts = [threading.Thread(target=reader, args=(i,)) for i in range(a.readers)]
        for t in ts:
            t.start()
        time.sleep(a.write_at)
        w0 = time.perf_counter() - t_start
        try:
            ix.add_batch(list(range(N - hold, N)), np.ascontiguousarray(tail))
            w = "ok"
        except Exception as e:                      # noqa: BLE001
            w = type(e).__name__ + ": " + str(e)[:50]
        w1 = time.perf_counter() - t_start
        for t in ts:
            t.join()

        allst = sorted(x for s in stamps for x in s)
        seg = lambda lo, hi: len([x for x in allst if lo <= x < hi]) / max(hi - lo, 1e-9)
        assert [x for x in allst if w0 <= x < w1] or w != "ok" or (w1 - w0) < 1e-3, \
            "no reader activity during the write window - test is vacuous"
        print(f"\n  {name}")
        print(f"    write   : {w}   window=[{w0:.2f},{w1:.2f}]s "
              f"({(w1-w0)*1000:.0f}ms)")
        print(f"    reader qps  before={seg(0,w0):>8.0f}  "
              f"DURING={seg(w0,w1):>8.0f}  after={seg(w1,a.duration):>8.0f}")
        print(f"    reader errors={len(err)}" + (f" -> {err[0]}" if err else ""))
        print(f"    len={len(ix):,}  expected={N:,}"
              + ("   <-- the write silently did not happen" if len(ix) != N else ""))
        out[name] = dict(write=w, len=len(ix), expected=N, errors=len(err),
                         window_ms=round((w1 - w0) * 1000))
        del ix

    print("\n" + "=" * 78)
    print("[2] concurrent read scaling (is the GIL released during search?)")
    print("=" * 78)
    ix = build(aigvs_rust.ConcurrentIndex, head, a.profile, seal_at=10 ** 9)

    def run(nt, per=150):
        def w(_i):
            for j in range(per):
                ix.search(q[j % len(q)], 10, 64, 1.0, 8.0)
        ts = [threading.Thread(target=w, args=(i,)) for i in range(nt)]
        t0 = time.perf_counter()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return nt * per / (time.perf_counter() - t0)

    b1 = run(1)
    print(f"  {'threads':>8} {'qps':>10} {'scaling':>9}")
    scale = {}
    for n in (1, 2, 4, 8, 16):
        s = run(n)
        scale[n] = round(s)
        print(f"  {n:>8} {s:>10.0f} {s/b1:>8.2f}x")
    out["read_scaling"] = scale

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"concurrency_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
