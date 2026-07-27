"""bench_scaling.py: does work at fixed recall grow with N?

Fits scanned ~ N^alpha for two policies:

  fixed   cells = N/128     (what we use)
  sqrt    cells = sqrt(N)   (the usual IVF rule)

The sqrt arm is the control and should come out near alpha 0.5. If it doesn't, the measurement is
broken.
"""
import argparse, json, math, os
import numpy as np
from common import build_index, load_corpus, sweep, RESULTS_DIR, TARGET_CELL_SIZE


def interp(rows, tgt, field):
    pts = sorted((r["recall@10"], r["nprobe"], r[field]) for r in rows)
    for i in range(1, len(pts)):
        (r0, n0, v0), (r1, n1, v1) = pts[i - 1], pts[i]
        if r0 <= tgt <= r1 and r1 > r0:
            f = (tgt - r0) / (r1 - r0)
            return n0 + f * (n1 - n0), v0 + f * (v1 - v0)
    return (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--sizes", default="125000,250000,500000,999000")
    ap.add_argument("--targets", default="0.95,0.98")
    ap.add_argument("--profile", default="memory")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]
    targets = [float(x) for x in a.targets.split(",")]

    out = {}
    for policy in ("fixed", "sqrt"):
        out[policy] = {}
        for N in sizes:
            base, q, gt, _ = load_corpus(a.corpus, limit=N)
            n = len(base)
            cells = (n // TARGET_CELL_SIZE) if policy == "fixed" else int(math.sqrt(n))
            cells = max(16, cells)
            csize = n // cells
            print(f"\n[{policy}] N={n:,} cells={cells:,} cell_size={csize}")
            ix, _t = build_index(base, a.profile, cells=cells)
            probes = (8, 16, 32, 64, 128, 256, 512) if csize < 400 else (1, 4, 8, 16, 32, 64, 128)
            rows = sweep(ix, q, gt, 100, probes, policy)
            for r in rows:
                r["scanned"] = r["nprobe"] * csize
            out[policy][n] = rows
            del ix

    print(f"\n{'='*78}\nSCALING EXPONENT   scanned ~ N^alpha at fixed recall\n{'='*78}")
    for policy in out:
        print(f"\npolicy = {policy}")
        for tgt in targets:
            xs, ys = [], []
            for N, rows in out[policy].items():
                _np_, sc = interp(rows, tgt, "scanned")
                if sc:
                    xs.append(N); ys.append(sc)
            if len(xs) < 2:
                print(f"  R@10>={tgt}: not reached at every N"); continue
            alpha = np.polyfit(np.log(xs), np.log(ys), 1)[0]
            span = xs[-1] / xs[0]
            print(f"  R@10>={tgt}: alpha={alpha:.3f}  (flat=0, sqrt=0.5, linear=1) "
                  f"| N x{span:.0f} -> scanned x{ys[-1]/ys[0]:.2f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "scaling.json"), "w") as f:
        json.dump({p: {str(k): v for k, v in d.items()} for p, d in out.items()}, f, indent=2)


if __name__ == "__main__":
    main()
