"""results_tables.py: render the head-to-head tables from saved bench_vs_faiss JSON, so no number in
the docs is typed by hand.

    python bench/results_tables.py                        # results/windows
    python bench/results_tables.py bench/results/*.json   # fresh runs

If a system appears in several files, the last file wins.
"""
import glob
import json
import os
import sys

import numpy as np

TARGETS = (0.90, 0.95, 0.98, 0.99, 0.995)


def interp(curve, key, target):
    pts = sorted(curve, key=lambda r: r["recall"])
    for a, b in zip(pts, pts[1:]):
        if a["recall"] <= target <= b["recall"]:
            t = (target - a["recall"]) / max(b["recall"] - a["recall"], 1e-12)
            return float(np.exp(np.log(a[key]) + t * (np.log(b[key]) - np.log(a[key]))))
    return float("nan")


def fmt(x, w=7):
    return f"{x:>{w},.0f}" if x == x else f"{' - ':>{w}}"


def main(paths):
    merged = {}
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        key = (d["corpus"], d["n"])
        merged.setdefault(key, {}).update(d["systems"])
    for (corpus, n), systems in sorted(merged.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        print(f"\n### {corpus}, N = {n:,}\n")
        print("| system | build s | B/vec |")
        print("|---|---|---|")
        for s, r in systems.items():
            print(f"| {s} | {r['build_s']:.1f} | {r['mem_bytes'] / n:,.0f} |")
        for qn in ("IND", "OOD"):
            if not any(qn in r["curves"] for r in systems.values()):
                continue
            print(f"\n**{qn} queries - batch QPS / p50 us at matched R@10**\n")
            print("| R@10 | " + " | ".join(systems) + " |")
            print("|---|" + "---|" * len(systems))
            for t in TARGETS:
                cells = []
                for s, r in systems.items():
                    c = r["curves"].get(qn, [])
                    q, l = interp(c, "qps", t), interp(c, "p50_us", t)
                    cells.append(f"{fmt(q, 0)} / {fmt(l, 0)}" if q == q else " - ")
                print(f"| {t} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "windows")
    main(sys.argv[1:] or sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)))
