"""Matched-recall tables from the fairbench.py results, as markdown on stdout.

    python bench/fair/summarise.py [RESULTS_DIR] > summary_tables.md

Throughput: the Pareto frontier of (R@10, batch QPS) over each system's own grid,
log-interpolated at each target and never extrapolated ("-" = target not reached). Latency: the
same over (R@10, single-thread p50), i.e. each system at its own best latency setting.
"""
import glob
import json
import math
import os
import sys

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.environ.get("FAIR_DIR", os.path.expanduser("~/maivs-fair")), "results")
TARGETS = (0.95, 0.98, 0.99, 0.995)
ORDER = ["maivs-quality", "maivs-memory", "faiss-hnsw", "hnswlib", "usearch", "scann", "scann-soar",
         "faiss-ivf-pqfs", "faiss-ivf-rabitqfs", "faiss-ivf-rabitqfs4"]
LABEL = {"maivs-quality": "MAIVS quality", "maivs-memory": "MAIVS memory", "faiss-hnsw": "faiss HNSW32",
         "hnswlib": "hnswlib HNSW32", "usearch": "USearch HNSW32", "scann": "ScaNN", "scann-soar": "ScaNN + SOAR",
         "faiss-ivf-pqfs": "faiss IVF-PQ4fs", "faiss-ivf-rabitqfs": "faiss IVF-RaBitQfs (1-bit)",
         "faiss-ivf-rabitqfs4": "faiss IVF-RaBitQfs4 (4-bit)"}


def frontier(pts, rk, vk, higher):
    """Points not dominated in (recall up, value up if higher else down)."""
    s = sorted(pts, key=lambda p: (-p[rk], -p[vk] if higher else p[vk]))
    out, best = [], None
    for p in s:
        v = p[vk]
        if best is None or (v > best if higher else v < best):
            out.append(p)
            best = v
    return sorted(out, key=lambda p: p[rk])


def interp(front, rk, vk, t):
    for a, b in zip(front, front[1:]):
        if a[rk] <= t <= b[rk]:
            f = (t - a[rk]) / max(b[rk] - a[rk], 1e-12)
            return math.exp(math.log(a[vk]) + f * (math.log(b[vk]) - math.log(a[vk])))
    if front and abs(front[-1][rk] - t) < 1e-9:
        return front[-1][vk]
    return float("nan")


def fmt(x, d=0):
    return "-" if x != x else f"{x:,.{d}f}"


def load(n, tag=""):
    out = {}
    for s in ORDER:
        p = os.path.join(R, f"{s}_{n}{('_' + tag) if tag else ''}.json")
        if os.path.exists(p):
            out[s] = json.load(open(p))
    return out


def table(n):
    res = load(n)
    if not res:
        return
    print(f"\n## HotpotQA {n // 1000}k  (MiniLM d=384; 16 threads; WSL2)\n")
    print("| system | build s | B/vec | best R@10 IND / OOD | peak RSS MB |")
    print("|---|---|---|---|---|")
    for s, r in res.items():
        bs = r.get("build_s_range") or [r["build_s"]]
        b = f"{min(bs):.1f}" + (f"-{max(bs):.1f}" if len(bs) > 1 else "")
        best = " / ".join(f"{max(p['recall'] for p in r['grid'][q]):.4f}" for q in ("IND", "OOD"))
        print(f"| {LABEL[s]} | {b} | {fmt(r['mem_bytes'] / n)} | {best} | {fmt(r['peak_rss_mb'])} |")
    for q in ("IND", "OOD"):
        print(f"\n**{q}: batch QPS (1,000 queries, 16 threads) at matched R@10**\n")
        print("| system | " + " | ".join(f"{t}" for t in TARGETS) + " |")
        print("|---|" + "---|" * len(TARGETS))
        rows = {}
        for s, r in res.items():
            pts = r["grid"][q]
            rows[s] = [interp(frontier(pts, "recall", "qps", True), "recall", "qps", t) for t in TARGETS]
            print(f"| {LABEL[s]} | " + " | ".join(fmt(x) for x in rows[s]) + " |")
            if s.startswith("maivs"):
                pq = [interp(frontier(pts, "recall_perquery", "qps_perquery", True), "recall_perquery",
                             "qps_perquery", t) for t in TARGETS]
                rows[s + "/pq"] = pq
                print(f"| {LABEL[s]}, per-query (no batching) | " + " | ".join(fmt(x) for x in pq) + " |")
        base = [s for s in res if not s.startswith("maivs")]
        if base:
            best = [max((rows[s][i] for s in base if rows[s][i] == rows[s][i]), default=float("nan"))
                    for i in range(len(TARGETS))]
            who = [max((s for s in base if rows[s][i] == rows[s][i]), key=lambda s: rows[s][i], default=None)
                   for i in range(len(TARGETS))]
            print("| *best baseline* | " + " | ".join(
                f"{fmt(b)} ({LABEL[w]})" if w else "-" for b, w in zip(best, who)) + " |")
            for m in ("maivs-quality", "maivs-quality/pq"):
                if m in rows:
                    nm = "MAIVS quality / best" + (", per-query" if m.endswith("/pq") else ", cell-major")
                    print(f"| *{nm}* | " + " | ".join(
                        "-" if not (a == a and b == b) else f"{a / b:.2f}x" for a, b in zip(rows[m], best)) + " |")
        print(f"\n**{q}: single-thread p50 latency (us) at matched R@10, each system's latency-best setting**\n")
        print("| system | " + " | ".join(f"{t}" for t in TARGETS) + " |")
        print("|---|" + "---|" * len(TARGETS))
        for s, r in res.items():
            pts = r["grid"][q]
            v = [interp(frontier(pts, "recall", "p50_us", False), "recall", "p50_us", t) for t in TARGETS]
            print(f"| {LABEL[s]} | " + " | ".join(fmt(x) for x in v) + " |")


def drift():
    out = []
    for s in ("maivs-quality", "faiss-hnsw"):
        a, b = load(500000).get(s), load(500000, "repeat").get(s)
        if not (a and b):
            continue
        for q in ("IND", "OOD"):
            for t in (0.95, 0.99):
                x = interp(frontier(a["grid"][q], "recall", "qps", True), "recall", "qps", t)
                y = interp(frontier(b["grid"][q], "recall", "qps", True), "recall", "qps", t)
                out.append(f"| {LABEL[s]} | {q} {t} | {fmt(x)} | {fmt(y)} | {100 * (y - x) / x:+.0f}% |")
    if out:
        print("\n## Run-to-run drift (500k, same system re-run last in a fresh process)\n")
        print("| system | point | first run QPS | repeat QPS | change |\n|---|---|---|---|---|")
        print("\n".join(out))


if __name__ == "__main__":
    env = os.path.join(R, "environment.json")
    if os.path.exists(env):
        e = json.load(open(env))
        print("Environment: " + ", ".join(f"{k} {v}" for k, v in e.items()))
    for n in (250000, 500000):
        table(n)
    drift()
