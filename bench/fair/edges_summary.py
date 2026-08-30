"""Markdown tables for the memory-matched, churn and filtered experiments.

    python bench/fair/edges_summary.py [RESULTS_DIR] > edges_tables.md
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarise import frontier, interp, fmt, R  # noqa: E402


def j(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def at(pts, t, vk="qps"):
    return interp(frontier(pts, "recall", vk, vk == "qps"), "recall", vk, t)


def memory_matched():
    rows = [("MAIVS memory (SQ8 rerank)", "maivs-memory_500000.json"),
            ("ScaNN + SOAR, INT8 reorder", "scann-soar-int8_500000.json"),
            ("ScaNN, INT8 reorder", "scann-int8_500000.json"),
            ("faiss HNSW32, SQ8", "faiss-hnsw-sq8_500000.json"),
            ("faiss IVF-PQ4fs + SQ8 refine", "faiss-ivf-pqfs-sq8_500000.json"),
            ("(ref) MAIVS quality, f32", "maivs-quality_500000.json"),
            ("(ref) ScaNN + SOAR, f32", "scann-soar_500000.json")]
    T = (0.95, 0.98, 0.99)
    print("\n## 1. Same memory budget (~600 B/vec), HotpotQA 500k\n")
    print("| system | B/vec | build s | best R@10 IND / OOD | " +
          " | ".join(f"QPS {q} {t}" for q in ("IND", "OOD") for t in T) + " | p50 us IND 0.95 / 0.98 |")
    print("|---|---|---|---|" + "---|" * (2 * len(T)) + "---|")
    for lab, f in rows:
        r = j(f)
        if not r:
            continue
        g = r["grid"]
        bs = r.get("build_s_range") or [r["build_s"]]
        b = f"{min(bs):.1f}" + (f"-{max(bs):.1f}" if len(bs) > 1 else "")
        best = " / ".join(f"{max(p['recall'] for p in g[q]):.4f}" for q in ("IND", "OOD"))
        q = [fmt(at(g[qn], t)) for qn in ("IND", "OOD") for t in T]
        lat = " / ".join(fmt(at(g["IND"], t, "p50_us")) for t in (0.95, 0.98))
        print(f"| {lab} | {fmt(r['mem_bytes'] / r['n'])} | {b} | {best} | " + " | ".join(q) + f" | {lat} |")


def churn():
    print("\n## 2. Churn: 25% turnover (100k deletes + 100k inserts, 10 rounds) of a 400k index\n")
    print("| system | build S0 s | deletes/s | inserts/s | maintenance | " +
          " | ".join(f"{ph} QPS@{q} 0.95 / 0.98" for ph in ("churned", "after maint.", "fresh rebuild")
                     for q in ("IND", "OOD")) + " |")
    print("|---|---|---|---|---|" + "---|" * 6)
    CH = (("MAIVS quality", "churn_maivs.json"), ("MAIVS memory", "churn_maivs-memory.json"),
          ("ScaNN (no SOAR)", "churn_scann.json"), ("hnswlib HNSW32", "churn_hnswlib.json"),
          ("faiss HNSW32", "churn_faiss-hnsw.json"))
    for lab, f in CH:
        r = j(f)
        if not r:
            continue
        dps = fmt(r["churn"] / r["delete_s"]) if "delete_s" in r else "not supported"
        ips = fmt(r["churn"] / r["insert_s"]) if "insert_s" in r else "not supported"
        mt = r.get("maintain", "")
        if "maintain_s" in r:
            mt += f", {r['maintain_s']:.1f} s"
        cells = []
        for ph in ("churned", "maintained", "fresh"):
            for qn in ("IND", "OOD"):
                if ph in r:
                    cells.append(" / ".join(fmt(at(r[ph][qn], t)) for t in (0.95, 0.98)))
                else:
                    cells.append("-")
        b0 = fmt(r["build_s0_s"], 1) if "build_s0_s" in r else "-"
        print(f"| {lab} | {b0} | {dps} | {ips} | {mt} | " + " | ".join(cells) + " |")
    print("| ScaNN + SOAR | - | crashes | crashes | - | ScaNN 1.4.2: segfault in delete() (f32 reorder); "
          "heap corruption in upsert() at 400k (INT8 reorder) | | | | | |")
    print("\nFresh-rebuild build times (400k): " + ", ".join(
        f"{lab} {j(f)['fresh_build_s']:.1f} s" for lab, f in CH if j(f)))
    print("\nBest R@10 reached (IND/OOD), churned / after maintenance / fresh: " + "; ".join(
        f"{lab} " + ", ".join("/".join(f"{max(p['recall'] for p in j(f)[ph][q]):.4f}" for q in ("IND", "OOD"))
                               for ph in ("churned", "maintained", "fresh") if ph in j(f))
        for lab, f in CH if j(f)))


def filtered():
    print("\n## 3. Filtered search, HotpotQA 500k, random attribute, batch QPS at matched R@10\n")
    T = (0.90, 0.95, 0.99)
    sysl = (("MAIVS (filter in scan)", "filtered_maivs.json"),
            ("faiss HNSW32 (IDSelector)", "filtered_faiss-hnsw.json"),
            ("ScaNN + SOAR (over-fetch + post-filter)", "filtered_scann-soar.json"),
            ("brute force over matching subset", "filtered_bruteforce.json"))
    data = {lab: j(f) for lab, f in sysl}
    for s in ("0.5", "0.1", "0.01", "0.001"):
        print(f"\n**Filter keeps {float(s) * 100:g}% ({int(500000 * float(s)):,} vectors)**\n")
        print("| system | best R@10 IND / OOD | " + " | ".join(f"{q} {t}" for q in ("IND", "OOD") for t in T) + " |")
        print("|---|---|" + "---|" * (2 * len(T)))
        for lab, r in data.items():
            if not r or s not in r["sel"]:
                continue
            g = r["sel"][s]
            best = " / ".join(f"{max(p['recall'] for p in g[q]):.4f}" for q in ("IND", "OOD"))
            if lab.startswith("brute"):
                vals = [fmt(g[q][0]["qps"]) for q in ("IND", "OOD") for _ in T]
            else:
                vals = [fmt(at(g[q], t)) for q in ("IND", "OOD") for t in T]
            print(f"| {lab} | {best} | " + " | ".join(vals) + " |")


if __name__ == "__main__":
    memory_matched()
    churn()
    filtered()
