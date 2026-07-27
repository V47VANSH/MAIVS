"""run_all.py: run every benchmark against one corpus and write a single markdown table plus a JSON
file, so the numbers in the docs and in bench/results/ come from the same run.

    python bench/run_all.py msmarco_embeddings_cache_v2_1000000.npy

--quick runs on a 100k slice with a shorter churn cycle for iterating. Quote the full run.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

STEPS = [
    ("build", ["bench_build.py", "{corpus}", "--faiss"]),
    ("recall", ["bench_recall.py", "{corpus}", "--cells", "{cells}",
                "--rerank", "32", "--per-dim-sq", "--two-stage", "--adaptive",
                "--systems", "v2-binary,faiss-hnsw,faiss-ivf,faiss-hnsw-sq"]),
    ("latency", ["bench_latency.py", "{corpus}"]),
    ("filter", ["bench_filter.py", "{corpus}"]),
    ("churn", ["bench_churn.py", "{corpus}", "--cycles", "{cycles}"]),
    ("concurrency", ["bench_concurrency.py", "{corpus}"]),
    ("scaling", ["bench_scaling.py", "{corpus}", "--sizes", "{sizes}"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated step names")
    a = ap.parse_args()

    n = 100_000 if a.quick else 999_000
    ctx = {
        "corpus": a.corpus,
        "cells": str(max(16, n // 128)),
        "cycles": "3" if a.quick else "10",
        "sizes": "125000,250000" if a.quick else "125000,250000,500000,999000",
    }
    want = set(a.only.split(",")) if a.only else {s for s, _ in STEPS}

    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")
    summary = {}
    for name, argv in STEPS:
        if name not in want:
            continue
        cmd = [sys.executable, os.path.join(HERE, argv[0])] + \
              [x.format(**ctx) for x in argv[1:]]
        if a.quick and name in ("build", "recall", "latency", "filter",
                                "concurrency"):
            cmd += ["--limit", str(n)]
        print(f"\n{'='*78}\n### {name}\n{'='*78}", flush=True)
        t0 = time.perf_counter()
        rc = subprocess.call(cmd, env=env)
        summary[name] = {"ok": rc == 0, "seconds": round(time.perf_counter() - t0, 1)}
        if rc != 0:
            print(f"  !! {name} exited {rc}")

    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    for k, v in summary.items():
        print(f"  {k:<14} {'ok' if v['ok'] else 'FAILED':<8} {v['seconds']:>8.1f}s")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "run_all_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  per-benchmark JSON in {RESULTS}")


if __name__ == "__main__":
    main()
