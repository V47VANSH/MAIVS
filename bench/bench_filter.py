"""bench_filter.py: predicate pushdown.

The scan checks the tag before computing a distance, so a selective filter should make the query
cheaper, not slower. Recall is measured against exact ground truth over the filtered subset.
"""
import argparse, json, os
import numpy as np
import faiss
from common import build_index, load_corpus, timed_serial, RESULTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--profile", default="memory")
    ap.add_argument("--probes", default="128,512")
    a = ap.parse_args()

    base, q, _gt, tag = load_corpus(a.corpus, limit=a.limit)
    ix, _t = build_index(base, a.profile)

    rng = np.random.default_rng(7)
    tags = rng.integers(0, 100, size=len(base)).astype(np.uint32)
    ids = list(range(len(base)))
    ix.set_tags(ids, tags.tolist())

    print(f"\n  {'selectivity':>12} {'nprobe':>7} {'R@10 (filtered GT)':>20} {'serial qps':>11}")
    rows = []
    for pct, hi in ((100, 99), (50, 49), (10, 9), (1, 0)):
        ix.set_filter(0, hi)
        sub = np.where(tags <= hi)[0]
        _d, gi = faiss.knn(q, np.ascontiguousarray(base[sub]), 10,
                           metric=faiss.METRIC_INNER_PRODUCT)
        fgt = sub[gi.astype(np.int64)]
        for nprobe in [int(x) for x in a.probes.split(",")]:
            ret, lat = timed_serial(lambda x, n=nprobe: ix.search(x, 10, n, 1.0, 8.0)[0], q, 10)
            r10 = float(np.mean([len(set(ret[i][:10]) & set(fgt[i])) / 10
                                 for i in range(len(q))]))
            qps = 1000.0 / float(np.mean(lat))
            rows.append(dict(selectivity=pct, nprobe=nprobe,
                             recall10=round(r10, 4), serial_qps=round(qps, 1)))
            print(f"  {pct:>11}% {nprobe:>7} {r10:>20.4f} {qps:>11.0f}")
    ix.set_filter(None, None)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"filter_{tag}.json"), "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
