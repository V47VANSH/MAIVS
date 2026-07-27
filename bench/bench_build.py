"""bench_build.py: build time split into training (hierarchical k-means, src/tree.rs) and ingestion
(tree-routed placement, O(N log M)).
"""
import argparse, json, os
from common import build_index, load_corpus, rss_mb, RESULTS_DIR, PROFILES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--profile", default="memory", choices=list(PROFILES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--faiss", action="store_true", help="also time faiss-hnsw")
    a = ap.parse_args()

    base, _q, _gt, tag = load_corpus(a.corpus, limit=a.limit)
    m0 = rss_mb()
    ix, t = build_index(base, a.profile)
    mem = rss_mb() - m0
    st = ix.stats()

    print(f"\n  N={len(base):,}  cells={t['cells']:,}  "
          f"cell_size={len(base)//t['cells']}  arena={st['arena_bytes_per_vec']:.0f} B/vec")
    print(f"  train   {t['train_s']:>7.1f}s   (hierarchical k-means, Rust)")
    print(f"  ingest  {t['ingest_s']:>7.1f}s   (tree-routed placement)")
    print(f"  compact {t['compact_s']:>7.1f}s   (cluster-major layout)")
    print(f"  TOTAL   {t['total_s']:>7.1f}s   RSS {mem:.0f} MB")

    row = dict(t, n=len(base), rss_mb=round(mem, 1), profile=a.profile,
               arena_b_per_vec=st["arena_bytes_per_vec"])

    if a.faiss:
        import time, faiss
        faiss.omp_set_num_threads(int(os.environ.get("BENCH_THREADS", 8)))
        t0 = time.perf_counter()
        h = faiss.IndexHNSWFlat(base.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        h.hnsw.efConstruction = 200
        h.add(base)
        hb = time.perf_counter() - t0
        print(f"\n  faiss-hnsw build {hb:.1f}s  ->  we are {hb/t['total_s']:.1f}x faster")
        row["faiss_hnsw_build_s"] = round(hb, 1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"build_{tag}.json"), "w") as f:
        json.dump(row, f, indent=2)


if __name__ == "__main__":
    main()
