"""Build faiss HNSW (M=32, efConstruction=200, inner product) on the first N rows of a corpus and cache
it, timing the full build.

    python bench/build_hnsw_cache.py hotpot_5M.npy 1000000
"""
import os, sys, time, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, faiss
path, n = sys.argv[1], int(sys.argv[2])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_index_cache",
                   f"hnsw32_{os.path.basename(path).replace('.npy','')}_n{n}.faissindex")
X = np.ascontiguousarray(np.load(path, mmap_mode="r")[:n], dtype=np.float32)
ix = faiss.IndexHNSWFlat(X.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
ix.hnsw.efConstruction = 200
t0 = time.perf_counter(); ix.add(X); t = time.perf_counter() - t0
faiss.write_index(ix, out)
json.dump({"n": n, "build_s": t, "threads": faiss.omp_get_max_threads(), "M": 32, "efC": 200},
          open(out + ".json", "w"))
print(f"HNSW n={n} build {t:.1f}s -> {out}")
