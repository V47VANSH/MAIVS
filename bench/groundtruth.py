"""groundtruth.py: exact nearest-neighbour ground truth for measuring ANN recall.

Until now the benchmarks here counted MSMARCO qrel hits, which measures retrieval quality and is
capped by the embedding model. That isn't ANN recall and can't be used to evaluate an index. ANN
recall asks: of the K vectors actually closest to the query, how many did the index return? It's
measured against exhaustive search and tops out at 1.0.

Query modes:
    holdout   the last Q corpus vectors become queries and are removed from the base
              (same distribution as the corpus)
    external  queries come from a separate .npy, e.g. embedded questions (a different
              distribution from the passages)

Running both measures the question/passage mismatch directly.

Similarity is inner product on L2-normalised vectors, the same as the index computes.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from typing import Optional, Tuple

import numpy as np

# Don't import torch here: faiss and torch in one process segfault on Windows (clashing OpenMP
# runtimes). benchmark_aigvs.py works around it with KMP_DUPLICATE_LIB_OK. This module only needs
# numpy and faiss.

CACHE_DIR = os.path.join(os.path.dirname(__file__), "_gt_cache")


# --- helpers ---

def normalize(X: np.ndarray, inplace: bool = False) -> np.ndarray:
    """L2-normalise rows (zero rows stay zero).

    inplace=True avoids a full copy, which matters at 1M+. Only use it on arrays you own, not mmap
    views.
    """
    if not inplace:
        X = np.ascontiguousarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    if inplace:
        X /= norms
        return X
    return X / norms


def _fingerprint(*parts: object) -> str:
    """Stable short hash identifying a ground-truth configuration."""
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, np.ndarray):
            # Hash shape + a deterministic sample rather than the whole array;
            # full hashing of 1.5 GB is slower than recomputing the GT.
            h.update(str(p.shape).encode())
            h.update(p[:: max(1, len(p) // 64)].tobytes())
        else:
            h.update(str(p).encode())
    return h.hexdigest()[:16]


def split_holdout(X: np.ndarray, n_queries: int, seed: int = 0
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Split a corpus into (base, queries).

    Queries are sampled at random instead of taken from the tail, because some corpora are ordered
    (the MSMARCO cache puts judged passages first) and a tail slice wouldn't be representative.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    q_idx, b_idx = idx[:n_queries], idx[n_queries:]
    return X[np.sort(b_idx)], X[np.sort(q_idx)]


# --- helpers ---

def compute_groundtruth(
    base: np.ndarray,
    queries: np.ndarray,
    k: int = 100,
    use_faiss: bool = True,
    base_block: int = 65_536,
    faiss_max_n: int = 200_000,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact top-k search, streaming over the base set.

    faiss.IndexFlatIP.add copies the whole base, which at 1M x 384 needs ~1.5 GB on top of the
    corpus and hits std::bad_alloc. So faiss is only used below faiss_max_n; above that we block
    over the base and keep a running top-k. Extra memory is len(queries) * base_block * 4 bytes
    (~262 MB at Q=1000, block=65536) regardless of corpus size.

    Returns
        ids   (Q, k) int32, indices into base, sorted by decreasing similarity
        sims  (Q, k) float32 cosine similarities
    """
    base = normalize(base)
    queries = normalize(queries)
    n, d = base.shape
    q = len(queries)
    k = min(k, n)

    t0 = time.perf_counter()

    if use_faiss and n <= faiss_max_n:
        try:
            import faiss  # noqa: PLC0415  (optional dependency, imported late)
            index = faiss.IndexFlatIP(d)
            index.add(base)
            sims, ids = index.search(queries, k)
            if verbose:
                print(f"[gt] faiss IndexFlatIP: {q} queries x {n:,} base "
                      f"in {time.perf_counter() - t0:.1f}s")
            return ids.astype(np.int32), sims.astype(np.float32)
        except (ImportError, MemoryError, RuntimeError) as e:
            if verbose:
                print(f"[gt] faiss unusable ({type(e).__name__}) - using numpy")

    # Streaming top-k over base blocks: keep the best k so far and merge each block's top k into it.
    run_ids = np.full((q, k), -1, dtype=np.int64)
    run_sims = np.full((q, k), -np.inf, dtype=np.float32)

    for start in range(0, n, base_block):
        stop = min(start + base_block, n)
        s = queries @ base[start:stop].T                       # (q, b)
        b = stop - start
        kk = min(k, b)
        part = np.argpartition(-s, kth=kk - 1, axis=1)[:, :kk]  # (q, kk)
        part_s = np.take_along_axis(s, part, axis=1)
        del s

        # Merge running best with this block's best.
        cand_ids = np.concatenate([run_ids, part.astype(np.int64) + start], axis=1)
        cand_sims = np.concatenate([run_sims, part_s], axis=1)
        sel = np.argpartition(-cand_sims, kth=k - 1, axis=1)[:, :k]
        run_ids = np.take_along_axis(cand_ids, sel, axis=1)
        run_sims = np.take_along_axis(cand_sims, sel, axis=1)

        if verbose:
            print(f"[gt] scanned {stop:,}/{n:,}", end="\r", flush=True)

    order = np.argsort(-run_sims, axis=1)
    ids = np.take_along_axis(run_ids, order, axis=1).astype(np.int32)
    sims = np.take_along_axis(run_sims, order, axis=1).astype(np.float32)
    if verbose:
        print(f"\n[gt] numpy exact: {q} queries x {n:,} base "
              f"in {time.perf_counter() - t0:.1f}s")
    return ids, sims


def load_or_build(
    corpus_path: str,
    k: int = 100,
    n_queries: int = 1000,
    query_path: Optional[str] = None,
    limit: Optional[int] = None,
    seed: int = 0,
    verbose: bool = True,
):
    """Load a corpus, derive a query set, and return cached exact ground truth.

    Returns (base, queries, gt_ids, gt_sims, tag); tag names the configuration for result files.
    """
    # Keep the corpus memory-mapped and make exactly one copy. At 1M x 384 f32 each extra copy is
    # 1.5 GB, and the naive load -> asarray -> split -> normalise chain peaks near 4.5 GB.
    Xmm = np.load(corpus_path, mmap_mode="r")
    if limit is not None:
        Xmm = Xmm[:limit]

    if query_path is not None:
        base = normalize(np.ascontiguousarray(Xmm, dtype=np.float32), inplace=True)
        queries = normalize(np.load(query_path).astype(np.float32))[:n_queries]
        mode = "external"
    else:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(Xmm))
        q_idx = np.sort(idx[:n_queries])
        b_idx = np.sort(idx[n_queries:])
        queries = normalize(np.asarray(Xmm[q_idx], dtype=np.float32), inplace=True)
        base = normalize(np.asarray(Xmm[b_idx], dtype=np.float32), inplace=True)
        del idx, q_idx, b_idx
        mode = "holdout"
    del Xmm

    tag = (f"{os.path.basename(corpus_path).replace('.npy', '')}"
           f"_{mode}_n{len(base)}_q{len(queries)}_k{k}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(
        CACHE_DIR, f"gt_{tag}_{_fingerprint(base, queries, k)}.npz")

    if os.path.exists(cache):
        if verbose:
            print(f"[gt] cache hit: {os.path.basename(cache)}")
        z = np.load(cache)
        return base, queries, z["ids"], z["sims"], tag

    if verbose:
        print(f"[gt] computing exact top-{k} for {len(queries)} queries "
              f"over {len(base):,} base vectors (mode={mode})...")
    ids, sims = compute_groundtruth(base, queries, k=k, verbose=verbose)
    np.savez_compressed(cache, ids=ids, sims=sims)
    if verbose:
        print(f"[gt] cached -> {os.path.basename(cache)}")
    return base, queries, ids, sims, tag


# --- helpers ---

def recall_at_k(retrieved: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    """ANN recall@k: |retrieved_k & groundtruth_k| / k, averaged over queries.

    `retrieved` can be shorter than k; missing slots count as misses.
    """
    assert len(retrieved) == len(gt_ids), "query count mismatch"
    total = 0.0
    for r, g in zip(retrieved, gt_ids):
        truth = set(g[:k].tolist())
        got = set(int(x) for x in r[:k] if x >= 0)
        total += len(truth & got) / k
    return total / len(gt_ids)


def mean_reciprocal_rank(retrieved: np.ndarray, gt_ids: np.ndarray) -> float:
    """MRR of the single true nearest neighbour within the returned list."""
    total = 0.0
    for r, g in zip(retrieved, gt_ids):
        nn = int(g[0])
        for rank, cand in enumerate(r, 1):
            if int(cand) == nn:
                total += 1.0 / rank
                break
    return total / len(gt_ids)


# --- helpers ---

def main() -> None:
    p = argparse.ArgumentParser(description="Build exact ANN ground truth.")
    p.add_argument("corpus", help="path to .npy of shape (N, D)")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--query-path", default=None,
                   help="external query .npy; omit for in-distribution holdout")
    p.add_argument("--limit", type=int, default=None,
                   help="use only the first L corpus vectors")
    args = p.parse_args()

    base, queries, ids, sims, tag = load_or_build(
        args.corpus, k=args.k, n_queries=args.queries,
        query_path=args.query_path, limit=args.limit,
    )
    print(f"\n[gt] {tag}")
    print(f"     base    {base.shape}")
    print(f"     queries {queries.shape}")
    print(f"     gt ids  {ids.shape}")
    print(f"     mean sim@1  {sims[:, 0].mean():.4f}")
    print(f"     mean sim@10 {sims[:, 9].mean():.4f}"
          if ids.shape[1] >= 10 else "")


if __name__ == "__main__":
    main()
