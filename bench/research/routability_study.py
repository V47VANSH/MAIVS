"""routability_study.py: is a better cell ranker possible at all?

An oracle that probes the right cells reaches R@10 0.9997 at nprobe=8, while ranking by centroid
cosine needs nprobe=128 for 0.9942. That's a 7-16x gap, but the oracle knows the answer. Before
building anything: is "does this cell contain one of the query's true neighbours" predictable from
features available at query time? If the best feature set does no better than plain cosine, the
information isn't there.

For every (query, cell) pair we compute query-time features and the label "cell holds at least one
of the true top-10", then compare ranking functions by the recall they reach at a given nprobe (not
just AUC).

  cosine       rank by cos(q, mu_c)                        (baseline)
  bound_max    rank by cos(max(0, angle(q,mu_c) - r_max))  (reachability bound)
  bound_mean   same with the mean member radius
  learned      gradient-boosted trees over all features
  oracle       cells that actually contain neighbours      (upper bound)

The bound rankers use the usual IVF argument: every member of c is within r_max radians of mu_c, so
nothing in c can score better than cos(angle(q,mu_c) - r_max). A large diffuse cell can hold
neighbours even when its centroid is far, which plain cosine ignores.
"""

from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from groundtruth import load_or_build  # noqa: E402

CORPUS = "msmarco_embeddings_cache_v2_100000.npy"
NCELLS = 314
K = 10
PROBES = (1, 2, 4, 8, 16, 32, 64)


def recall_at_nprobe(rank: np.ndarray, assign: np.ndarray, gt: np.ndarray,
                     t: int, k: int = K) -> float:
    """Fraction of true top-k that live in the t highest-ranked cells."""
    return float(np.mean([
        np.isin(assign[gt[i, :k]], rank[i, :t]).mean()
        for i in range(len(rank))
    ]))


def main() -> None:
    import faiss

    base, queries, gt, _, _ = load_or_build(
        CORPUS, k=100, n_queries=1000, verbose=False)
    d = base.shape[1]

    km = faiss.Kmeans(d, NCELLS, niter=15, spherical=True, seed=0)
    km.train(base)
    cent = np.ascontiguousarray(km.centroids, dtype=np.float32)
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)

    sims_base = base @ cent.T
    assign = np.argmax(sims_base, axis=1)

    # --- per-cell geometry (computed at build time) ---
    size = np.zeros(NCELLS, dtype=np.float32)
    r_max = np.zeros(NCELLS, dtype=np.float32)
    r_mean = np.zeros(NCELLS, dtype=np.float32)
    r_std = np.zeros(NCELLS, dtype=np.float32)
    for c in range(NCELLS):
        idx = np.where(assign == c)[0]
        size[c] = len(idx)
        if len(idx) == 0:
            continue
        # angular distance of each member from its centroid
        cs = np.clip(base[idx] @ cent[c], -1.0, 1.0)
        ang = np.arccos(cs)
        r_max[c], r_mean[c], r_std[c] = ang.max(), ang.mean(), ang.std()

    # --- query-time features ---
    qsim = queries @ cent.T                       # (Q, C) cos(q, mu)
    qang = np.arccos(np.clip(qsim, -1.0, 1.0))
    bound_max = np.cos(np.maximum(0.0, qang - r_max[None, :]))
    bound_mean = np.cos(np.maximum(0.0, qang - r_mean[None, :]))

    rank_cos = np.argsort(-qsim, axis=1)
    rank_bmax = np.argsort(-bound_max, axis=1)
    rank_bmean = np.argsort(-bound_mean, axis=1)

    # oracle: cells containing true neighbours first, then by cosine
    oracle_key = np.zeros_like(qsim)
    for i in range(len(queries)):
        oracle_key[i] = qsim[i]
        oracle_key[i, np.unique(assign[gt[i, :K]])] += 10.0
    rank_oracle = np.argsort(-oracle_key, axis=1)

    # --- learned ranker (train/test split over queries, not over pairs) ---
    ntr = len(queries) // 2
    label = np.zeros_like(qsim, dtype=np.int8)
    for i in range(len(queries)):
        label[i, np.unique(assign[gt[i, :K]])] = 1

    def feats(sl):
        n = sl.stop - sl.start if isinstance(sl, slice) else len(sl)
        return np.stack([
            qsim[sl].ravel(),
            bound_max[sl].ravel(),
            bound_mean[sl].ravel(),
            np.tile(r_max, (n, 1)).ravel(),
            np.tile(r_mean, (n, 1)).ravel(),
            np.tile(r_std, (n, 1)).ravel(),
            np.tile(size, (n, 1)).ravel(),
            (qsim[sl] - qsim[sl].max(axis=1, keepdims=True)).ravel(),  # gap to best
        ], axis=1)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    Xtr, ytr = feats(slice(0, ntr)), label[:ntr].ravel()
    Xte, yte = feats(slice(ntr, len(queries))), label[ntr:].ravel()
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    clf.fit(Xtr, ytr)
    p_te = clf.predict_proba(Xte)[:, 1]

    print(f"\n{'='*74}\nP0 ROUTABILITY GATE  (corpus=99k, cells={NCELLS}, k={K})\n{'='*74}")
    print(f"positive rate: {yte.mean()*100:.2f}% of (query,cell) pairs hold a true top-{K}\n")
    print("AUC for predicting 'cell holds a true neighbour' (held-out queries):")
    for nm, col in (("cosine", 0), ("bound_max", 1), ("bound_mean", 2)):
        print(f"  {nm:12s} {roc_auc_score(yte, Xte[:, col]):.4f}")
    print(f"  {'learned':12s} {roc_auc_score(yte, p_te):.4f}")

    # learned ranking on the held-out half
    rank_learn = np.argsort(-p_te.reshape(-1, NCELLS), axis=1)

    print(f"\n{'-'*74}\nR@{K} REACHABLE AT EACH nprobe  (held-out queries only)\n{'-'*74}")
    te = slice(ntr, len(queries))
    gt_te, = (gt[te],)
    hdr = "  ".join(f"np={t:<4}" for t in PROBES)
    print(f"{'ranker':<12} {hdr}")
    for nm, rk in (("cosine", rank_cos[te]), ("bound_max", rank_bmax[te]),
                   ("bound_mean", rank_bmean[te]), ("learned", rank_learn),
                   ("ORACLE", rank_oracle[te])):
        row = "  ".join(f"{recall_at_nprobe(rk, assign, gt_te, t):.4f}"
                        for t in PROBES)
        print(f"{nm:<12} {row}")

    # headline: probes needed to reach the cosine baseline's nprobe=32 recall
    target = recall_at_nprobe(rank_cos[te], assign, gt_te, 32)
    print(f"\nProbes needed to reach cosine@np=32 recall ({target:.4f}):")
    for nm, rk in (("cosine", rank_cos[te]), ("bound_max", rank_bmax[te]),
                   ("bound_mean", rank_bmean[te]), ("learned", rank_learn),
                   ("ORACLE", rank_oracle[te])):
        need = next((t for t in range(1, NCELLS + 1)
                     if recall_at_nprobe(rk, assign, gt_te, t) >= target), None)
        speed = f"{32/need:.2f}x" if need else "n/a"
        print(f"  {nm:<12} nprobe={need!s:<5} ({speed} fewer probes)")


if __name__ == "__main__":
    main()
