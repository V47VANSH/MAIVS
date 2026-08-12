"""residual_lab.py: a residual attractor channel.

Each vector x is in cell i (attractor mu_i) and gets residual codes j1 (and j2) from residual
codebooks: x ~ mu_i + rho_j1 (+ psi_j2). A cell's members are stored grouped by code, so each (cell,
code) group is a contiguous run. The query visits groups in decreasing
    s(i, j) = q.mu_i + q.rho_j (+ q.psi_k)
Cost = vectors examined before each true neighbour is reached. Compared with plain cells in centroid
order at the same budget.

Usage: python residual_lab.py <hotpot_5M.npy> <N> <questions.npy>
"""
import sys, time, json, os, numpy as np, faiss

t0 = time.time()
CELL, K = 128, 10
path, N, qpath = sys.argv[1], int(sys.argv[2]), sys.argv[3]
HERE = os.path.dirname(os.path.abspath(__file__))
mm = np.load(path, mmap_mode="r")
X = np.ascontiguousarray(mm[:N], dtype=np.float32)
Qs = {"OOD": np.load(qpath).astype(np.float32)[:500],
      "IND": np.ascontiguousarray(mm[-500:], dtype=np.float32)}
d = X.shape[1]
fi = faiss.IndexFlatIP(d); fi.add(X)
GT = {k: fi.search(v, K)[1] for k, v in Qs.items()}; del fi
M = N // CELL
km = faiss.Kmeans(d, M, niter=12, spherical=True, seed=0, verbose=False, max_points_per_centroid=512)
km.train(X)
C = km.centroids.astype(np.float32)
C /= np.linalg.norm(C, axis=1, keepdims=True)
fc = faiss.IndexFlatIP(d); fc.add(C)
a1 = fc.search(X, 1)[1][:, 0]
R1 = X - C[a1]
print(f"[cells] M={M} {time.time()-t0:.1f}s", flush=True)

def vq(R, k, seed):
    kmr = faiss.Kmeans(d, k, niter=10, seed=seed, verbose=False, max_points_per_centroid=256)
    kmr.train(R)
    cb = kmr.centroids.astype(np.float32)
    ix = faiss.IndexFlatL2(d); ix.add(cb)
    return cb, ix.search(R, 1)[1][:, 0]

books = {}
for k in ((256,) if os.environ.get("NO4096") else (256, 4096)):
    cb, j = vq(R1, k, 1)
    books[k] = (cb, j)
    ev = 1 - ((R1 - cb[j]) ** 2).sum() / (R1 ** 2).sum()
    print(f"[rvq1 K={k}] residual variance explained {ev:.3f}  {time.time()-t0:.1f}s", flush=True)
cb2, j2 = vq(R1 - books[256][0][books[256][1]], 256, 2)

def arm_scores(Q, arm):
    S = Q @ C.T
    if arm == "cells":
        return S[:, a1], True
    if arm.startswith("rvq1-"):
        k = int(arm.split("-")[1]); cb, j = books[k]
        return S[:, a1] + (Q @ cb.T)[:, j], False
    cb, j = books[256]
    return S[:, a1] + (Q @ cb.T)[:, j] + (Q @ cb2.T)[:, j2], False

out = {"N": N, "M": M}
budgets = np.array([2000, 4000, 8000, 16000, 32000, 64000])
for qn, Q in Qs.items():
    g = GT[qn]; out[qn] = {}
    print(f"\n{qn} N={N}: recall@10 at a budget of vectors examined")
    print(f"{'arm':10s} " + " ".join(f"{b:>7d}" for b in budgets) + "   vectors for R@10 0.95 / 0.98 / 0.99")
    for arm in (os.environ.get("ARMS", "cells,rvq1-256,rvq1-4096,rvq2-256x256").split(",")):
        ranks = []
        for s in range(0, len(Q), 50):
            E, whole_cell = arm_scores(Q[s:s + 50], arm)
            for qi in range(E.shape[0]):
                e = E[qi]; es = np.sort(e)
                tv = e[g[s + qi]]
                # examined before reaching x*: everything scoring above it, plus
                # (whole_cell) its entire cell / (groups) its own group's tie mass
                rk = len(e) - np.searchsorted(es, tv, side="left" if whole_cell else "right") \
                    if whole_cell else len(e) - np.searchsorted(es, tv, side="left")
                ranks.append(rk)
        ranks = np.concatenate(ranks)
        recs = [(ranks <= b).mean() for b in budgets]
        srt = np.sort(ranks)
        need = [int(srt[int(np.ceil(t * len(srt))) - 1]) for t in (0.95, 0.98, 0.99)]
        out[qn][arm] = {"recall_at_budget": [float(r) for r in recs], "need": need}
        print(f"{arm:10s} " + " ".join(f"{r:7.4f}" for r in recs) + f"   {need[0]:7d} / {need[1]:7d} / {need[2]:7d}", flush=True)
json.dump(out, open(os.path.join(HERE, f"residual_N{N}.json"), "w"), indent=1)
print(f"[done] {time.time()-t0:.1f}s")
