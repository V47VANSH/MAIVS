"""polysemy_ood.py: orthogonal polysemy on HotpotQA, held-out passages and real questions.

base = first N rows of the shuffled corpus, IND = last 1000 rows, OOD = 1000 HotpotQA questions
embedded with the same MiniLM. Exact ground truth by brute force. Same cost convention as
polysemy_lab.py.

Usage: python polysemy_ood.py <hotpot_5M.npy> <N> <questions.npy>
"""
import sys, time, json, os, numpy as np, faiss

t0 = time.time()
path, N, qpath = sys.argv[1], int(sys.argv[2]), sys.argv[3]
CELL, T, K = 128, 16, 10
mm = np.load(path, mmap_mode="r")
X = np.ascontiguousarray(mm[:N], dtype=np.float32)
Qi = np.ascontiguousarray(mm[-1000:], dtype=np.float32)
Qo = np.load(qpath).astype(np.float32)
d = X.shape[1]
print(f"[load] N={N} {time.time()-t0:.1f}s", flush=True)

def exact(Q):
    fi = faiss.IndexFlatIP(d); fi.add(X)
    return fi.search(Q, K)[1]
gt = {"IND": exact(Qi), "OOD": exact(Qo)}
print(f"[gt] {time.time()-t0:.1f}s", flush=True)

M = N // CELL
km = faiss.Kmeans(d, M, niter=12, spherical=True, seed=0, verbose=False,
                  max_points_per_centroid=CELL * 4)
km.train(X)
C = km.centroids.astype(np.float32)
C /= np.sqrt(np.einsum("ij,ij->i", C, C))[:, None]
fc = faiss.IndexFlatIP(d); fc.add(C)
S_top, I_top = fc.search(X, T)
a1, s1 = I_top[:, 0], S_top[:, 0]
G = C @ C.T
r1n = np.sqrt(np.maximum(2-2 * s1, 1e-12))
print(f"[part+assign] M={M} {time.time()-t0:.1f}s", flush=True)

def soar(lam):
    cand, sc = I_top[:, 1:], S_top[:, 1:]
    proj = (1 - s1[:, None] - sc + G[a1[:, None], cand]) / r1n[:, None]
    return cand[np.arange(N), ((2-2 * sc) + lam * proj ** 2).argmin(1)]

arms = {"none": [a1], "near": [a1, I_top[:, 1]],
        "soar1": [a1, soar(1.0)], "soar4": [a1, soar(4.0)]}
def soar3(lam, a2):
    cand, sc = I_top[:, 1:], S_top[:, 1:]
    pos2 = cand == a2[:, None]
    s2 = (sc * pos2).sum(1)
    r2n = np.sqrt(np.maximum(2-2 * s2, 1e-12))
    p1 = (1 - s1[:, None] - sc + G[a1[:, None], cand]) / r1n[:, None]
    p2 = (1 - s2[:, None] - sc + G[a2[:, None], cand]) / r2n[:, None]
    loss = (2-2 * sc) + lam * (p1 ** 2 + p2 ** 2)
    loss[pos2] = np.inf
    return cand[np.arange(N), loss.argmin(1)]

arms["soar4x3"] = [a1, arms["soar4"][1], soar3(4.0, arms["soar4"][1])]
arms["near3"] = [a1, I_top[:, 1], I_top[:, 2]]

targets = (0.90, 0.95, 0.98, 0.99)
out = {"N": N, "M": M}
for qn, Q in (("IND", Qi), ("OOD", Qo)):
    SQ = Q @ C.T
    ordr = np.argsort(-SQ, axis=1)
    rank = np.empty_like(ordr)
    rank[np.arange(len(Q))[:, None], ordr] = np.arange(M)[None, :]
    g = gt[qn]
    print(f"\n{qn}  N={N} M={M}  vectors scanned to reach R@10")
    print(f"{'arm':10s} {'repl':>5s} " + " ".join(f"{t:>8.2f}" for t in targets))
    out[qn] = {}
    for name, cells in arms.items():
        sizes = np.bincount(cells[0], minlength=M).astype(np.int64)
        for a in cells[1:]:
            sizes += np.bincount(a[a >= 0], minlength=M)
        dep = rank[np.arange(len(Q))[:, None], cells[0][g]]
        for a in cells[1:]:
            aa = a[g]
            dep = np.minimum(dep, np.where(aa >= 0, rank[np.arange(len(Q))[:, None], np.maximum(aa, 0)], M))
        dep = dep.ravel() + 1
        cum = np.cumsum(sizes[ordr], axis=1).mean(0)
        rec = np.cumsum(np.bincount(dep, minlength=M + 2)[1:M + 1]) / dep.size
        vals = [float(cum[np.searchsorted(rec, t)]) if np.searchsorted(rec, t) < M else None for t in targets]
        out[qn][name] = vals
        diag = ""
        if len(cells) >= 2:
            # independence check on the two first homes: corr of log-ranks, and
            # P(both missed at depth P*) vs product of marginals.
            ri = np.arange(len(Q))[:, None]
            l1 = np.log1p(rank[ri, cells[0][g]]).ravel()
            l2 = np.log1p(rank[ri, cells[1][g]]).ravel()
            Ps = 64
            m1 = rank[ri, cells[0][g]].ravel() >= Ps
            m2 = rank[ri, cells[1][g]].ravel() >= Ps
            diag = (f"   corr(logrank)={np.corrcoef(l1, l2)[0,1]:.2f}  "
                    f"P(both miss@{Ps})={np.mean(m1 & m2):.3f} vs indep {np.mean(m1)*np.mean(m2):.3f}")
        print(f"{name:10s} {sizes.sum()/N:5.2f} " + " ".join(f"{(v or float('nan')):8.0f}" for v in vals) + diag)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"ood_N{N}.json"), "w"), indent=1)
print(f"\n[done] {time.time()-t0:.1f}s")
