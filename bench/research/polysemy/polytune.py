"""polytune.py: try to make the orthogonal home good, not just independent.

  pool T     number of candidate cells searched for the orthogonal home (16 vs 64)
  lambda     closeness vs orthogonality
  pk-means   train attractors on their whole basin: assign primary = nearest, secondary =
             orthogonal (SOAR loss), then attractor = normalise(sum primaries + w * sum secondaries)

Cost = vectors scanned at fixed nprobe (exact centroid ranking), real questions and IND.

Usage: python polytune.py <hotpot_5M.npy> <N> <questions.npy>
"""
import sys, time, json, os, numpy as np, faiss

t0 = time.time()
CELL, K = 128, 10
path, N, qpath = sys.argv[1], int(sys.argv[2]), sys.argv[3]
HERE = os.path.dirname(os.path.abspath(__file__))
mm = np.load(path, mmap_mode="r")
X = np.ascontiguousarray(mm[:N], dtype=np.float32)
Qs = {"OOD": np.load(qpath).astype(np.float32)[:1000],
      "IND": np.ascontiguousarray(mm[-1000:], dtype=np.float32)}
d = X.shape[1]
def nrm(a):
    a /= np.sqrt(np.einsum("ij,ij->i", a, a))[:, None] + 1e-12
    return a
fi = faiss.IndexFlatIP(d); fi.add(X)
GT = {k: fi.search(v, K)[1] for k, v in Qs.items()}; del fi
M = N // CELL
km = faiss.Kmeans(d, M, niter=12, spherical=True, seed=0, verbose=False, max_points_per_centroid=512)
km.train(X)
C0 = nrm(km.centroids.astype(np.float32))
print(f"[kmeans] M={M} {time.time()-t0:.1f}s", flush=True)

def assign(C, T, lam):
    fc = faiss.IndexFlatIP(d); fc.add(C)
    S, I = fc.search(X, T)
    G = C @ C.T
    a1, s1 = I[:, 0], S[:, 0]
    r1n = np.sqrt(np.maximum(2-2 * s1, 1e-12))
    cand, sc = I[:, 1:], S[:, 1:]
    proj = (1 - s1[:, None] - sc + G[a1[:, None], cand]) / r1n[:, None]
    a2 = cand[np.arange(N), ((2-2 * sc) + lam * proj ** 2).argmin(1)]
    return a1, a2

def recenter(a1, a2, w):
    S = np.zeros((M, d), np.float64)
    np.add.at(S, a1, X)
    if w > 0:
        np.add.at(S, a2, w * X)
    S = S.astype(np.float32)
    nz = np.einsum("ij,ij->i", S, S) > 1e-12
    out = C0.copy(); out[nz] = nrm(S[nz])
    return out

def evaluate(C, homes, Q, g):
    sizes = sum(np.bincount(h, minlength=M) for h in homes).astype(np.int64)
    SQ = Q @ C.T
    ordr = np.argsort(-SQ, axis=1)
    rank = np.empty_like(ordr)
    rank[np.arange(len(Q))[:, None], ordr] = np.arange(M)[None, :]
    ri = np.arange(len(Q))[:, None]
    dep = np.min(np.stack([rank[ri, h[g]] for h in homes]), axis=0).ravel() + 1
    cum = np.cumsum(sizes[ordr], axis=1).mean(0)
    rec = np.cumsum(np.bincount(dep, minlength=M + 2)[1:M + 1]) / dep.size
    vals = []
    for t in (0.95, 0.98, 0.99):
        P = np.searchsorted(rec, t)
        vals.append(float(cum[P]) if P < M else float("nan"))
    m = [np.mean(rank[ri, h[g]].ravel() >= 64) for h in homes]
    return vals, m

rows = []
def report(name, C, homes):
    line = f"{name:24s}"
    for qn, Q in Qs.items():
        vals, m = evaluate(C, homes, Q, GT[qn])
        rows.append({"arm": name, "q": qn, "scan@.95/.98/.99": vals, "marg_miss@64": m})
        line += f" | {qn} " + " ".join(f"{v:7.0f}" for v in vals) + " miss@64 " + "/".join(f"{x:.3f}" for x in m)
    print(line, flush=True)

a1, _ = assign(C0, 2, 0)
report("none", C0, [a1])
for T in (16, 64):
    for lam in (1.0, 4.0, 16.0):
        a1, a2 = assign(C0, T, lam)
        report(f"soar T={T} lam={lam:g}", C0, [a1, a2])
# polysemic attractor training (fixed point of the whole basin)
for w in (0.5, 1.0):
    C = C0.copy()
    for it in range(3):
        a1, a2 = assign(C, 64, 4.0)
        C = recenter(a1, a2, w)
    a1, a2 = assign(C, 64, 4.0)
    report(f"pk-means w={w:g} it=3", C, [a1, a2])
json.dump(rows, open(os.path.join(HERE, f"polytune_N{N}.json"), "w"), indent=1)
print(f"[done] {time.time()-t0:.1f}s")
