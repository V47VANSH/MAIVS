"""polysemy_lab.py: does decorrelated multi-membership change the tail exponent?

Pure routing, same convention as the other research benches: exact f32 centroid ranking, exact
in-cell scan, recall@10 vs exact ground truth, cost = vectors scanned at a fixed nprobe (spilled
entries count).

Arms:
  none      one cell per vector
  near      + second-nearest cell (the old spill rule)
  soar      + the cell minimising ||r'||^2 + lam*(r'.r_hat)^2
  soar@f    soar, but only for the fraction f of worst-placed vectors

Usage: python polysemy_lab.py <corpus.npy> <limit or 0> <part.npz or ''> <gt.npz>
"""
import sys, time, numpy as np, faiss

t0 = time.time()
corpus, limit, part_path, gt_path = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
CELL = 128

Xmm = np.load(corpus, mmap_mode="r")
if limit:
    Xmm = Xmm[:limit]
rng = np.random.default_rng(0)
idx = rng.permutation(len(Xmm))
q_idx, b_idx = np.sort(idx[:1000]), np.sort(idx[1000:])
def nrm(a):
    n = np.sqrt(np.einsum("ij,ij->i", a, a)) + 1e-12
    a /= n[:, None]
    return a
Q = nrm(np.asarray(Xmm[q_idx], dtype=np.float32))
X = nrm(np.asarray(Xmm[b_idx], dtype=np.float32))
del Xmm
N, d = X.shape
gt = np.load(gt_path)["ids"][:, :10]
# sanity: GT must match this split
chk = (Q[:20] @ X.T).argmax(1)
assert (chk == gt[:20, 0]).mean() > 0.9, "GT does not match split"
print(f"[load] N={N} {time.time()-t0:.1f}s", flush=True)

if part_path:
    p = np.load(part_path)
    C = np.ascontiguousarray(p["cents"], dtype=np.float32)
else:
    M = max(16, N // CELL)
    km = faiss.Kmeans(d, M, niter=12, spherical=True, seed=0, verbose=False,
                      max_points_per_centroid=CELL * 4)
    km.train(X)
    C = km.centroids.astype(np.float32)
C = nrm(C)
M = len(C)
print(f"[part] M={M} {time.time()-t0:.1f}s", flush=True)

# top-T centroids per base vector
T = 16
fi = faiss.IndexFlatIP(d); fi.add(C)
S_top, I_top = fi.search(X, T)          # (N,T) sims, ids
a1 = I_top[:, 0]
s1 = S_top[:, 0]
print(f"[assign] {time.time()-t0:.1f}s", flush=True)

G = C @ C.T                              # centroid gram, for residual geometry
r1n = np.sqrt(np.maximum(2-2 * s1, 1e-12))   # ||x - mu1||

def soar_pick(lam, exclude=None):
    """Secondary cell among top-T: argmin ||r'||^2 + lam*(r'.r1_hat)^2."""
    cand = I_top[:, 1:]; sc = S_top[:, 1:]
    rr = 2-2 * sc                                   # ||r'||^2
    g = G[a1[:, None], cand]                          # mu1 . mu_j
    proj = (1 - s1[:, None] - sc + g) / r1n[:, None]  # r' . r1_hat
    loss = rr + lam * proj ** 2
    if exclude is not None:
        loss[cand == exclude[:, None]] = np.inf
    j = loss.argmin(1)
    return cand[np.arange(N), j]

arms = {"none": [a1]}
arms["near"] = [a1, I_top[:, 1]]
arms["near3"] = [a1, I_top[:, 1], I_top[:, 2]]
for lam in (1.0, 4.0):
    arms[f"soar{lam:g}"] = [a1, soar_pick(lam)]

def soar_third(lam, a2):
    """Third cell, decorrelated from BOTH earlier residuals."""
    cand = I_top[:, 1:]; sc = S_top[:, 1:]
    pos2 = (cand == a2[:, None])
    s2 = (sc * pos2).sum(1)
    r2n = np.sqrt(np.maximum(2-2 * s2, 1e-12))
    g1 = G[a1[:, None], cand]; g2 = G[a2[:, None], cand]
    p1 = (1 - s1[:, None] - sc + g1) / r1n[:, None]
    p2 = (1 - s2[:, None] - sc + g2) / r2n[:, None]
    loss = (2-2 * sc) + lam * (p1 ** 2 + p2 ** 2)
    loss[pos2] = np.inf
    return cand[np.arange(N), loss.argmin(1)]

arms["soar1x3"] = [a1, arms["soar1"][1], soar_third(1.0, arms["soar1"][1])]
# selective: only the worst-placed fraction gets a second home
s_l4 = arms["soar4"][1]
order = np.argsort(s1)                    # lowest x.mu1 first = worst placed
for f in (0.25, 0.5):
    a2 = np.full(N, -1, dtype=np.int64)
    sel = order[: int(f * N)]
    a2[sel] = s_l4[sel]
    arms[f"soar4@{f:g}"] = [a1, a2]
print(f"[arms] {time.time()-t0:.1f}s", flush=True)

# query-side ranking (exact f32 centroid scores)
SQ = Q @ C.T
ordr = np.argsort(-SQ, axis=1)
rank = np.empty_like(ordr)
rows = np.arange(len(Q))[:, None]
rank[rows, ordr] = np.arange(M)[None, :]

targets = (0.90, 0.95, 0.98, 0.99)
print(f"\nN={N} M={M}  vectors scanned at fixed nprobe to reach R@10 target")
print(f"{'arm':12s} {'repl':>5s} " + " ".join(f"{t:>9.2f}" for t in targets) + "   maxR")
res = {}
for name, cells in arms.items():
    sizes = np.bincount(cells[0], minlength=M).astype(np.int64)
    for a in cells[1:]:
        sizes += np.bincount(a[a >= 0], minlength=M)
    repl = sizes.sum() / N
    # depth at which each true neighbour is first found
    dep = rank[np.arange(len(Q))[:, None], cells[0][gt]]
    for a in cells[1:]:
        aa = a[gt]
        r2 = np.where(aa >= 0, rank[np.arange(len(Q))[:, None], np.maximum(aa, 0)], M)
        dep = np.minimum(dep, r2)
    dep = dep.ravel() + 1
    cum = np.cumsum(sizes[ordr], axis=1).mean(0)       # mean scanned at depth P
    hist = np.bincount(dep, minlength=M + 2)[1:M + 1]
    rec = np.cumsum(hist) / dep.size                    # recall at depth P
    out = []
    for t in targets:
        P = np.searchsorted(rec, t)
        out.append(cum[P] if P < M else np.nan)
    res[name] = out
    print(f"{name:12s} {repl:5.2f} " + " ".join(f"{v:9.0f}" for v in out) + f"   {rec[-1]:.4f}")
print(f"\n[done] {time.time()-t0:.1f}s")
import json, os
outp = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"poly_N{N}.json")
json.dump({"N": N, "M": M, "targets": targets,
           "part": "lab-cache" if part_path else "faiss-kmeans",
           "res": {k: [None if np.isnan(v) else float(v) for v in vs] for k, vs in res.items()}},
          open(outp, "w"), indent=1)
