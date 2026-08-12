"""multipart.py: several independently trained partitions, probed with one joint ranking over all R*M
attractors (not split per table).

Arms (entries per vector in brackets):
  1part        1 partition, 1 home                         (1)
  soar         1 partition, nearest + orthogonal home      (2)
  2part        2 partitions, nearest home in each          (2)
  2part+soar   2 partitions, nearest + orthogonal in each  (4)
  4part        4 partitions, nearest home in each          (4)

Cost = vectors scanned (full member list of every probed cell). Also prints joint miss vs product of
marginals at depth 64 for the first two homes.

Usage: python multipart.py <hotpot_5M.npy> <N> <questions.npy>
"""
import sys, time, json, os, numpy as np, faiss

t0 = time.time()
CELL, T, K = 128, 16, 10
path, N, qpath = sys.argv[1], int(sys.argv[2]), sys.argv[3]
HERE = os.path.dirname(os.path.abspath(__file__))
NQ = 1000
mm = np.load(path, mmap_mode="r")
X = np.ascontiguousarray(mm[:N], dtype=np.float32)
Qs = {"OOD": np.load(qpath).astype(np.float32)[:NQ],
      "IND": np.ascontiguousarray(mm[-NQ:], dtype=np.float32)}
d = X.shape[1]
def nrm(a):
    a /= np.sqrt(np.einsum("ij,ij->i", a, a))[:, None] + 1e-12
    return a
fi = faiss.IndexFlatIP(d); fi.add(X)
GT = {k: fi.search(v, K)[1] for k, v in Qs.items()}; del fi
M = N // CELL

parts = []
for seed in range(4):
    km = faiss.Kmeans(d, M, niter=12, spherical=True, seed=1000 + seed, verbose=False,
                      max_points_per_centroid=512)
    km.train(X)
    C = nrm(km.centroids.astype(np.float32))
    fc = faiss.IndexFlatIP(d); fc.add(C)
    S, I = fc.search(X, T)
    parts.append((C, S, I))
    print(f"[part {seed}] {time.time()-t0:.1f}s", flush=True)

def soar2(C, S, I, lam=4.0):
    G = C @ C.T
    a1, s1 = I[:, 0], S[:, 0]
    r1n = np.sqrt(np.maximum(2-2 * s1, 1e-12))
    cand, sc = I[:, 1:], S[:, 1:]
    proj = (1 - s1[:, None] - sc + G[a1[:, None], cand]) / r1n[:, None]
    return cand[np.arange(N), ((2-2 * sc) + lam * proj ** 2).argmin(1)]

# global cell id = part*M + local id
arms = {
    "1part": [(0, parts[0][2][:, 0])],
    "soar": [(0, parts[0][2][:, 0]), (0, soar2(*parts[0]))],
    "2part": [(0, parts[0][2][:, 0]), (1, parts[1][2][:, 0])],
    "2part+soar": [(0, parts[0][2][:, 0]), (0, soar2(*parts[0])),
                   (1, parts[1][2][:, 0]), (1, soar2(*parts[1]))],
    "4part": [(p, parts[p][2][:, 0]) for p in range(4)],
}
targets = (0.90, 0.95, 0.98, 0.99)
out = {"N": N, "M": M}
for qn, Q in Qs.items():
    g = GT[qn]
    print(f"\n{qn}  N={N} M={M}  vectors scanned to reach R@10 (joint ranking over all attractors)")
    print(f"{'arm':11s} {'ent':>4s} " + " ".join(f"{t:>8.2f}" for t in targets) + "   P(both miss@64) vs indep")
    out[qn] = {}
    for name, homes in arms.items():
        used = sorted({p for p, _ in homes})
        Call = np.concatenate([parts[p][0] for p in used])
        off = {p: i * M for i, p in enumerate(used)}
        MM = len(Call)
        gh = [off[p] + a for p, a in homes]
        sizes = sum(np.bincount(h, minlength=MM) for h in gh).astype(np.int64)
        SQ = Q @ Call.T
        ordr = np.argsort(-SQ, axis=1)
        rank = np.empty_like(ordr)
        rank[np.arange(len(Q))[:, None], ordr] = np.arange(MM)[None, :]
        ri = np.arange(len(Q))[:, None]
        dep = np.min(np.stack([rank[ri, h[g]] for h in gh]), axis=0).ravel() + 1
        cum = np.cumsum(sizes[ordr], axis=1).mean(0)
        rec = np.cumsum(np.bincount(dep, minlength=MM + 2)[1:MM + 1]) / dep.size
        vals = []
        for t in targets:
            P = np.searchsorted(rec, t)
            vals.append(float(cum[P]) if P < MM else None)
        diag = ""
        if len(gh) >= 2:
            m1 = rank[ri, gh[0][g]].ravel() >= 64
            m2 = rank[ri, gh[1][g]].ravel() >= 64
            diag = f"   {np.mean(m1 & m2):.3f} vs {np.mean(m1) * np.mean(m2):.3f}  (marginals {np.mean(m1):.3f}, {np.mean(m2):.3f})"
        out[qn][name] = vals
        print(f"{name:11s} {len(homes):4d} " + " ".join(f"{(v or float('nan')):8.0f}" for v in vals) + diag, flush=True)
json.dump(out, open(os.path.join(HERE, f"multipart_N{N}.json"), "w"), indent=1)
print(f"\n[done] {time.time()-t0:.1f}s")
