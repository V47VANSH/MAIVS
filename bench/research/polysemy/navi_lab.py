"""navi_lab.py: leaf-level navigation for the attractor DAG.

Each point stores the cells that hold its KG nearest neighbours (point -> cell links; cell ids
survive deletes). Query:
  round 0   scan the top-P0 cells by centroid score
  round r   the L best points found so far vote for the cells their neighbours live in
            (weighted by rank); scan the best P_r unscanned cells
Compared with flat top-P at equal vectors scanned, across N.

Usage: python navi_lab.py <hotpot_5M.npy> <N> <questions.npy>
"""
import sys, time, json, os, numpy as np, faiss, torch

t0 = time.time()
CELL, T, K = 128, 16, 10
path, N, qpath = sys.argv[1], int(sys.argv[2]), sys.argv[3]
KG = int(os.environ.get("KG", 16))
NQ = 500
mm = np.load(path, mmap_mode="r")
X = np.ascontiguousarray(mm[:N], dtype=np.float32)
Qs = {"IND": np.ascontiguousarray(mm[-NQ:], dtype=np.float32),
      "OOD": np.load(qpath).astype(np.float32)[:NQ]}
d = X.shape[1]
def nrm(a):
    a /= np.sqrt(np.einsum("ij,ij->i", a, a))[:, None] + 1e-12
    return a
fi = faiss.IndexFlatIP(d); fi.add(X)
GT = {k: fi.search(v, K)[1] for k, v in Qs.items()}
del fi

# point kNN graph on GPU (build-time cost, exact here for the lab)
dev = "cuda" if torch.cuda.is_available() else "cpu"
Xt = torch.from_numpy(X).to(dev)
KNN = np.empty((N, KG), np.int64)
CH = max(128, int(2e8 // N))
for s in range(0, N, CH):
    sim = Xt[s:s + CH] @ Xt.T
    sim[torch.arange(sim.shape[0]), torch.arange(s, s + sim.shape[0])] = -9
    KNN[s:s + CH] = sim.topk(KG, dim=1).indices.cpu().numpy()
del Xt, sim
torch.cuda.empty_cache()
print(f"[knn] KG={KG} {time.time()-t0:.1f}s", flush=True)

M = N // CELL
km = faiss.Kmeans(d, M, niter=12, spherical=True, seed=0, verbose=False, max_points_per_centroid=512)
km.train(X)
C = nrm(km.centroids.astype(np.float32))
fc = faiss.IndexFlatIP(d); fc.add(C)
S_top, I_top = fc.search(X, T)
a1, s1 = I_top[:, 0], S_top[:, 0]
G = C @ C.T
r1n = np.sqrt(np.maximum(2-2 * s1, 1e-12))
cand, sc = I_top[:, 1:], S_top[:, 1:]
proj = (1 - s1[:, None] - sc + G[a1[:, None], cand]) / r1n[:, None]
a2 = cand[np.arange(N), ((2-2 * sc) + 4.0 * proj ** 2).argmin(1)]
print(f"[part] M={M} {time.time()-t0:.1f}s", flush=True)

def members_csr(home_lists):
    par = np.concatenate(home_lists); pt = np.concatenate([np.arange(N)] * len(home_lists))
    o = np.argsort(par, kind="stable")
    ptr = np.zeros(M + 1, np.int64); np.add.at(ptr, par + 1, 1)
    return np.cumsum(ptr), pt[o]

def run(Q, g, homes, schedule, L):
    ptr, mem = members_csr(homes)
    sizes = np.diff(ptr)
    NB = homes[0][KNN]                         # cells of each point's neighbours (primary)
    rec = 0.0; scan = 0.0
    for qi in range(len(Q)):
        q = Q[qi]
        cs = C @ q
        probed = np.zeros(M, bool)
        pts = []
        sel = np.argsort(-cs)[:schedule[0]]
        for r, Pr in enumerate(schedule):
            if r > 0:
                allp = np.unique(np.concatenate(pts))
                ps = X[allp] @ q
                best = allp[np.argsort(-ps)[:L]]
                w = np.repeat(1.0 / (1 + np.arange(len(best))), KG)
                votes = np.bincount(NB[best].ravel(), weights=w, minlength=M)
                votes[probed] = 0
                cands = np.nonzero(votes > 0)[0]
                key = votes[cands] + 1e-3 * cs[cands]     # votes first, attractor score tie-break
                sel = cands[np.argsort(-key)[:Pr]]
            probed[sel] = True
            for c in sel:
                pts.append(mem[ptr[c]:ptr[c + 1]])
        hit = np.zeros(K, bool)
        for a in homes:
            hit |= probed[a[g[qi]]]
        rec += hit.mean(); scan += sizes[probed].sum()
    return rec / len(Q), scan / len(Q)

def flat(Q, g, homes, P):
    ptr, _ = members_csr(homes); sizes = np.diff(ptr)
    rec = 0.0; scan = 0.0
    for qi in range(len(Q)):
        o = np.argsort(-(C @ Q[qi]))[:P]
        probed = np.zeros(M, bool); probed[o] = True
        hit = np.zeros(K, bool)
        for a in homes:
            hit |= probed[a[g[qi]]]
        rec += hit.mean(); scan += sizes[probed].sum()
    return rec / len(Q), scan / len(Q)

out = {"N": N, "M": M, "KG": KG, "rows": []}
for qn, Q in Qs.items():
    g = GT[qn]
    for hn, homes in (("leaf1", [a1]), ("leaf2", [a1, a2])):
        for P in (16, 32, 64, 128):
            r, s = flat(Q, g, homes, P)
            out["rows"].append([qn, hn, "flat", P, r, s])
            print(f"{qn} {hn} flat       P={P:4d}           R@10={r:.4f}  scanned={s:7.0f}", flush=True)
        SCH = ((16, 16, 16), (16, 16, 16, 16), (32, 32, 32)) if os.environ.get("FAST") else ((8, 8), (8, 8, 8), (16, 16), (16, 16, 16), (16, 16, 16, 16), (32, 32, 32))
        for sched in SCH:
            for L in ((30,) if os.environ.get("FAST") else (10, 30)):
                r, s = run(Q, g, homes, sched, L)
                out["rows"].append([qn, hn, "navi", list(sched), L, r, s])
                print(f"{qn} {hn} navi {str(sched):16s} L={L:2d}  R@10={r:.4f}  scanned={s:7.0f}", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"navi_N{N}_KG{KG}.json"), "w"), indent=1)
print(f"[done] {time.time()-t0:.1f}s")
