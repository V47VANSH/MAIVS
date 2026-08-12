"""navi2.py: Hopfield-weighted navigation over cell links.

Cached per N: exact point kNN graph (KG=32), partition, orthogonal second home. Query (budget = sum
of the schedule):
  round 0   top-P0 cells by attractor score
  round r   the L best points found so far (exact q.x) add activation to the cells holding
            their KG nearest neighbours: act(cell) += softmax_beta(q.x_voter)
            (beta=0 means rank weights); scan the P_r most activated unscanned cells
Reports R@10 and vectors scanned, plus the flat curve, on real questions.

Usage: python navi2.py <hotpot_5M.npy> <N> <questions.npy>
"""
import sys, time, json, os, numpy as np, faiss

t0 = time.time()
CELL, T, K, KGMAX = 128, 16, 10, 32
path, N, qpath = sys.argv[1], int(sys.argv[2]), sys.argv[3]
HERE = os.path.dirname(os.path.abspath(__file__))
NQ = int(os.environ.get("NQ", 500))
mm = np.load(path, mmap_mode="r")
X = np.ascontiguousarray(mm[:N], dtype=np.float32)
Q = np.load(qpath).astype(np.float32)[:NQ]
if os.environ.get("IND"):
    Q = np.ascontiguousarray(mm[-NQ:], dtype=np.float32)
d = X.shape[1]
def nrm(a):
    a /= np.sqrt(np.einsum("ij,ij->i", a, a))[:, None] + 1e-12
    return a

cache = os.path.join(HERE, f"cache_hotpot_N{N}.npz")
if os.path.exists(cache):
    z = np.load(cache); KNN, C, a1, a2 = z["KNN"], z["C"], z["a1"], z["a2"]
else:
    import torch
    Xt = torch.from_numpy(X).cuda()
    KNN = np.empty((N, KGMAX), np.int32)
    CH = max(128, int(2e8 // N))
    for s in range(0, N, CH):
        sim = Xt[s:s + CH] @ Xt.T
        sim[torch.arange(sim.shape[0]), torch.arange(s, s + sim.shape[0])] = -9
        KNN[s:s + CH] = sim.topk(KGMAX, dim=1).indices.cpu().numpy()
    del Xt, sim; torch.cuda.empty_cache()
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
    np.savez(cache, KNN=KNN, C=C, a1=a1, a2=a2)
M = len(C)
fi = faiss.IndexFlatIP(d); fi.add(X)
GT = fi.search(Q, K)[1]; del fi
print(f"[prep] N={N} M={M} {time.time()-t0:.1f}s", flush=True)

def csr(homes):
    par = np.concatenate(homes); pt = np.concatenate([np.arange(N)] * len(homes))
    o = np.argsort(par, kind="stable")
    ptr = np.zeros(M + 1, np.int64); np.add.at(ptr, par + 1, 1)
    return np.cumsum(ptr), pt[o]

CS = Q @ C.T                                   # attractor scores, all queries
ORD = np.argsort(-CS, axis=1)

def evaluate(homes, sched, L, KG, beta):
    ptr, mem = csr(homes); sizes = np.diff(ptr)
    NB = homes[0][KNN[:, :KG]].astype(np.int64)   # cells of each point's KG neighbours
    rec = scan = 0.0
    for qi in range(len(Q)):
        q = Q[qi]; probed = np.zeros(M, bool)
        sel = ORD[qi, :sched[0]]
        seen = []
        for r, Pr in enumerate(sched):
            if r > 0:
                allp = np.unique(np.concatenate(seen))
                ps = X[allp] @ q
                top = np.argsort(-ps)[:L]
                best, bs = allp[top], ps[top]
                w = np.exp(beta * (bs - bs[0])) if beta > 0 else 1.0 / (1 + np.arange(len(best)))
                act = np.bincount(NB[best].ravel(), weights=np.repeat(w, KG), minlength=M)
                act[probed] = 0
                c = np.nonzero(act > 0)[0]
                sel = c[np.argsort(-(act[c] + 1e-4 * CS[qi, c]))[:Pr]]
                if len(sel) < Pr:                      # fall back to attractor order
                    rest = ORD[qi][~probed[ORD[qi]]]
                    rest = rest[~np.isin(rest, sel)][:Pr - len(sel)]
                    sel = np.concatenate([sel, rest])
            probed[sel] = True
            seen.extend(mem[ptr[c0]:ptr[c0 + 1]] for c0 in sel)
        hit = np.zeros(K, bool)
        for a in homes:
            hit |= probed[a[GT[qi]]]
        rec += hit.mean(); scan += sizes[probed].sum()
    return rec / len(Q), scan / len(Q)

def flat_curve(homes):
    ptr, _ = csr(homes); sizes = np.diff(ptr)
    out = []
    for P in (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512):
        rec = scan = 0.0
        for qi in range(len(Q)):
            pr = np.zeros(M, bool); pr[ORD[qi, :P]] = True
            hit = np.zeros(K, bool)
            for a in homes:
                hit |= pr[a[GT[qi]]]
            rec += hit.mean(); scan += sizes[pr].sum()
        out.append((P, rec / len(Q), scan / len(Q)))
    return out

def flat_cost_at(curve, r):
    """Vectors flat ranking needs to reach recall r (log-interpolated)."""
    for (p0, r0, s0), (p1, r1, s1) in zip(curve, curve[1:]):
        if r0 <= r <= r1:
            t = (r - r0) / max(r1 - r0, 1e-9)
            return float(np.exp(np.log(s0) + t * (np.log(s1) - np.log(s0))))
    return float("nan")

res = {"N": N, "M": M, "ood": not os.environ.get("IND"), "rows": []}
CONFIGS = [((32, 32, 32), 30, 16, 0), ((32, 32, 32), 100, 32, 0), ((32, 32, 32), 100, 32, 30),
           ((16, 16, 16, 16, 16), 100, 32, 30), ((32, 32, 32, 32), 100, 32, 30),
           ((24, 24, 24, 24, 24, 24), 100, 32, 30)]
for hn, homes in (("leaf1", [a1]), ("leaf2", [a1, a2])):
    curve = flat_curve(homes)
    res["rows"].append({"homes": hn, "flat": curve})
    print(f"-- {hn} flat: " + "  ".join(f"P{p}:{r:.3f}/{s:.0f}" for p, r, s in curve if p in (32, 64, 128, 256)), flush=True)
    for sched, L, KG, beta in CONFIGS:
        r, s = evaluate(homes, sched, L, KG, beta)
        fc_ = flat_cost_at(curve, r)
        res["rows"].append({"homes": hn, "sched": sched, "L": L, "KG": KG, "beta": beta,
                            "recall": r, "scanned": s, "flat_equiv": fc_})
        print(f"{hn} navi {str(sched):26s} L={L:3d} KG={KG} beta={beta:2d}  R@10={r:.4f}  "
              f"scanned={s:7.0f}  flat-needs={fc_:7.0f}  gain={fc_/s:4.2f}x", flush=True)
tag = "IND" if os.environ.get("IND") else "OOD"
json.dump(res, open(os.path.join(HERE, f"navi2_{tag}_N{N}.json"), "w"), indent=1)
print(f"[done] {time.time()-t0:.1f}s")
