"""tree_lab.py: O(log N) test for the multi-parent attractor DAG.

Work per level is fixed (beam B x fanout f nodes scored, P leaf cells scanned), so if recall at
fixed (B, P) doesn't drop as N grows, cost at fixed recall is O(depth) = O(log N).

  leaves     cells of CELL vectors; each vector has R_leaf homes (orthogonal polysemy)
  internal   attractors built bottom-up with spherical k-means over the level below, fanout
             f; each child gets R_int parents chosen with the same orthogonal rule
  query      score all top nodes; keep the top B at each level and expand their children;
             at the cell level keep the top P and scan them

Usage: python tree_lab.py hotpot <hotpot_5M.npy> <N> <questions.npy>
"""
import sys, time, json, os, numpy as np, faiss

t0 = time.time()
CELL, T, K, F, TOPMAX = 128, 16, 10, 32, 64
path, N, qpath = sys.argv[2], int(sys.argv[3]), sys.argv[4]
NQ = 500
mm = np.load(path, mmap_mode="r")
X = np.ascontiguousarray(mm[:N], dtype=np.float32)
Qs = {"IND": np.ascontiguousarray(mm[-NQ:], dtype=np.float32),
      "OOD": np.load(qpath).astype(np.float32)[:NQ]}
d = X.shape[1]

def nrm(a):
    a /= np.sqrt(np.einsum("ij,ij->i", a, a))[:, None] + 1e-12
    return a

def exact(Q):
    fi = faiss.IndexFlatIP(d); fi.add(X)
    return fi.search(Q, K)[1]
GT = {k: exact(v) for k, v in Qs.items()}

def kmeans(Y, k, seed=0):
    km = faiss.Kmeans(d, k, niter=12, spherical=True, seed=seed, verbose=False,
                      max_points_per_centroid=512)
    km.train(Y)
    return nrm(km.centroids.astype(np.float32))

def homes(Y, U, R, lam):
    """R decorrelated homes in U for every row of Y (orthogonal polysemy)."""
    fi = faiss.IndexFlatIP(d); fi.add(U)
    S, I = fi.search(Y, min(T, len(U)))
    out = [I[:, 0]]
    if R == 1:
        return out
    n = len(Y); G = U @ U.T
    s1 = S[:, 0]; r1n = np.sqrt(np.maximum(2-2 * s1, 1e-12))
    cand, sc = I[:, 1:], S[:, 1:]
    p1 = (1 - s1[:, None] - sc + G[I[:, :1], cand]) / r1n[:, None]
    loss = (2-2 * sc) + lam * p1 ** 2
    out.append(cand[np.arange(n), loss.argmin(1)])
    return out

def csr(assign_lists, n_parent):
    par = np.concatenate([a[a >= 0] for a in assign_lists])
    ch = np.concatenate([np.nonzero(a >= 0)[0] for a in assign_lists])
    o = np.argsort(par, kind="stable")
    ptr = np.zeros(n_parent + 1, np.int64)
    np.add.at(ptr, par + 1, 1)
    return np.cumsum(ptr), ch[o]

M = N // CELL
C = kmeans(X, M)
leaf = {1: homes(X, C, 1, 0), 2: homes(X, C, 2, 4.0)}
print(f"[cells] N={N} M={M} {time.time()-t0:.1f}s", flush=True)

def build_tree(R_int):
    levels = []                          # coarsest first: (cent, ptr, idx)
    cur = C
    while len(cur) > TOPMAX:
        U = kmeans(cur, int(np.ceil(len(cur) / F)), seed=len(levels) + 1)
        ptr, idx = csr(homes(cur, U, R_int, 1.0), len(U))
        levels.insert(0, (U, ptr, idx))
        cur = U
    return levels
trees = {1: build_tree(1), 2: build_tree(2)}
print(f"[tree] depth={len(trees[1])} sizes={[len(l[0]) for l in trees[1]]} {time.time()-t0:.1f}s", flush=True)

def descend(q, levels, B):
    """Return candidate cells (ids) with scores, after beam descent."""
    if not levels:
        return np.arange(M), C @ q
    top = levels[0][0]
    front = np.argsort(-(top @ q))[:B]
    for li, (U, ptr, idx) in enumerate(levels):
        ch = np.unique(np.concatenate([idx[ptr[u]:ptr[u + 1]] for u in front]))
        below = levels[li + 1][0] if li + 1 < len(levels) else C
        s = below[ch] @ q
        if li + 1 < len(levels):
            front = ch[np.argsort(-s)[:B]]
        else:
            return ch, s
    return None

Bs, Ps = (4, 8, 16, 32), (16, 32, 64, 128, 256)
res = {"N": N, "M": M, "depth": len(trees[1]), "cfg": {}}
for qn, Q in Qs.items():
    g = GT[qn]
    for (Ri, Rl) in ((1, 1), (1, 2), (2, 2)):
        lh = leaf[Rl]
        sizes = sum(np.bincount(a, minlength=M) for a in lh)
        for B in Bs:
            rec = np.zeros(len(Ps)); scan = np.zeros(len(Ps)); route = 0
            for qi in range(len(Q)):
                q = Q[qi]
                ch, s = descend(q, trees[Ri], B)
                route += len(ch)
                o = ch[np.argsort(-s)]
                for pi, P in enumerate(Ps):
                    probed = np.zeros(M, bool); probed[o[:P]] = True
                    hit = np.zeros(K, bool)
                    for a in lh:
                        hit |= probed[a[g[qi]]]
                    rec[pi] += hit.mean(); scan[pi] += sizes[o[:P]].sum()
            key = f"{qn}|int{Ri}|leaf{Rl}|B{B}"
            res["cfg"][key] = {"route": route / len(Q), "P": list(Ps),
                               "recall": (rec / len(Q)).round(4).tolist(),
                               "scan": (scan / len(Q)).round(0).tolist()}
        # flat baseline (global ranking, no tree) for the same leaf arm
        rec = np.zeros(len(Ps)); scan = np.zeros(len(Ps))
        for qi in range(len(Q)):
            o = np.argsort(-(C @ Q[qi]))
            for pi, P in enumerate(Ps):
                probed = np.zeros(M, bool); probed[o[:P]] = True
                hit = np.zeros(K, bool)
                for a in lh:
                    hit |= probed[a[g[qi]]]
                rec[pi] += hit.mean(); scan[pi] += sizes[o[:P]].sum()
        res["cfg"][f"{qn}|flat|leaf{Rl}"] = {"route": M, "P": list(Ps),
            "recall": (rec / len(Q)).round(4).tolist(), "scan": (scan / len(Q)).round(0).tolist()}
    print(f"[{qn}] {time.time()-t0:.1f}s", flush=True)

json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"tree_N{N}.json"), "w"), indent=1)
# compact print: recall at P=64 and P=128 for each arm, B=8/16
for k, v in res["cfg"].items():
    if "|B4" in k or "|B32" in k:
        continue
    print(f"{k:28s} route={v['route']:7.0f}  " +
          "  ".join(f"P{P}:{r:.3f}/{s:.0f}" for P, r, s in zip(v["P"], v["recall"], v["scan"]) if P in (32, 64, 128)))
print(f"[done] {time.time()-t0:.1f}s")
