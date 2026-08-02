"""summariser_lab: offline bench for cell summaries.

Routing is the last part of query cost that still grows with N, and it's running ~75x above its
floor: at 999k / 7,804 cells the true top-10 sit in 4.01 cells, but ranking by centroid cosine needs
~60 probes for R@10 0.95 and ~404 for 0.99.

The mismatch: routing wants

    s*(C) = max over x in C of q.x      (could anything in this cell match?)

and a centroid gives

    q.cbar                              (what is this cell like on average?)

So every summary idea is just a different estimate of s*, and comparing them is a ranking problem
over one (queries x cells) matrix. That doesn't need the Rust engine. This computes s* once, caches
it, and scores a candidate in seconds instead of a rebuild.

Metric: routing recall@k, the fraction of a query's true top-k that sit in the nprobe cells chosen.
It's the ceiling for the scan and separates routing loss from scan/rerank loss; end-to-end R@10
lands slightly below it. Both nprobe and vectors scanned are reported, since cells differ in size
and the cost model is in bytes.

Usage:
    python bench/summariser_lab.py msmarco_embeddings_cache_v2_1000000.npy --queries 1000 --cell-size 128 --variants mean,cone,gamma,box

Cached results are reused; --rebuild retrains the partition.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groundtruth as GT  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover
    print("this bench needs torch (GPU strongly preferred)")
    raise

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_lab_cache")

# Use exact precision wherever a ranking is produced: TF32 has ~1e-3 relative error, enough to
# reorder near-tied cells. It's only enabled for k-means, where the error washes out.
torch.backends.cuda.matmul.allow_tf32 = False


def _dev() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _log(msg: str) -> None:
    print(f"[lab] {msg}", flush=True)


# --- partition: spherical k-means, streamed so a 4 GB GPU handles 1M x 384 ---

def spherical_kmeans(
    X: np.ndarray,
    n_cells: int,
    iters: int = 20,
    seed: int = 0,
    chunk: int = 8192,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm with inner product on unit vectors.

    The base stays in host memory and is streamed in chunks, so device memory is chunk x n_cells for
    the score block instead of n x n_cells (31 GB at 1M x 7,804).

    Returns (centroids [n_cells, d] unit-norm, assign [n] int32).
    """
    dev = _dev()
    n, d = X.shape
    rng = np.random.default_rng(seed)

    # Random points as init. k-means++ over 7,804 centres costs k full passes and doesn't buy much
    # once Lloyd's has converged on the whole corpus (faiss trains on a subsample; this doesn't).
    init = rng.choice(n, size=n_cells, replace=False)
    C = torch.from_numpy(np.ascontiguousarray(X[np.sort(init)])).to(dev)
    C = torch.nn.functional.normalize(C, dim=1)

    Xt = torch.from_numpy(X)  # host, no copy
    assign = torch.empty(n, dtype=torch.int32, device=dev)

    prev_obj = -1.0
    tf32_was = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        for it in range(iters):
            t0 = time.time()
            acc = torch.zeros(n_cells, d, device=dev)
            cnt = torch.zeros(n_cells, device=dev)
            obj = 0.0
            for s in range(0, n, chunk):
                xb = Xt[s:s + chunk].to(dev, non_blocking=True)
                sc = xb @ C.T                      # [b, n_cells]
                best, idx = sc.max(dim=1)
                assign[s:s + chunk] = idx.to(torch.int32)
                acc.index_add_(0, idx, xb)
                cnt.index_add_(0, idx, torch.ones_like(best))
                obj += float(best.sum())
                del sc, xb
            obj /= n

            empty = (cnt == 0)
            n_empty = int(empty.sum())
            C = torch.nn.functional.normalize(acc, dim=1)
            if n_empty:
                # Re-seed empty cells from random points instead of dropping them; the number of
                # cells sets the scaling behaviour, so the partition has to keep the requested size.
                pick = rng.choice(n, size=n_empty, replace=False)
                C[empty] = torch.nn.functional.normalize(
                    torch.from_numpy(np.ascontiguousarray(X[np.sort(pick)])).to(dev), dim=1)
            if verbose:
                _log(f"kmeans it {it:2d}  obj={obj:.5f}  empty={n_empty}  "
                     f"{time.time() - t0:.1f}s")
            if prev_obj > 0 and obj - prev_obj < 1e-5:
                break
            prev_obj = obj
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32_was

    return C.cpu().numpy(), assign.cpu().numpy()


# --- CSR view of the partition ---

class Partition:
    """Cells as CSR: order[start[c]:start[c+1]] are cell c's rows.

    assign can be [n] (one cell per vector) or [n, r] (replication, each vector in r cells).
    Replication lowers the floor itself, since a neighbour on a boundary becomes reachable from more
    than one cell, at the cost of r-fold occupancy. Whether that's worth it is a bytes question, so
    it's measured on the same axis as everything else.
    """

    def __init__(self, cents: np.ndarray, assign: np.ndarray, n_cells: int):
        self.cents = cents
        self.assign = assign                       # [n] or [n, r]
        self.reps = 1 if assign.ndim == 1 else assign.shape[1]
        self.n_cells = n_cells
        flat = assign.reshape(-1).astype(np.int64)
        rows = np.repeat(np.arange(assign.shape[0], dtype=np.int64), self.reps)
        srt = np.argsort(flat, kind="stable")
        self.order = rows[srt]                     # base row per CSR position
        counts = np.bincount(flat, minlength=n_cells).astype(np.int64)
        self.start = np.zeros(n_cells + 1, dtype=np.int64)
        np.cumsum(counts, out=self.start[1:])
        self.sizes = counts
        # flat per-CSR-position cell id, for the oracle and sub-clustering
        self.flat_cell = flat[srt]

    def members(self, c: int) -> np.ndarray:
        return self.order[self.start[c]:self.start[c + 1]]


# --- oracle: s*(q, C) = max over members, exact ---

def oracle_max(X: np.ndarray, Q: np.ndarray, part: Partition,
               chunk: int = 8192, verbose: bool = True) -> np.ndarray:
    """Exact max over members of q.x for every (query, cell), [n_queries, n_cells].

    This is what every summary is trying to approximate, and ranking by it gives the probe floor.
    """
    dev = _dev()
    nq = Q.shape[0]
    Qd = torch.from_numpy(Q).to(dev)
    Xt = torch.from_numpy(X)
    asg = torch.from_numpy(part.assign.astype(np.int64)).to(dev)
    out = torch.full((nq, part.n_cells), -2.0, device=dev)
    t0 = time.time()
    for s in range(0, X.shape[0], chunk):
        xb = Xt[s:s + chunk].to(dev, non_blocking=True)
        sc = Qd @ xb.T                                  # [nq, b]
        idx = asg[s:s + chunk].unsqueeze(0).expand(nq, -1)
        out.scatter_reduce_(1, idx, sc, reduce="amax")
        del sc, xb
    if verbose:
        _log(f"oracle s* computed in {time.time() - t0:.1f}s")
    return out.cpu().numpy()


# --- CSR view of the partition ---

def routing_curve(S: np.ndarray, gt_cells: np.ndarray, sizes: np.ndarray,
                  probes: List[int]) -> Dict[int, Tuple[float, float]]:
    """For each nprobe: (routing recall@k, mean vectors scanned).

    S is the candidate's cell scores [nq, M]; gt_cells is the cell holding each true neighbour [nq,
    k]. A neighbour counts as reachable if its cell is in the top nprobe by S.
    """
    dev = _dev()
    Sd = torch.from_numpy(S).to(dev)
    gc = torch.from_numpy(gt_cells.astype(np.int64)).to(dev)
    sz = torch.from_numpy(sizes.astype(np.float64)).to(dev)
    nq, M = S.shape
    out: Dict[int, Tuple[float, float]] = {}
    # topk, not argsort: at 1M with m=32 a full sort of [nq, M*m] would
    # materialise a 1 GB index tensor and OOM the card.
    pmax = min(max(probes), M)
    order = Sd.topk(pmax, dim=1).indices
    # rank[q, cell] = position of `cell` in q's probe order (M if unprobed)
    rank = torch.full((nq, M), M, dtype=torch.int32, device=dev)
    rank.scatter_(1, order, torch.arange(pmax, dtype=torch.int32,
                                         device=dev).expand(nq, -1))
    gt_rank = torch.gather(rank, 1, gc)                 # [nq, k]
    scanned = torch.cumsum(torch.gather(sz.expand(nq, -1), 1, order), dim=1)
    for p in probes:
        if p > M:
            continue
        out[p] = (float((gt_rank < p).float().mean()),
                  float(scanned[:, p - 1].mean()))
    return out


def effective_eps(S: np.ndarray, oracle: np.ndarray, top: int = 384) -> float:
    """The candidate's error in the same units as the noise table.

    A summary doesn't need to reproduce s* on any particular scale, only order cells the same way,
    so raw differences mean nothing. What counts is the residual after the best affine fit,
    restricted to cells that are actually in contention (the 5,000th-ranked cell doesn't need an
    accurate score). That residual is comparable to v_noise's eps, so "this candidate needs N
    probes" becomes "this candidate is accurate to eps, and eps costs N probes".
    """
    dev = _dev()
    Sd = torch.from_numpy(S).to(dev)
    Od = torch.from_numpy(oracle).to(dev)
    idx = Od.topk(min(top, S.shape[1]), dim=1).indices
    x = torch.gather(Sd, 1, idx).reshape(-1).double()
    y = torch.gather(Od, 1, idx).reshape(-1).double()
    xm, ym = x.mean(), y.mean()
    a = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum().clamp(min=1e-12)
    return float(((y - ym) - a * (x - xm)).pow(2).mean().sqrt())


def probes_for(curve: Dict[int, Tuple[float, float]], target: float
               ) -> Optional[Tuple[int, float]]:
    """Smallest evaluated nprobe reaching `target` recall, and its scan size."""
    for p in sorted(curve):
        if curve[p][0] >= target:
            return p, curve[p][1]
    return None


# --- cost in bytes, which is what predicts throughput ---
#
# Throughput follows bytes moved per query, so probes are the wrong unit for comparing summaries: a
# richer summary saves probes by spending routing bytes, and only the total says if that was worth
# it. Every candidate is scored on
#
#   stage 1   M * 64 B                     Hamming over all centroid codes
#   stage 2   nprobe * mult * S            re-order the shortlist, S = summary size
#   scan      scanned_vectors * 64 B       binary codes in the probed cells
#   rerank    k * rf * rerank_bytes
#
# With sub-summaries stored as 64 B binary codes, S = m*64, so m=1 is the current router and
# m=cell_size is the full binary scan. We've only ever measured the two ends of that range.

BYTES_PER_CODE = 64
RERANK_BYTES = 1536


def query_bytes(curve: Dict[int, Tuple[float, float]], target: float,
                n_cells: int, summary_bytes: int, mult: int = 2,
                k: int = 10, rf: int = 32) -> Optional[Tuple[int, float]]:
    """Total bytes moved for the cheapest nprobe that reaches `target`."""
    got = probes_for(curve, target)
    if got is None:
        return None
    p, scanned = got
    total = (n_cells * BYTES_PER_CODE
             + p * mult * summary_bytes
             + scanned * BYTES_PER_CODE
             + k * rf * RERANK_BYTES)
    return p, total


# --- candidate summarisers --- Each takes the shared context and returns cell scores [nq, M].
# They're all estimates of s* and differ in what they assume about how a cell's members spread
# around the centroid.

class Ctx:
    def __init__(self, X, Q, part: Partition, oracle: np.ndarray):
        self.X, self.Q, self.part, self.oracle = X, Q, part, oracle
        self.dev = _dev()
        self.Qd = torch.from_numpy(Q).to(self.dev)
        self.Cd = torch.from_numpy(part.cents).to(self.dev)
        self._cos = None
        self._radii = None

    @property
    def cos(self) -> torch.Tensor:
        """q . cbar for every (query, cell), i.e. the current router's score."""
        if self._cos is None:
            self._cos = self.Qd @ self.Cd.T
        return self._cos

    def radii(self) -> Dict[str, torch.Tensor]:
        """Per-cell angular radius statistics of the members about cbar."""
        if self._radii is not None:
            return self._radii
        X, part = self.X, self.part
        dev = self.dev
        alpha = np.empty(X.shape[0], dtype=np.float32)
        Xt = torch.from_numpy(X)
        asg = torch.from_numpy(part.assign.astype(np.int64))
        for s in range(0, X.shape[0], 65536):
            xb = Xt[s:s + 65536].to(dev)
            cb = self.Cd[asg[s:s + 65536].to(dev)]
            alpha[s:s + 65536] = (xb * cb).sum(1).cpu().numpy()
        ang = np.arccos(np.clip(alpha, -1.0, 1.0))
        srt = ang[part.order]
        r99 = np.zeros(part.n_cells, dtype=np.float32)
        rmx = np.zeros(part.n_cells, dtype=np.float32)
        rmean = np.zeros(part.n_cells, dtype=np.float32)
        for c in range(part.n_cells):
            a = np.sort(srt[part.start[c]:part.start[c + 1]])
            if a.size == 0:
                continue
            rmx[c] = a[-1]
            rmean[c] = a.mean()
            r99[c] = a[min(int(np.ceil(a.size * 0.99)) - 1, a.size - 1)]
        self._radii = {k: torch.from_numpy(v).to(dev)
                       for k, v in (("p99", r99), ("max", rmx), ("mean", rmean))}
        return self._radii


def v_mean(ctx: Ctx, **kw) -> np.ndarray:
    """The shipped router: rank by centroid cosine."""
    return ctx.cos.cpu().numpy()


def v_oracle(ctx: Ctx, **kw) -> np.ndarray:
    """The floor.  Perfect knowledge of s*."""
    return ctx.oracle


def v_cone(ctx: Ctx, radius: str = "p99", **kw) -> np.ndarray:
    """Spherical bound: cos(max(0, theta - r)).

    An upper bound on s* for members inside the cone, but the slack grows with r, so wide cells look
    reachable whatever q is. Same bound the index uses for early termination; it lost to plain
    cosine as a ranker before.
    """
    r = ctx.radii()[radius]
    th = torch.arccos(ctx.cos.clamp(-1, 1))
    return torch.cos((th - r.unsqueeze(0)).clamp(min=0)).cpu().numpy()


def v_gamma(ctx: Ctx, gamma: float = 0.5, radius: str = "p99", **kw) -> np.ndarray:
    """cos(theta - gamma*r): interpolates between the centroid (gamma=0) and the full cone bound
    (gamma=1).

    The cheapest summary there is, 4 bytes per cell. Tests whether the routing loss is explained by
    how spread out a cell is, or whether we need the direction of the spread. If sweeping gamma
    recovers most of the headroom, nothing richer is needed.
    """
    r = ctx.radii()[radius]
    th = torch.arccos(ctx.cos.clamp(-1, 1))
    return torch.cos((th - gamma * r.unsqueeze(0)).clamp(min=0)).cpu().numpy()


def v_box(ctx: Ctx, **kw) -> np.ndarray:
    """Per-dimension min/max box bound: sum over j of max(q_j*lo_j, q_j*hi_j).

    A real upper bound and cheap, but a box around 128 points in 384 dimensions covers far more than
    their hull, so it should be loose. A loose bound isn't necessarily a bad ranker though, so it's
    worth checking.
    """
    dev, part, X = ctx.dev, ctx.part, ctx.X
    d = X.shape[1]
    lo = np.full((part.n_cells, d), np.inf, dtype=np.float32)
    hi = np.full((part.n_cells, d), -np.inf, dtype=np.float32)
    Xs = X[part.order]
    for c in range(part.n_cells):
        blk = Xs[part.start[c]:part.start[c + 1]]
        if blk.size:
            lo[c] = blk.min(0)
            hi[c] = blk.max(0)
    lo = np.where(np.isfinite(lo), lo, 0.0)
    hi = np.where(np.isfinite(hi), hi, 0.0)
    lod = torch.from_numpy(lo).to(dev)
    hid = torch.from_numpy(hi).to(dev)
    qp = ctx.Qd.clamp(min=0)
    qn = ctx.Qd.clamp(max=0)
    return (qp @ hid.T + qn @ lod.T).cpu().numpy()


# --- directional summaries --- The scalar-radius ones (cone, gamma) say a cell is reachable if it's
# wide, which doesn't depend on the query, so they can't really re-rank cells. These spend their
# bytes on where the cell's mass is instead.

def _sub_kmeans(ctx: Ctx, m: int, iters: int = 12, chunk: int = 65536,
                seed: int = 0) -> torch.Tensor:
    """m sub-centroids per cell, by Lloyd's restricted to each cell.

    Done over the CSR-sorted arena with scatter ops instead of a per-cell loop: 7,804 cells x 12
    iterations of tiny GPU kernels is all launch overhead, while this is m streaming passes per
    iteration.

    Returns (unit-norm [n_cells*m, d], raw mean [n_cells*m, d]); sub-centroid j of cell c is row
    c*m+j. Both are returned because it's not obvious which to score with: normalising divides by
    ||mean||, which inflates the loose sub-clusters, while the raw mean scores a sub-cluster by its
    members' average similarity.
    """
    dev, part, X = ctx.dev, ctx.part, ctx.X
    n, d = X.shape
    M = part.n_cells
    Xs = np.ascontiguousarray(X[part.order])
    cell_of = np.repeat(np.arange(M, dtype=np.int64), np.diff(part.start))
    rng = np.random.default_rng(seed)
    # Init: m distinct members per cell (wrapping when a cell is smaller).
    off = np.stack([rng.integers(0, np.maximum(part.sizes, 1)) for _ in range(m)], 1)
    init = (part.start[:M, None] + off % np.maximum(part.sizes, 1)[:, None]).ravel()
    S = torch.from_numpy(np.ascontiguousarray(Xs[np.clip(init, 0, n - 1)])).to(dev)
    S = torch.nn.functional.normalize(S, dim=1)

    Xt = torch.from_numpy(Xs)
    cod = torch.from_numpy(cell_of).to(dev)
    for _ in range(iters):
        acc = torch.zeros(M * m, d, device=dev)
        cnt = torch.zeros(M * m, device=dev)
        for s in range(0, n, chunk):
            xb = Xt[s:s + chunk].to(dev, non_blocking=True)
            cb = cod[s:s + chunk]
            sc = torch.empty(xb.shape[0], m, device=dev)
            for j in range(m):
                sc[:, j] = (xb * S[cb * m + j]).sum(1)
            tgt = cb * m + sc.argmax(1)
            acc.index_add_(0, tgt, xb)
            cnt.index_add_(0, tgt, torch.ones(xb.shape[0], device=dev))
            del xb, sc
        nz = acc.norm(dim=1) > 1e-8
        newS = torch.nn.functional.normalize(acc, dim=1)
        S = torch.where(nz.unsqueeze(1), newS, S)   # keep dead sub-centroids put
    raw = acc / cnt.clamp(min=1).unsqueeze(1)
    raw = torch.where(nz.unsqueeze(1), raw, S)

    # Per-group scale and isotropic residual variance for the order-statistic correction in
    # v_subvar. nu = ||raw mean||, so q.raw = nu * (q.unit).
    nu = raw.norm(dim=1)
    tau2 = torch.zeros(M * m, device=dev)
    for s in range(0, n, chunk):
        xb = Xt[s:s + chunk].to(dev, non_blocking=True)
        cb = cod[s:s + chunk]
        sc = torch.empty(xb.shape[0], m, device=dev)
        for j in range(m):
            sc[:, j] = (xb * S[cb * m + j]).sum(1)
        tgt = cb * m + sc.argmax(1)
        p = (xb * S[tgt]).sum(1)
        tau2.index_add_(0, tgt, (1.0 - p * p).clamp(min=0))
        del xb, sc
    tau2 = tau2 / cnt.clamp(min=1) / max(d - 1, 1)

    # Sub-group of every point in original row order. Needed to score the hierarchical path, where
    # the scan unit is a sub-group rather than a cell.
    sub = torch.empty(n, dtype=torch.int32, device=dev)
    for s in range(0, n, chunk):
        xb = Xt[s:s + chunk].to(dev, non_blocking=True)
        cb = cod[s:s + chunk]
        sc = torch.empty(xb.shape[0], m, device=dev)
        for j in range(m):
            sc[:, j] = (xb * S[cb * m + j]).sum(1)
        sub[s:s + chunk] = (cb * m + sc.argmax(1)).to(torch.int32)
        del xb, sc
    sub_orig = np.empty(n, dtype=np.int32)
    sub_orig[part.order] = sub.cpu().numpy()
    return S, raw, nu, tau2, sub_orig


def v_subc(ctx: Ctx, m: int = 4, norm: bool = True, **kw) -> np.ndarray:
    """Rank by max over j of q . subcentroid_j.

    The router picks at cell granularity (128 vectors) while the neighbours sit in a ~4-cell,
    ~700-vector neighbourhood. Sub-centroids decouple the two: stage 1 still scans M coarse codes,
    but the probe order is decided at M*m resolution. Splitting the cells themselves would give the
    same resolution but multiply the stage-1 scan by m.

    It's also a lower bound on s* (each sub-centroid is a real direction in the cell), whereas the
    variants above are upper bounds, and those didn't work.
    """
    key = f"_sub{m}"
    got = getattr(ctx, key, None)
    if got is None:
        got = _sub_kmeans(ctx, m, iters=kw.get("iters", 12))
        setattr(ctx, key, got)
    S = got[0] if norm else got[1]
    nq, M = ctx.Qd.shape[0], ctx.part.n_cells
    out = torch.empty(nq, M, device=ctx.dev)
    # Chunk over sub-centroids to keep the [nq, M*m] block off the card.
    for s in range(0, M, 4096):
        e = min(s + 4096, M)
        blk = ctx.Qd @ S[s * m:e * m].T          # [nq, (e-s)*m]
        out[:, s:e] = blk.view(nq, e - s, m).max(dim=2).values
        del blk
    return out.cpu().numpy()


def _residual_pcs(ctx: Ctx, m: int, iters: int = 8, chunk: int = 65536
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top m principal directions of each cell's residual around its centroid, by batched power
    iteration with deflation, plus the extent along each direction and the leftover residual norm.

    Batched like _sub_kmeans: an SVD per cell over 7,804 cells would be launch-bound, while a power
    iteration is t_c = sum_i (r_i.u_c) r_i, one scatter-add per pass.

    Returns (U [M, m, d] roughly orthonormal, rho [M, m], rest [M]).
    """
    dev, part, X = ctx.dev, ctx.part, ctx.X
    n, d = X.shape
    M = part.n_cells
    Xs = np.ascontiguousarray(X[part.order])
    cell_of = torch.from_numpy(
        np.repeat(np.arange(M, dtype=np.int64), np.diff(part.start))).to(dev)
    Xt = torch.from_numpy(Xs)
    C = ctx.Cd

    U = torch.zeros(M, m, d, device=dev)
    rho = torch.zeros(M, m, device=dev)
    g = torch.Generator(device=dev).manual_seed(0)

    for j in range(m):
        u = torch.nn.functional.normalize(
            torch.randn(M, d, device=dev, generator=g), dim=1)
        for _ in range(iters):
            acc = torch.zeros(M, d, device=dev)
            for s in range(0, n, chunk):
                xb = Xt[s:s + chunk].to(dev, non_blocking=True)
                cb = cell_of[s:s + chunk]
                r = xb - (xb * C[cb]).sum(1, keepdim=True) * C[cb]
                for jj in range(j):        # deflate directions already taken
                    r = r - (r * U[cb, jj]).sum(1, keepdim=True) * U[cb, jj]
                acc.index_add_(0, cb, (r * u[cb]).sum(1, keepdim=True) * r)
                del xb, r
            nz = acc.norm(dim=1) > 1e-12
            u = torch.where(nz.unsqueeze(1),
                            torch.nn.functional.normalize(acc, dim=1), u)
        U[:, j] = u

    # extents along each kept direction, and the norm left outside the basis
    rest = torch.zeros(M, device=dev)
    for s in range(0, n, chunk):
        xb = Xt[s:s + chunk].to(dev, non_blocking=True)
        cb = cell_of[s:s + chunk]
        r = xb - (xb * C[cb]).sum(1, keepdim=True) * C[cb]
        for j in range(m):
            p = (r * U[cb, j]).sum(1)
            rho[:, j].scatter_reduce_(0, cb, p.abs(), reduce="amax")
            r = r - p.unsqueeze(1) * U[cb, j]
        rest.scatter_reduce_(0, cb, r.norm(dim=1), reduce="amax")
        del xb, r
    return U, rho, rest


def v_rank(ctx: Ctx, m: int = 2, **kw) -> np.ndarray:
    """Anisotropic cone: cos(theta) + sum_j |q.u_j| rho_j + ||q_perp|| * rest.

    The isotropic cone assumes a residual could point anywhere and charges the full radius. In
    practice 128 residuals in 384 dimensions occupy a few directions, so the reachable set is a thin
    ellipsoid, not a ball. This bounds each kept direction by its measured extent and only treats
    the rest isotropically.
    """
    key = f"_pc{m}"
    got = getattr(ctx, key, None)
    if got is None:
        got = _residual_pcs(ctx, m)
        setattr(ctx, key, got)
    U, rho, rest = got
    nq = ctx.Qd.shape[0]
    M = ctx.part.n_cells
    out = ctx.cos.clone()
    used = ctx.cos ** 2
    for j in range(m):
        p = ctx.Qd @ U[:, j].T                      # [nq, M]
        out = out + p.abs() * rho[:, j].unsqueeze(0)
        used = used + p ** 2
    qperp = (1.0 - used).clamp(min=0).sqrt()
    out = out + qperp * rest.unsqueeze(0)
    return out.cpu().numpy()


def _residual_moments(ctx: Ctx, m: int, iters: int = 8, chunk: int = 65536):
    """Per-cell residual covariance as low rank plus an isotropic remainder:

        Sigma_C ~= sum_j lam_j u_j u_j^T + tau2_C * I_rest

    Same batched power iteration as _residual_pcs, but keeping the variance along each direction
    instead of the extreme extent (see v_var for why).

    Returns (U [M,m,d], lam [M,m], tau2 [M]).
    """
    dev, part, X = ctx.dev, ctx.part, ctx.X
    n, d = X.shape
    M = part.n_cells
    Xs = np.ascontiguousarray(X[part.order])
    cell_of = torch.from_numpy(
        np.repeat(np.arange(M, dtype=np.int64), np.diff(part.start))).to(dev)
    Xt = torch.from_numpy(Xs)
    C = ctx.Cd
    cnt = torch.from_numpy(part.sizes.astype(np.float32)).to(dev).clamp(min=1)

    U = torch.zeros(M, max(m, 1), d, device=dev)
    lam = torch.zeros(M, max(m, 1), device=dev)
    g = torch.Generator(device=dev).manual_seed(0)

    for j in range(m):
        u = torch.nn.functional.normalize(
            torch.randn(M, d, device=dev, generator=g), dim=1)
        for _ in range(iters):
            acc = torch.zeros(M, d, device=dev)
            for s in range(0, n, chunk):
                xb = Xt[s:s + chunk].to(dev, non_blocking=True)
                cb = cell_of[s:s + chunk]
                r = xb - (xb * C[cb]).sum(1, keepdim=True) * C[cb]
                for jj in range(j):
                    r = r - (r * U[cb, jj]).sum(1, keepdim=True) * U[cb, jj]
                acc.index_add_(0, cb, (r * u[cb]).sum(1, keepdim=True) * r)
                del xb, r
            nz = acc.norm(dim=1) > 1e-12
            u = torch.where(nz.unsqueeze(1),
                            torch.nn.functional.normalize(acc, dim=1), u)
        U[:, j] = u

    # variance along each kept direction, and mean leftover variance per dim
    tot = torch.zeros(M, device=dev)
    for s in range(0, n, chunk):
        xb = Xt[s:s + chunk].to(dev, non_blocking=True)
        cb = cell_of[s:s + chunk]
        r = xb - (xb * C[cb]).sum(1, keepdim=True) * C[cb]
        tot.index_add_(0, cb, (r * r).sum(1))
        for j in range(m):
            p = (r * U[cb, j]).sum(1)
            lam[:, j].index_add_(0, cb, p * p)
            r = r - p.unsqueeze(1) * U[cb, j]
        del xb, r
    lam = lam / cnt.unsqueeze(1)
    tot = tot / cnt
    rest_var = (tot - lam[:, :m].sum(1)).clamp(min=0)
    tau2 = rest_var / max(d - 1 - m, 1)
    return U, lam, tau2


def v_var(ctx: Ctx, m: int = 4, beta: float = 3.0, **kw) -> np.ndarray:
    """Order-statistic estimate: s* ~= mu + beta * sigma_C(q).

    Over a cell's members the scores q.x_i have mean ~q.cbar and standard deviation sigma_C(q) =
    sqrt(q^T Sigma_C q). The max of n_c such scores sits about beta deviations above the mean, so
    what routing wants is q.cbar + beta*sqrt(q^T Sigma_C q).

    The earlier candidates are cruder versions of this:
        cone / gamma   replace sqrt(q^T Sigma q) with a scalar radius, i.e. assume an isotropic
                       Sigma, which removes the query dependence
        box / rank     use a worst-case extent, a bound rather than a spread, dominated by outliers

    With Sigma_C kept as m principal directions plus an isotropic remainder,

        q^T Sigma_C q = sum_j lam_j (q.u_j)^2 + tau2 * ||q_rest||^2

    and m=0 is the isotropic case, included as a control that should reproduce gamma's result.
    """
    key = f"_mom{m}"
    got = getattr(ctx, key, None)
    if got is None:
        got = _residual_moments(ctx, m)
        setattr(ctx, key, got)
    U, lam, tau2 = got
    cos = ctx.cos
    var = torch.zeros_like(cos)
    used = cos ** 2
    for j in range(m):
        p = ctx.Qd @ U[:, j].T
        var = var + lam[:, j].unsqueeze(0) * p ** 2
        used = used + p ** 2
    var = var + tau2.unsqueeze(0) * (1.0 - used).clamp(min=0)
    return (cos + beta * var.sqrt()).cpu().numpy()


# --- support-function summaries --- s*(q,C) = max over x in C of q.x is the support function of the
# convex hull of the cell's members: convex and piecewise linear in q. Two things follow, and they
# explain the negative results above:
#
#   1. No single vector can represent it, so the centroid isn't a bad choice of vector, it's the
#      wrong kind of object. lrv fits the best possible single linear functional, which bounds
#      what any one vector can do.
#
#   2. A max of m linear functions has the right form, but a support function only depends on
#      the hull's vertices, and a k-means sub-centroid is an interior mean. Averaging four
#      vectors where one matches at 0.8 and three at 0.4 gives 0.5, so the max gets washed out,
#      which is what subc32 showed.
#
# So keep the max, drop the averaging, and pick the m members that best cover the query
# distribution. Expected max over a set of linear functions is submodular, so greedy gets within
# (1-1/e) of optimal.

def _train_queries(ctx: Ctx, n_train: int, seed: int = 7) -> np.ndarray:
    """In-distribution training queries sampled from the base.

    Not the evaluation queries: hull and lrv are fitted to a query distribution, so reusing the eval
    set would leak and give a number that doesn't hold for new queries.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(ctx.X.shape[0], size=min(n_train, ctx.X.shape[0]),
                     replace=False)
    return np.ascontiguousarray(ctx.X[np.sort(idx)])


def _relevant_queries(ctx: Ctx, Qtr: np.ndarray, top_p: int
                      ) -> List[np.ndarray]:
    """For each cell, which training queries could ever probe it.

    A cell's score only has to be right for queries that rank it near the top; fitting the other 99%
    would spend the whole model on cases that never change a probe decision.
    """
    dev = ctx.dev
    M = ctx.part.n_cells
    rows: List[List[int]] = [[] for _ in range(M)]
    Qd = torch.from_numpy(Qtr).to(dev)
    for s in range(0, Qtr.shape[0], 2048):
        sc = Qd[s:s + 2048] @ ctx.Cd.T
        top = sc.topk(min(top_p, M), dim=1).indices.cpu().numpy()
        for i in range(top.shape[0]):
            for c in top[i]:
                rows[c].append(s + i)
        del sc
    return [np.asarray(r, dtype=np.int64) for r in rows]


def v_hull(ctx: Ctx, m: int = 4, n_train: int = 8192, top_p: int = 64,
           **kw) -> np.ndarray:
    """Greedy support-function representatives: keep m actual members per cell, chosen to maximise the
    expected max over the query distribution, and rank cells by max over j of q.x_{s_j}.

    Unlike the averaging variants:
      * every representative is a real member, so the score is a lower bound on s* and never
        overstates a cell
      * no dilution: a member matching at 0.8 reports 0.8
      * it's an m-vertex approximation of the hull's support function, which is what routing needs

    Costs m vectors per cell instead of one, on the routing side (1/128th of the data).
    """
    key = f"_hull{m}_{n_train}_{top_p}"
    reps = getattr(ctx, key, None)
    if reps is None:
        part, X = ctx.part, ctx.X
        Qtr = _train_queries(ctx, n_train)
        rel = _relevant_queries(ctx, Qtr, top_p)
        reps = np.zeros((part.n_cells, m), dtype=np.int64)
        t0 = time.time()
        for c in range(part.n_cells):
            mem = part.members(c)
            if mem.size == 0:
                continue
            qi = rel[c]
            # A cell no training query ever considered still needs summarising;
            # cover its own members' directions instead.
            Qs = Qtr[qi] if qi.size else X[mem]
            A = Qs @ X[mem].T                       # [n_rel, n_mem]
            cur = np.full(A.shape[0], -np.inf, dtype=np.float32)
            chosen = []
            for _ in range(min(m, mem.size)):
                gain = np.maximum(A, cur[:, None]).sum(0) - cur.sum()
                if chosen:
                    gain[chosen] = -np.inf
                j = int(np.argmax(gain))
                chosen.append(j)
                cur = np.maximum(cur, A[:, j])
            while len(chosen) < m:
                chosen.append(chosen[-1])
            reps[c] = mem[np.asarray(chosen)]
        _log(f"hull m={m}: greedy over {part.n_cells} cells in "
             f"{time.time() - t0:.1f}s")
        setattr(ctx, key, reps)

    dev = ctx.dev
    M, nq = ctx.part.n_cells, ctx.Qd.shape[0]
    R = torch.from_numpy(np.ascontiguousarray(ctx.X[reps.ravel()])).to(dev)
    out = torch.empty(nq, M, device=dev)
    for s in range(0, M, 4096):
        e = min(s + 4096, M)
        blk = ctx.Qd @ R[s * m:e * m].T
        out[:, s:e] = blk.view(nq, e - s, m).max(dim=2).values
        del blk
    del R
    return out.cpu().numpy()


def v_lrv(ctx: Ctx, n_train: int = 8192, ridge: float = 1e-2, **kw) -> np.ndarray:
    """Learned routing vector: replace the centroid with the ridge-regression best linear predictor of
    s*(q, C).

    Same bytes, kernel and query cost as the current router, just a different vector. It's the best
    any single vector per cell can do, so it measures the ceiling of the one-vector design.

    Cheap to fit because all cells share the Gram matrix: W = (Q'Q + lam I)^-1 Q'Y with Q [T,d] and
    Y [T,M], so one d x d solve and one matmul covers every cell. Y is accumulated in a stream.
    """
    dev = ctx.dev
    d, M = ctx.X.shape[1], ctx.part.n_cells
    Qtr = _train_queries(ctx, n_train)
    Qd = torch.from_numpy(Qtr).to(dev)
    T = Qtr.shape[0]

    # targets: exact s* for the training queries, in query blocks
    G = torch.zeros(d + 1, M, device=dev)
    A = torch.zeros(d + 1, d + 1, device=dev)
    Xt = torch.from_numpy(ctx.X)
    asg = torch.from_numpy(ctx.part.assign.astype(np.int64)).to(dev)
    for s in range(0, T, 2048):
        qb = Qd[s:s + 2048]
        y = torch.full((qb.shape[0], M), -2.0, device=dev)
        for t in range(0, ctx.X.shape[0], 8192):
            xb = Xt[t:t + 8192].to(dev, non_blocking=True)
            sc = qb @ xb.T
            idx = asg[t:t + 8192].unsqueeze(0).expand(qb.shape[0], -1)
            y.scatter_reduce_(1, idx, sc, reduce="amax")
            del sc, xb
        qa = torch.cat([qb, torch.ones(qb.shape[0], 1, device=dev)], 1)
        A += qa.T @ qa
        G += qa.T @ y
        del y, qa
    A += ridge * T * torch.eye(d + 1, device=dev)
    W = torch.linalg.solve(A, G)                    # [d+1, M]
    qa = torch.cat([ctx.Qd, torch.ones(ctx.Qd.shape[0], 1, device=dev)], 1)
    return (qa @ W).cpu().numpy()


def v_subvar(ctx: Ctx, m: int = 8, beta: float = 2.0, **kw) -> np.ndarray:
    """Cover every member, but estimate the max inside each group instead of the mean.

    subc reports each group's mean, which is where it loses: a group of four with one match at 0.85
    and three at 0.45 reports 0.55. A group should be summarised by its expected max, which for
    scores with mean q.mu and spread sqrt(q^T Sigma q) is

        s_j ~= nu_j * (q.uhat_j) + beta * sqrt(tau2_j * (1 - (q.uhat_j)^2))

    nu_j is the raw mean's norm (so the first term is the real mean member score, undoing the
    normalisation inflation subc has) and tau2_j is the group's isotropic residual variance;
    isotropic because var4 vs var0 showed the anisotropic part is worth nothing.

    Costs two scalars per group on top of subc.
    """
    key = f"_sub{m}"
    got = getattr(ctx, key, None)
    if got is None:
        got = _sub_kmeans(ctx, m, iters=kw.get("iters", 12))
        setattr(ctx, key, got)
    S, _raw, nu, tau2 = got[0], got[1], got[2], got[3]
    nq, M = ctx.Qd.shape[0], ctx.part.n_cells
    out = torch.empty(nq, M, device=ctx.dev)
    for s in range(0, M, 4096):
        e = min(s + 4096, M)
        p = ctx.Qd @ S[s * m:e * m].T                    # [nq, (e-s)*m]
        nub = nu[s * m:e * m].unsqueeze(0)
        tb = tau2[s * m:e * m].unsqueeze(0)
        est = nub * p + beta * (tb * (1.0 - p * p).clamp(min=0)).sqrt()
        out[:, s:e] = est.view(nq, e - s, m).max(dim=2).values
        del p, est
    return out.cpu().numpy()


def v_noise(ctx: Ctx, eps: float = 0.01, seed: int = 0, **kw) -> np.ndarray:
    """The oracle plus Gaussian noise of scale eps.

    Not a real candidate, more of a spec: how accurate does a summary have to be before probes
    collapse to the floor? That tells us whether any compact summary can close the gap at all.
    """
    g = np.random.default_rng(seed)
    return ctx.oracle + g.normal(0.0, eps, ctx.oracle.shape).astype(np.float32)


VARIANTS = {
    "mean": v_mean,
    "hull": v_hull,
    "subvar": v_subvar,
    "noise": v_noise,
    "lrv": v_lrv,
    "var": v_var,
    "oracle": v_oracle,
    "cone": v_cone,
    "cone_max": lambda ctx, **kw: v_cone(ctx, radius="max"),
    "gamma": v_gamma,
    "box": v_box,
    "subc": v_subc,
    "subcraw": lambda ctx, m=4, **kw: v_subc(ctx, m=m, norm=False),
    "rank": v_rank,
}


# --- CSR view of the partition ---

def hier_eval(ctx: Ctx, gt_ids: np.ndarray, m: int, beta: float, L: int,
              probes: List[int], k_eval: int, kw_bits: int = 0
              ) -> Dict[int, Tuple[float, float]]:
    """Two-level evaluation where the sub-group is the scan unit.

        stage 1   Hamming over all M cell centroid codes  -> top L cells
        stage 2   read those L cells' m sub-summaries     -> score L*m groups
        scan      the best P sub-groups, ~cell_size/m vectors each

    The cell becomes only the unit of shortlisting. With m=8, at most one or two sub-groups in a
    probed cell hold the match, and after compact() a sub-group is already contiguous, so skipping
    the rest is easy to implement.

    L is set by how far the stage-1 router reaches: a cell the Hamming pass never shortlists can't
    be recovered later, so scoring as if the shortlist were as small as the final probe count would
    be wrong.
    """
    dev = ctx.dev
    part = ctx.part
    M = part.n_cells
    got = getattr(ctx, f"_sub{m}", None)
    if got is None:
        got = _sub_kmeans(ctx, m)
        setattr(ctx, f"_sub{m}", got)
    S, _raw, nu, tau2, sub_orig = got
    nq = ctx.Qd.shape[0]

    # per-subgroup sizes, and the subgroup holding each true neighbour
    gsz = np.bincount(sub_orig, minlength=M * m).astype(np.float64)
    gt_sub = sub_orig[gt_ids[:, :k_eval]].astype(np.int64)

    # stage 1: which cells survive
    keep = ctx.cos.topk(min(L, M), dim=1).indices           # [nq, L]

    # bits=1 scores sub-centroids as the engine would store them (1-bit codes, 64 B) instead of f32
    # (1,536 B). The byte argument for sub-groups assumes 64 B, so measuring at f32 would be
    # measuring something we couldn't afford to build.
    bits = int(kw_bits)
    if bits == 1:
        R = _rand_rot(ctx.X.shape[1], 1)
        qsg = torch.sign(ctx.Qd @ R)
        ssg = torch.sign(S @ R)
        dpad = ctx.X.shape[1]

    scores = torch.full((nq, M * m), -1e9, device=dev)
    for s in range(0, M, 2048):
        e = min(s + 2048, M)
        if bits == 1:
            ham = (dpad - qsg @ ssg[s * m:e * m].T) * 0.5
            p = torch.cos(torch.pi * ham / dpad)
            del ham
        else:
            p = ctx.Qd @ S[s * m:e * m].T
        nub = nu[s * m:e * m].unsqueeze(0)
        tb = tau2[s * m:e * m].unsqueeze(0)
        scores[:, s * m:e * m] = nub * p + beta * (
            tb * (1.0 - p * p).clamp(min=0)).sqrt()
        del p
    # mask out sub-groups whose cell did not survive stage 1
    mask = torch.full((nq, M), False, device=dev)
    mask.scatter_(1, keep, True)
    scores = torch.where(mask.repeat_interleave(m, dim=1), scores,
                         torch.tensor(-1e9, device=dev))

    gszd = torch.from_numpy(gsz).to(dev)
    gc = torch.from_numpy(gt_sub).to(dev)
    pmax = min(max(probes), M * m)
    order = scores.topk(pmax, dim=1).indices
    rank = torch.full((nq, M * m), M * m, dtype=torch.int32, device=dev)
    rank.scatter_(1, order, torch.arange(pmax, dtype=torch.int32,
                                         device=dev).expand(nq, -1))
    gt_rank = torch.gather(rank, 1, gc)
    scanned = torch.cumsum(torch.gather(gszd.expand(nq, -1), 1, order), dim=1)
    out = {}
    for p in probes:
        if p > pmax:
            continue
        out[p] = (float((gt_rank < p).float().mean()),
                  float(scanned[:, p - 1].mean()))
    del scores, order, rank
    torch.cuda.empty_cache()
    return out


# --- routing as the engine actually runs it ---
#
# Everything above compares summaries at f32, which is right for comparing estimators but isn't what
# the engine does. search_one with set_two_stage(True, 2) runs:
#
#     stage 1   Hamming over all M centroid codes           M * 64 B
#     stage 2   exact f32 re-order of nprobe*mult centroids np*mult*1536 B
#
# At 1M with nprobe=384 the second line is 768 * 1536 = 1,180 KB per query, a quarter of the budget,
# spent at f32 on a step that only orders cells. That ordering already carries ~0.05 of structured
# error (section 4 of SUMMARISER_STUDY.md), so f32 precision is wasted there.
#
# This measures how much precision that step needs. Dropping precision here has no downside
# elsewhere: memory goes down (smaller centroid copy), latency follows the bytes, and recall is
# whatever this shows.

def _rand_rot(d: int, seed: int = 0) -> torch.Tensor:
    """Random orthogonal matrix, standing in for RaBitQ's Hadamard+signs."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    return q.to(_dev())


def _quantize(C: torch.Tensor, mode: str) -> torch.Tensor:
    """Round-trip centroids through a storage format, per-dimension."""
    if mode == "f32":
        return C
    if mode == "f16":
        return C.half().float()
    if mode in ("sq8", "sq6", "sq4"):
        bits = {"sq8": 8, "sq6": 6, "sq4": 4}[mode]
        lo = C.min(dim=0).values
        hi = C.max(dim=0).values
        step = ((hi - lo) / (2 ** bits - 1)).clamp(min=1e-12)
        return ((C - lo) / step).round().clamp(0, 2 ** bits - 1) * step + lo
    raise ValueError(mode)


def twostage_curve(ctx: Ctx, gt_cells: np.ndarray, probes: List[int],
                   mode: str, mult: int) -> Dict[int, Tuple[float, float]]:
    """Recall/scan curve for "Hamming shortlist, then re-order at `mode` precision".

    The shortlist is rebuilt at every nprobe because its size is nprobe*mult, so this matches what
    the engine does instead of approximating it with one fixed shortlist.
    """
    dev = ctx.dev
    M = ctx.part.n_cells
    R = _rand_rot(ctx.X.shape[1], 0)
    qs = torch.sign(ctx.Qd @ R)
    cs = torch.sign(ctx.Cd @ R)
    d = ctx.X.shape[1]
    # SimHash estimate: E[hamming/d] = theta/pi, so cos ~= cos(pi*h/d)
    ham = (d - qs @ cs.T) * 0.5
    coarse = torch.cos(torch.pi * ham / d)
    exact = ctx.Qd @ _quantize(ctx.Cd, mode).T

    sz = torch.from_numpy(ctx.part.sizes.astype(np.float64)).to(dev)
    gc = torch.from_numpy(gt_cells.astype(np.int64)).to(dev)
    nq = ctx.Qd.shape[0]
    out: Dict[int, Tuple[float, float]] = {}
    for p in probes:
        if p > M:
            continue
        want = min(p * mult, M)
        cand = coarse.topk(want, dim=1).indices
        sc = torch.gather(exact, 1, cand)
        sel = cand.gather(1, sc.topk(min(p, want), dim=1).indices)   # [nq, p]
        hit = (gc.unsqueeze(2) == sel.unsqueeze(1)).any(dim=2)
        scanned = torch.gather(sz.expand(nq, -1), 1, sel).sum(1)
        out[p] = (float(hit.float().mean()), float(scanned.mean()))
        del cand, sc, sel, hit
    return out


def build_context(args) -> Tuple[Ctx, np.ndarray]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    base, queries, gt_ids, _gt_sims, tag = GT.load_or_build(
        args.corpus, k=args.k, n_queries=args.queries, limit=args.limit)
    base = np.ascontiguousarray(base, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)

    n_cells = args.cells or max(1, len(base) // args.cell_size)
    pcache = os.path.join(CACHE_DIR, f"part_{tag}_m{n_cells}_s{args.seed}.npz")
    if os.path.exists(pcache) and not args.rebuild:
        z = np.load(pcache)
        cents, assign = z["cents"], z["assign"]
        _log(f"partition cache hit: {os.path.basename(pcache)}")
    else:
        _log(f"training {n_cells} cells over {len(base):,} vectors ...")
        cents, assign = spherical_kmeans(base, n_cells, iters=args.kmeans_iters,
                                         seed=args.seed)
        np.savez(pcache, cents=cents, assign=assign)
        _log(f"partition cached -> {os.path.basename(pcache)}")

    part = Partition(cents, assign, n_cells)
    _log(f"cells: {n_cells}  size mean={part.sizes.mean():.1f} "
         f"p50={np.median(part.sizes):.0f} max={part.sizes.max()} "
         f"empty={(part.sizes == 0).sum()}")

    ocache = os.path.join(CACHE_DIR, f"oracle_{tag}_m{n_cells}_s{args.seed}.npy")
    if os.path.exists(ocache) and not args.rebuild:
        oracle = np.load(ocache)
        _log(f"oracle cache hit: {os.path.basename(ocache)}")
    else:
        oracle = oracle_max(base, queries, part)
        np.save(ocache, oracle)

    gt_cells = part.assign[gt_ids[:, :args.k_eval]]
    occ = np.array([len(np.unique(r)) for r in gt_cells])
    _log(f"FLOOR: true top-{args.k_eval} occupy {occ.mean():.2f} cells "
         f"({sum(part.sizes[np.unique(r)].sum() for r in gt_cells) / len(gt_cells):.0f} vectors)")
    return Ctx(base, queries, part, oracle), gt_cells, gt_ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus")
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=100, help="ground-truth depth")
    ap.add_argument("--k-eval", type=int, default=10, help="recall@ this k")
    ap.add_argument("--cell-size", type=int, default=128)
    ap.add_argument("--cells", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kmeans-iters", type=int, default=20)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--variants", default="mean,oracle,cone,gamma,box")
    ap.add_argument("--gammas", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--subs", default="2,4,8")
    ap.add_argument("--ranks", default="1,2,4")
    ap.add_argument("--betas", default="1.0,2.0,3.0")
    ap.add_argument("--eps", default="0.005,0.01,0.02,0.04,0.08")
    ap.add_argument("--mult", type=int, default=2, help="shortlist multiplier")
    ap.add_argument("--rerank", type=int, default=32)
    ap.add_argument("--rerank-bytes", type=int, default=384,
                    help="384 = per-dim SQ8 (the shipped efficient config)")
    ap.add_argument("--hier", action="store_true",
                    help="evaluate with the sub-group as the scan unit")
    ap.add_argument("--target", type=float, default=0.99)
    ap.add_argument("--shortlist", type=int, default=None)
    ap.add_argument("--hier-bits", type=int, default=0,
                    help="1 = score sub-centroids from 1-bit codes (64 B), as "
                         "the engine would store them; 0 = f32 (1,536 B)")
    ap.add_argument("--precision", action="store_true",
                    help="what precision the stage-2 centroid re-order needs")
    ap.add_argument("--modes", default="f32,f16,sq8,sq6,sq4")
    ap.add_argument("--mults", default="2,1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ctx, gt_cells, gt_ids = build_context(args)
    probes = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384,
              512, 768, 1024, 1536, 2048]

    if args.precision:
        M = ctx.part.n_cells
        print()
        print(f"{'re-order':<10} {'mult':>5} {'B/cell':>7} {'np@0.99':>8} "
              f"{'scan@0.99':>10} {'stage2 KB':>10} {'KB/query':>9} {'vs f32x2':>9}")
        print("-" * 76)
        base = None
        for mode in args.modes.split(","):
            nb = {"f32": 1536, "f16": 768, "sq8": 384, "sq6": 288, "sq4": 192}[mode]
            for mult in [int(x) for x in args.mults.split(",")]:
                cur = twostage_curve(ctx, gt_cells, probes, mode, mult)
                got = probes_for(cur, args.target)
                if got is None:
                    print(f"{mode:<10} {mult:>5} {nb:>7} {'misses target':>19}")
                    continue
                p, scanned = got
                st2 = p * mult * nb
                tot = (M * BYTES_PER_CODE + st2 + scanned * BYTES_PER_CODE
                       + args.k_eval * args.rerank * args.rerank_bytes)
                if base is None:
                    base = tot
                print(f"{mode:<10} {mult:>5} {nb:>7} {p:>8} {scanned:>10,.0f} "
                      f"{st2 / 1024:>10.0f} {tot / 1024:>9.0f} "
                      f"{base / tot:>8.2f}x")
        print(f"\nRow 1 (f32, mult=2) is the shipped configuration. "
              f"target R@{args.k_eval}={args.target}")
        return

    if args.hier:
        # Baseline for comparison: the shipped cell-level router.
        base_curve = routing_curve(v_mean(ctx), gt_cells, ctx.part.sizes, probes)
        M = ctx.part.n_cells
        bp = probes_for(base_curve, args.target)
        L = args.shortlist or (bp[0] if bp else 512)
        base_tot = (M * BYTES_PER_CODE + L * 68
                    + bp[1] * BYTES_PER_CODE
                    + args.k_eval * args.rerank * args.rerank_bytes)
        _log(f"stage-1 shortlist L={L} (cells the centroid router needs at "
             f"R@{args.k_eval}={args.target})")
        hprobes = sorted({int(round(2 ** (e / 4.0))) for e in range(12, 61)})
        print()
        print(f"{'config':<14} {'groups':>8} {'scan@t':>9} {'stage2 KB':>10} "
              f"{'scan KB':>9} {'KB/query':>9} {'vs mean':>8}")
        print("-" * 74)
        print(f"{'mean (cells)':<14} {bp[0]:>8} {bp[1]:>9,.0f} "
              f"{L * 68 / 1024:>10.0f} {bp[1] * 64 / 1024:>9.0f} "
              f"{base_tot / 1024:>9.0f} {'1.00x':>8}")
        for m in [int(x) for x in args.subs.split(",")]:
            for beta in [float(b) for b in args.betas.split(",")]:
                cur = hier_eval(ctx, gt_ids, m, beta, L, hprobes,
                                args.k_eval, kw_bits=args.hier_bits)
                got = probes_for(cur, args.target)
                if got is None:
                    print(f"{f'subvar{m} b{beta}':<14} {'-':>8} "
                          f"{'misses target':>9}")
                    continue
                p, scanned = got
                st2 = L * m * (BYTES_PER_CODE + 8)
                tot = (M * BYTES_PER_CODE + st2 + scanned * BYTES_PER_CODE
                       + args.k_eval * args.rerank * args.rerank_bytes)
                print(f"{f'subvar{m} b{beta}':<14} {p:>8} {scanned:>9,.0f} "
                      f"{st2 / 1024:>10.0f} {scanned * 64 / 1024:>9.0f} "
                      f"{tot / 1024:>9.0f} {base_tot / tot:>7.2f}x")
        print(f"\nbytes: M*64 (stage 1) + L*m*72 (sub-summaries for the "
              f"shortlist) + scanned*64 + {args.k_eval}*{args.rerank}*"
              f"{args.rerank_bytes} (rerank)")
        return

    # (vectors, scalars) stored per cell by each variant, for the byte model. oracle and noise are
    # bounds, not implementable summaries, so they're marked as such instead of given a made-up
    # size.
    def shape_of(label: str, kw: dict) -> Optional[Tuple[int, int]]:
        if label.startswith(("oracle", "noise")):
            return None
        if label.startswith("mean") or label.startswith("lrv"):
            return (1, 1)
        m = int(kw.get("m", 1))
        if label.startswith(("subc", "subvar", "hull")):
            return (m, 2 * m if label.startswith("subvar") else 0)
        if label.startswith(("var", "rank")):
            return (1 + m, 1 + m)
        if label.startswith(("cone", "gamma", "box")):
            return (2 if label.startswith("box") else 1, 1)
        return (1, 1)

    rows = []
    for name in args.variants.split(","):
        name = name.strip()
        if not name:
            continue
        specs = [(name, {})]
        if name == "gamma":
            specs = [(f"gamma{g}", {"gamma": float(g)})
                     for g in args.gammas.split(",")]
        elif name in ("subc", "subcraw"):
            specs = [(f"{name}{m}", {"m": int(m)}) for m in args.subs.split(",")]
        elif name == "rank":
            specs = [(f"rank{m}", {"m": int(m)}) for m in args.ranks.split(",")]
        elif name == "hull":
            specs = [(f"hull{m}", {"m": int(m)}) for m in args.subs.split(",")]
        elif name == "subvar":
            specs = [(f"subvar{m}b{b}", {"m": int(m), "beta": float(b)})
                     for m in args.subs.split(",") for b in args.betas.split(",")]
        elif name == "noise":
            specs = [(f"noise{e}", {"eps": float(e)}) for e in args.eps.split(",")]
        elif name == "var":
            specs = [(f"var{m}b{b}", {"m": int(m), "beta": float(b)})
                     for m in args.ranks.split(",") for b in args.betas.split(",")]
        for label, kw in specs:
            fn = VARIANTS[name]
            t0 = time.time()
            S = np.ascontiguousarray(fn(ctx, **kw), dtype=np.float32)
            curve = routing_curve(S, gt_cells, ctx.part.sizes, probes)
            eps = (None if label.startswith(("oracle", "noise"))
                   else effective_eps(S, ctx.oracle))
            rows.append((label, curve, shape_of(label, kw), eps))
            del S
            torch.cuda.empty_cache()

    M = ctx.part.n_cells
    base_bytes = None
    print()
    print(f"{'variant':<13} {'S/cell':>8} {'eps':>7} {'np@0.95':>8} "
          f"{'np@0.99':>8} {'scan@0.99':>10} {'KB/q@0.99':>10} {'vs mean':>8}")
    print("-" * 78)
    for label, curve, shape, eps in rows:
        cells = []
        for t in (0.95, 0.99):
            got = probes_for(curve, t)
            cells.append(f"{got[0]}" if got else ">2048")
        s99 = probes_for(curve, 0.99)
        if shape is None:
            sb, sb_txt, tot = 0, "bound", None
        else:
            sb = shape[0] * BYTES_PER_CODE + shape[1] * 4
            sb_txt = f"{sb}B"
            got = query_bytes(curve, 0.99, M, sb, mult=args.mult,
                              k=args.k_eval, rf=args.rerank)
            tot = got[1] if got else None
        if label == "mean" and tot:
            base_bytes = tot
        spd = (f"{base_bytes / tot:.2f}x"
               if (tot and base_bytes) else "-")
        print(f"{label:<13} {sb_txt:>8} "
              f"{(f'{eps:.4f}' if eps is not None else '-'):>7} "
              f"{cells[0]:>8} {cells[1]:>8} "
              f"{(f'{s99[1]:,.0f}' if s99 else '-'):>10} "
              f"{(f'{tot / 1024:,.0f}' if tot else '-'):>10} {spd:>8}")
    print("\nbytes model: M*64 (stage 1) + nprobe*mult*S (re-order) + "
          "scanned*64 (scan) + k*rf*1536 (rerank)")
    print("S/cell counts every stored vector as one 64 B binary code and every "
          "scalar as 4 B.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({lbl: {str(p): v for p, v in c.items()}
                       for lbl, c, _s, _ in rows}, f, indent=2)
        _log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
