"""scan_cascade.py: how cheap can the first pass over a probed cell's vectors be?

After the SQ8 re-order change the scan is ~77% of a query's bytes, at 64 B per vector. Sub-group
pruning tries to scan fewer vectors, but ranking the sub-groups well enough costs more bytes than it
saves (1.25x with 64 B sub-centroids, negative with SQ8 ones).

Other direction: still scan every vector in the probed cells, but with a shorter code first, and
only promote a small shortlist to the full 64 B code:

    b-bit prefilter over all probed vectors   ->  top T
    64 B binary over those T                  ->  top k*rf
    SQ8 rerank                                ->  top k

Nice property: it doesn't touch the partition, so compact(), delete, split/merge and persistence
stay as they are. It's one more arena read in the same loop order.

Measures stage 1's accuracy/bytes trade-off: for each code length, the shortlist T needed to keep
recall, and the total bytes.

    python bench/scan_cascade.py msmarco_embeddings_cache_v2_1000000.npy --queries 1000 --nprobe 256 --bits 32,64,128,256
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groundtruth as GT  # noqa: E402
from summariser_lab import (CACHE_DIR, Partition, _dev, _log,  # noqa: E402
                            _rand_rot, spherical_kmeans)

torch.backends.cuda.matmul.allow_tf32 = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--k-eval", type=int, default=10)
    ap.add_argument("--cell-size", type=int, default=128)
    ap.add_argument("--nprobe", type=int, default=256)
    ap.add_argument("--bits", default="32,64,128,256")
    ap.add_argument("--shortlists", default="256,512,1024,2048,4096,8192")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=8192)
    args = ap.parse_args()

    dev = _dev()
    base, queries, gt_ids, _s, tag = GT.load_or_build(
        args.corpus, k=100, n_queries=args.queries)
    base = np.ascontiguousarray(base, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    n, d = base.shape
    M = max(1, n // args.cell_size)

    pc = os.path.join(CACHE_DIR, f"part_{tag}_m{M}_s{args.seed}.npz")
    if os.path.exists(pc):
        z = np.load(pc)
        cents, assign = z["cents"], z["assign"]
        _log(f"partition cache hit: {os.path.basename(pc)}")
    else:
        cents, assign = spherical_kmeans(base, M, seed=args.seed)
        np.savez(pc, cents=cents, assign=assign)
    part = Partition(cents, assign, M)

    Qd = torch.from_numpy(queries).to(dev)
    Cd = torch.from_numpy(cents).to(dev)
    nq = queries.shape[0]

    # which cells each query probes, picked the same way the engine does
    probe = (Qd @ Cd.T).topk(min(args.nprobe, M), dim=1).indices
    inprobe = torch.zeros(nq, M, dtype=torch.bool, device=dev)
    inprobe.scatter_(1, probe, True)
    asg = torch.from_numpy(part.assign.astype(np.int64)).to(dev)
    scanned = float(torch.from_numpy(
        part.sizes.astype(np.float64)).to(dev)[probe].sum(1).mean())
    _log(f"nprobe={args.nprobe}: {scanned:,.0f} vectors scanned per query")

    R = _rand_rot(d, 3)
    Xt = torch.from_numpy(base)
    gt = torch.from_numpy(gt_ids[:, :args.k_eval].astype(np.int64)).to(dev)
    Ts = [int(x) for x in args.shortlists.split(",")]
    bitlist = [int(b) for b in args.bits.split(",")]
    Tmax = max(Ts)

    print()
    print(f"{'code':>6} {'B/vec':>6} {'T':>7} {'recall':>8} {'stage1 KB':>10} "
          f"{'stage2 KB':>10} {'total KB':>9} {'vs 64B':>7}")
    print("-" * 70)
    # Baseline: today's path scans every probed vector at 64 B.
    base_kb = (scanned * 64 + args.k_eval * 32 * 384) / 1024

    for b in bitlist:
        t0 = time.time()
        Rb = R[:, :b]
        qs = torch.sign(Qd @ Rb)
        # running top-Tmax over the b-bit estimate, restricted to probed cells
        bv = torch.full((nq, Tmax), -1e9, device=dev)
        bi = torch.zeros(nq, Tmax, dtype=torch.long, device=dev)
        for s in range(0, n, args.chunk):
            xb = Xt[s:s + args.chunk].to(dev, non_blocking=True)
            cb = torch.sign(xb @ Rb)
            est = qs @ cb.T                       # ~ b - 2*hamming
            keep = inprobe.gather(1, asg[s:s + args.chunk].unsqueeze(0)
                                  .expand(nq, -1))
            est = torch.where(keep, est, torch.tensor(-1e9, device=dev))
            idx = torch.arange(s, min(s + args.chunk, n), device=dev
                               ).unsqueeze(0).expand(nq, -1)
            cv = torch.cat([bv, est], 1)
            ci = torch.cat([bi, idx], 1)
            tv, ti = cv.topk(Tmax, dim=1)
            bv, bi = tv, ci.gather(1, ti)
            del xb, cb, est, keep, idx, cv, ci
        for T in Ts:
            sel = bi[:, :T]
            hit = (gt.unsqueeze(2) == sel.unsqueeze(1)).any(dim=2)
            rec = float(hit.float().mean())
            s1 = scanned * (b / 8)
            s2 = T * 64
            tot = (s1 + s2 + args.k_eval * 32 * 384) / 1024
            print(f"{b:>6} {b / 8:>6.0f} {T:>7} {rec:>8.4f} {s1 / 1024:>10.0f} "
                  f"{s2 / 1024:>10.0f} {tot:>9.0f} {base_kb / tot:>6.2f}x")
        _log(f"b={b} in {time.time() - t0:.0f}s")
    print(f"\nbaseline (today): scan every probed vector at 64 B = "
          f"{scanned * 64 / 1024:,.0f} KB + rerank -> {base_kb:,.0f} KB")
    print("recall here is the fraction of true top-10 surviving stage 1; the "
          "64 B pass and SQ8 rerank then run on T candidates.")


if __name__ == "__main__":
    main()
