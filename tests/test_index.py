"""Correctness tests for the flat-cell index.

Not benchmarks: these check invariants that would make the index silently return wrong answers if
broken. A benchmark won't catch that, since a corrupt index can still be fast.

Run this file directly with python.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import aigvs_rust  # noqa: E402

# Binary codes need enough bits to rank with. At d=64 a code is one 64-bit word, so Hamming has only
# 65 possible values and can't order 4000 vectors (recall 0.467). That's the known low-dimension
# limitation, not a bug; real embeddings are 384-1536d.
DIM = 256
N = 4000
# Set so neighbour similarities look like real embeddings (MSMARCO: sim@1 = 0.738, sim@10 = 0.502).
SPREAD = 0.7
NCLUSTERS = 200
NCELLS = 32
K = 10

_fail = 0


def check(cond: bool, msg: str) -> None:
    global _fail
    if cond:
        print(f"  PASS  {msg}")
    else:
        _fail += 1
        print(f"  FAIL  {msg}")


def unit(rng, n, d):
    x = rng.standard_normal((n, d)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return np.ascontiguousarray(x)


def clustered(rng, n, d, n_clusters=200, spread=0.7, skew=True, centers=None):
    """Clustered unit vectors.

    Uniform random points on a sphere make a bad ANN test: neighbour similarities are so tightly
    packed (gaps ~0.005 at d=64) that quantization noise legitimately reshuffles the top-k, so
    recall looks broken when nothing is wrong. Real embeddings are clustered.

    `skew` gives clusters geometrically decreasing sizes so the partition is actually unbalanced;
    otherwise maintain() has nothing to do.
    """
    # Queries have to come from the same centres as the base. With fresh centres the queries land
    # nowhere near any indexed cluster, neighbour similarity drops to the uniform-random level
    # (~0.2), and the data has no structure again.
    if centers is None:
        centers = unit(rng, n_clusters, d)
    n_clusters = len(centers)
    if skew:
        w = np.geomspace(1.0, 0.05, n_clusters)
    else:
        w = np.ones(n_clusters)
    w = w / w.sum()
    # Largest-remainder allocation, exact without a fix-up loop. Clusters can get zero points (the
    # query set is smaller than the cluster count), so no minimum of one is forced; that would be
    # impossible whenever n < n_clusters.
    counts = np.floor(w * n).astype(int)
    rem = n - counts.sum()
    if rem > 0:
        counts[np.argsort(-(w * n - counts))[:rem]] += 1
    assert counts.sum() == n, (counts.sum(), n)

    parts = []
    # Noise is scaled by 1/sqrt(d). An unscaled N(0,1) vector has norm ~sqrt(d) (16 at d=256), which
    # swamps the unit-norm centre and makes the data uniform-random, which then looks like a
    # quantizer bug. With the scaling, `spread` is the noise size relative to the centre, and
    # cos(point, centre) ~ 1/sqrt(1+spread^2).
    sigma = spread / np.sqrt(d)
    for c, m in zip(centers, counts):
        if m <= 0:
            continue
        p = c[None, :] + sigma * rng.standard_normal((m, d)).astype(np.float32)
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        parts.append(p)
    x = np.concatenate(parts).astype(np.float32)
    rng.shuffle(x)
    return np.ascontiguousarray(x), centers


def build(vecs, ids, scan_mode="binary", **kw):
    import faiss
    km = faiss.Kmeans(vecs.shape[1], NCELLS, niter=10, spherical=True, seed=0)
    km.train(vecs)
    c = np.ascontiguousarray(km.centroids)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    ix = aigvs_rust.FlatCellIndex(vecs.shape[1], scan_mode, 8, 0x5EED,
                                  kw.get("code_bits", 0))
    if kw.get("f16_rerank"):
        ix.set_f16_rerank(True)
    elif kw.get("i8_rerank", True):
        ix.set_i8_rerank(True)
    if kw.get("adaptive"):
        ix.set_adaptive(True, 0.02)
    if kw.get("two_stage"):
        ix.set_two_stage(True, 4)
    ix.set_centroids(c)
    ix.add_batch([int(i) for i in ids], vecs)
    return ix


def search_all(ix, queries, nprobe=NCELLS):
    ids, _ = ix.search_batch(np.ascontiguousarray(queries), K, nprobe, 1.0, 8.0)
    return np.asarray(ids, dtype=np.int64).reshape(len(queries), K)


def assert_realistic(base, queries, tag):
    """Check the test data before testing the index.

    A generator that quietly produces uniform-random points makes a correct quantizer look broken:
    binary scan gets 0.94 on real MSMARCO (sim@1 0.505) but 0.39 on uniform random (sim@1 0.203),
    while f32 gets 1.0000 on both. Two generator bugs did exactly that (noise not scaled by
    1/sqrt(d), and queries drawn from different centres than the base), and I first took both for
    engine bugs.

    Real embedding corpora sit around sim@1 0.5-0.75. Anything near 0.2 means uniform-random data
    and should fail here, not show up later as a confusing recall number.
    """
    s = np.sort(queries @ base.T, axis=1)[:, ::-1]
    s1, s10 = float(s[:, 0].mean()), float(s[:, 9].mean())

    # Compare against uniform-random data of the same shape instead of a fixed constant, since the
    # baseline depends on d and n; a hard-coded threshold would miss degenerate data at high d or
    # reject good data at low d.
    r = np.random.default_rng(12345)
    u = r.standard_normal((len(base), base.shape[1])).astype(np.float32)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    uq = r.standard_normal((min(64, len(queries)), base.shape[1])).astype(np.float32)
    uq /= np.linalg.norm(uq, axis=1, keepdims=True)
    baseline = float(np.sort(uq @ u.T, axis=1)[:, -1].mean())

    ok = s1 > 2.0 * baseline
    check(ok, f"{tag} data has resolvable neighbour structure "
              f"(sim@1={s1:.3f}, sim@10={s10:.3f}; uniform baseline={baseline:.3f})")
    return s1, s10


def exact_topk(base, ids, queries, k=K):
    sims = queries @ base.T
    order = np.argsort(-sims, axis=1)[:, :k]
    return ids[order]


def main() -> None:
    rng = np.random.default_rng(0)
    base, centers = clustered(rng, N, DIM, n_clusters=NCLUSTERS, spread=SPREAD)
    ids = np.arange(N, dtype=np.uint64)
    queries, _ = clustered(rng, 100, DIM, spread=SPREAD, skew=False,
                           centers=centers)
    truth = exact_topk(base, ids, queries)

    print("\n[0] synthetic data must resemble real embeddings")
    assert_realistic(base, queries, "synthetic")

    print("\n[1] f32 mode scanning every cell must equal brute force")
    ixf = build(base, ids, scan_mode="f32", i8_rerank=False)
    gotf = search_all(ixf, queries)
    ov = np.mean([len(set(g) & set(t)) / K for g, t in zip(gotf, truth)])
    check(ov > 0.999, f"f32 exhaustive recall@{K} = {ov:.4f} (>0.999)")

    print("\n[1b] binary + int8 rerank, every cell probed")
    ix = build(base, ids)
    got = search_all(ix, queries)
    overlap = np.mean([len(set(g) & set(t)) / K for g, t in zip(got, truth)])
    check(overlap > 0.97, f"binary recall@{K} vs brute force = {overlap:.4f}")

    print("\n[2] save/load round-trip is bit-identical")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "idx.aigvs")
        ix.save(p)
        ix2 = aigvs_rust.FlatCellIndex.load(p)
        check(len(ix2) == len(ix), f"n_live preserved ({len(ix2)} == {len(ix)})")
        got2 = search_all(ix2, queries)
        check(np.array_equal(got, got2), "search results identical after reload")
        s1, s2 = ix.stats(), ix2.stats()
        check(abs(s1["quant_scale"] - s2["quant_scale"]) < 1e-9,
              "quant_scale preserved (rotation/codes stay consistent)")
        check(s1["n_cells"] == s2["n_cells"], "cell count preserved")

    print("\n[3] delete removes exactly the requested ids")
    ixd = build(base, ids)
    victims = [int(v) for v in rng.choice(N, 500, replace=False)]
    for v in victims:
        check_ok = ixd.remove(v)
        if not check_ok:
            break
    check(len(ixd) == N - 500, f"n_live after 500 deletes = {len(ixd)}")
    got = search_all(ixd, queries)
    vs = set(victims)
    leaked = sum(1 for row in got for x in row if int(x) in vs)
    check(leaked == 0, f"no deleted id returned by search (leaked={leaked})")
    check(ixd.remove(victims[0]) is False, "re-deleting an id returns False")

    print("\n[4] deleted ids stay deleted across save/load")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "idx.aigvs")
        ixd.save(p)
        ixl = aigvs_rust.FlatCellIndex.load(p)
        check(len(ixl) == N - 500, "n_live preserved after reload")
        got = search_all(ixl, queries)
        leaked = sum(1 for row in got for x in row if int(x) in vs)
        check(leaked == 0, f"no resurrected ids after reload (leaked={leaked})")

    print("\n[5] maintain() rebalances without corrupting results")
    ixm = build(base, ids)
    before = search_all(ixm, queries)
    st0 = ixm.stats()
    splits, merges = ixm.maintain(0, 1.5, 0.3)
    st1 = ixm.stats()
    check(len(ixm) == N, f"no vectors lost by maintenance (n_live={len(ixm)})")
    spread0 = st0["cell_size_max"] / max(st0["cell_size_mean"], 1)
    spread1 = st1["cell_size_max"] / max(st1["cell_size_mean"], 1)
    print(f"        splits={splits} merges={merges} "
          f"max/mean {spread0:.2f} -> {spread1:.2f}")
    check(spread1 <= spread0 + 1e-6, "size skew did not get worse")
    after = search_all(ixm, queries)
    ov = np.mean([len(set(a) & set(b)) / K for a, b in zip(before, after)])
    check(ov > 0.99, f"results stable across maintenance (overlap {ov:.4f})")

    print("\n[6] adaptive probing must not change results vs full scan")
    ixa = build(base, ids, adaptive=True)
    full = search_all(build(base, ids), queries, nprobe=NCELLS)
    ada = search_all(ixa, queries, nprobe=NCELLS)
    ov = np.mean([len(set(a) & set(b)) / K for a, b in zip(full, ada)])
    check(ov > 0.97, f"adaptive matches exhaustive (overlap {ov:.4f})")

    print("\n[7] two-stage routing must not change results")
    ix2s = build(base, ids, two_stage=True)
    got2s = search_all(ix2s, queries, nprobe=NCELLS)
    ov = np.mean([len(set(a) & set(b)) / K for a, b in zip(full, got2s)])
    check(ov > 0.98, f"two-stage matches single-stage (overlap {ov:.4f})")

    print("\n[8] truncated codes still retrieve correctly")
    ixt = build(base, ids, code_bits=64)
    gott = search_all(ixt, queries, nprobe=NCELLS)
    ov = np.mean([len(set(a) & set(b)) / K for a, b in zip(truth, gott)])
    check(ov > 0.90, f"64-bit codes + rerank vs brute force ({ov:.4f})")
    print("        (64 bits for a 256-d vector = 4x truncation)")

    print("\n[8b] centroid-graph routing must match flat routing")
    # The graph only turns on past MIN_CELLS_FOR_GRAPH (4096), since below that the flat centroid
    # scan wins. Force it on with a small fixture so the code path actually runs.
    import faiss
    gcells = 128
    km = faiss.Kmeans(DIM, gcells, niter=10, spherical=True, seed=0)
    km.train(base)
    cg = np.ascontiguousarray(km.centroids)
    cg /= np.linalg.norm(cg, axis=1, keepdims=True)

    def build_g(graph: bool):
        ix = aigvs_rust.FlatCellIndex(DIM, "binary", 8, 0x5EED, 0)
        ix.set_i8_rerank(True)
        ix.set_graph_route(graph, 8 if graph else 10 ** 9)
        ix.set_centroids(cg)
        ix.add_batch([int(i) for i in ids], base)
        return ix

    ix_flat, ix_graph = build_g(False), build_g(True)
    st = ix_graph.stats()
    check(st["centroid_graph_nodes"] > 0,
          f"graph built ({st['centroid_graph_nodes']:.0f} nodes, "
          f"{st['centroid_graph_bytes']/1024:.0f} KB)")
    for npb in (8, 32):
        a = search_all(ix_flat, queries, nprobe=npb)
        b = search_all(ix_graph, queries, nprobe=npb)
        ov = np.mean([len(set(x) & set(y)) / K for x, y in zip(a, b)])
        check(ov > 0.95, f"graph vs flat routing at nprobe={npb}: {ov:.4f}")

    print("\n[8c] delete must work under every storage mode")
    # Delete is the only path that needs the original vector back (to subtract it from the cell's
    # centroid sum), so it has to handle every storage mode. Reading scan_i8 unconditionally
    # panicked under f16, where the int8 arena is empty. A benchmark found that, so now there's a
    # test.
    for mode, kw in (("f32", dict(scan_mode="f32", i8_rerank=False)),
                     ("int8 rerank", dict(i8_rerank=True)),
                     ("f16 rerank", dict(i8_rerank=False, f16_rerank=True))):
        ixq = build(base, ids, **kw)
        try:
            for v in range(50):
                ixq.remove(int(v))
            ok = len(ixq) == N - 50
            err = ""
        except BaseException as e:
            ok, err = False, f" -- {type(e).__name__}: {e}"
        check(ok, f"delete works with {mode} storage{err}")

    real = os.path.join(_ROOT, "msmarco_embeddings_cache_v2_100000.npy")
    if os.path.exists(real):
        print("\n[9] REAL embeddings (msmarco) end-to-end")
        X = np.load(real)[:20200].astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        rb = np.ascontiguousarray(X[:20000])
        rq = np.ascontiguousarray(X[20000:20200])
        rids = np.arange(len(rb), dtype=np.uint64)
        assert_realistic(rb, rq, "msmarco")
        rtruth = exact_topk(rb, rids, rq)
        rf32 = build(rb, rids, scan_mode="f32", i8_rerank=False)
        ov = np.mean([len(set(a) & set(b)) / K
                      for a, b in zip(search_all(rf32, rq), rtruth)])
        check(ov > 0.999, f"real data, f32 exhaustive = {ov:.4f}")
        rbin = build(rb, rids)
        ov = np.mean([len(set(a) & set(b)) / K
                      for a, b in zip(search_all(rbin, rq), rtruth)])
        check(ov > 0.90, f"real data, binary + int8 rerank = {ov:.4f}")
    else:
        print("\n[9] SKIPPED: real-embedding corpus not found")

    print(f"\n{'='*60}")
    if _fail:
        print(f"{_fail} CHECK(S) FAILED")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
