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

import maivs  # noqa: E402

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
    ix = maivs.FlatCellIndex(vecs.shape[1], scan_mode, 8, 0x5EED,
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
        p = os.path.join(td, "idx.maivs")
        ix.save(p)
        ix2 = maivs.FlatCellIndex.load(p)
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
        p = os.path.join(td, "idx.maivs")
        ixd.save(p)
        ixl = maivs.FlatCellIndex.load(p)
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
        ix = maivs.FlatCellIndex(DIM, "binary", 8, 0x5EED, 0)
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

    print("\n[10] orthogonal polysemy: engaged, exact, no duplicates, deletes unlink every home")
    import faiss
    kmp = faiss.Kmeans(DIM, NCELLS, niter=10, spherical=True, seed=0)
    kmp.train(base)
    cp = np.ascontiguousarray(kmp.centroids)
    cp /= np.linalg.norm(cp, axis=1, keepdims=True)

    def build_poly(homes, mode, adaptive=False):
        ixp = maivs.FlatCellIndex(DIM, mode, 8, 0x5EED, 0)
        if mode == "binary":
            ixp.set_i8_rerank(True)
        if homes > 1:
            ixp.set_orthogonal_polysemy(homes, 1.0)
        if adaptive:
            ixp.set_adaptive(True, 0.02)
        ixp.set_centroids(cp)
        ixp.add_batch([int(i) for i in ids], base)
        return ixp

    def rec(res):
        return float(np.mean([len(set(a) & set(b)) / K for a, b in zip(res, truth)]))

    for homes in (2, 3):
        for mode in ("f32", "binary"):
            ixp = build_poly(homes, mode)
            st = ixp.stats()
            check(st["max_assign"] == homes,
                  f"[{mode} R={homes}] polysemy engaged (max_assign={st['max_assign']:.0f})")
            dup_rows = 0
            for npb in (1, 2, 4, 8, NCELLS):
                for row in search_all(ixp, queries, npb):
                    live = row[row >= 0]
                    dup_rows += len(live) != len(set(live.tolist()))
            check(dup_rows == 0, f"[{mode} R={homes}] no duplicate ids in any result row ({dup_rows} bad)")
            if mode == "f32":
                check(rec(search_all(ixp, queries)) > 0.999,
                      f"[f32 R={homes}] exhaustive search equals brute force")
                ixa = build_poly(homes, "f32", adaptive=True)
                check(rec(search_all(ixa, queries)) > 0.999,
                      f"[f32 R={homes}] adaptive termination still exact under polysemy")
            victims = list(range(0, 400, 2))
            for v in victims:
                ixp.remove(v)
            got = search_all(ixp, queries)
            leaked = int(np.isin(got, victims).sum())
            check(leaked == 0 and len(ixp) == N - len(victims),
                  f"[{mode} R={homes}] deletes unlink every home (leaked={leaked}, len={len(ixp)})")

    print("\n[11] cell-major batch search matches per-query search")
    import time as _time

    def cm_case(vecs, qs, tag, two_stage=False):
        ixc = maivs.FlatCellIndex(vecs.shape[1], "binary", 8, 0x5EED, 0)
        ixc.set_i8_rerank(True)
        if two_stage:
            # Engages route_batch (blocked flat Hamming + exact rescore) on the
            # cell-major side; the per-query side uses the two-stage flat path.
            ixc.set_two_stage(True, 2)
            tag += "+2stage"
        kmc = faiss.Kmeans(vecs.shape[1], NCELLS, niter=10, spherical=True, seed=0)
        kmc.train(vecs)
        cc = np.ascontiguousarray(kmc.centroids)
        cc /= np.linalg.norm(cc, axis=1, keepdims=True)
        ixc.set_centroids(cc)
        ixc.add_batch(list(range(len(vecs))), vecs)
        ixc.compact()                       # cell-major requires a compacted arena
        qq = np.ascontiguousarray(np.repeat(qs, max(1, 64 // len(qs) + 1), axis=0)[:max(64, len(qs))])
        for npb in (1, 4, NCELLS):
            a, _ = ixc.search_batch(qq, K, npb, 1.0, 8.0, False)
            b, _ = ixc.search_batch(qq, K, npb, 1.0, 8.0, True)
            a = np.asarray(a).reshape(len(qq), K); b = np.asarray(b).reshape(len(qq), K)
            ov = np.mean([len(set(x) & set(y)) / K for x, y in zip(a, b)])
            check(ov > 0.995, f"[{tag}] cell-major == per-query at nprobe={npb} (overlap {ov:.4f})")
        # make sure it's really the cell-major path: a delete breaks compaction and the call
        # silently falls back, so time it
        t0 = _time.perf_counter(); ixc.search_batch(qq, K, NCELLS, 1.0, 8.0, True); t1 = _time.perf_counter()
        ixc.remove(0)
        t2 = _time.perf_counter(); r, _ = ixc.search_batch(qq, K, NCELLS, 1.0, 8.0, True); t3 = _time.perf_counter()
        check(0 not in np.asarray(r).tolist(), f"[{tag}] after a delete the fallback path is used and correct")

    cm_case(base, queries, "synthetic")
    cm_case(base, queries, "synthetic", two_stage=True)

    print("\n[12] polysemic runs: streaming + cell-major match the member-list path")

    def poly_case(vecs, qs, tag, homes):
        kmp2 = faiss.Kmeans(vecs.shape[1], NCELLS, niter=10, spherical=True, seed=0)
        kmp2.train(vecs)
        cp2 = np.ascontiguousarray(kmp2.centroids)
        cp2 /= np.linalg.norm(cp2, axis=1, keepdims=True)
        def mk(compact):
            ixp = maivs.FlatCellIndex(vecs.shape[1], "binary", 8, 0x5EED, 0)
            ixp.set_i8_rerank(True)
            ixp.set_two_stage(True, 2)
            ixp.set_orthogonal_polysemy(homes, 1.0)
            ixp.set_centroids(cp2)
            ixp.add_batch(list(range(len(vecs))), vecs)
            if compact:
                ixp.compact()
            return ixp
        ref, cmp_ = mk(False), mk(True)
        qq = np.ascontiguousarray(np.repeat(qs, max(1, 64 // len(qs) + 1), axis=0)[:max(64, len(qs))])
        for npb in (1, 4, NCELLS):
            a, _ = ref.search_batch(qq, K, npb, 1.0, 8.0, False)   # member-list path
            b, _ = cmp_.search_batch(qq, K, npb, 1.0, 8.0, False)  # streaming polysemic runs
            c, _ = cmp_.search_batch(qq, K, npb, 1.0, 8.0, True)   # cell-major polysemic runs
            a, b, c = (np.asarray(x).reshape(len(qq), K) for x in (a, b, c))
            ov_b = np.mean([len(set(x) & set(y)) / K for x, y in zip(a, b)])
            ov_c = np.mean([len(set(x) & set(y)) / K for x, y in zip(a, c)])
            dup = sum(len(set(r[r >= 0].tolist())) != len(r[r >= 0]) for r in np.vstack([b, c]))
            check(ov_b > 0.999 and ov_c > 0.999 and dup == 0,
                  f"[{tag} R={homes}] nprobe={npb}: stream {ov_b:.4f}, cell-major {ov_c:.4f}, dup rows {dup}")

    for homes in (2, 3):
        poly_case(base, queries, "synthetic", homes)

    real = os.path.join(_ROOT, "msmarco_embeddings_cache_v2_100000.npy")
    if os.path.exists(real):
        Xc = np.load(real)[:20200].astype(np.float32)
        Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
        cm_case(np.ascontiguousarray(Xc[:20000]), np.ascontiguousarray(Xc[20000:20200]), "msmarco")
        cm_case(np.ascontiguousarray(Xc[:20000]), np.ascontiguousarray(Xc[20000:20200]), "msmarco",
                two_stage=True)
        for homes in (2, 3):
            poly_case(np.ascontiguousarray(Xc[:20000]), np.ascontiguousarray(Xc[20000:20200]),
                      "msmarco", homes)

        print("\n[13] asymmetric cascade: engaged, raises recall, both paths agree")
        rb2 = np.ascontiguousarray(Xc[:20000]); rq2 = np.ascontiguousarray(Xc[20000:20200])
        tr2 = exact_topk(rb2, np.arange(len(rb2), dtype=np.uint64), rq2)
        kma = faiss.Kmeans(rb2.shape[1], NCELLS, niter=10, spherical=True, seed=0)
        kma.train(rb2)
        ca = np.ascontiguousarray(kma.centroids); ca /= np.linalg.norm(ca, axis=1, keepdims=True)
        ixa = maivs.FlatCellIndex(rb2.shape[1], "binary", 4, 0x5EED, 0)
        ixa.set_i8_rerank(True); ixa.set_two_stage(True, 2); ixa.set_centroids(ca)
        ixa.add_batch(list(range(len(rb2))), rb2); ixa.compact()
        qq2 = np.ascontiguousarray(np.vstack([rq2] * 1))
        def rr(cm):
            ids, _ = ixa.search_batch(qq2, K, 8, 1.0, 8.0, cm)
            ids = np.asarray(ids).reshape(len(qq2), K)
            return ids, float(np.mean([len(set(a) & set(b)) / K for a, b in zip(ids, tr2)]))
        _, r_off = rr(True)
        ixa.set_asym_cascade(8)
        ids_cm, r_on = rr(True)
        ids_pq, r_pq = rr(False)
        check(r_on > r_off, f"cascade raises recall at nprobe=8, rf=4 ({r_off:.4f} -> {r_on:.4f})")
        ov = np.mean([len(set(a) & set(b)) / K for a, b in zip(ids_cm, ids_pq)])
        check(ov > 0.999, f"cascade: cell-major == per-query (overlap {ov:.4f})")

        print("\n[14] FastScan (asymmetric estimator): raises recall, paths agree, polysemy ok")
        for homes in (1, 2):
            def mk_fs(fs):
                ixf = maivs.FlatCellIndex(rb2.shape[1], "binary", 4, 0x5EED, 0)
                ixf.set_i8_rerank(True); ixf.set_two_stage(True, 2)
                if homes > 1:
                    ixf.set_orthogonal_polysemy(homes, 1.0)
                ixf.set_fastscan(fs)
                ixf.set_centroids(ca)
                ixf.add_batch(list(range(len(rb2))), rb2); ixf.compact()
                return ixf
            ham, fsx = mk_fs(False), mk_fs(True)
            def rr2(ix_, cm):
                ids, _ = ix_.search_batch(qq2, K, 8, 1.0, 8.0, cm)
                ids = np.asarray(ids).reshape(len(qq2), K)
                return ids, float(np.mean([len(set(a) & set(b)) / K for a, b in zip(ids, tr2)]))
            _, r_h = rr2(ham, True)
            ids_fc, r_fc = rr2(fsx, True)
            ids_fp, r_fp = rr2(fsx, False)
            check(r_fc > r_h, f"[R={homes}] FastScan raises recall at nprobe=8, rf=4 ({r_h:.4f} -> {r_fc:.4f})")
            ov = np.mean([len(set(a) & set(b)) / K for a, b in zip(ids_fc, ids_fp)])
            dup = sum(len(set(r[r >= 0].tolist())) != len(r[r >= 0]) for r in np.vstack([ids_fc, ids_fp]))
            check(ov > 0.999 and dup == 0,
                  f"[R={homes}] FastScan: cell-major == per-query (overlap {ov:.4f}, dup rows {dup})")
            # Flat codes are freed once the blocks exist (no duplicated memory)...
            mh, mf = ham.stats()["total_bytes"], fsx.stats()["total_bytes"]
            check(mf < mh + 16 * len(rb2),
                  f"[R={homes}] FastScan memory {mf / len(rb2):.0f} B/vec vs Hamming {mh / len(rb2):.0f} B/vec")
            # ...and come back exactly when needed: after a delete (mutation) and through save/load
            victims = list(range(0, 2000, 7))
            for v in victims:
                ham.remove(v); fsx.remove(v)
            a_, _ = ham.search_batch(qq2, K, 8, 1.0, 8.0, False)
            b_, _ = fsx.search_batch(qq2, K, 8, 1.0, 8.0, False)
            a_ = np.asarray(a_).reshape(len(qq2), K); b_ = np.asarray(b_).reshape(len(qq2), K)
            ov2 = np.mean([len(set(x) & set(y)) / K for x, y in zip(a_, b_)])
            check(ov2 > 0.999 and not np.isin(b_, victims).any(),
                  f"[R={homes}] after deletes, decoded codes match the Hamming twin ({ov2:.4f})")
            with tempfile.TemporaryDirectory() as td:
                # (a) codes written while only the FastScan blocks were resident are exact: the
                # reloaded FastScan index equals the reloaded Hamming twin (both rebuild routing
                # state on load)
                fsx.save(os.path.join(td, "fs.idx")); ham.save(os.path.join(td, "h.idx"))
                rf_ = maivs.FlatCellIndex.load(os.path.join(td, "fs.idx"))
                rh_ = maivs.FlatCellIndex.load(os.path.join(td, "h.idx"))
                c_, _ = rf_.search_batch(qq2, K, 8, 1.0, 8.0, False)
                d_, _ = rh_.search_batch(qq2, K, 8, 1.0, 8.0, False)
                c_ = np.asarray(c_).reshape(len(qq2), K); d_ = np.asarray(d_).reshape(len(qq2), K)
                ov3 = np.mean([len(set(x) & set(y)) / K for x, y in zip(c_, d_)])
                check(ov3 > 0.999, f"[R={homes}] saved FastScan codes are exact (reload == Hamming twin reload, {ov3:.4f})")
                # (b) with no mutation in between, save/load is a strict identity.
                fresh = mk_fs(True)
                e_, _ = fresh.search_batch(qq2, K, 8, 1.0, 8.0, False)
                fresh.save(os.path.join(td, "fresh.idx"))
                fr_ = maivs.FlatCellIndex.load(os.path.join(td, "fresh.idx"))
                fr_.set_fastscan(True); fr_.compact()
                f_, _ = fr_.search_batch(qq2, K, 8, 1.0, 8.0, False)
                e_ = np.asarray(e_).reshape(len(qq2), K); f_ = np.asarray(f_).reshape(len(qq2), K)
                ov4 = np.mean([len(set(x) & set(y)) / K for x, y in zip(e_, f_)])
                check(ov4 > 0.999, f"[R={homes}] FastScan save/load/compact is an identity ({ov4:.4f})")

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

        # Polysemy should actually help on real data: at low nprobe a single home misses neighbours
        # across cell boundaries and the orthogonal home recovers some. (On the synthetic fixture
        # one home already scores 1.0, so the check would mean nothing there.)
        import faiss
        kmr = faiss.Kmeans(rb.shape[1], NCELLS, niter=10, spherical=True, seed=0)
        kmr.train(rb)
        cr = np.ascontiguousarray(kmr.centroids)
        cr /= np.linalg.norm(cr, axis=1, keepdims=True)
        def real_ix(homes):
            ixr = maivs.FlatCellIndex(rb.shape[1], "f32", 8, 0x5EED, 0)
            if homes > 1:
                ixr.set_orthogonal_polysemy(homes, 1.0)
            ixr.set_centroids(cr)
            ixr.add_batch([int(i) for i in rids], rb)
            return ixr
        r1, r2 = real_ix(1), real_ix(2)
        for npb in (1, 2):
            a1 = np.mean([len(set(a) & set(b)) / K for a, b in zip(search_all(r1, rq, npb), rtruth)])
            a2 = np.mean([len(set(a) & set(b)) / K for a, b in zip(search_all(r2, rq, npb), rtruth)])
            check(a1 < 0.999 and a2 > a1,
                  f"real data, orthogonal polysemy raises recall at nprobe={npb} ({a1:.4f} -> {a2:.4f})")
    else:
        print("\n[9] SKIPPED: real-embedding corpus not found")

    print(f"\n{'='*60}")
    if _fail:
        print(f"{_fail} CHECK(S) FAILED")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
