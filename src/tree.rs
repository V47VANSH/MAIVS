//! centroid_tree: a dynamic-depth tree over the cell centroids.
//!
//! With a fixed cell size the leaf scan doesn't grow with N (identical 8,192-vector scans from 124k
//! to 999k), so routing is the only part that still grows. The router used to scan every centroid,
//! O(M) = O(N/c):
//!
//! ```text
//!   N       cells M    flat routing @ 64 B/centroid
//!   1M        7,804      488 KB      fine
//!   10M      78,125      4.8 MB      ~10x the leaf scan
//!   1B     7,812,500      500 MB      not workable
//! ```
//!
//! An HNSW over the centroids was 2x slower at 7,804 nodes (3,677 qps flat vs 1,874 graph at R@10
//! >= 0.95): routing wants the top 1-7% of cells, where greedy traversal visits much of the graph
//! anyway, by pointer chasing, while the flat version is a contiguous Hamming scan. So the takeaway
//! was to keep a hierarchy but make every level a flat contiguous scan, which is what this is.
//!
//! Levels are built bottom-up by clustering the level below by a factor of `fanout`, until the top
//! fits in TOP_MAX nodes. Depth is ceil(log_fanout(M / TOP_MAX)) and never set by hand: 1 level at
//! 1M, 2 at 10M, 3-4 at 1B.
//!
//! Search keeps a beam at each level, scans the beam's children as one contiguous run of codes, and
//! descends. Routing work is depth * beam * fanout * 64 B, which is O(log N) as long as beam and
//! fanout are constants, so MIN_BEAM must not scale with M.
//!
//! At 999k / 7,804 cells / 88 chapters the cells a query needs are spread over 13.4 chapters on
//! average, but ranking chapters by centroid cosine needs 32 of them to cover 95%. That ~2.4x loss
//! shows up at every level, and DEFAULT_BEAM_FRAC accounts for it.

use rayon::prelude::*;

use crate::quantize as rabitq;

/// Children per node.  64 codes x 64 B = 4 KB per expansion: one comfortable
/// sequential burst, and small enough that a wide beam stays cheap.
pub(crate) const DEFAULT_FANOUT: usize = 64;

/// Stop adding levels once the top fits in this many nodes.  A scan this
/// small is free, and another level would only add routing error.
pub(crate) const TOP_MAX: usize = 256;

/// Beam floor.  **This constant is the O(log N) claim.**  It must not grow
/// with M; if it ever does, routing goes back to linear.
pub(crate) const MIN_BEAM: usize = 32;

/// Beam at a level, as a fraction of the nodes wanted at the level below.
///
/// The obvious need / fanout (assuming the wanted cells sit in need/fanout parents) is wrong. At
/// 999k with 88 chapters of ~89 cells:
///
/// ```text
///   cells wanted     chapters they occupy     chapters to probe for 95%
///          64             13.4                        32
///         256             36.5                        64
/// ```
///
/// 64 cells occupy 13.4 parents, not 64/89 = 0.7, because neighbouring cells scatter across parents
/// instead of nesting. Probing 256/4 = 64 chapters gets 95% coverage, so the beam is a fraction of
/// need, independent of fanout. need/fanout starved the beam (48 of 488 nodes at fanout=16) and
/// cost 1.7 recall points at 1M.
///
/// 0.5 rather than the 0.25 the table suggests: at 999k, serial qps at nprobe=512 went 823 (0.25)
/// -> 883 (0.50) -> 893 (0.75) and recall went up too (0.9836 -> 0.9859 -> 0.9859). A wider beam at
/// one level saves wasted probes further down.
pub(crate) const DEFAULT_BEAM_FRAC: f32 = 0.5;

/// Below this many cells a flat centroid scan beats any hierarchy (7,804 x 64 B is only 488 KB and
/// streams).
pub(crate) const MIN_CELLS_FOR_TREE: usize = 2048;

/// One level of the tree. Node i's children are child_idx[child_start[i] .. child_start[i] +
/// child_len[i]], indexing the next level down (or leaf positions for the finest level).
struct Level {
    n: usize,
    /// Row-major node centroids, `n * dim`.  Kept for exact re-scoring.
    cents: Vec<f32>,
    /// Row-major 1-bit codes, `n * n_words`.  What the scan actually reads.
    codes: Vec<u64>,
    child_start: Vec<u32>,
    child_len: Vec<u32>,
    child_idx: Vec<u32>,
}

pub(crate) struct CentroidTree {
    dim: usize,
    n_words: usize,
    fanout: usize,
    beam_frac: f32,
    /// Coarsest first; `levels.last()` has leaf positions as children.
    levels: Vec<Level>,
    /// Codes for the cells themselves, by leaf position.
    leaf_codes: Vec<u64>,
    /// Leaf position -> caller's cell id.
    leaf_ids: Vec<u32>,
    n_leaf: usize,
}

/// Per-query scratch so the descent allocates nothing.
pub(crate) struct TreeScratch {
    frontier: Vec<u32>,
    /// (Hamming distance, node); integer distances, so selection is a counting pass
    scored: Vec<(u32, u32)>,
    expand: Vec<u32>,
    hist: Vec<u32>,
}

impl TreeScratch {
    pub(crate) fn new() -> Self {
        TreeScratch { frontier: Vec::new(), scored: Vec::new(), expand: Vec::new(),
                      hist: Vec::new() }
    }
}

/// Keep the `keep` entries with the smallest Hamming distance, in O(n).
///
/// Distances are integers in [0, bits], so one histogram pass finds the cut-off and a second pass
/// compacts the survivors, with no sort and no float compares. Ties at the cut-off are kept in
/// encounter order (select_nth kept an arbitrary subset; either is valid).
fn keep_smallest(v: &mut Vec<(u32, u32)>, keep: usize, hist: &mut Vec<u32>, bits: usize) {
    if keep >= v.len() {
        return;
    }
    if hist.len() < bits + 1 {
        hist.resize(bits + 1, 0);
    }
    for &(h, _) in v.iter() {
        hist[h as usize] += 1;
    }
    let mut cum = 0usize;
    let mut t = 0usize;
    while t < bits {
        cum += hist[t] as usize;
        if cum >= keep {
            break;
        }
        t += 1;
    }
    if t == bits {
        cum += hist[t] as usize;
    }
    let mut need_eq = keep - (cum - hist[t] as usize);
    let mut w = 0;
    for r in 0..v.len() {
        let e = v[r];
        let h = e.0 as usize;
        if h < t || (h == t && need_eq > 0) {
            if h == t {
                need_eq -= 1;
            }
            v[w] = e;
            w += 1;
        }
    }
    v.truncate(w);
    for x in hist[..=bits].iter_mut() {
        *x = 0;
    }
}

#[inline]
fn code_of(codes: &[u64], i: usize, nw: usize) -> &[u64] {
    &codes[i * nw..(i + 1) * nw]
}

fn normalize(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// Spherical Lloyd k-means over unit vectors.
///
/// Kept simple: this clusters centroids, not data (7,804 points at 1M, 7.8M at 1B), and only runs
/// when the partition changes, not per insert.
fn kmeans_spherical(x: &[f32], n: usize, dim: usize, k: usize, iters: usize,
                    seed: u64) -> (Vec<u32>, Vec<f32>) {
    let k = k.max(1).min(n);
    let mut cents = vec![0f32; k * dim];

    // Deterministic spread-out seeding: stride through the input.  Cheap, and
    // adequate because the input is itself already a set of cluster means.
    let stride = (n / k).max(1);
    for j in 0..k {
        let src = (j * stride).min(n - 1);
        cents[j * dim..(j + 1) * dim].copy_from_slice(&x[src * dim..(src + 1) * dim]);
    }

    let mut assign = vec![0u32; n];
    let mut state = seed | 1;
    let mut next_u = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    for _ in 0..iters {
        // --- assign (parallel, read-only over centroids) ---
        assign
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, a)| {
                let v = &x[i * dim..(i + 1) * dim];
                let mut best = 0u32;
                let mut best_s = f32::NEG_INFINITY;
                for j in 0..k {
                    let c = &cents[j * dim..(j + 1) * dim];
                    let s = crate::index::dot_f32_pub(v, c);
                    if s > best_s {
                        best_s = s;
                        best = j as u32;
                    }
                }
                *a = best;
            });

        // --- recompute means ---
        let mut sums = vec![0f32; k * dim];
        let mut counts = vec![0u32; k];
        for i in 0..n {
            let j = assign[i] as usize;
            counts[j] += 1;
            let v = &x[i * dim..(i + 1) * dim];
            let s = &mut sums[j * dim..(j + 1) * dim];
            for (o, &q) in s.iter_mut().zip(v) {
                *o += q;
            }
        }
        for j in 0..k {
            if counts[j] == 0 {
                // Empty cluster: reseed onto an arbitrary point rather than
                // leaving a dead node that can never win a probe.
                let src = (next_u() as usize) % n;
                cents[j * dim..(j + 1) * dim]
                    .copy_from_slice(&x[src * dim..(src + 1) * dim]);
                continue;
            }
            let c = &mut cents[j * dim..(j + 1) * dim];
            c.copy_from_slice(&sums[j * dim..(j + 1) * dim]);
            normalize(c);
        }
    }
    (assign, cents)
}

/// Two-level balanced k-means: cluster into ~sqrt(k) groups, then subdivide each group separately.
///
/// Flat k-means is O(N * k * d) per iteration, which at 999k x 7,804 x 384 was 56.7 s of a 67.9 s
/// build, and with a fixed cell size it grows as O(N^1.5). Two levels cost O(N * sqrt(k) * d).
///
/// Slightly worse clusters than flat k-means (points can't move between top-level groups), the
/// usual hierarchical trade-off and the same one SPANN makes.
pub(crate) fn hierarchical_kmeans(x: &[f32], n: usize, dim: usize, k: usize,
                                  seed: u64) -> Vec<f32> {
    let k = k.max(1).min(n);
    if k <= 64 || n <= k * 4 {
        let (_, c) = kmeans_spherical(x, n, dim, k, 12, seed);
        return c;
    }
    let k1 = (k as f64).sqrt().ceil() as usize;
    let (assign, _) = kmeans_spherical(x, n, dim, k1, 8, seed);

    // Bucket point indices by top-level group.
    let mut groups: Vec<Vec<u32>> = vec![Vec::new(); k1];
    for (i, &a) in assign.iter().enumerate() {
        groups[a as usize].push(i as u32);
    }

    // Sub-clusters proportional to group size, so cells stay balanced even if the top level isn't.
    // Two phases, because the budget arithmetic is sequential but the clustering isn't.
    //
    // The single loop this replaces carried a running `written` counter, which forced the
    // sub-clusterings to run one after another. kmeans_spherical only parallelises over its own
    // points, and at the second level a group has ~n/sqrt(k) points (3,300 at k=90,000), not enough
    // to keep 16 cores busy. Training went from 7.8 s at 62,500 cells to over 550 s at 90,000,
    // about 25x worse than the FLOP count, which made small cells unusable at scale even though
    // they scan 2.35-2.95x fewer vectors (RESULTS_v2.md 1.3).
    //
    // Phase 1 computes each group's share and output offset; phase 2 runs the clusterings in
    // parallel into disjoint slices. Same output, since the share arithmetic is unchanged and each
    // group still gets its own deterministic seed.
    let mut subs = vec![0usize; groups.len()];
    let mut offs = vec![0usize; groups.len()];
    let mut written = 0usize;
    for (g, idx) in groups.iter().enumerate() {
        if idx.is_empty() || written >= k {
            continue;
        }
        let remaining_groups = k1 - g;
        let want = (((k - written) as f64) * (idx.len() as f64)
            / (n - groups[..g].iter().map(|v| v.len()).sum::<usize>()).max(1) as f64)
            .round() as usize;
        let sub = want.max(1).min(k - written).min(idx.len());
        let sub = if remaining_groups == 1 { (k - written).min(idx.len()).max(1) } else { sub };
        if sub == 0 {
            continue;
        }
        subs[g] = sub.min(k - written);
        offs[g] = written;
        written += subs[g];
    }

    let mut out = vec![0f32; k * dim];
    {
        // Hand each group a disjoint output slice so the clusterings can run
        // concurrently without sharing anything.
        let mut rest: &mut [f32] = &mut out;
        let mut cursor = 0usize;
        let mut chunks: Vec<(usize, &mut [f32])> = Vec::new();
        let mut order: Vec<usize> = (0..groups.len()).filter(|&g| subs[g] > 0).collect();
        order.sort_by_key(|&g| offs[g]);
        for &g in &order {
            let skip = offs[g] * dim - cursor;
            let (_, tail) = rest.split_at_mut(skip);
            let (mine, tail) = tail.split_at_mut(subs[g] * dim);
            chunks.push((g, mine));
            cursor = (offs[g] + subs[g]) * dim;
            rest = tail;
        }
        chunks.par_iter_mut().for_each(|(g, dst)| {
            let idx = &groups[*g];
            let mut buf = vec![0f32; idx.len() * dim];
            for (r, &i) in idx.iter().enumerate() {
                buf[r * dim..(r + 1) * dim]
                    .copy_from_slice(&x[i as usize * dim..(i as usize + 1) * dim]);
            }
            let (_, cs) = kmeans_spherical(
                &buf, idx.len(), dim, subs[*g], 12,
                seed ^ (*g as u64 + 1).wrapping_mul(0x9E3779B9));
            dst.copy_from_slice(&cs[..subs[*g] * dim]);
        });
    }
    // Any shortfall (empty groups) is filled from the data so no centroid row
    // is left as zeros, which would be a cell nothing can ever route to.
    let mut fill = 0usize;
    while written < k {
        let src = (fill * 7919) % n;
        out[written * dim..(written + 1) * dim]
            .copy_from_slice(&x[src * dim..(src + 1) * dim]);
        written += 1;
        fill += 1;
    }
    out
}

impl CentroidTree {
    /// Build over the live cell centroids. cell_ids[i] is the caller's id for row i of cell_cents.
    ///
    /// Returns None if a flat scan would be faster (too few cells) or the input is degenerate.
    pub(crate) fn build(
        cell_cents: &[f32],
        cell_ids: &[u32],
        dim: usize,
        rot_signs: &[Vec<f32>],
        dim_pad: usize,
        n_words: usize,
        fanout: usize,
        beam_frac: f32,
    ) -> Option<CentroidTree> {
        let m = cell_ids.len();
        // The threshold is the caller's (tree_min_cells, tunable). Checking the compile-time
        // constant here as well overrode it, so lowering it did nothing and the tree couldn't be
        // tested below 2048 cells.
        if m < 2 || n_words == 0 || cell_cents.len() != m * dim {
            return None;
        }
        let fanout = fanout.max(2);

        let code = |v: &[f32], rot: &mut Vec<f32>, tmp: &mut Vec<u64>| {
            rabitq::rotate(v, rot_signs, rot);
            rabitq::pack_signs_trunc(rot, tmp);
        };

        let mut rot = vec![0f32; dim_pad];
        let mut tmp = vec![0u64; n_words];

        let mut leaf_codes = Vec::with_capacity(m * n_words);
        for i in 0..m {
            code(&cell_cents[i * dim..(i + 1) * dim], &mut rot, &mut tmp);
            leaf_codes.extend_from_slice(&tmp);
        }

        // Build bottom-up, then reverse so the coarsest level is first.
        let mut rev: Vec<Level> = Vec::new();
        let mut cur_cents: Vec<f32> = cell_cents.to_vec();
        let mut cur_n = m;

        while cur_n > TOP_MAX {
            let k = ((cur_n + fanout - 1) / fanout).max(1);
            if k >= cur_n {
                break; // cannot coarsen any further
            }
            let (assign, parents) =
                kmeans_spherical(&cur_cents, cur_n, dim, k, 10, 0x9E3779B97F4A7C15);

            // Group children by parent into a flat CSR layout.
            let mut counts = vec![0u32; k];
            for &a in assign.iter() {
                counts[a as usize] += 1;
            }
            let mut child_start = vec![0u32; k];
            let mut acc = 0u32;
            for j in 0..k {
                child_start[j] = acc;
                acc += counts[j];
            }
            let mut fill = child_start.clone();
            let mut child_idx = vec![0u32; cur_n];
            for (i, &a) in assign.iter().enumerate() {
                let p = a as usize;
                child_idx[fill[p] as usize] = i as u32;
                fill[p] += 1;
            }

            let mut codes = Vec::with_capacity(k * n_words);
            for j in 0..k {
                code(&parents[j * dim..(j + 1) * dim], &mut rot, &mut tmp);
                codes.extend_from_slice(&tmp);
            }

            rev.push(Level {
                n: k,
                cents: parents.clone(),
                codes,
                child_start,
                child_len: counts,
                child_idx,
            });
            cur_cents = parents;
            cur_n = k;
        }

        if rev.is_empty() {
            return None;
        }
        rev.reverse();
        let mut t = CentroidTree {
            dim,
            n_words,
            fanout,
            beam_frac: if beam_frac > 0.0 { beam_frac } else { DEFAULT_BEAM_FRAC },
            levels: rev,
            leaf_codes,
            leaf_ids: cell_ids.to_vec(),
            n_leaf: m,
        };
        t.compact_layout();
        Some(t)
    }

    /// Renumber every level so a node's children are contiguous.
    ///
    /// Otherwise expanding a node gathers its children's codes from scattered positions, one cache
    /// line per child with no prefetch. Same problem the data arena had (fixing it there was 3x),
    /// and the reason a 3x reduction in bytes routed was still slower than the flat scan.
    ///
    /// After this, child_idx is the identity and expanding a node reads one sequential run.
    fn compact_layout(&mut self) {
        let d = self.levels.len();
        for i in 0..d {
            let order = self.levels[i].child_idx.clone();
            if i + 1 < d {
                self.permute_level(i + 1, &order);
            } else {
                self.permute_leaves(&order);
            }
            // Children are now stored in visit order, so the indirection is
            // the identity and the reads are sequential.
            for (k, v) in self.levels[i].child_idx.iter_mut().enumerate() {
                *v = k as u32;
            }
        }
    }

    fn permute_level(&mut self, j: usize, order: &[u32]) {
        let dim = self.dim;
        let nw = self.n_words;
        let l = &mut self.levels[j];
        let n = l.n;
        if order.len() != n {
            return;
        }
        let mut cents = vec![0f32; n * dim];
        let mut codes = vec![0u64; n * nw];
        let mut new_start = vec![0u32; n];
        let mut new_len = vec![0u32; n];
        let mut new_idx: Vec<u32> = Vec::with_capacity(l.child_idx.len());
        let mut acc = 0u32;
        for (np, &old) in order.iter().enumerate() {
            let o = old as usize;
            cents[np * dim..(np + 1) * dim]
                .copy_from_slice(&l.cents[o * dim..(o + 1) * dim]);
            codes[np * nw..(np + 1) * nw]
                .copy_from_slice(&l.codes[o * nw..(o + 1) * nw]);
            let st = l.child_start[o] as usize;
            let ln = l.child_len[o] as usize;
            new_start[np] = acc;
            new_len[np] = ln as u32;
            acc += ln as u32;
            new_idx.extend_from_slice(&l.child_idx[st..st + ln]);
        }
        l.cents = cents;
        l.codes = codes;
        l.child_start = new_start;
        l.child_len = new_len;
        l.child_idx = new_idx;
    }

    fn permute_leaves(&mut self, order: &[u32]) {
        let nw = self.n_words;
        let n = order.len();
        let mut codes = vec![0u64; n * nw];
        let mut ids = vec![0u32; n];
        for (np, &old) in order.iter().enumerate() {
            let o = old as usize;
            codes[np * nw..(np + 1) * nw]
                .copy_from_slice(&self.leaf_codes[o * nw..(o + 1) * nw]);
            ids[np] = self.leaf_ids[o];
        }
        self.leaf_codes = codes;
        self.leaf_ids = ids;
        self.n_leaf = n;
    }

    pub(crate) fn depth(&self) -> usize {
        self.levels.len()
    }

    pub(crate) fn n_leaf(&self) -> usize {
        self.n_leaf
    }

    /// Nodes at the coarsest level, which every query scans first.
    pub(crate) fn top_width(&self) -> usize {
        self.levels.first().map(|l| l.n).unwrap_or(0)
    }

    pub(crate) fn bytes(&self) -> usize {
        self.leaf_codes.len() * 8
            + self.leaf_ids.len() * 4
            + self
                .levels
                .iter()
                .map(|l| {
                    l.cents.len() * 4
                        + l.codes.len() * 8
                        + (l.child_start.len() + l.child_len.len() + l.child_idx.len()) * 4
                })
                .sum::<usize>()
    }

    /// Beam width per level, derived top-down from the `want` leaf cells: each level up needs about
    /// want / fanout^depth nodes, times the measured slack, and never fewer than MIN_BEAM. The
    /// floor is what keeps this O(log N).
    fn beams(&self, want: usize) -> Vec<usize> {
        let d = self.levels.len();
        let mut beams = vec![0usize; d];
        let mut need = want.max(1);
        for i in (0..d).rev() {
            let raw = (need as f32 * self.beam_frac).ceil() as usize;
            let b = raw.max(MIN_BEAM).min(self.levels[i].n);
            beams[i] = b;
            need = b;
        }
        beams
    }

    /// Descend the tree and return up to `want` candidate cell ids, best first by Hamming. The
    /// caller should re-order them exactly.
    pub(crate) fn search(&self, q_bits: &[u64], want: usize,
                         s: &mut TreeScratch, out: &mut Vec<u32>) {
        out.clear();
        if self.levels.is_empty() {
            return;
        }
        let nw = self.n_words;
        let bits = nw * 64;
        let beams = self.beams(want);

        // Start with every node at the coarsest level (at most TOP_MAX by construction).
        s.frontier.clear();
        s.frontier.extend(0..self.levels[0].n as u32);

        for (li, level) in self.levels.iter().enumerate() {
            // Score the current frontier at this level.
            s.scored.clear();
            for &node in s.frontier.iter() {
                let h = rabitq::hamming(q_bits, code_of(&level.codes, node as usize, nw));
                s.scored.push((h, node));
            }
            let keep = beams[li].min(s.scored.len());
            if keep == 0 {
                return;
            }
            keep_smallest(&mut s.scored, keep, &mut s.hist, bits);

            // Expand the kept nodes' children into the next frontier.
            s.expand.clear();
            for &(_, node) in s.scored.iter() {
                let st = level.child_start[node as usize] as usize;
                let ln = level.child_len[node as usize] as usize;
                s.expand.extend_from_slice(&level.child_idx[st..st + ln]);
            }
            std::mem::swap(&mut s.frontier, &mut s.expand);

            // After the finest level the frontier holds leaf positions.
            if li + 1 == self.levels.len() {
                break;
            }
        }

        // Final scan over candidate cells.
        s.scored.clear();
        for &leaf in s.frontier.iter() {
            let h = rabitq::hamming(q_bits, code_of(&self.leaf_codes, leaf as usize, nw));
            s.scored.push((h, leaf));
        }
        let keep = want.min(s.scored.len());
        if keep == 0 {
            return;
        }
        keep_smallest(&mut s.scored, keep, &mut s.hist, bits);
        out.extend(s.scored.iter().map(|&(_, leaf)| self.leaf_ids[leaf as usize]));
    }

    /// Nodes touched by one query for a given `want` (for the byte-budget numbers).
    pub(crate) fn probed_nodes(&self, want: usize) -> usize {
        let beams = self.beams(want);
        let mut total = self.levels[0].n;
        for i in 0..self.levels.len() {
            let expand = beams[i] * self.fanout;
            if i + 1 < self.levels.len() {
                total += expand;
            } else {
                total += expand; // leaf candidates
            }
        }
        total
    }
}
