//! graph: HNSW over the centroids, so routing doesn't have to scan every cell.
//!
//! Graphs were removed from the leaves on purpose: at ~315 vectors per cell a flat SIMD scan is
//! faster, the adjacency is bigger than the vectors, and a graph would rule out O(1) delete. None
//! of that applies to centroids: there are ~300x fewer of them, they only change on split/merge (so
//! vector deletes stay O(1)), and routing is just a top-k lookup.
//!
//! Why it might be needed: the router scans every centroid. At 1M with 999 cells that's already
//! ~30% of the per-query bytes, and at 10M it would be ~10k centroids, 15 MB of f32 per query.
//! Two-stage binary routing lowers the constant but it's still linear in the number of cells.
//!
//! Below a few thousand centroids the flat scan wins (sequential, vectorised, no pointer chasing),
//! so this only turns on past MIN_CELLS_FOR_GRAPH.

use rayon::prelude::*;
use std::collections::BinaryHeap;

/// Below this many cells, a flat SIMD scan of the centroids is faster than
/// graph traversal and this index is not built.
pub(crate) const MIN_CELLS_FOR_GRAPH: usize = 4096;

/// f32 ordered wrapper so scores can go in a BinaryHeap.
#[derive(Clone, Copy, PartialEq)]
struct Ord32(f32);
impl Eq for Ord32 {}
impl PartialOrd for Ord32 {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&o.0)
    }
}
impl Ord for Ord32 {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.partial_cmp(o).unwrap_or(std::cmp::Ordering::Equal)
    }
}

pub(crate) struct CentroidIndex {
    dim: usize,
    m: usize,
    ef_construction: usize,
    /// Row-major centroid copy; contiguous so the distance loop streams.
    vecs: Vec<f32>,
    /// `layers[l][node]` = neighbours of `node` at layer `l`.  Layer 0 holds
    /// every node; higher layers are progressively sparser.
    layers: Vec<Vec<Vec<u32>>>,
    node_level: Vec<usize>,
    entry: u32,
    n: usize,
}

impl CentroidIndex {
    #[inline]
    fn vec_of(&self, i: u32) -> &[f32] {
        let s = i as usize * self.dim;
        &self.vecs[s..s + self.dim]
    }

    #[inline]
    fn sim(&self, q: &[f32], i: u32) -> f32 {
        crate::index_v2::dot_f32_pub(q, self.vec_of(i))
    }

    /// Greedy search in one layer, returning the best `ef` nodes found.
    ///
    /// Standard HNSW layer search: a max-heap of frontier candidates and a min-heap of the current
    /// best ef, stopping when the best frontier candidate can't beat the worst kept result.
    fn search_layer(&self, q: &[f32], entries: &[u32], ef: usize, layer: usize,
                    visited: &mut [u32], stamp: u32) -> Vec<(f32, u32)> {
        let mut cand: BinaryHeap<(Ord32, u32)> = BinaryHeap::new();
        let mut best: BinaryHeap<(std::cmp::Reverse<Ord32>, u32)> = BinaryHeap::new();

        // Entry points always go into the frontier, even if already stamped. The visited array is
        // shared across layers, so a layer-0 entry point was usually marked during the descent;
        // skipping it left `cand` and `best` empty and the search returned nothing.
        for &e in entries {
            visited[e as usize] = stamp;
            let s = self.sim(q, e);
            cand.push((Ord32(s), e));
            best.push((std::cmp::Reverse(Ord32(s)), e));
        }
        while best.len() > ef {
            best.pop();
        }

        while let Some((Ord32(cs), c)) = cand.pop() {
            let worst = best.peek().map(|&(std::cmp::Reverse(Ord32(s)), _)| s);
            if let Some(w) = worst {
                if best.len() >= ef && cs < w {
                    break;
                }
            }
            for &nb in self.layers[layer][c as usize].iter() {
                if visited[nb as usize] == stamp {
                    continue;
                }
                visited[nb as usize] = stamp;
                let s = self.sim(q, nb);
                let w = best.peek().map(|&(std::cmp::Reverse(Ord32(x)), _)| x);
                if best.len() < ef || w.map_or(true, |w| s > w) {
                    cand.push((Ord32(s), nb));
                    best.push((std::cmp::Reverse(Ord32(s)), nb));
                    while best.len() > ef {
                        best.pop();
                    }
                }
            }
        }
        let mut out: Vec<(f32, u32)> = best
            .into_iter()
            .map(|(std::cmp::Reverse(Ord32(s)), i)| (s, i))
            .collect();
        out.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        out
    }

    /// Build over `centroids` (row-major, unit-norm, `n * dim`).
    pub(crate) fn build(centroids: &[f32], dim: usize, m: usize,
                        ef_construction: usize, seed: u64) -> Self {
        let n = centroids.len() / dim;
        // Level assignment: geometric with mL = 1/ln(m), the standard HNSW
        // choice giving ~1/m of the nodes at each successive layer.
        let ml = 1.0f64 / (m as f64).ln();
        let mut state = seed | 1;
        let mut next_f = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let node_level: Vec<usize> = (0..n)
            .map(|_| {
                let u: f64 = next_f().max(1e-12);
                ((-u.ln() * ml).floor() as usize).min(16)
            })
            .collect();
        let max_level = node_level.iter().copied().max().unwrap_or(0);

        let mut ix = CentroidIndex {
            dim,
            m,
            ef_construction,
            vecs: centroids.to_vec(),
            layers: (0..=max_level)
                .map(|l| {
                    (0..n)
                        .map(|i| {
                            if node_level[i] >= l {
                                Vec::with_capacity(m)
                            } else {
                                Vec::new()
                            }
                        })
                        .collect()
                })
                .collect(),
            node_level,
            entry: 0,
            n,
        };
        ix.entry = (0..n)
            .max_by_key(|&i| ix.node_level[i])
            .unwrap_or(0) as u32;

        let mut visited = vec![0u32; n];
        for i in 0..n {
            if i as u32 == ix.entry {
                continue;
            }
            let q: Vec<f32> = ix.vec_of(i as u32).to_vec();
            let li = ix.node_level[i];
            let mut ep = vec![ix.entry];

            // Descend the layers above the node's own level greedily.
            let mut l = max_level;
            while l > li {
                let r = ix.search_layer(&q, &ep, 1, l, &mut visited, (i as u32) + 1);
                if let Some(&(_, best)) = r.first() {
                    ep = vec![best];
                }
                if l == 0 { break; }
                l -= 1;
            }

            // Connect at every layer at or below the node's level.
            let mut l = li.min(max_level) as i64;
            while l >= 0 {
                let found = ix.search_layer(&q, &ep, ef_construction, l as usize,
                                            &mut visited, (i as u32) + 1);
                let take = m.min(found.len());
                let nbrs: Vec<u32> = found[..take].iter().map(|&(_, x)| x).collect();
                ix.layers[l as usize][i] = nbrs.clone();
                // Back-links, pruned to the m best so degree stays bounded.
                for &nb in &nbrs {
                    let lst = &mut ix.layers[l as usize][nb as usize];
                    if !lst.contains(&(i as u32)) {
                        lst.push(i as u32);
                    }
                }
                ep = found.iter().map(|&(_, x)| x).collect();
                if ep.is_empty() {
                    ep = vec![ix.entry];
                }
                l -= 1;
            }
        }

        // Prune over-connected nodes once, after construction.
        for l in 0..ix.layers.len() {
            for i in 0..n {
                if ix.layers[l][i].len() <= m {
                    continue;
                }
                let q: Vec<f32> = ix.vec_of(i as u32).to_vec();
                let mut nb: Vec<(f32, u32)> = ix.layers[l][i]
                    .iter()
                    .map(|&x| (ix.sim(&q, x), x))
                    .collect();
                nb.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                nb.truncate(m);
                ix.layers[l][i] = nb.into_iter().map(|(_, x)| x).collect();
            }
        }
        ix
    }

    /// Top-`k` centroids for `q`, best first.  `ef` controls the beam.
    pub(crate) fn search(&self, q: &[f32], k: usize, ef: usize,
                         visited: &mut Vec<u32>, stamp: u32) -> Vec<(u32, f32)> {
        if self.n == 0 {
            return Vec::new();
        }
        if visited.len() < self.n {
            visited.resize(self.n, 0);
        }
        let mut ep = vec![self.entry];
        let top = self.layers.len().saturating_sub(1);
        let mut l = top;
        while l > 0 {
            let r = self.search_layer(q, &ep, 1, l, visited, stamp);
            if let Some(&(_, b)) = r.first() {
                ep = vec![b];
            }
            l -= 1;
        }
        let ef = ef.max(k);
        let mut r = self.search_layer(q, &ep, ef, 0, visited, stamp);
        r.truncate(k);
        r.into_iter().map(|(s, i)| (i, s)).collect()
    }

    pub(crate) fn len(&self) -> usize {
        self.n
    }

    /// Approximate resident bytes, for stats.
    pub(crate) fn bytes(&self) -> usize {
        self.vecs.len() * 4
            + self
                .layers
                .iter()
                .map(|l| l.iter().map(|v| v.capacity() * 4 + 24).sum::<usize>())
                .sum::<usize>()
    }
}

/// Build in parallel-friendly form from a slice of centroid rows.
pub(crate) fn build_from_cells(centroids: &[f32], dim: usize) -> CentroidIndex {
    // M=16 / efC=100 are the usual HNSW defaults and plenty here: the centroid set is small and we
    // only need the top few tens of cells, since the exact re-order and the cell scan come after.
    let _ = rayon::current_num_threads();
    CentroidIndex::build(centroids, dim, 16, 100, 0x9E3779B9)
}
