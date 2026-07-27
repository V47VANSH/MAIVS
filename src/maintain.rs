//! maintain: online cell split / merge.
//!
//! The churn benchmark keeps recall flat under uniform churn because random deletes and inserts
//! keep the size distribution. Real workloads aren't uniform (a tenant grows, a topic trends, a
//! date range gets purged), so some cells inflate and others empty out. Latency is set by the
//! largest probed cell, so an unbalanced partition hurts tail latency well before recall.
//!
//! faiss IVF can't fix that without retraining the quantizer. SPFresh (SOSP'23) and Quake (OSDI'25)
//! showed local rebalancing is enough, since only the affected cells need to move. This is that for
//! the flat-cell index.
//!
//! Scan work per query is the sum of the probed cell sizes, so with nprobe roughly fixed it follows
//! the mean probed size, which is smallest when sizes are even. So instead of a full access-cost
//! model this just pushes sizes into a band:
//!
//!   * size > target * split_ratio  -> split in two with spherical 2-means
//!   * size < target * merge_ratio  -> merge into the nearest surviving cell
//!
//! Emptied cells are left in place. slot_cell and slot_cells point at cell indices, so removing
//! cells would mean rewriting every slot's back-pointer; an empty cell only costs one centroid dot
//! during routing and the scan skips it.

use crate::index_v2::{Cell, FlatCellIndex};

/// Fetch a member's vector as f32, from whichever representation exists.
fn vec_of(ix: &FlatCellIndex, slot: u32) -> Vec<f32> {
    let off = slot as usize * ix.dim;
    if ix.keep_f32 {
        ix.vecs_f32[off..off + ix.dim].to_vec()
    } else if ix.f16_rerank {
        ix.rr_f16[off..off + ix.dim]
            .iter()
            .map(|&h| crate::index_v2::f16_to_f32(h))
            .collect()
    } else if ix.per_dim_sq {
        (0..ix.dim)
            .map(|j| (ix.scan_i8[off + j] as u8) as f32 * ix.sq_step[j] + ix.sq_min[j])
            .collect()
    } else {
        let inv = if ix.quant_scale > 0.0 { 1.0 / ix.quant_scale } else { 1.0 };
        ix.scan_i8[off..off + ix.dim]
            .iter()
            .map(|&x| x as f32 * inv)
            .collect()
    }
}

fn normalize(v: &mut [f32]) {
    let mut n = 0.0f32;
    for &x in v.iter() {
        n += x * x;
    }
    n = n.sqrt();
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// Spherical 2-means over one cell's members.
///
/// Seeded with the two members farthest apart, which is deterministic and avoids the lopsided split
/// random seeds give on tight cells.
fn split_members(ix: &FlatCellIndex, members: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let dim = ix.dim;
    let vecs: Vec<Vec<f32>> = members.iter().map(|&s| vec_of(ix, s)).collect();

    // Farthest-point seeding: pick an extreme, then the point least like it.
    let mut mean = vec![0.0f32; dim];
    for v in &vecs {
        for j in 0..dim {
            mean[j] += v[j];
        }
    }
    normalize(&mut mean);
    let dot = |a: &[f32], b: &[f32]| -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    };
    let a_i = (0..vecs.len())
        .min_by(|&i, &j| dot(&vecs[i], &mean).partial_cmp(&dot(&vecs[j], &mean)).unwrap())
        .unwrap_or(0);
    let b_i = (0..vecs.len())
        .min_by(|&i, &j| {
            dot(&vecs[i], &vecs[a_i])
                .partial_cmp(&dot(&vecs[j], &vecs[a_i]))
                .unwrap()
        })
        .unwrap_or(0);

    let mut ca = vecs[a_i].clone();
    let mut cb = vecs[b_i].clone();
    let mut left: Vec<u32> = Vec::new();
    let mut right: Vec<u32> = Vec::new();

    for _ in 0..8 {
        left.clear();
        right.clear();
        let mut sa = vec![0.0f32; dim];
        let mut sb = vec![0.0f32; dim];
        for (i, v) in vecs.iter().enumerate() {
            if dot(v, &ca) >= dot(v, &cb) {
                left.push(members[i]);
                for j in 0..dim {
                    sa[j] += v[j];
                }
            } else {
                right.push(members[i]);
                for j in 0..dim {
                    sb[j] += v[j];
                }
            }
        }
        if left.is_empty() || right.is_empty() {
            break;
        }
        normalize(&mut sa);
        normalize(&mut sb);
        ca = sa;
        cb = sb;
    }
    (left, right)
}

/// Run one rebalancing pass.  Returns (splits, merges) performed.
pub(crate) fn maintain(
    ix: &mut FlatCellIndex,
    target: usize,
    split_ratio: f32,
    merge_ratio: f32,
) -> (usize, usize) {
    let mut splits = 0usize;
    let mut merges = 0usize;
    let dim = ix.dim;
    let hi = (target as f32 * split_ratio) as usize;
    let lo = (target as f32 * merge_ratio) as usize;

    // --- split oversized cells --- Take a snapshot of the cell list first: splitting appends
    // cells, and a fresh half is already in band, so it shouldn't be looked at again.
    let candidates: Vec<usize> = (0..ix.cells.len())
        .filter(|&c| ix.cells[c].members.len() > hi.max(2))
        .collect();

    for ci in candidates {
        let members = ix.cells[ci].members.clone();
        let (left, right) = split_members(ix, &members);
        if left.is_empty() || right.is_empty() {
            continue;
        }
        let new_ci = ix.cells.len() as u32;

        // Rebuild the two member sets and their exact sums.
        let mut sum_l = vec![0.0f32; dim];
        for &s in &left {
            let v = vec_of(ix, s);
            for j in 0..dim {
                sum_l[j] += v[j];
            }
        }
        let mut sum_r = vec![0.0f32; dim];
        for &s in &right {
            let v = vec_of(ix, s);
            for j in 0..dim {
                sum_r[j] += v[j];
            }
        }

        // Re-point every moved slot, primary and replica entries alike.
        for &s in &right {
            if ix.slot_cell[s as usize] == ci as u32 {
                ix.slot_cell[s as usize] = new_ci;
            }
            let base = s as usize * ix.max_assign;
            for j in 0..ix.max_assign {
                if ix.slot_cells[base + j] == ci as u32 {
                    ix.slot_cells[base + j] = new_ci;
                }
            }
        }

        let mut cent_l = sum_l.clone();
        normalize(&mut cent_l);
        let mut cent_r = sum_r.clone();
        normalize(&mut cent_r);

        ix.cells[ci].members = left;
        ix.cells[ci].sum = sum_l;
        ix.cells[ci].centroid = cent_l;
        ix.cells.push(Cell {
            members: right,
            sum: sum_r,
            centroid: cent_r,
            r_bound: 0.0,
            r_max: 0.0,
        });
        splits += 1;
    }

    // --- merge undersized cells --- Finding the nearest surviving cell by scanning every other
    // cell makes this O(M^2) full-dimension dots. That didn't show at ~1000 cells, but with
    // 128-vector cells at 1M (~7800 cells) it's ~61M dots per pass and grows quadratically.
    //
    // So build one graph over the live centroids and query it instead. Candidates are re-checked
    // against live membership, and anything the graph misses falls back to the exact scan, so the
    // result is the same, just faster.
    let n_under = ix
        .cells
        .iter()
        .filter(|c| !c.members.is_empty() && c.members.len() < lo.max(1))
        .count();
    let mut merge_graph = None;
    let mut merge_ids: Vec<u32> = Vec::new();
    if n_under > 0 && ix.cells.len() > 1024 {
        let live: Vec<usize> = (0..ix.cells.len())
            .filter(|&i| !ix.cells[i].members.is_empty())
            .collect();
        if live.len() > 1 {
            let mut flat = Vec::with_capacity(live.len() * dim);
            for &i in &live {
                flat.extend_from_slice(&ix.cells[i].centroid);
            }
            merge_graph = Some(crate::centroid_index::build_from_cells(&flat, dim));
            merge_ids = live.iter().map(|&i| i as u32).collect();
        }
    }
    let mut gvis: Vec<u32> = Vec::new();
    let mut gstamp: u32 = 0;

    for ci in 0..ix.cells.len() {
        let n = ix.cells[ci].members.len();
        if n == 0 || n >= lo.max(1) {
            continue;
        }
        // Nearest non-trivial neighbour by centroid cosine.
        let mut best = usize::MAX;
        let mut best_s = f32::NEG_INFINITY;

        if let Some(g) = merge_graph.as_ref() {
            gstamp = gstamp.wrapping_add(1).max(1);
            let q = ix.cells[ci].centroid.clone();
            for (node, _) in g.search(&q, None, 16, 64, &mut gvis, gstamp) {
                let cj = merge_ids[node as usize] as usize;
                if cj == ci || ix.cells[cj].members.is_empty() {
                    continue;
                }
                let s: f32 = q.iter().zip(ix.cells[cj].centroid.iter())
                    .map(|(a, b)| a * b).sum();
                if s > best_s {
                    best_s = s;
                    best = cj;
                }
            }
        }
        // No graph, or every candidate it returned has since been merged away.
        if best == usize::MAX {
            for cj in 0..ix.cells.len() {
                if cj == ci || ix.cells[cj].members.is_empty() {
                    continue;
                }
                let s: f32 = ix.cells[ci]
                    .centroid
                    .iter()
                    .zip(ix.cells[cj].centroid.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                if s > best_s {
                    best_s = s;
                    best = cj;
                }
            }
        }
        if best == usize::MAX {
            continue;
        }

        let moving = std::mem::take(&mut ix.cells[ci].members);
        for &s in &moving {
            if ix.slot_cell[s as usize] == ci as u32 {
                ix.slot_cell[s as usize] = best as u32;
            }
            let base = s as usize * ix.max_assign;
            for j in 0..ix.max_assign {
                if ix.slot_cells[base + j] == ci as u32 {
                    ix.slot_cells[base + j] = best as u32;
                }
            }
        }
        let moved_sum = std::mem::replace(&mut ix.cells[ci].sum, vec![0.0; dim]);
        ix.cells[best].members.extend_from_slice(&moving);
        for j in 0..dim {
            ix.cells[best].sum[j] += moved_sum[j];
        }
        let mut c = ix.cells[best].sum.clone();
        normalize(&mut c);
        ix.cells[best].centroid = c;
        ix.cells[ci].r_bound = 0.0;
        ix.cells[ci].r_max = 0.0;
        merges += 1;
    }

    ix.refresh_derived();
    (splits, merges)
}
