//! index: the flat-cell partitioned index (see AIGVS_MASTER_PLAN.md sec. 3 and Phase 2).
//!
//! ```text
//!   centroids (f32, flat SIMD scan)
//!        |  adaptive nucleus selection, capped
//!        v
//!   cells: Vec<u32> offsets into a single global arena
//!        |  flat SIMD scan, no graph
//!        v
//!   optional f32 rerank of top (k * rerank_factor)
//! ```
//!
//! No graph in the leaf: at d=384 a flat int8 scan beats an NSW traversal with ef=100 up to about
//! 10k vectors per cell, the NSW adjacency at M=64 is 544 B/node vs 384 B for the vector itself,
//! and a graph would rule out O(1) delete.
//!
//! Delete: swap_remove from the owning cell's member list plus a push onto the free list. O(1),
//! exact, no tombstones, so recall can't decay under churn.
//!
//! Scan modes: F32 scans full-precision vectors (exact within scanned cells). I8 scans a
//! global-scale int8 copy and reranks the top k * rerank_factor in f32. Having both means the cost
//! of quantization is measured, not assumed (that's the Phase 4 question).

use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use rayon::prelude::*;
use std::collections::HashMap;

use crate::quantize as rabitq;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// --- SIMD kernels ---

/// Horizontal sum of eight i32 lanes.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum_epi32(v: __m256i) -> i32 {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let s = _mm_add_epi32(lo, hi);
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_01_00_01));
    _mm_cvtsi128_si32(s)
}

/// Horizontal sum of eight f32 lanes.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum_ps(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    _mm_cvtss_f32(s)
}

/// int8 dot product, 16 lanes per iteration with widening multiply-add.
///
/// _mm256_madd_epi16 on sign-extended i8 avoids the unsigned-operand restriction of
/// _mm256_maddubs_epi16 and can't overflow: the largest partial is 127*127*2 = 32258.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
    let n = a.len();
    let mut acc = _mm256_setzero_si256();
    let mut i = 0;
    while i + 16 <= n {
        let va = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr().add(i) as *const __m128i));
        let vb = _mm256_cvtepi8_epi16(_mm_loadu_si128(b.as_ptr().add(i) as *const __m128i));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
        i += 16;
    }
    let mut s = hsum_epi32(acc);
    while i < n {
        s += *a.get_unchecked(i) as i32 * *b.get_unchecked(i) as i32;
        i += 1;
    }
    s
}

/// f32 dot product, 8 lanes/iteration, FMA when available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(va, vb, acc);
        i += 8;
    }
    let mut s = hsum_ps(acc);
    while i < n {
        s += a.get_unchecked(i) * b.get_unchecked(i);
        i += 1;
    }
    s
}

/// Asymmetric dot: f32 query against an int8 document vector.
///
/// Quantizing the query too throws away precision we get for free (it's one vector). Keeping it f32
/// costs no memory and no extra bytes read, since the documents stay int8. In isolation: 98.90%
/// top-10 agreement with exact vs 98.45% for symmetric int8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_i8_avx2(q: &[f32], v: &[i8]) -> f32 {
    let n = v.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let vi = _mm_loadl_epi64(v.as_ptr().add(i) as *const __m128i);
        let vf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vi));
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(q.as_ptr().add(i)), vf, acc);
        i += 8;
    }
    let mut s = hsum_ps(acc);
    while i < n {
        s += *q.get_unchecked(i) * (*v.get_unchecked(i) as f32);
        i += 1;
    }
    s
}

#[inline(always)]
fn dot_f32_i8(q: &[f32], v: &[i8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_f32_i8_avx2(q, v) };
        }
    }
    q.iter().zip(v).map(|(&x, &y)| x * y as f32).sum()
}

/// IEEE-754 binary16 conversion (round-to-nearest-even), scalar.
/// Only used on the insert path, which is not latency critical.
#[inline]
pub(crate) fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let mut exp = ((b >> 23) & 0xff) as i32-127 + 15;
    let mant = b & 0x7f_ffff;
    if exp >= 0x1f {
        return sign | 0x7c00; // inf / overflow
    }
    if exp <= 0 {
        // Subnormal or underflow; embeddings never land here in practice.
        return sign;
    }
    let mut m = (mant >> 13) as u16;
    // round to nearest even
    if (mant & 0x1000) != 0 && ((mant & 0x0fff) != 0 || (m & 1) != 0) {
        m += 1;
        if m == 0x400 {
            m = 0;
            exp += 1;
            if exp >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((exp as u16) << 10) | m
}

/// Asymmetric dot: f32 query against f16 documents.
///
/// f16 halves the rerank store compared to f32 and keeps ~11 mantissa bits, well above int8.
/// _mm256_cvtph_ps (F16C) widens 8 halves per instruction, so the loop stays bandwidth-bound.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_f32_f16_avx2(q: &[f32], v: &[u16]) -> f32 {
    let n = v.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let h = _mm_loadu_si128(v.as_ptr().add(i) as *const __m128i);
        let vf = _mm256_cvtph_ps(h);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(q.as_ptr().add(i)), vf, acc);
        i += 8;
    }
    let mut s = hsum_ps(acc);
    while i < n {
        s += *q.get_unchecked(i) * f16_to_f32(*v.get_unchecked(i));
        i += 1;
    }
    s
}

#[inline]
pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    if exp == 0 {
        return f32::from_bits(sign);
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
}

#[inline(always)]
fn dot_f32_f16(q: &[f32], v: &[u16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
        {
            return unsafe { dot_f32_f16_avx2(q, v) };
        }
    }
    q.iter().zip(v).map(|(&a, &b)| a * f16_to_f32(b)).sum()
}

/// Asymmetric dot for per-dimension affine SQ8 codes.
///
/// Codes are u8 over a per-dimension range [vmin_d, vmax_d]:
///     x_d ~= code_d/255 * diff_d + vmin_d
/// so  <q,x> = sum_d q_d*diff_d/255 * code_d + <q, vmin>
/// The caller folds q_d*diff_d/255 into `qs` and adds <q,vmin> once per query, which leaves a plain
/// f32 x u8 dot here. Same cost as global-scale int8, but each dimension gets its full 8-bit range.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_u8_avx2(qs: &[f32], v: &[i8]) -> f32 {
    let n = v.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let b = _mm_loadl_epi64(v.as_ptr().add(i) as *const __m128i);
        let vf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b));
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(qs.as_ptr().add(i)), vf, acc);
        i += 8;
    }
    let mut s = hsum_ps(acc);
    while i < n {
        s += *qs.get_unchecked(i) * (*v.get_unchecked(i) as u8) as f32;
        i += 1;
    }
    s
}

#[inline(always)]
fn dot_f32_u8(qs: &[f32], v: &[i8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_f32_u8_avx2(qs, v) };
        }
    }
    qs.iter().zip(v).map(|(&a, &b)| a * (b as u8) as f32).sum()
}

#[inline(always)]
fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(&x, &y)| x as i32 * y as i32).sum()
}

#[inline(always)]
fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

#[inline(always)]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_i8_avx2(a, b) };
        }
    }
    dot_i8_scalar(a, b)
}

/// Re-export of the SIMD dot for sibling modules (graph, tree).
#[inline(always)]
pub(crate) fn dot_f32_pub(a: &[f32], b: &[f32]) -> f32 {
    dot_f32(a, b)
}

#[inline(always)]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_f32_avx2(a, b) };
        }
    }
    dot_f32_scalar(a, b)
}

// --- SIMD kernels ---

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum ScanMode {
    F32,
    I8,
    /// Rotated 1-bit signs ranked by Hamming distance, then reranked exactly. 64 B/vec at d=384
    /// (padded to 512 bits). See `rabitq`.
    Binary,
}

/// One cell: member slots plus a running exact centroid.
///
/// The centroid is kept as (sum, count) instead of an EMA. The old 0.9*old + 0.1*new let centroids
/// drift away from their members across ingest batches. sum/count is exact, O(1) per insert, and
/// has no parameter.
pub(crate) struct Cell {
    pub(crate) members: Vec<u32>,
    pub(crate) sum: Vec<f32>,
    pub(crate) centroid: Vec<f32>,
    /// Angular radius used by the early-termination bound.
    ///
    /// This is the p99 of member angles, not the max. A max radius is set by a single outlier,
    /// which makes every large cell look reachable (ranking by it scored 0.9550 vs 0.9656 for plain
    /// cosine). The <=1% of members past p99 are covered by the caller's margin.
    pub(crate) r_bound: f32,
    /// True max radius, retained for diagnostics.
    pub(crate) r_max: f32,
}

impl Cell {
    fn new(dim: usize, centroid: Vec<f32>) -> Self {
        Cell { members: Vec::new(), sum: vec![0.0; dim], centroid, r_bound: 0.0, r_max: 0.0 }
    }

    fn recompute_centroid(&mut self) {
        if self.members.is_empty() {
            return;
        }
        let mut norm = 0.0f32;
        for &v in &self.sum {
            norm += v * v;
        }
        norm = norm.sqrt();
        if norm > 1e-12 {
            for (c, &s) in self.centroid.iter_mut().zip(self.sum.iter()) {
                *c = s / norm;
            }
        }
    }
}

/// Cells shortlisted by the tree when placing a vector during ingestion.
///
/// Much more than a query uses: a misplaced vector stays misplaced, while a badly probed query only
/// loses one result. All candidates are re-scored exactly.
pub(crate) const ASSIGN_CANDIDATES: usize = 64;

/// Per-thread scratch for tree-routed ingestion, so assigning 1M vectors does
/// not perform 4M heap allocations.
pub(crate) struct AssignScratch {
    rot: Vec<f32>,
    bits: Vec<u64>,
    ts: crate::tree::TreeScratch,
    out: Vec<u32>,
    scored: Vec<(u32, f32)>,
}

impl AssignScratch {
    fn new(dim_pad: usize, n_words: usize) -> Self {
        AssignScratch {
            rot: vec![0f32; dim_pad],
            bits: vec![0u64; n_words],
            ts: crate::tree::TreeScratch::new(),
            out: Vec::new(),
            scored: Vec::new(),
        }
    }
}

/// Target vectors per cell for recommend()/autotune().
///
/// 128 was the best of {128, 256, 512, 1024} at 999k, both for how the scan grows with N and for
/// the absolute scan at R@10 0.99 (see recommend()).
pub(crate) const TARGET_CELL_SIZE: usize = 128;

#[pyclass]
pub struct FlatCellIndex {
    pub(crate) dim: usize,
    pub(crate) scan_mode: ScanMode,
    pub(crate) rerank_factor: usize,

    // --- binary scan state (ScanMode::Binary) ---
    pub(crate) dim_pad: usize,
    pub(crate) n_words: usize,
    /// Kept so the rotation can be regenerated on load; a query has to be rotated with exactly the
    /// operator the stored codes were built with. 8 bytes instead of storing the sign vectors.
    pub(crate) seed: u64,
    pub(crate) rot_signs: Vec<Vec<f32>>,

    // --- global arena: one contiguous store, cells hold u32 offsets ---
    pub(crate) bits: Vec<u64>,
    pub(crate) scan_i8: Vec<i8>,
    /// f16 rerank store, used when rerank_prec == F16.
    pub(crate) rr_f16: Vec<u16>,
    pub(crate) vecs_f32: Vec<f32>,
    pub(crate) ext_ids: Vec<u64>,
    pub(crate) slot_cell: Vec<u32>,
    pub(crate) free_slots: Vec<u32>,
    pub(crate) id_to_slot: HashMap<u64, u32>,

    pub(crate) cells: Vec<Cell>,
    /// Binary codes for the centroids, for two-stage routing.
    ///
    /// The centroid scan was the last unquantized part of a query, ~30% of its bytes (314 x 1536 B
    /// = 482 KB at d=384). Same idea as the cell scan: Hamming over binary centroids (20 KB)
    /// shortlists a few times nprobe cells, then an exact dot re-orders just the shortlist. Recall
    /// is unchanged because the exact pass decides the order.
    pub(crate) cent_bits: Vec<u64>,
    /// Per-dimension SQ8 copy of the centroids, used by the stage-2 re-order.
    ///
    /// The re-order was the last f32 read on the query path and the biggest term after the scan:
    /// nprobe * mult * 1536 B, which is 1,536 KB per query at 1M / nprobe=512, a quarter of the
    /// budget. All it does is order cells, and that order already has ~0.05 of structured error
    /// (SUMMARISER_STUDY.md section 4), so f32 precision buys nothing.
    ///
    /// At 999k on two corpora (target R@10 0.99) the probe count is the same at f32, f16, sq8, sq6
    /// and sq4, and vectors scanned differ by 3 in 66,555 (MSMARCO) and 4 in 200,497 (HotpotQA). So
    /// it's 1.22-1.24x fewer query bytes at no recall cost, and memory goes down (the f32 copy is
    /// 12 MB at 1M, this is 3 MB).
    pub(crate) cent_sq8: Vec<u8>,
    pub(crate) csq_min: Vec<f32>,
    pub(crate) csq_step: Vec<f32>,
    /// Whether the stage-2 re-order reads `cent_sq8` instead of f32 centroids.
    pub(crate) centroid_sq8: bool,
    /// HNSW over the centroids, built only past MIN_CELLS_FOR_GRAPH.  Not
    /// persisted: it is a pure function of the centroids and rebuilds fast.
    pub(crate) cent_graph: Option<crate::graph::CentroidIndex>,
    /// Dynamic-depth tree over the centroids. Depth follows from the cell count (1 level at 1M, 2
    /// at 10M, 3-4 at 1B). Derived state: rebuilt on load, not saved.
    pub(crate) cent_tree: Option<crate::tree::CentroidTree>,
    pub(crate) tree_route: bool,
    pub(crate) tree_min_cells: usize,
    pub(crate) tree_fanout: usize,
    pub(crate) tree_beam_frac: f32,
    /// Cell count the tree was last built for. The tree depends on the partition, not the data, so
    /// inserting into existing cells doesn't invalidate it; only splits and merges do.
    pub(crate) tree_cells: usize,
    /// Scan the probed cells across cores within one query.
    ///
    /// The cells a query probes are independent, so this parallelises trivially. An HNSW search
    /// can't do this because each hop depends on the previous one.
    pub(crate) par_scan: bool,
    /// Graph node index -> cell index.  The graph holds only non-empty cells,
    /// so node ids and cell ids are not the same thing.
    pub(crate) graph_ids: Vec<u32>,
    pub(crate) graph_route: bool,
    /// Cell count at which graph routing switches on.  Overridable so tests
    /// can exercise the graph without building a 4096-cell fixture.
    pub(crate) graph_min_cells: usize,
    pub(crate) two_stage: bool,
    pub(crate) shortlist_mult: usize,
    pub(crate) quant_scale: f32,
    pub(crate) n_live: usize,

    // --- redundant assignment (spilling) --- A replica is a u32 in a second cell's member list;
    // the vector isn't copied, so a replica costs 4 B. (The old engine copied whole vectors into
    // both branches.) slot_cells is a flat array with stride max_assign holding every cell a slot
    // is in (u32::MAX = unused), so delete can unlink all of them.
    pub(crate) max_assign: usize,
    pub(crate) spill_margin: f32,
    pub(crate) slot_cells: Vec<u32>,
    /// Per-slot attribute for predicate pushdown.  One u32 beside the arena;
    /// empty when unused, so the unfiltered path pays nothing.
    pub(crate) tags: Vec<u32>,
    /// Inclusive [lo, hi] predicate on `tags`, applied inside the scan.
    pub(crate) tag_filter: Option<(u32, u32)>,
    pub(crate) i8_rerank: bool,
    /// Per-dimension affine SQ8 (faiss QT_8bit style) instead of one global
    /// scale.  Same byte budget, materially better accuracy.
    pub(crate) per_dim_sq: bool,
    pub(crate) sq_min: Vec<f32>,
    pub(crate) sq_step: Vec<f32>,
    pub(crate) f16_rerank: bool,
    pub(crate) keep_f32: bool,
    pub(crate) adaptive: bool,
    pub(crate) at_margin: f32,
}

/// Per-query scratch, reused for every cell of one query.
///
/// The old flat_map(...).collect() allocated a Vec per probed cell plus one for the merge, 33
/// allocations per query at nprobe=32. Everything here is allocated once per task and cleared.
pub(crate) struct SearchBuf {
    cscores: Vec<(u32, f32)>,
    chosen: Vec<u32>,
    cands: Vec<(f32, u32)>,
    q_i8: Vec<i8>,
    rot: Vec<f32>,
    q_bits: Vec<u64>,
    /// Epoch-stamped visited set, only used with redundant assignment. With replication a slot can
    /// appear in several probed cells, and without dedup it gets scored twice and the top-k fills
    /// with duplicate ids. Bumping `epoch` avoids clearing a bitset per query. Allocated lazily so
    /// the single-home path doesn't pay for it.
    visited: Vec<u32>,
    epoch: u32,
    /// Separate visited stamps for the centroid graph; its nodes are cells, not slots.
    gvisited: Vec<u32>,
    /// Descent scratch for the centroid tree, so routing allocates nothing.
    tscratch: crate::tree::TreeScratch,
    tout: Vec<u32>,
    /// Query pre-scaled by the centroid SQ8 step, so the re-order's inner loop
    /// is a plain dot over u8 codes rather than a dequantize per element.
    cqs: Vec<f32>,
}

impl SearchBuf {
    pub(crate) fn new(dim: usize, dim_pad: usize, n_words: usize) -> Self {
        SearchBuf {
            cscores: Vec::new(),
            chosen: Vec::new(),
            cands: Vec::new(),
            q_i8: vec![0i8; dim],
            rot: vec![0f32; dim_pad],
            q_bits: vec![0u64; n_words],
            visited: Vec::new(),
            epoch: 0,
            gvisited: Vec::new(),
            tscratch: crate::tree::TreeScratch::new(),
            tout: Vec::new(),
            cqs: Vec::new(),
        }
    }
}

impl FlatCellIndex {
    /// (Re)build the centroid graph if the cell count warrants one.
    pub(crate) fn rebuild_centroid_graph(&mut self) {

        // Only non-empty cells are routable targets; merged-away cells would
        // otherwise occupy graph nodes that can never be a useful answer.
        let live: Vec<usize> = (0..self.cells.len())
            .filter(|&i| !self.cells[i].members.is_empty())
            .collect();
        if !self.graph_route || live.len() < self.graph_min_cells {
            self.cent_graph = None;
            self.graph_ids = Vec::new();
            return;
        }
        let mut flat = Vec::with_capacity(live.len() * self.dim);
        for &i in &live {
            flat.extend_from_slice(&self.cells[i].centroid);
        }
        let mut g = crate::graph::build_from_cells(&flat, self.dim);

        // Attach 1-bit codes in node order so traversal reads 64 B per node instead of 1536 B.
        // Built here rather than taken from cent_bits, because those are indexed by cell id and
        // only exist with two_stage, while graph nodes are indexed by position among live cells.
        let nw = self.n_words;
        if nw > 0 {
            let mut codes = Vec::with_capacity(live.len() * nw);
            let mut rot = vec![0f32; self.dim_pad];
            let mut tmp = vec![0u64; nw];
            for &i in &live {
                rabitq::rotate(&self.cells[i].centroid, &self.rot_signs, &mut rot);
                rabitq::pack_signs_trunc(&rot, &mut tmp);
                codes.extend_from_slice(&tmp);
            }
            g.set_codes(codes, nw);
        }

        self.cent_graph = Some(g);
        self.graph_ids = live.iter().map(|&i| i as u32).collect();
    }

    /// (Re)build the centroid tree.
    ///
    /// Only live cells become leaves; an empty, merged-away cell would just pull its ancestors'
    /// centroids toward a region with no data.
    pub(crate) fn maybe_rebuild_centroid_tree(&mut self) {
        // Compare the live cell count, not the total. The leaves are the non-empty cells, so a
        // fresh index (all cells empty, tree over all of them) and the same index after ingestion
        // (tree over live cells) are different trees. Using the total made a reloaded index route
        // differently from the saved one.
        let live = self.live_cell_count();
        if self.cent_tree.is_some() && self.tree_cells == live {
            return;
        }
        self.rebuild_centroid_tree();
    }

    #[inline]
    fn live_cell_count(&self) -> usize {
        self.cells.iter().filter(|c| !c.members.is_empty()).count()
    }

    pub(crate) fn rebuild_centroid_tree(&mut self) {
        self.tree_cells = self.live_cell_count();
        let mut live: Vec<u32> = (0..self.cells.len())
            .filter(|&i| !self.cells[i].members.is_empty())
            .map(|i| i as u32)
            .collect();
        // Fresh index: centroids are set but every cell is still empty. Build the tree anyway,
        // because ingestion routes through it (that's what makes placement O(N log M) instead of
        // O(N*M)).
        if live.is_empty() && !self.cells.is_empty() {
            live = (0..self.cells.len() as u32).collect();
        }
        if !self.tree_route || live.len() < self.tree_min_cells {
            self.cent_tree = None;
            return;
        }
        let mut flat = Vec::with_capacity(live.len() * self.dim);
        for &i in &live {
            flat.extend_from_slice(&self.cells[i as usize].centroid);
        }
        self.cent_tree = crate::tree::CentroidTree::build(
            &flat, &live, self.dim, &self.rot_signs, self.dim_pad,
            self.n_words, self.tree_fanout, self.tree_beam_frac,
        );
    }

    /// (Re)build the binary codes used by the first routing stage.
    fn rebuild_centroid_codes(&mut self) {
        let nw = self.n_words;
        let mut rot = vec![0f32; self.dim_pad];
        let mut tmp = vec![0u64; nw];
        self.cent_bits.clear();
        self.cent_bits.reserve(self.cells.len() * nw);
        for ci in 0..self.cells.len() {
            let c = self.cells[ci].centroid.clone();
            rabitq::rotate(&c, &self.rot_signs, &mut rot);
            rabitq::pack_signs_trunc(&rot, &mut tmp);
            self.cent_bits.extend_from_slice(&tmp);
        }
    }

    /// (Re)build the SQ8 centroid copy used by the stage-2 re-order.
    ///
    /// The per-dimension range is fitted over live centroids only. Merged-away cells keep a stale
    /// centroid so their indices stay valid, and letting them widen the range would coarsen the
    /// step for every real cell.
    fn rebuild_centroid_sq8(&mut self) {
        let d = self.dim;
        let mut lo = vec![f32::INFINITY; d];
        let mut hi = vec![f32::NEG_INFINITY; d];
        let mut any = false;
        for c in self.cells.iter() {
            if c.members.is_empty() {
                continue;
            }
            any = true;
            for j in 0..d {
                let v = c.centroid[j];
                if v < lo[j] {
                    lo[j] = v;
                }
                if v > hi[j] {
                    hi[j] = v;
                }
            }
        }
        if !any {
            self.cent_sq8.clear();
            self.csq_min.clear();
            self.csq_step.clear();
            return;
        }
        self.csq_step = (0..d)
            .map(|j| {
                let s = (hi[j] - lo[j]) / 255.0;
                if s > 1e-12 { s } else { 1e-12 }
            })
            .collect();
        self.csq_min = lo;
        self.cent_sq8.clear();
        self.cent_sq8.resize(self.cells.len() * d, 0);
        for ci in 0..self.cells.len() {
            let off = ci * d;
            for j in 0..d {
                let c = ((self.cells[ci].centroid[j] - self.csq_min[j])
                    / self.csq_step[j])
                    .round()
                    .clamp(0.0, 255.0) as u8;
                self.cent_sq8[off + j] = c;
            }
        }
    }

    /// Stage-2 re-order score for one cell from the SQ8 centroid copy.
    ///
    /// cqs[j] = q[j] * csq_step[j] and base = sum_j q[j] * csq_min[j] are computed once per query,
    /// so this is a plain dot over u8.
    #[inline]
    fn dot_cent_sq8(&self, ci: usize, cqs: &[f32], base: f32) -> f32 {
        let off = ci * self.dim;
        let code = &self.cent_sq8[off..off + self.dim];
        let mut acc = 0.0f32;
        for j in 0..self.dim {
            acc += code[j] as f32 * cqs[j];
        }
        acc + base
    }

    /// Score a cell for the stage-2 re-order, at whatever precision is on.
    #[inline]
    fn cent_score(&self, ci: usize, q: &[f32], cqs: &[f32], base: f32) -> f32 {
        if self.centroid_sq8 && !self.cent_sq8.is_empty() {
            self.dot_cent_sq8(ci, cqs, base)
        } else {
            dot_f32(q, &self.cells[ci].centroid)
        }
    }

    #[inline]
    fn cent_bits_of(&self, ci: usize) -> &[u64] {
        let s = ci * self.n_words;
        &self.cent_bits[s..s + self.n_words]
    }

    /// Recompute every cell's angular radius from its current members.
    ///
    /// Uses the f32 arena if present, otherwise dequantizes the int8 copy. Run after a batch
    /// insert, once centroids are updated.
    fn recompute_radii(&mut self) {
        let inv = if self.quant_scale > 0.0 { 1.0 / self.quant_scale } else { 1.0 };
        let dim = self.dim;
        let keep_f32 = self.keep_f32;
        for ci in 0..self.cells.len() {
            let mut angs: Vec<f32> = Vec::with_capacity(self.cells[ci].members.len());
            for mi in 0..self.cells[ci].members.len() {
                let slot = self.cells[ci].members[mi] as usize;
                let off = slot * dim;
                let dot = if self.f16_rerank {
                    let mut d = 0.0f32;
                    for j in 0..dim {
                        d += f16_to_f32(self.rr_f16[off + j]) * self.cells[ci].centroid[j];
                    }
                    d
                } else if self.per_dim_sq {
                    let mut d = 0.0f32;
                    for j in 0..dim {
                        let c = (self.scan_i8[off + j] as u8) as f32;
                        d += (c * self.sq_step[j] + self.sq_min[j])
                            * self.cells[ci].centroid[j];
                    }
                    d
                } else if keep_f32 {
                    let mut d = 0.0f32;
                    for j in 0..dim {
                        d += self.vecs_f32[off + j] * self.cells[ci].centroid[j];
                    }
                    d
                } else {
                    let mut d = 0.0f32;
                    for j in 0..dim {
                        d += self.scan_i8[off + j] as f32 * inv * self.cells[ci].centroid[j];
                    }
                    d
                };
                angs.push(dot.clamp(-1.0, 1.0).acos());
            }
            if angs.is_empty() {
                self.cells[ci].r_bound = 0.0;
                self.cells[ci].r_max = 0.0;
                continue;
            }
            angs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let p99 = ((angs.len() as f32 * 0.99).ceil() as usize)
                .saturating_sub(1)
                .min(angs.len() - 1);
            self.cells[ci].r_bound = angs[p99];
            self.cells[ci].r_max = angs[angs.len() - 1];
        }
    }

    /// Recompute radii and centroid codes after cells change.
    pub(crate) fn refresh_derived(&mut self) {
        self.recompute_radii();
        if self.two_stage {
            self.rebuild_centroid_codes();
        }
        if self.centroid_sq8 {
            self.rebuild_centroid_sq8();
        }
        self.rebuild_centroid_graph();
        self.maybe_rebuild_centroid_tree();
    }

    /// Reorder the arena so every cell's members are contiguous, in cell order.
    ///
    /// Slots are handed out in insertion order, so a cell's members are spread across the arena and
    /// scanning a cell is a random gather: one cache line per vector at 64 B/vec, nothing for the
    /// prefetcher to follow. At 1M a query moved ~10 MB but took 4.1 ms, ~2 GB/s, against the ~25
    /// GB/s a streaming scan should get.
    ///
    /// After compaction each cell is one sequential run, and the byte-based cost model actually
    /// matches. With 128 vectors per cell it's the difference between one 8 KB read and 128
    /// scattered cache lines.
    ///
    /// Deletes (swap_remove + free list) and later inserts scatter the layout again, so this is
    /// safe and cheap to re-run as maintenance.
    pub(crate) fn compact_arena(&mut self) {
        let old_n = self.ext_ids.len();
        if old_n == 0 {
            return;
        }
        let dim = self.dim;
        let nw = self.n_words;
        let ma = self.max_assign;

        // Placement order: cells in order, and within each cell its primary members. A replicated
        // slot appears in several member lists but can only be contiguous in one, so its primary
        // cell decides where it lives; replica entries stay as u32 references.
        let mut new_of_old = vec![u32::MAX; old_n];
        let mut old_of_new: Vec<u32> = Vec::with_capacity(self.n_live);
        for ci in 0..self.cells.len() {
            for idx in 0..self.cells[ci].members.len() {
                let s = self.cells[ci].members[idx];
                if self.slot_cell[s as usize] as usize != ci {
                    continue;
                }
                if new_of_old[s as usize] != u32::MAX {
                    continue;
                }
                new_of_old[s as usize] = old_of_new.len() as u32;
                old_of_new.push(s);
            }
        }
        // Any live slot the loop above missed (one its primary cell doesn't list) still gets a
        // slot, just not a well-placed one, instead of having its data dropped.
        for (_, &s) in self.id_to_slot.iter() {
            if new_of_old[s as usize] == u32::MAX {
                new_of_old[s as usize] = old_of_new.len() as u32;
                old_of_new.push(s);
            }
        }

        let n_new = old_of_new.len();

        // --- permute every per-slot array ---
        if self.keep_f32 && !self.vecs_f32.is_empty() {
            let mut v = vec![0f32; n_new * dim];
            for (ns, &os) in old_of_new.iter().enumerate() {
                let o = os as usize * dim;
                v[ns * dim..(ns + 1) * dim].copy_from_slice(&self.vecs_f32[o..o + dim]);
            }
            self.vecs_f32 = v;
        }
        if !self.scan_i8.is_empty() {
            let mut v = vec![0i8; n_new * dim];
            for (ns, &os) in old_of_new.iter().enumerate() {
                let o = os as usize * dim;
                v[ns * dim..(ns + 1) * dim].copy_from_slice(&self.scan_i8[o..o + dim]);
            }
            self.scan_i8 = v;
        }
        if !self.rr_f16.is_empty() {
            let mut v = vec![0u16; n_new * dim];
            for (ns, &os) in old_of_new.iter().enumerate() {
                let o = os as usize * dim;
                v[ns * dim..(ns + 1) * dim].copy_from_slice(&self.rr_f16[o..o + dim]);
            }
            self.rr_f16 = v;
        }
        if !self.bits.is_empty() && nw > 0 {
            let mut v = vec![0u64; n_new * nw];
            for (ns, &os) in old_of_new.iter().enumerate() {
                let o = os as usize * nw;
                v[ns * nw..(ns + 1) * nw].copy_from_slice(&self.bits[o..o + nw]);
            }
            self.bits = v;
        }

        let mut ext = vec![0u64; n_new];
        let mut scell = vec![0u32; n_new];
        let mut scells = vec![u32::MAX; n_new * ma];
        for (ns, &os) in old_of_new.iter().enumerate() {
            ext[ns] = self.ext_ids[os as usize];
            scell[ns] = self.slot_cell[os as usize];
            let ob = os as usize * ma;
            scells[ns * ma..(ns + 1) * ma].copy_from_slice(&self.slot_cells[ob..ob + ma]);
        }
        if !self.tags.is_empty() {
            let mut tg = vec![0u32; n_new];
            for (ns, &os) in old_of_new.iter().enumerate() {
                tg[ns] = self.tags[os as usize];
            }
            self.tags = tg;
        }
        self.ext_ids = ext;
        self.slot_cell = scell;
        self.slot_cells = scells;

        // --- remap member lists, then sort so each cell scans forwards ---
        for c in self.cells.iter_mut() {
            c.members.retain(|&m| new_of_old[m as usize] != u32::MAX);
            for m in c.members.iter_mut() {
                *m = new_of_old[*m as usize];
            }
            c.members.sort_unstable();
        }

        self.id_to_slot.clear();
        for (ns, &id) in self.ext_ids.iter().enumerate() {
            self.id_to_slot.insert(id, ns as u32);
        }
        // The arena is dense again, so nothing is free.
        self.free_slots.clear();
    }

    /// Rebuild derived state after a load.  `cent_bits` is not stored on
    /// disk because it is a pure function of the centroids.
    pub(crate) fn finish_load(&mut self) {
        if self.two_stage {
            self.rebuild_centroid_codes();
        }
        if self.centroid_sq8 {
            self.rebuild_centroid_sq8();
        }
        self.rebuild_centroid_tree();
    }

    #[inline]
    fn code_len(&self) -> usize {
        self.n_words * 64
    }

    /// Map a candidate score back to an approximate cosine, so the early-stop threshold can be
    /// compared with the reachability bound.
    ///
    /// f32 scores are already cosine. Asymmetric int8 scores are cosine times quant_scale. Binary
    /// scores are dim_pad - 2*hamming, and for randomly rotated unit vectors the expected angle is
    /// pi * hamming / dim_pad. That's an estimate, not a bound, hence the margin on the binary
    /// path.
    #[inline]
    fn score_to_cos(&self, s: f32) -> f32 {
        match self.scan_mode {
            ScanMode::F32 => s,
            ScanMode::I8 => {
                // Per-dimension SQ8 scores are already an estimate of the
                // cosine; only the globally-scaled symmetric path needs
                // dividing back out.
                if self.per_dim_sq {
                    s
                } else if self.quant_scale > 0.0 {
                    s / self.quant_scale
                } else {
                    s
                }
            }
            ScanMode::Binary => {
                let dp = self.code_len() as f32;
                let h = (dp - s) * 0.5;
                (std::f32::consts::PI * h / dp).cos()
            }
        }
    }

    /// Core single-query path.  Returns top-k as (score, slot), best first.
    fn search_one(
        &self,
        q: &[f32],
        k: usize,
        nprobe: usize,
        target_confidence: f32,
        beta: f32,
        buf: &mut SearchBuf,
    ) -> Vec<(f32, u32)> {
        // Disjoint field borrows so the quantized query and the candidate
        // buffer can be held at once.
        let SearchBuf { cscores, chosen, cands, q_i8, rot, q_bits, visited, epoch,
                        gvisited, tscratch, tout, cqs } = buf;
        // Pre-scale the query for the SQ8 centroid re-order, once per query.
        let mut cqbase = 0.0f32;
        let cent_q8 = self.centroid_sq8 && !self.cent_sq8.is_empty();
        if cent_q8 {
            if cqs.len() != self.dim {
                cqs.resize(self.dim, 0.0);
            }
            for j in 0..self.dim {
                cqs[j] = q[j] * self.csq_step[j];
                cqbase += q[j] * self.csq_min[j];
            }
        }
        let dedup = self.max_assign > 1;
        if dedup {
            if visited.len() < self.ext_ids.len() {
                visited.resize(self.ext_ids.len(), 0);
            }
            *epoch += 1;
            if *epoch == u32::MAX {
                // Wrapped: reset stamps so stale marks cannot alias.
                for v in visited.iter_mut() { *v = 0; }
                *epoch = 1;
            }
        }
        let ep = *epoch;
        // Filter check inside the scan: one u32 compare per member, before the distance is
        // computed, so a selective filter makes the query cheaper. (In a graph index, removing
        // nodes breaks the connectivity the search relies on.)
        let (flo, fhi) = self.tag_filter.unwrap_or((0, u32::MAX));
        let filtering = self.tag_filter.is_some() && !self.tags.is_empty();
        macro_rules! keep {
            ($slot:expr) => {
                !filtering || {
                    let t = self.tags[$slot as usize];
                    t >= flo && t <= fhi
                }
            };
        }
        macro_rules! seen {
            ($slot:expr) => {
                dedup && {
                    let s = $slot as usize;
                    if visited[s] == ep { true } else { visited[s] = ep; false }
                }
            };
        }

        // --- 1. centroid scan (single stage, or binary shortlist + exact) ---
        cscores.clear();
        let mut q_coded = false;
        if let Some(t) = self.cent_tree.as_ref() {
            // Tree routing: a beam descent where each level is a contiguous Hamming scan. Work is
            // depth * beam * fanout instead of O(cells) for the flat scan.
            rabitq::rotate(q, &self.rot_signs, rot);
            rabitq::pack_signs_trunc(rot, q_bits);
            q_coded = true;
            let want_c = (nprobe * self.shortlist_mult.max(2)).min(t.n_leaf());
            t.search(q_bits, want_c, tscratch, tout);
            for &ci in tout.iter() {
                if !self.cells[ci as usize].members.is_empty() {
                    // Exact f32 re-order of the shortlist decides probe order,
                    // so the coarse descent only has to get the SET right.
                    cscores.push((ci, self.cent_score(ci as usize, q, cqs, cqbase)));
                }
            }
        } else if let Some(g) = self.cent_graph.as_ref() {
            // Graph routing: greedy descent over the centroid graph instead of scanning every
            // centroid. ef is set wider than nprobe so the beam doesn't collapse onto a local
            // optimum; the cells it returns are scanned exactly, so a loose beam only costs time.
            *epoch = epoch.wrapping_add(1).max(1);
            // The beam must not grow with the graph. The old .max(g.len()/16) term made routing
            // O(M) (488 at 8k cells, 6250 at 100k), which is the linear cost the graph is supposed
            // to avoid. Size it from what was asked for (nprobe).
            let want_c = (nprobe * self.shortlist_mult.max(2)).min(g.len());
            let ef = (want_c * 2).max(128);

            // Traverse on binary codes when available; the exact re-order below decides the probe
            // order anyway, so a coarse metric only costs candidates.
            let qb = if self.n_words > 0 {
                rabitq::rotate(q, &self.rot_signs, rot);
                rabitq::pack_signs_trunc(rot, q_bits);
                q_coded = true;
                Some(&q_bits[..])
            } else {
                None
            };

            for (node, _) in g.search(q, qb, want_c, ef, gvisited, *epoch) {
                let ci = self.graph_ids[node as usize];
                if !self.cells[ci as usize].members.is_empty() {
                    // Exact f32 re-order of the shortlist only.
                    cscores.push((ci, self.cent_score(ci as usize, q, cqs, cqbase)));
                }
            }
        } else if self.two_stage && !self.cent_bits.is_empty() {
            rabitq::rotate(q, &self.rot_signs, rot);
            rabitq::pack_signs_trunc(rot, q_bits);
            q_coded = true;
            let cl = self.code_len();
            let mut coarse: Vec<(u32, f32)> = (0..self.cells.len())
                .map(|i| {
                    let h = rabitq::hamming(q_bits, self.cent_bits_of(i));
                    (i as u32, rabitq::hamming_score(h, cl))
                })
                .collect();
            let want_c = (nprobe * self.shortlist_mult).min(coarse.len());
            if want_c < coarse.len() {
                coarse.select_nth_unstable_by(
                    want_c - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
                coarse.truncate(want_c);
            }
            // Exact re-order of the shortlist only.
            for (i, _) in coarse.iter() {
                if self.cells[*i as usize].members.is_empty() {
                    continue;
                }
                cscores.push((*i, self.cent_score(*i as usize, q, cqs, cqbase)));
            }
        } else {
            for (i, c) in self.cells.iter().enumerate() {
                // Merged-away cells stay in the vector to keep cell indices
                // stable for slot back-pointers; they must not consume probes.
                if c.members.is_empty() {
                    continue;
                }
                cscores.push((i as u32, self.cent_score(i, q, cqs, cqbase)));
            }
        }
        let cap = nprobe.min(cscores.len());
        if cap == 0 {
            return Vec::new();
        }
        // Partial selection: we only need the top `cap`, not a full sort.
        if cap < cscores.len() {
            cscores.select_nth_unstable_by(cap - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
            cscores.truncate(cap);
        }
        cscores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // --- 2. nucleus selection over the capped candidates ---
        chosen.clear();
        if target_confidence >= 1.0 {
            chosen.extend(cscores.iter().map(|&(i, _)| i));
        } else {
            let max_s = cscores[0].1;
            let mut z = 0.0f32;
            for &(_, s) in cscores.iter() {
                z += ((s - max_s) * beta).exp();
            }
            let mut cum = 0.0f32;
            for &(i, s) in cscores.iter() {
                chosen.push(i);
                cum += ((s - max_s) * beta).exp() / (z + 1e-9);
                if cum >= target_confidence {
                    break;
                }
            }
        }

        // --- 2b. reachability bounds for early termination --- bound[i] is the best cosine any
        // member of chosen[i] could have, and suffix_max[i] the best bound over cells i..end. Once
        // the running k-th best beats suffix_max[i], no later cell can contribute and we stop.
        // Cells stay in centroid-cosine order, which worked better than ordering by the bound
        // (outlier radii dominate it).
        let want0 = if self.scan_mode == ScanMode::F32 {
            k
        } else {
            k * self.rerank_factor
        };
        let mut suffix: Vec<f32> = Vec::new();
        if self.adaptive {
            let n = chosen.len();
            suffix = vec![f32::NEG_INFINITY; n + 1];
            let mut bounds = vec![0.0f32; n];
            for (i, &ci) in chosen.iter().enumerate() {
                let cell = &self.cells[ci as usize];
                let cs = cscores
                    .iter()
                    .find(|&&(c, _)| c == ci)
                    .map(|&(_, s)| s)
                    .unwrap_or(0.0);
                let ang = cs.clamp(-1.0, 1.0).acos();
                bounds[i] = (ang - cell.r_bound).max(0.0).cos();
            }
            for i in (0..n).rev() {
                suffix[i] = bounds[i].max(suffix[i + 1]);
            }
        }

        // --- 3. scan into the reusable candidate buffer ---
        cands.clear();
        let mut threshold = f32::NEG_INFINITY;

        // Parallel scan within one query. Probed cells are independent, so they split across cores
        // cleanly. Not used with adaptive termination (the stop condition depends on the running
        // k-th best, so it's sequential) or with replication (needs the shared visited stamps).
        // Chunked by thread rather than by cell, since one 128-vector cell is only 8 KB of work.
        let use_par = self.par_scan && !self.adaptive && !dedup && chosen.len() >= 8;
        if use_par {
            if self.scan_mode == ScanMode::Binary && !q_coded {
                rabitq::rotate(q, &self.rot_signs, rot);
                rabitq::pack_signs_trunc(rot, q_bits);
            }
            if self.scan_mode == ScanMode::I8 && !self.per_dim_sq {
                for (o, &x) in q_i8.iter_mut().zip(q) {
                    *o = (x * self.quant_scale).round().clamp(-127.0, 127.0) as i8;
                }
            }
            let mut qs: Vec<f32> = Vec::new();
            let mut qbase = 0f32;
            if self.scan_mode == ScanMode::I8 && self.per_dim_sq {
                qs = vec![0f32; self.dim];
                for j in 0..self.dim {
                    qs[j] = q[j] * self.sq_step[j];
                    qbase += q[j] * self.sq_min[j];
                }
            }
            let dp = self.code_len();
            let nthreads = rayon::current_num_threads().max(1);
            let per = (chosen.len() + nthreads - 1) / nthreads;
            let qb: &[u64] = q_bits;
            let qi: &[i8] = q_i8;
            let parts: Vec<Vec<(f32, u32)>> = chosen
                .par_chunks(per.max(1))
                .map(|group| {
                    let mut local: Vec<(f32, u32)> = Vec::new();
                    for &ci in group {
                        for &slot in self.cells[ci as usize].members.iter() {
                            if filtering {
                                let t = self.tags[slot as usize];
                                if t < flo || t > fhi {
                                    continue;
                                }
                            }
                            let s = match self.scan_mode {
                                ScanMode::Binary => {
                                    rabitq::hamming_score(
                                        rabitq::hamming(qb, self.vec_bits(slot)), dp)
                                }
                                ScanMode::I8 => {
                                    if self.per_dim_sq {
                                        qbase + dot_f32_u8(&qs, self.vec_i8(slot))
                                    } else {
                                        dot_i8(qi, self.vec_i8(slot)) as f32
                                    }
                                }
                                ScanMode::F32 => dot_f32(q, self.vec_f32(slot)),
                            };
                            local.push((s, slot));
                        }
                    }
                    local
                })
                .collect();
            for p in parts {
                cands.extend_from_slice(&p);
            }
        } else {
        match self.scan_mode {
            ScanMode::I8 => {
                // Asymmetric per-dimension SQ8: the query stays f32, pre-scaled once per query, so
                // the inner loop is as cheap as the int8 path but the scores are already final.
                let mut qs: Vec<f32> = Vec::new();
                let mut qbase = 0f32;
                if self.per_dim_sq {
                    qs = vec![0f32; self.dim];
                    for j in 0..self.dim {
                        qs[j] = q[j] * self.sq_step[j];
                        qbase += q[j] * self.sq_min[j];
                    }
                } else {
                    for (o, &x) in q_i8.iter_mut().zip(q) {
                        *o = (x * self.quant_scale).round().clamp(-127.0, 127.0) as i8;
                    }
                }
                for (ci_i, &ci) in chosen.iter().enumerate() {
                    for &slot in self.cells[ci as usize].members.iter() {
                        if !keep!(slot) { continue; }
                        if seen!(slot) { continue; }
                        let s = if self.per_dim_sq {
                            qbase + dot_f32_u8(&qs, self.vec_i8(slot))
                        } else {
                            dot_i8(q_i8, self.vec_i8(slot)) as f32
                        };
                        cands.push((s, slot));
                    }
                    if self.adaptive {
                        // Trim to `want0` so the k-th best score is exact,
                        // then stop if no remaining cell can beat it.
                        if cands.len() >= 2 * want0 && want0 > 0 {
                            cands.select_nth_unstable_by(
                                want0-1, |a, b| b.0.partial_cmp(&a.0).unwrap());
                            cands.truncate(want0);
                            threshold = self.score_to_cos(cands[want0-1].0);
                        }
                        if threshold > f32::NEG_INFINITY
                            && suffix[ci_i + 1] < threshold - self.at_margin
                        {
                            break;
                        }
                    }
                }
            }
            ScanMode::Binary => {
                if !q_coded {
                    rabitq::rotate(q, &self.rot_signs, rot);
                    rabitq::pack_signs_trunc(rot, q_bits);
                }
                let dp = self.code_len();
                for (ci_i, &ci) in chosen.iter().enumerate() {
                    for &slot in self.cells[ci as usize].members.iter() {
                        if !keep!(slot) { continue; }
                        if seen!(slot) { continue; }
                        let h = rabitq::hamming(q_bits, self.vec_bits(slot));
                        cands.push((rabitq::hamming_score(h, dp), slot));
                    }
                    if self.adaptive {
                        // Trim to `want0` so the k-th best score is exact,
                        // then stop if no remaining cell can beat it.
                        if cands.len() >= 2 * want0 && want0 > 0 {
                            cands.select_nth_unstable_by(
                                want0-1, |a, b| b.0.partial_cmp(&a.0).unwrap());
                            cands.truncate(want0);
                            threshold = self.score_to_cos(cands[want0-1].0);
                        }
                        if threshold > f32::NEG_INFINITY
                            && suffix[ci_i + 1] < threshold - self.at_margin
                        {
                            break;
                        }
                    }
                }
            }
            ScanMode::F32 => {
                for (ci_i, &ci) in chosen.iter().enumerate() {
                    for &slot in self.cells[ci as usize].members.iter() {
                        if !keep!(slot) { continue; }
                        if seen!(slot) { continue; }
                        cands.push((dot_f32(q, self.vec_f32(slot)), slot));
                    }
                    if self.adaptive {
                        // Trim to `want0` so the k-th best score is exact,
                        // then stop if no remaining cell can beat it.
                        if cands.len() >= 2 * want0 && want0 > 0 {
                            cands.select_nth_unstable_by(
                                want0-1, |a, b| b.0.partial_cmp(&a.0).unwrap());
                            cands.truncate(want0);
                            threshold = self.score_to_cos(cands[want0-1].0);
                        }
                        if threshold > f32::NEG_INFINITY
                            && suffix[ci_i + 1] < threshold - self.at_margin
                        {
                            break;
                        }
                    }
                }
            }
        }

        }

        // --- 4. top-k, with exact rerank when the scan was quantized ---
        let want = want0.min(cands.len());
        if want == 0 {
            return Vec::new();
        }
        if want < cands.len() {
            cands.select_nth_unstable_by(want - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());
            cands.truncate(want);
        }

        let mut out: Vec<(f32, u32)> = if self.scan_mode == ScanMode::F32 && self.keep_f32 {
            cands.clone()
        } else if self.f16_rerank {
            cands
                .iter()
                .map(|&(_, slot)| {
                    let o = slot as usize * self.dim;
                    (dot_f32_f16(q, &self.rr_f16[o..o + self.dim]), slot)
                })
                .collect()
        } else if self.per_dim_sq {
            let mut qs = vec![0f32; self.dim];
            let mut base = 0f32;
            for j in 0..self.dim {
                qs[j] = q[j] * self.sq_step[j];
                base += q[j] * self.sq_min[j];
            }
            // Parallelise the rerank too: it's independent per candidate, and at rerank_factor 32
            // it's 3,200 vectors of 384 B, about as much as the scan. Leaving it sequential capped
            // the speedup.
            if use_par {
                cands
                    .par_iter()
                    .map(|&(_, slot)| (base + dot_f32_u8(&qs, self.vec_i8(slot)), slot))
                    .collect()
            } else {
                cands
                    .iter()
                    .map(|&(_, slot)| (base + dot_f32_u8(&qs, self.vec_i8(slot)), slot))
                    .collect()
            }
        } else if self.i8_rerank {
            // Asymmetric: query stays f32, documents stay int8.
            cands
                .iter()
                .map(|&(_, slot)| (dot_f32_i8(q, self.vec_i8(slot)), slot))
                .collect()
        } else if use_par {
            cands
                .par_iter()
                .map(|&(_, slot)| (dot_f32(q, self.vec_f32(slot)), slot))
                .collect()
        } else {
            cands
                .iter()
                .map(|&(_, slot)| (dot_f32(q, self.vec_f32(slot)), slot))
                .collect()
        };
        let kk = k.min(out.len());
        out.select_nth_unstable_by(kk - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());
        out.truncate(kk);
        out.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        out
    }

    /// Search from Rust without a Python token.
    ///
    /// search() is the pyo3 entry point and handles the GIL; the concurrent layer needs the same
    /// thing from a plain thread, so the body lives here and both call it. Returns (score, external
    /// id).
    pub(crate) fn search_rust(&self, q: &[f32], k: usize, nprobe: usize,
                              target_confidence: f32, beta: f32,
                              buf: &mut SearchBuf) -> Vec<(f32, u64)> {
        if self.cells.is_empty() || self.n_live == 0 {
            return Vec::new();
        }
        self.search_one(q, k, nprobe, target_confidence, beta, buf)
            .into_iter()
            .map(|(sc, slot)| (sc, self.ext_ids[slot as usize]))
            .collect()
    }

    /// A new empty index with this one's configuration and centroids.
    ///
    /// Used when sealing a segment: the new head reuses the partition instead of retraining, which
    /// is much cheaper, and centroids fitted on a representative sample work fine for later data
    /// from the same distribution.
    pub(crate) fn fork_empty(&self) -> FlatCellIndex {
        let mut f = FlatCellIndex {
            dim: self.dim,
            scan_mode: self.scan_mode,
            rerank_factor: self.rerank_factor,
            dim_pad: self.dim_pad,
            n_words: self.n_words,
            seed: self.seed,
            rot_signs: self.rot_signs.clone(),
            bits: Vec::new(),
            scan_i8: Vec::new(),
            rr_f16: Vec::new(),
            vecs_f32: Vec::new(),
            ext_ids: Vec::new(),
            slot_cell: Vec::new(),
            free_slots: Vec::new(),
            id_to_slot: HashMap::new(),
            cells: Vec::new(),
            cent_bits: Vec::new(),
            cent_sq8: Vec::new(),
            csq_min: Vec::new(),
            csq_step: Vec::new(),
            cent_graph: None,
            graph_ids: Vec::new(),
            graph_route: self.graph_route,
            graph_min_cells: self.graph_min_cells,
            cent_tree: None,
            tree_route: self.tree_route,
            tree_min_cells: self.tree_min_cells,
            tree_fanout: self.tree_fanout,
            tree_beam_frac: self.tree_beam_frac,
            tree_cells: 0,
            par_scan: self.par_scan,
            two_stage: self.two_stage,
            centroid_sq8: self.centroid_sq8,
            shortlist_mult: self.shortlist_mult,
            quant_scale: self.quant_scale,
            n_live: 0,
            max_assign: self.max_assign,
            spill_margin: self.spill_margin,
            slot_cells: Vec::new(),
            tags: Vec::new(),
            tag_filter: None,
            i8_rerank: self.i8_rerank,
            per_dim_sq: self.per_dim_sq,
            sq_min: self.sq_min.clone(),
            sq_step: self.sq_step.clone(),
            f16_rerank: self.f16_rerank,
            keep_f32: self.keep_f32,
            adaptive: self.adaptive,
            at_margin: self.at_margin,
        };
        f.cells = self.cells.iter()
            .map(|c| Cell::new(self.dim, c.centroid.clone()))
            .collect();
        f.refresh_derived();
        f
    }

    pub(crate) fn n_live(&self) -> usize { self.n_live }
    pub(crate) fn dim_of(&self) -> usize { self.dim }
    pub(crate) fn dim_pad_of(&self) -> usize { self.dim_pad }
    pub(crate) fn n_words_of(&self) -> usize { self.n_words }
    pub(crate) fn has_id(&self, id: u64) -> bool { self.id_to_slot.contains_key(&id) }

    #[inline]
    fn vec_f32(&self, slot: u32) -> &[f32] {
        let s = slot as usize * self.dim;
        &self.vecs_f32[s..s + self.dim]
    }

    #[inline]
    fn vec_bits(&self, slot: u32) -> &[u64] {
        let s = slot as usize * self.n_words;
        &self.bits[s..s + self.n_words]
    }

    #[inline]
    fn vec_i8(&self, slot: u32) -> &[i8] {
        let s = slot as usize * self.dim;
        &self.scan_i8[s..s + self.dim]
    }

    fn quantize_into(&self, v: &[f32], out: &mut [i8]) {
        for (o, &x) in out.iter_mut().zip(v) {
            *o = (x * self.quant_scale).round().clamp(-127.0, 127.0) as i8;
        }
    }

    /// Cells this vector belongs to, best first: the nearest centroid plus up to max_assign - 1
    /// more whose similarity is within spill_margin of the best. Routed through the tree, so O(log
    /// M) per vector instead of O(M).
    ///
    /// Ingestion used to scan every centroid for every vector, O(N*M), which took the build from 20
    /// s at 999 cells to 143 s at 7,804. The candidate list is much bigger than a query's: a vector
    /// placed in the wrong cell stays there, while a query that probes a slightly wrong cell only
    /// loses one result. Candidates are re-scored exactly, so the tree just has to get the right
    /// cell into the shortlist.
    fn assign_cells_routed(&self, v: &[f32], sc: &mut AssignScratch) -> Vec<u32> {
        let t = match self.cent_tree.as_ref() {
            Some(t) => t,
            None => return self.assign_cells(v),
        };
        rabitq::rotate(v, &self.rot_signs, &mut sc.rot);
        rabitq::pack_signs_trunc(&sc.rot, &mut sc.bits);
        let want = ASSIGN_CANDIDATES.max(self.max_assign * 8);
        t.search(&sc.bits, want, &mut sc.ts, &mut sc.out);
        if sc.out.is_empty() {
            return self.assign_cells(v);
        }

        // exact re-score of the shortlist; this decides placement
        sc.scored.clear();
        for &ci in sc.out.iter() {
            sc.scored.push((ci, dot_f32(v, &self.cells[ci as usize].centroid)));
        }
        if self.max_assign <= 1 {
            let mut best = (sc.scored[0].0, sc.scored[0].1);
            for &(i, s) in sc.scored.iter() {
                if s > best.1 {
                    best = (i, s);
                }
            }
            return vec![best.0];
        }
        let r = self.max_assign.min(sc.scored.len());
        sc.scored.select_nth_unstable_by(r - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
        sc.scored.truncate(r);
        sc.scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let cutoff = sc.scored[0].1 - self.spill_margin;
        sc.scored.iter().filter(|&&(_, s)| s >= cutoff).map(|&(i, _)| i).collect()
    }

    fn assign_cells(&self, v: &[f32]) -> Vec<u32> {
        let mut scored: Vec<(u32, f32)> = self
            .cells
            .iter()
            .enumerate()
            .map(|(i, c)| (i as u32, dot_f32(v, &c.centroid)))
            .collect();
        if self.max_assign <= 1 {
            let mut best = (0u32, f32::NEG_INFINITY);
            for &(i, s) in scored.iter() {
                if s > best.1 {
                    best = (i, s);
                }
            }
            return vec![best.0];
        }
        let r = self.max_assign.min(scored.len());
        scored.select_nth_unstable_by(r - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(r);
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let cutoff = scored[0].1 - self.spill_margin;
        scored
            .into_iter()
            .filter(|&(_, s)| s >= cutoff)
            .map(|(i, _)| i)
            .collect()
    }
}

#[pymethods]
impl FlatCellIndex {
    /// `scan_mode`: "f32" (exact within scanned cells) or "i8" (quantized
    /// scan + f32 rerank).  `rerank_factor` is ignored in f32 mode.
    #[new]
    #[pyo3(signature = (dim, scan_mode="f32", rerank_factor=4, seed=0x5EED, code_bits=0))]
    pub(crate) fn new(dim: usize, scan_mode: &str, rerank_factor: usize, seed: u64,
           code_bits: usize) -> PyResult<Self> {
        let mode = match scan_mode {
            "f32" => ScanMode::F32,
            "i8" => ScanMode::I8,
            "binary" => ScanMode::Binary,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "scan_mode must be 'f32', 'i8' or 'binary', got '{other}'"
                )))
            }
        };
        let dim_pad = rabitq::pad_dim(dim);
        // code_bits=0 means full length. Shorter codes cut the scan cost proportionally; after the
        // rotation any prefix is an unbiased sample, and the exact rerank fixes the coarser
        // ordering.
        let n_words = rabitq::code_words(code_bits, dim_pad);
        Ok(FlatCellIndex {
            dim,
            scan_mode: mode,
            rerank_factor: rerank_factor.max(1),
            dim_pad,
            n_words,
            // deterministic from `seed`, so the same rotation can be rebuilt on load
            seed,
            rot_signs: rabitq::make_signs(dim_pad, seed),
            bits: Vec::new(),
            scan_i8: Vec::new(),
            rr_f16: Vec::new(),
            vecs_f32: Vec::new(),
            ext_ids: Vec::new(),
            slot_cell: Vec::new(),
            free_slots: Vec::new(),
            id_to_slot: HashMap::new(),
            cells: Vec::new(),
            cent_bits: Vec::new(),
            cent_sq8: Vec::new(),
            csq_min: Vec::new(),
            csq_step: Vec::new(),
            cent_graph: None,
            graph_ids: Vec::new(),
            graph_route: true,
            graph_min_cells: crate::graph::MIN_CELLS_FOR_GRAPH,
            cent_tree: None,
            tree_route: true,
            tree_min_cells: crate::tree::MIN_CELLS_FOR_TREE,
            tree_fanout: crate::tree::DEFAULT_FANOUT,
            tree_beam_frac: crate::tree::DEFAULT_BEAM_FRAC,
            tree_cells: 0,
            par_scan: false,
            two_stage: false,
            centroid_sq8: true,
            shortlist_mult: 4,
            quant_scale: 0.0,
            n_live: 0,
            tags: Vec::new(),
            tag_filter: None,
            max_assign: 1,
            spill_margin: 0.0,
            slot_cells: Vec::new(),
            i8_rerank: false,
            per_dim_sq: false,
            sq_min: Vec::new(),
            sq_step: Vec::new(),
            f16_rerank: false,
            keep_f32: true,
            adaptive: false,
            at_margin: 0.02,
        })
    }

    /// Train cell centroids in-process with two-level balanced k-means and install them.
    ///
    /// Flat k-means is O(N*k*d) and took 56.7 s of a 67.9 s build at 1M; two-level is
    /// O(N*sqrt(k)*d). `sample` caps the training set; more than a few hundred points per centroid
    /// doesn't help.
    #[pyo3(signature = (vectors, n_cells, sample=262_144, seed=0x5EED))]
    pub(crate) fn train_centroids(&mut self, vectors: PyReadonlyArray2<f32>, n_cells: usize,
                       sample: usize, seed: u64) -> PyResult<()> {
        let arr = vectors.as_array();
        if arr.shape()[1] != self.dim {
            return Err(pyo3::exceptions::PyValueError::new_err("dim mismatch"));
        }
        let n_all = arr.shape()[0];
        let data: &[f32] = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("vectors must be C-contiguous")
        })?;

        // deterministic strided subsample over the whole corpus (not a prefix)
        let (train, n_train) = if sample > 0 && n_all > sample {
            let stride = n_all / sample;
            let mut t = Vec::with_capacity(sample * self.dim);
            for i in 0..sample {
                let s = (i * stride).min(n_all - 1);
                t.extend_from_slice(&data[s * self.dim..(s + 1) * self.dim]);
            }
            (t, sample)
        } else {
            (data.to_vec(), n_all)
        };

        let mut cents = crate::tree::hierarchical_kmeans(
            &train, n_train, self.dim, n_cells, seed);
        for c in cents.chunks_exact_mut(self.dim) {
            let nrm: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
            if nrm > 1e-12 {
                for x in c.iter_mut() {
                    *x /= nrm;
                }
            }
        }
        self.cells = cents
            .chunks_exact(self.dim)
            .map(|c| Cell::new(self.dim, c.to_vec()))
            .collect();
        self.refresh_derived();
        Ok(())
    }

    fn set_centroids(&mut self, centroids: PyReadonlyArray2<f32>) -> PyResult<()> {
        let arr = centroids.as_array();
        if arr.shape()[1] != self.dim {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "centroid dim mismatch",
            ));
        }
        self.cells = (0..arr.shape()[0])
            .map(|i| {
                let c: Vec<f32> = arr.row(i).iter().copied().collect();
                Cell::new(self.dim, c)
            })
            .collect();
        self.refresh_derived();
        Ok(())
    }

    /// Rerank against the int8 copy instead of f32.
    ///
    /// Only valid when an int8 copy exists (scan_mode="i8"), or when an int8 rerank store was
    /// requested for binary mode.
    fn set_i8_rerank(&mut self, on: bool) {
        self.i8_rerank = on;
        // Dropping the f32 arena is what saves the memory: at d=384 it's 1536 B of the 1984 B/vec.
        // Without it we're at ~448 B/vec vs ~1700 for HNSW.
        self.keep_f32 = !on;
    }

    /// Turn tree routing on or off and set its parameters.
    ///
    /// min_cells is where a tree starts beating a flat centroid scan. fanout is children per node;
    /// with the cell count it determines the depth, so there's no depth setting and no two-level
    /// limit. The tree gets deeper on its own as data grows.
    #[pyo3(signature = (on, min_cells=0, fanout=0, beam_frac=0.0))]
    pub(crate) fn set_tree_route(&mut self, on: bool, min_cells: usize, fanout: usize,
                      beam_frac: f32) {
        if beam_frac > 0.0 {
            self.tree_beam_frac = beam_frac;
        }
        self.tree_route = on;
        if min_cells > 0 {
            self.tree_min_cells = min_cells;
        }
        if fanout > 1 {
            self.tree_fanout = fanout;
        }
        self.rebuild_centroid_tree();
    }

    /// Turn graph routing over the centroids on or off.
    ///
    /// Only takes effect past MIN_CELLS_FOR_GRAPH and when no tree is active (the tree was faster,
    /// so it takes priority).
    #[pyo3(signature = (on, min_cells=0))]
    pub(crate) fn set_graph_route(&mut self, on: bool, min_cells: usize) {
        self.graph_route = on;
        if min_cells > 0 {
            self.graph_min_cells = min_cells;
        }
        self.rebuild_centroid_graph();
    }

    /// Read the stage-2 centroid re-order from an SQ8 copy instead of f32.
    ///
    /// On by default. At 999k on MSMARCO and HotpotQA the probe count for R@10 0.99 is the same at
    /// f32/f16/sq8/sq6/sq4 and vectors scanned differ by ~4 in 10^5, so it saves 1.22-1.24x of
    /// query bytes and some memory. Kept as a switch so it can be checked on new data.
    pub(crate) fn set_centroid_sq8(&mut self, on: bool) {
        self.centroid_sq8 = on;
        if on {
            self.rebuild_centroid_sq8();
        } else {
            self.cent_sq8.clear();
        }
    }

    pub(crate) fn set_two_stage(&mut self, on: bool, mult: usize) {
        self.two_stage = on;
        self.shortlist_mult = mult.max(1);
        if on {
            self.rebuild_centroid_codes();
        }
    }

    /// Enable adaptive probing.
    ///
    /// nprobe becomes an upper bound: stop as soon as no unscanned cell can hold anything better
    /// than the current k-th best. Most queries are easy (the cells holding a query's true top-10
    /// are within the first 7 centroid-ranked cells for the median query, but up to the 66th at
    /// p90), so a fixed budget overpays for most queries.
    ///
    /// `margin` is subtracted from the bound before comparing. Only the binary scan needs it,
    /// because its scores are mapped back to cosine approximately; 0 is exact for f32 and int8.
    pub(crate) fn set_adaptive(&mut self, on: bool, margin: f32) {
        self.adaptive = on;
        self.at_margin = margin;
    }

    /// Use per-dimension affine SQ8 for the rerank store.
    ///
    /// Call before the first add_batch: the ranges are fitted on the first batch and every later
    /// vector uses them.
    pub(crate) fn set_per_dim_sq(&mut self, on: bool) {
        self.per_dim_sq = on;
        if on {
            self.i8_rerank = true;
            self.f16_rerank = false;
            self.keep_f32 = false;
        }
    }

    /// Rerank against an f16 copy: 2 B/dim, vs 1 B for int8 or 4 B for f32.
    ///
    /// With a 1-bit scan the rerank precision is what limits recall: at 100k, f32 rerank reaches
    /// 0.9942 at nprobe=128 while int8 needs nprobe=314 for 0.9946. f16 keeps ~11 mantissa bits, so
    /// it should track f32 closely at half the memory.
    fn set_f16_rerank(&mut self, on: bool) {
        self.f16_rerank = on;
        if on {
            self.i8_rerank = false;
            self.keep_f32 = false;
        }
    }

    /// Set the global int8 scale. If unset it's taken from the first batch as 127 / max|component|.
    ///
    /// The old engine used a hard-coded x * 127.0, which on d=384 unit vectors (component std
    /// 0.051, max 0.277) only used 66 of 255 levels, effectively 6 bits, and cost ~3.6 recall
    /// points.
    fn set_quant_scale(&mut self, scale: f32) {
        self.quant_scale = scale;
    }

    /// Rerank depth as a multiple of k: the top k * rerank_factor by scan score are rescored
    /// exactly.
    ///
    /// Query-time only, so it can be changed on a built index and a sweep needs one build.
    ///
    /// It shouldn't really be a constant. With exhaustive probing every vector is scored by its
    /// 1-bit code and only the top k*rf are reranked, so a true neighbour is lost if its Hamming
    /// rank is above k*rf. Staying in the top 320 of 10k is easy, of 5M it isn't, which is why the
    /// recall ceiling drops with N at a fixed rf. Better to scale it with the number of vectors
    /// scanned.
    fn set_rerank_factor(&mut self, rf: usize) {
        self.rerank_factor = rf.max(1);
    }

    fn set_replication(&mut self, max_assign: usize, spill_margin: f32) {
        self.max_assign = max_assign.max(1);
        self.spill_margin = spill_margin.max(0.0);
    }

    /// Insert a batch.  Vectors must be L2-normalised.
    pub(crate) fn add_batch(
        &mut self,
        ids: Vec<u64>,
        vectors: PyReadonlyArray2<f32>,
    ) -> PyResult<()> {
        if self.cells.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "call set_centroids() before add_batch()",
            ));
        }
        let arr = vectors.as_array();
        let n = ids.len();
        if arr.shape()[0] != n || arr.shape()[1] != self.dim {
            return Err(pyo3::exceptions::PyValueError::new_err("shape mismatch"));
        }

        if self.quant_scale == 0.0 {
            let mut max_abs = 0.0f32;
            for &x in arr.iter() {
                let a = x.abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
            self.quant_scale = if max_abs > 0.0 { 127.0 / max_abs } else { 1.0 };
        }

        // Work on the caller's contiguous buffer directly; building a Vec<Vec<f32>> meant one
        // allocation per row (999k allocations and ~1.5 GB at 1M).
        let data: &[f32] = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "vectors must be C-contiguous (use np.ascontiguousarray)",
            )
        })?;

        if self.per_dim_sq && self.sq_min.is_empty() {
            // Fit [vmin_d, vmax_d] per dimension from the first batch.  A tiny
            // floor on the range avoids a divide-by-zero on constant dims.
            let mut lo = vec![f32::INFINITY; self.dim];
            let mut hi = vec![f32::NEG_INFINITY; self.dim];
            for row in data.chunks_exact(self.dim) {
                for (j, &x) in row.iter().enumerate() {
                    if x < lo[j] { lo[j] = x; }
                    if x > hi[j] { hi[j] = x; }
                }
            }
            self.sq_step = (0..self.dim)
                .map(|j| ((hi[j] - lo[j]).max(1e-9)) / 255.0)
                .collect();
            self.sq_min = lo;
        }
        // Assign in parallel (read-only over centroids), then commit serially. Placement goes
        // through the centroid tree when there is one. Scratch is per rayon task, not per vector.
        let (dp, nw) = (self.dim_pad, self.n_words);
        let assignments: Vec<Vec<u32>> = data
            .par_chunks_exact(self.dim)
            .map_init(|| AssignScratch::new(dp, nw), |sc, v| self.assign_cells_routed(v, sc))
            .collect();

        // scratch hoisted out of the loop; it runs 10^6 times at 1M
        let mut rot = vec![0f32; self.dim_pad];
        let mut bits_tmp = vec![0u64; self.n_words];
        let mut i8_tmp = vec![0i8; self.dim];
        let mut f16_tmp = vec![0u16; self.dim];

        for (i, id) in ids.into_iter().enumerate() {
            let v = &data[i * self.dim..(i + 1) * self.dim];
            let cell = assignments[i][0];

            // Encode first (immutable reads of self), then place.
            match self.scan_mode {
                ScanMode::I8 => {
                    if self.per_dim_sq {
                        // Per-dimension SQ8 for the scan itself, not just the rerank. One array
                        // serves both, so the f32 arena isn't needed (1936 -> ~448 B/vec), and the
                        // scan gets more accurate too, since a global scale wastes range on most
                        // dimensions.
                        for (j, o) in i8_tmp.iter_mut().enumerate() {
                            let c = ((v[j] - self.sq_min[j]) / self.sq_step[j])
                                .round()
                                .clamp(0.0, 255.0) as u8;
                            *o = c as i8;
                        }
                    } else {
                        for (o, &x) in i8_tmp.iter_mut().zip(v) {
                            *o = (x * self.quant_scale).round().clamp(-127.0, 127.0) as i8;
                        }
                    }
                }
                ScanMode::Binary => {
                    rabitq::rotate(v, &self.rot_signs, &mut rot);
                    rabitq::pack_signs_trunc(&rot, &mut bits_tmp);
                    if self.per_dim_sq {
                        for (j, o) in i8_tmp.iter_mut().enumerate() {
                            let c = ((v[j] - self.sq_min[j]) / self.sq_step[j])
                                .round()
                                .clamp(0.0, 255.0) as u8;
                            *o = c as i8;
                        }
                    }
                    if self.f16_rerank {
                        for (o, &x) in f16_tmp.iter_mut().zip(v) {
                            *o = f32_to_f16(x);
                        }
                    }
                    if self.i8_rerank && !self.per_dim_sq {
                        for (o, &x) in i8_tmp.iter_mut().zip(v) {
                            *o = (x * self.quant_scale).round().clamp(-127.0, 127.0) as i8;
                        }
                    }
                }
                ScanMode::F32 => {}
            }

            let slot = match self.free_slots.pop() {
                Some(s) => {
                    let off = s as usize * self.dim;
                    if self.keep_f32 {
                        self.vecs_f32[off..off + self.dim].copy_from_slice(v);
                    }
                    match self.scan_mode {
                        ScanMode::I8 => {
                            self.scan_i8[off..off + self.dim].copy_from_slice(&i8_tmp)
                        }
                        ScanMode::Binary => {
                            let bo = s as usize * self.n_words;
                            self.bits[bo..bo + self.n_words].copy_from_slice(&bits_tmp);
                            if self.i8_rerank {
                                self.scan_i8[off..off + self.dim].copy_from_slice(&i8_tmp);
                            }
                            if self.f16_rerank {
                                self.rr_f16[off..off + self.dim].copy_from_slice(&f16_tmp);
                            }
                        }
                        ScanMode::F32 => {}
                    }
                    self.ext_ids[s as usize] = id;
                    self.slot_cell[s as usize] = cell;
                    let base = s as usize * self.max_assign;
                    for j in 0..self.max_assign {
                        self.slot_cells[base + j] =
                            assignments[i].get(j).copied().unwrap_or(u32::MAX);
                    }
                    s
                }
                None => {
                    let s = self.ext_ids.len() as u32;
                    if self.keep_f32 {
                        self.vecs_f32.extend_from_slice(v);
                    }
                    match self.scan_mode {
                        ScanMode::I8 => self.scan_i8.extend_from_slice(&i8_tmp),
                        ScanMode::Binary => {
                            self.bits.extend_from_slice(&bits_tmp);
                            if self.i8_rerank {
                                self.scan_i8.extend_from_slice(&i8_tmp);
                            }
                            if self.f16_rerank {
                                self.rr_f16.extend_from_slice(&f16_tmp);
                            }
                        }
                        ScanMode::F32 => {}
                    }
                    self.ext_ids.push(id);
                    self.slot_cell.push(cell);
                    for j in 0..self.max_assign {
                        self.slot_cells
                            .push(assignments[i].get(j).copied().unwrap_or(u32::MAX));
                    }
                    s
                }
            };

            // Every assigned cell gets a u32 member entry, but only the primary adds to `sum`, so
            // centroids stay the true mean of the vectors they own and replicas aren't
            // double-counted.
            for (j, &ci) in assignments[i].iter().enumerate() {
                self.cells[ci as usize].members.push(slot);
                if j == 0 {
                    for (sv, &x) in self.cells[ci as usize].sum.iter_mut().zip(v) {
                        *sv += x;
                    }
                }
            }
            self.id_to_slot.insert(id, slot);
            self.n_live += 1;
        }

        for c in self.cells.iter_mut() {
            c.recompute_centroid();
        }
        self.refresh_derived();
        Ok(())
    }

    /// O(1) delete: swap-remove from the owning cell and free the arena slot.
    ///
    /// No tombstones, so recall doesn't decay: -0.0046 R@10 over 100% turnover, with deletes flat
    /// at 1.9 us.
    ///
    /// Locality does decay. swap_remove plus slot reuse slowly undoes the cluster-major layout, and
    /// the same 100% turnover run lost half its throughput (43,908 -> 21,494 qps) at the same
    /// recall. compact() brings it back (0.19 s for 500k vectors, 37,893 qps), so it's periodic
    /// maintenance, not a rebuild.
    pub(crate) fn remove(&mut self, id: u64) -> bool {
        let slot = match self.id_to_slot.remove(&id) {
            Some(s) => s,
            None => return false,
        };
        let primary = self.slot_cell[slot as usize] as usize;

        // subtract from the owning centroid only; replicas never contributed
        let v_start = slot as usize * self.dim;
        // Subtract using whichever representation exists. Reading scan_i8 unconditionally panicked
        // under f16 rerank, where the int8 arena is empty. Delete is the only path that needs the
        // original vector back, so it has to handle every storage mode.
        if self.keep_f32 {
            for j in 0..self.dim {
                self.cells[primary].sum[j] -= self.vecs_f32[v_start + j];
            }
        } else if self.f16_rerank {
            for j in 0..self.dim {
                self.cells[primary].sum[j] -= f16_to_f32(self.rr_f16[v_start + j]);
            }
        } else if self.per_dim_sq {
            for j in 0..self.dim {
                let c = (self.scan_i8[v_start + j] as u8) as f32;
                self.cells[primary].sum[j] -= c * self.sq_step[j] + self.sq_min[j];
            }
        } else {
            let inv = 1.0 / self.quant_scale;
            for j in 0..self.dim {
                self.cells[primary].sum[j] -= self.scan_i8[v_start + j] as f32 * inv;
            }
        }

        // Unlink from every cell holding this slot, primary and replicas.
        let base = slot as usize * self.max_assign;
        for j in 0..self.max_assign {
            let ci = self.slot_cells[base + j];
            if ci == u32::MAX {
                continue;
            }
            if let Some(pos) =
                self.cells[ci as usize].members.iter().position(|&m| m == slot)
            {
                self.cells[ci as usize].members.swap_remove(pos);
            }
            self.slot_cells[base + j] = u32::MAX;
        }
        self.cells[primary].recompute_centroid();

        self.free_slots.push(slot);
        self.n_live -= 1;
        true
    }

    /// Search one query.
    ///
    /// Single-threaded on purpose (see search_batch). nprobe caps the number of cells scanned, so
    /// adaptive probing can't blow up p99. target_confidence picks cells by cumulative softmax
    /// mass; 1.0 always uses the full nprobe.
    #[pyo3(signature = (query, k, nprobe=16, target_confidence=0.95, beta=8.0))]
    fn search(
        &self,
        py: Python,
        query: PyReadonlyArray1<f32>,
        k: usize,
        nprobe: usize,
        target_confidence: f32,
        beta: f32,
    ) -> (Vec<u64>, Vec<f32>) {
        let q: Vec<f32> = query.as_array().iter().copied().collect();
        if self.cells.is_empty() || self.n_live == 0 {
            return (vec![], vec![]);
        }
        py.allow_threads(|| {
            let mut buf = SearchBuf::new(self.dim, self.dim_pad, self.n_words);
            let out = self.search_one(&q, k, nprobe, target_confidence, beta, &mut buf);
            (
                out.iter().map(|&(_, s)| self.ext_ids[s as usize]).collect(),
                out.iter().map(|&(sc, _)| sc).collect(),
            )
        })
    }

    /// Search many queries, parallelised across queries.
    ///
    /// Parallelising within a query was the wrong granularity: with ~315 vectors per cell one cell
    /// scan is ~10 us and a rayon task costs ~2-5 us to dispatch, so at nprobe=32 we spent about as
    /// long scheduling as computing. One task per query spreads the dispatch cost over the whole
    /// query and keeps its working set in one core's cache.
    ///
    /// Returns (ids, scores), each (n_queries * k) row-major, padded with u64::MAX / -inf when a
    /// query returns fewer than k results.
    #[pyo3(signature = (queries, k, nprobe=16, target_confidence=0.95, beta=8.0))]
    fn search_batch(
        &self,
        py: Python,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        nprobe: usize,
        target_confidence: f32,
        beta: f32,
    ) -> PyResult<(Vec<u64>, Vec<f32>)> {
        let arr = queries.as_array();
        let nq = arr.shape()[0];
        if arr.shape()[1] != self.dim {
            return Err(pyo3::exceptions::PyValueError::new_err("query dim mismatch"));
        }
        let data: &[f32] = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "queries must be C-contiguous (use np.ascontiguousarray)",
            )
        })?;
        if self.cells.is_empty() || self.n_live == 0 {
            return Ok((vec![u64::MAX; nq * k], vec![f32::NEG_INFINITY; nq * k]));
        }

        let res: Vec<(Vec<u64>, Vec<f32>)> = py.allow_threads(|| {
            data.par_chunks_exact(self.dim)
                .map(|q| {
                    // One buffer per task, reused across the whole cell scan.
                    let mut buf = SearchBuf::new(self.dim, self.dim_pad, self.n_words);
                    let out =
                        self.search_one(q, k, nprobe, target_confidence, beta, &mut buf);
                    let mut ids = vec![u64::MAX; k];
                    let mut scs = vec![f32::NEG_INFINITY; k];
                    for (i, &(sc, slot)) in out.iter().enumerate() {
                        ids[i] = self.ext_ids[slot as usize];
                        scs[i] = sc;
                    }
                    (ids, scs)
                })
                .collect()
        });

        let mut ids = Vec::with_capacity(nq * k);
        let mut scs = Vec::with_capacity(nq * k);
        for (i, s) in res {
            ids.extend_from_slice(&i);
            scs.extend_from_slice(&s);
        }
        Ok((ids, scs))
    }

    pub(crate) fn stats(&self) -> HashMap<String, f64> {
        let sizes: Vec<usize> = self.cells.iter().map(|c| c.members.len()).collect();
        let n = self.n_live.max(1) as f64;
        let mut m = HashMap::new();
        m.insert("n_live".into(), self.n_live as f64);
        m.insert("n_cells".into(), self.cells.len() as f64);
        m.insert("free_slots".into(), self.free_slots.len() as f64);
        m.insert("quant_scale".into(), self.quant_scale as f64);
        m.insert("scan_bytes_per_vec".into(), match self.scan_mode {
            ScanMode::F32 => (self.dim * 4) as f64,
            ScanMode::I8 => self.dim as f64,
            ScanMode::Binary => (self.n_words * 8) as f64,
        });
        m.insert(
            "cell_size_mean".into(),
            sizes.iter().sum::<usize>() as f64 / self.cells.len().max(1) as f64,
        );
        m.insert("cell_size_max".into(), *sizes.iter().max().unwrap_or(&0) as f64);
        m.insert("max_assign".into(), self.max_assign as f64);
        m.insert(
            "centroid_graph_nodes".into(),
            self.cent_graph.as_ref().map(|g| g.len()).unwrap_or(0) as f64,
        );
        m.insert(
            "centroid_graph_bytes".into(),
            self.cent_graph.as_ref().map(|g| g.bytes()).unwrap_or(0) as f64,
        );
        // Tree: depth is derived, so reporting it is how you verify the
        // structure actually grew with the corpus rather than staying flat.
        m.insert(
            "tree_depth".into(),
            self.cent_tree.as_ref().map(|t| t.depth()).unwrap_or(0) as f64,
        );
        m.insert(
            "tree_top_width".into(),
            self.cent_tree.as_ref().map(|t| t.top_width()).unwrap_or(0) as f64,
        );
        m.insert(
            "tree_bytes".into(),
            self.cent_tree.as_ref().map(|t| t.bytes()).unwrap_or(0) as f64,
        );
        m.insert("tree_fanout".into(), self.tree_fanout as f64);
        m.insert("tree_beam_frac".into(), self.tree_beam_frac as f64);
        m.insert(
            "replication_factor".into(),
            sizes.iter().sum::<usize>() as f64 / n,
        );
        m.insert("cell_size_min".into(), *sizes.iter().min().unwrap_or(&0) as f64);
        let bytes = self.vecs_f32.len() * 4 + self.scan_i8.len()
            + self.rr_f16.len() * 2 + self.bits.len() * 8
            + self.ext_ids.len() * 12 + self.slot_cells.len() * 4;
        m.insert("arena_bytes_per_vec".into(), bytes as f64 / n);
        m
    }

    fn __len__(&self) -> usize {
        self.n_live
    }

    /// Recommended configuration for the current size: (target_cell_size, want_graph_routing).
    ///
    /// The constants come from the benchmarks:
    ///   * Once the scan is quantized, smaller cells stop helping: they win 8x in f32 (scan-bound) but
    ///     lose at 64 B/vec because the cost moves to routing (1.86 ms -> 4.40 ms at 3000 cells).
    ///   * Graph routing only pays above ~50k centroids. At 314 cells it caps recall (greedy traversal
    ///     can't reach every cell), and at 5000 it was still worse: R@10 0.9080 vs 0.9332 flat.
    fn recommend(&self) -> (usize, bool) {
        let n = self.n_live.max(1);
        // Fixed cell size instead of sqrt(N) cells.
        //
        // sqrt(N) cells means sqrt(N) vectors per cell, i.e. an O(sqrt(N)) scan. At 999k with exact
        // ground truth over an 8x range of N, a fixed cell size needs N^0.08 more vectors scanned
        // at R@10 0.95 and N^0.30 at 0.99, vs N^0.23 / N^0.46 at 1024 per cell. So cell size
        // changes the exponent, not just the constant. 128 was the best of {128, 256, 512, 1024}
        // and scanned 3.3x fewer vectors than 1024 at R@10 0.99.
        let target_cells = (n / TARGET_CELL_SIZE).max(1);
        let want_graph = target_cells >= crate::graph::MIN_CELLS_FOR_GRAPH;
        (target_cells, want_graph)
    }

    /// Re-tune the partition for the current size, in place.
    ///
    /// After a lot of inserts the cells get bigger, and scan cost grows with cell size. This splits
    /// and merges toward the target size and turns graph routing on only when the centroid count is
    /// large enough. Incremental; it never retrains the quantizer (faiss IVF would need to).
    fn autotune(&mut self) -> (usize, usize) {
        let (_target_cells, want_graph) = self.recommend();
        if want_graph != self.graph_route {
            self.graph_route = want_graph;
        }
        // The band is tied to the fixed cell size directly; deriving it from a cell count would
        // bring back the sqrt(N) dependency.
        crate::maintain::maintain(self, TARGET_CELL_SIZE, 2.0, 0.25)
    }

    /// Scan the probed cells in parallel within one query.
    ///
    /// Trades cores for latency. Off by default because for batch workloads every core is already
    /// busy with other queries; turn it on for latency-sensitive single queries.
    pub(crate) fn set_parallel_scan(&mut self, on: bool) {
        self.par_scan = on;
    }

    /// Force a full tree rebuild.
    ///
    /// Routine refreshes skip it when the partition hasn't changed, so call this after heavy churn
    /// if routing quality has drifted; centroids move even when the cell count doesn't.
    fn rebuild_tree(&mut self) {
        self.rebuild_centroid_tree();
    }

    /// Attach a u32 attribute to each id, for predicate pushdown.
    ///
    /// One flat array beside the arena, indexed by slot.  Not persisted yet.
    fn set_tags(&mut self, ids: Vec<u64>, tags: Vec<u32>) -> PyResult<()> {
        if ids.len() != tags.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("ids/tags length mismatch"));
        }
        if self.tags.len() < self.ext_ids.len() {
            self.tags.resize(self.ext_ids.len(), 0);
        }
        for (id, t) in ids.into_iter().zip(tags) {
            if let Some(&slot) = self.id_to_slot.get(&id) {
                self.tags[slot as usize] = t;
            }
        }
        Ok(())
    }

    /// Restrict subsequent searches to vectors whose tag is in [lo, hi].
    /// Pass `None` (omit both) to clear.
    #[pyo3(signature = (lo=None, hi=None))]
    fn set_filter(&mut self, lo: Option<u32>, hi: Option<u32>) {
        self.tag_filter = match (lo, hi) {
            (None, None) => None,
            (a, b) => Some((a.unwrap_or(0), b.unwrap_or(u32::MAX))),
        };
    }

    /// Reorder the arena so each cell's members are contiguous.
    ///
    /// Turns the per-cell scan from a random gather into a sequential read. Call once after bulk
    /// ingestion, and again now and then under heavy churn, since deletes and inserts slowly
    /// scatter the layout. Only moves data, never changes membership, so recall is unaffected.
    pub(crate) fn compact(&mut self) {
        self.compact_arena();
    }

    /// Rebalance the partition in place: split oversized cells, merge small ones. Returns (splits,
    /// merges).
    ///
    /// Latency is set by the largest probed cell, so skewed churn hurts the tail before it hurts
    /// recall. Run this periodically, or when stats()["cell_size_max"] drifts away from the mean.
    /// Only the affected cells move; the quantizer isn't retrained.
    #[pyo3(signature = (target=0, split_ratio=2.0, merge_ratio=0.25))]
    fn maintain(&mut self, target: usize, split_ratio: f32,
                merge_ratio: f32) -> (usize, usize) {
        let live: usize = self.cells.iter().filter(|c| !c.members.is_empty()).count();
        let t = if target > 0 {
            target
        } else {
            (self.n_live / live.max(1)).max(1)
        };
        crate::maintain::maintain(self, t, split_ratio, merge_ratio)
    }

    // --- persistence (see crate::persist) ---

    /// Write the index to `path`.
    fn save(&self, path: &str) -> PyResult<()> {
        crate::persist::save_index(self, path)
    }

    /// Load an index written by `save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        crate::persist::load_index(path)
    }
}
