//! concurrent: lock-free reads over immutable segments.
//!
//! Every mutating method on FlatCellIndex takes &mut self, so pyo3 holds an exclusive borrow and a
//! remove() during an in-flight search raises "RuntimeError: Already borrowed". Nothing gets
//! corrupted, but it means deletes and inserts only work with serving stopped.
//!
//! Wrapping the index in an RwLock would make it legal but not concurrent: a writer blocks every
//! reader for a whole batch, and under steady read load the writer starves instead.
//!
//! So the index is a list of immutable segments plus one mutable head:
//!
//! ```text
//!    sealed:  [ Arc<Segment> ][ Arc<Segment> ][ Arc<Segment> ]   immutable
//!    head:    RwLock<FlatCellIndex>                              mutable
//!    dead:    RwLock<HashSet<u64>>                               tombstones
//! ```
//!
//! * A sealed segment is never written again. A reader clones the Arc<Vec<Arc<Segment>>> (one
//!   atomic increment) and scans it with no lock held.
//! * The head takes inserts and is the only thing a writer touches, so a write blocks for one batch
//!   at most.
//! * When the head reaches seal_at it is compacted, frozen and appended to the segment list. The
//!   new head reuses the centroids instead of retraining; a partition fitted on a representative
//!   sample works fine for later data from the same distribution.
//! * Deletes never modify a segment. They're tombstones, so a delete is a hash insert wherever the
//!   vector lives.
//!
//! This works because cells hold no cross-references: a cell is a list of offsets into its own
//! segment's arena. A graph index can't be split like this without cutting or duplicating edges.
//!
//! Limitations:
//! * Tombstones accumulate. Deleted vectors are still scanned and then thrown away, so a heavily
//!   churned index wastes scan time. purge_head() handles the head; sealed segments need a merge
//!   pass that isn't implemented.
//! * Search cost grows with the number of segments, since each one routes separately. Keep seal_at
//!   large; auto-merge isn't implemented.
//! * Deleted results are filtered after retrieval, not in the scan. Over-fetching covers it, but
//!   with a very high deletion ratio recall drops until a purge.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use parking_lot::RwLock;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::index::{FlatCellIndex, SearchBuf};

/// Over-fetch factor on k per segment, so dropping tombstoned hits afterwards doesn't eat into the
/// requested k.
const OVERFETCH: usize = 2;

#[pyclass]
pub struct ConcurrentIndex {
    /// Snapshot pointer.  The lock is held only long enough to clone the Arc;
    /// the scan itself runs entirely outside it.
    sealed: RwLock<Arc<Vec<Arc<FlatCellIndex>>>>,
    /// The one mutable segment.
    head: RwLock<FlatCellIndex>,
    /// Tombstones by external id.  Readers take a shared lock to filter,
    /// deleters take it exclusively for a single hash insert.
    dead: RwLock<HashSet<u64>>,
    /// Live vectors in sealed segments (head tracks its own).
    sealed_live: RwLock<usize>,
    seal_at: usize,
}

impl ConcurrentIndex {
    /// Snapshot of everything a query needs, taken with locks held for as
    /// short a time as possible.
    fn snapshot(&self) -> (Arc<Vec<Arc<FlatCellIndex>>>, usize) {
        let segs = self.sealed.read().clone();
        let ndead = self.dead.read().len();
        (segs, ndead)
    }

    /// Merge per-segment results, drop tombstones, dedup, take the top k.
    ///
    /// The dedup is needed: within one FlatCellIndex id_to_slot prevents repeated ids, but a caller
    /// can re-insert an id whose old copy is in a sealed segment. Without dedup that id would come
    /// back twice and take two of the k slots.
    fn finish(&self, mut hits: Vec<(f32, u64)>, k: usize) -> (Vec<u64>, Vec<f32>) {
        if !hits.is_empty() {
            let dead = self.dead.read();
            if !dead.is_empty() {
                hits.retain(|&(_, id)| !dead.contains(&id));
            }
        }
        if hits.len() > 1 {
            // Best score wins. The head is scanned last, so a re-inserted vector's newer copy only
            // replaces the old one if it actually scores higher.
            hits.sort_unstable_by(|a, b| {
                a.1.cmp(&b.1).then(
                    b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal))
            });
            hits.dedup_by_key(|h| h.1);
        }
        let kk = k.min(hits.len());
        if kk == 0 {
            return (Vec::new(), Vec::new());
        }
        if kk < hits.len() {
            hits.select_nth_unstable_by(
                kk - 1, |a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            hits.truncate(kk);
        }
        hits.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        (hits.iter().map(|&(_, i)| i).collect(), hits.iter().map(|&(s, _)| s).collect())
    }

    /// Query every sealed segment plus the head.
    fn query_all(&self, q: &[f32], k: usize, nprobe: usize, tc: f32, beta: f32)
                 -> Vec<(f32, u64)> {
        let (segs, ndead) = self.snapshot();
        // Over-fetch enough to survive tombstone filtering.
        let want = if ndead == 0 { k } else { k * OVERFETCH };
        let mut hits: Vec<(f32, u64)> = Vec::new();

        for seg in segs.iter() {
            // No lock: the segment is immutable and the Arc keeps it alive.
            let mut buf = SearchBuf::new(seg.dim_of(), seg.dim_pad_of(), seg.n_words_of());
            hits.extend(seg.search_rust(q, want, nprobe, tc, beta, &mut buf));
        }
        {
            let h = self.head.read();
            let mut buf = SearchBuf::new(h.dim_of(), h.dim_pad_of(), h.n_words_of());
            hits.extend(h.search_rust(q, want, nprobe, tc, beta, &mut buf));
        }
        hits
    }

    /// Freeze the head into an immutable segment. Caller holds no locks.
    fn seal_locked(&self, head: &mut FlatCellIndex) {
        if head.n_live() == 0 {
            return;
        }
        // Caller holds the head lock; `sealed` is taken after it (head -> sealed, same order as
        // everywhere else). Compact before freezing: the segment is read-only from here on, so this
        // is the last chance to make it cluster-major, and deletes will never fragment it again.
        head.compact();
        let fresh = head.fork_empty();
        let frozen = std::mem::replace(head, fresh);
        let n = frozen.n_live();

        let old = self.sealed.read().clone();
        let mut list = Vec::with_capacity(old.len() + 1);
        list.extend(old.iter().cloned());
        list.push(Arc::new(frozen));
        *self.sealed.write() = Arc::new(list);
        *self.sealed_live.write() += n;
    }
}

#[pymethods]
impl ConcurrentIndex {
    /// `seal_at` is the head size that triggers freezing.  Larger means fewer
    /// segments (cheaper queries) and longer exclusive windows on seal.
    #[new]
    #[pyo3(signature = (dim, scan_mode="binary", rerank_factor=8, seed=0x5EED,
                        code_bits=0, seal_at=1_000_000))]
    fn new(dim: usize, scan_mode: &str, rerank_factor: usize, seed: u64,
           code_bits: usize, seal_at: usize) -> PyResult<Self> {
        Ok(ConcurrentIndex {
            sealed: RwLock::new(Arc::new(Vec::new())),
            head: RwLock::new(FlatCellIndex::new(
                dim, scan_mode, rerank_factor, seed, code_bits)?),
            dead: RwLock::new(HashSet::new()),
            sealed_live: RwLock::new(0),
            seal_at: seal_at.max(1),
        })
    }

    // --- reads ---

    #[pyo3(signature = (query, k, nprobe=16, target_confidence=0.95, beta=8.0))]
    fn search(&self, py: Python, query: PyReadonlyArray1<f32>, k: usize,
              nprobe: usize, target_confidence: f32, beta: f32)
              -> (Vec<u64>, Vec<f32>) {
        let q: Vec<f32> = query.as_array().iter().copied().collect();
        py.allow_threads(|| {
            let hits = self.query_all(&q, k, nprobe, target_confidence, beta);
            self.finish(hits, k)
        })
    }

    #[pyo3(signature = (queries, k, nprobe=16, target_confidence=0.95, beta=8.0))]
    fn search_batch(&self, py: Python, queries: PyReadonlyArray2<f32>, k: usize,
                    nprobe: usize, target_confidence: f32, beta: f32)
                    -> PyResult<(Vec<u64>, Vec<f32>)> {
        let arr = queries.as_array();
        let nq = arr.shape()[0];
        let dim = arr.shape()[1];
        let data: Vec<f32> = arr.iter().copied().collect();
        py.allow_threads(|| {
            let per: Vec<(Vec<u64>, Vec<f32>)> = (0..nq)
                .into_par_iter()
                .map(|i| {
                    let q = &data[i * dim..(i + 1) * dim];
                    let hits = self.query_all(q, k, nprobe, target_confidence, beta);
                    self.finish(hits, k)
                })
                .collect();
            let mut ids = Vec::with_capacity(nq * k);
            let mut scores = Vec::with_capacity(nq * k);
            for (i, s) in per {
                // Pad short rows so the caller can reshape to (nq, k).
                let mut i = i;
                let mut s = s;
                i.resize(k, u64::MAX);
                s.resize(k, f32::NEG_INFINITY);
                ids.extend_from_slice(&i);
                scores.extend_from_slice(&s);
            }
            Ok((ids, scores))
        })
    }

    fn __len__(&self) -> usize {
        let head = self.head.read().n_live();
        let sealed = *self.sealed_live.read();
        let dead = self.dead.read().len();
        (sealed + head).saturating_sub(dead)
    }

    fn stats(&self) -> HashMap<String, f64> {
        // One guard at a time, released before taking the next. Taking the same RwLock twice in one
        // scope can deadlock in parking_lot if a writer queues in between.
        let (mut m, head_live) = {
            let h = self.head.read();
            (h.stats(), h.n_live())
        };
        let nseg = self.sealed.read().len();
        let sealed_live = *self.sealed_live.read();
        let ndead = self.dead.read().len();
        m.insert("segments".into(), nseg as f64);
        m.insert("head_live".into(), head_live as f64);
        m.insert("sealed_live".into(), sealed_live as f64);
        m.insert("tombstones".into(), ndead as f64);
        m.insert("n_live".into(), (sealed_live + head_live).saturating_sub(ndead) as f64);
        m
    }

    // --- reads ---

    /// Insert. Only the head is locked, and only for this batch.
    fn add_batch(&self, ids: Vec<u64>, vectors: PyReadonlyArray2<f32>) -> PyResult<()> {
        {
            let mut h = self.head.write();
            h.add_batch(ids, vectors)?;
        }
        // Seal outside the insert so the exclusive window stays one operation.
        let need_seal = self.head.read().n_live() >= self.seal_at;
        if need_seal {
            let mut h = self.head.write();
            if h.n_live() >= self.seal_at {
                self.seal_locked(&mut h);
            }
        }
        Ok(())
    }

    /// Delete: a tombstone, so it never touches a sealed segment and costs the
    /// same wherever the vector lives.
    fn remove(&self, id: u64) -> bool {
        // Check existence with each guard released before the next one, keeping the head -> sealed
        // -> dead order used in the rest of this file.
        let in_head = self.head.read().has_id(id);
        let known = in_head || {
            let segs = self.sealed.read().clone();
            segs.iter().any(|s| s.has_id(id))
        };
        if !known {
            return false;
        }
        self.dead.write().insert(id)
    }

    /// Delete many under one lock acquisition.
    fn remove_many(&self, ids: Vec<u64>) -> usize {
        let segs = self.sealed.read().clone();
        let head = self.head.read();
        let mut dead = self.dead.write();
        let mut n = 0;
        for id in ids {
            if head.has_id(id) || segs.iter().any(|s| s.has_id(id)) {
                if dead.insert(id) {
                    n += 1;
                }
            }
        }
        n
    }

    /// Freeze the current head immediately.
    fn seal(&self) {
        let mut h = self.head.write();
        self.seal_locked(&mut h);
    }

    /// Drop tombstoned vectors from the head and compact it.
    ///
    /// Sealed segments are left alone; reclaiming them needs a merge pass that isn't written yet.
    fn purge_head(&self) -> usize {
        let mut h = self.head.write();
        let mut dead = self.dead.write();   // head -> dead, as elsewhere
        let victims: Vec<u64> = dead.iter().copied().filter(|&id| h.has_id(id)).collect();
        for id in &victims {
            h.remove(*id);
            dead.remove(id);
        }
        h.compact();
        victims.len()
    }

    // --- configuration ---
    //
    // These only affect the head. A sealed segment keeps the settings it was built with, and
    // fork_empty copies the head's settings to the next head, so changing scan/rerank/routing after
    // data has been sealed gives a mixed index. Configure before the first insert.

    #[pyo3(signature = (vectors, n_cells, sample=262_144, seed=0x5EED))]
    fn train_centroids(&self, vectors: PyReadonlyArray2<f32>, n_cells: usize,
                       sample: usize, seed: u64) -> PyResult<()> {
        self.head.write().train_centroids(vectors, n_cells, sample, seed)
    }

    fn set_per_dim_sq(&self, on: bool) { self.head.write().set_per_dim_sq(on) }
    fn set_two_stage(&self, on: bool, mult: usize) { self.head.write().set_two_stage(on, mult) }
    fn set_adaptive(&self, on: bool, margin: f32) { self.head.write().set_adaptive(on, margin) }
    fn set_parallel_scan(&self, on: bool) { self.head.write().set_parallel_scan(on) }

    #[pyo3(signature = (on, min_cells=0))]
    fn set_graph_route(&self, on: bool, min_cells: usize) {
        self.head.write().set_graph_route(on, min_cells)
    }

    #[pyo3(signature = (on, min_cells=0, fanout=0, beam_frac=0.0))]
    fn set_tree_route(&self, on: bool, min_cells: usize, fanout: usize, beam_frac: f32) {
        self.head.write().set_tree_route(on, min_cells, fanout, beam_frac)
    }

    fn compact(&self) { self.head.write().compact() }
}
