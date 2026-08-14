//! prof: opt-in phase timers for the query and build paths.
//!
//! Off by default, when a phase boundary costs one relaxed atomic load. When on it's an
//! Instant::now() plus a relaxed fetch_add. Profile with serial queries (one thread) so the
//! counters aren't contended.

use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

static ON: AtomicBool = AtomicBool::new(false);
static NS: [AtomicU64; 24] = [const { AtomicU64::new(0) }; 24];

pub(crate) const Q_ROUTE: usize = 0;
pub(crate) const Q_SELECT: usize = 1;
pub(crate) const Q_BOUNDS: usize = 2;
pub(crate) const Q_SCAN: usize = 3;
pub(crate) const Q_TOPK: usize = 4;
pub(crate) const Q_RERANK: usize = 5;
pub(crate) const Q_COUNT: usize = 6;
pub(crate) const Q_R_ROT: usize = 7;
pub(crate) const Q_R_TREE: usize = 8;
pub(crate) const Q_R_RESCORE: usize = 9;
pub(crate) const B_TRAIN: usize = 10;
pub(crate) const B_SQFIT: usize = 11;
pub(crate) const B_ASSIGN: usize = 12;
pub(crate) const B_PLACE: usize = 13;
pub(crate) const B_CENTROIDS: usize = 14;
pub(crate) const B_RADII: usize = 15;
pub(crate) const B_CODES: usize = 16;
pub(crate) const B_TREE: usize = 17;
pub(crate) const B_COMPACT: usize = 18;

const NAMES: [(usize, &str); 19] = [
    (Q_R_ROT, "q.r_rotate"), (Q_R_TREE, "q.r_tree"), (Q_R_RESCORE, "q.r_rescore"),
    (Q_ROUTE, "q.route"), (Q_SELECT, "q.select"), (Q_BOUNDS, "q.bounds"),
    (Q_SCAN, "q.scan"), (Q_TOPK, "q.topk"), (Q_RERANK, "q.rerank"),
    (Q_COUNT, "q.count"), (B_TRAIN, "b.train"), (B_SQFIT, "b.sqfit"),
    (B_ASSIGN, "b.assign"), (B_PLACE, "b.encode_place"),
    (B_CENTROIDS, "b.centroids"), (B_RADII, "b.radii"), (B_CODES, "b.codes"),
    (B_TREE, "b.graph_tree"), (B_COMPACT, "b.compact"),
];

#[inline(always)]
pub(crate) fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

#[inline(always)]
pub(crate) fn add(idx: usize, ns: u64) {
    NS[idx].fetch_add(ns, Ordering::Relaxed);
}

/// Lap timer: mark(i) adds the time since the previous mark (or since start) to counter i.
pub(crate) struct Lap {
    t: Option<Instant>,
}

impl Lap {
    #[inline(always)]
    pub(crate) fn start() -> Self {
        Lap { t: if on() { Some(Instant::now()) } else { None } }
    }

    #[inline(always)]
    pub(crate) fn mark(&mut self, idx: usize) {
        if let Some(t) = self.t {
            let now = Instant::now();
            add(idx, (now - t).as_nanos() as u64);
            self.t = Some(now);
        }
    }
}

/// Turn phase profiling on or off (process-wide).
#[pyfunction]
pub(crate) fn profile(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// Accumulated phase times in milliseconds (`q.count` is a plain count).
#[pyfunction]
#[pyo3(signature = (reset=true))]
pub(crate) fn profile_read(reset: bool) -> Vec<(String, f64)> {
    let out = NAMES
        .iter()
        .map(|&(i, name)| {
            let v = NS[i].load(Ordering::Relaxed) as f64;
            (name.to_string(), if i == Q_COUNT { v } else { v / 1e6 })
        })
        .collect();
    if reset {
        for c in NS.iter() {
            c.store(0, Ordering::Relaxed);
        }
    }
    out
}
