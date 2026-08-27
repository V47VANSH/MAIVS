//! A partitioned vector index.
//!
//! Fixed-size cells over one flat arena, hierarchical routing over the cell centroids, a SIMD scan
//! of quantized codes inside the probed cells, and an exact rerank of the survivors. See
//! docs/ARCHITECTURE.md for the design.
//!
//! Modules:
//! * [`index`]      FlatCellIndex: arena, cells, scan kernels, routing, polysemy, cell-major
//!                  batch search, rerank, delete, compaction, filters
//! * [`tree`]       dynamic-depth tree over centroids, plus the spherical / hierarchical k-means
//!                  that builds it and the cells
//! * [`graph`]      HNSW over centroids. Slower than a flat scan up to 7,804 nodes; kept for the
//!                  untested >65k range, behind MIN_CELLS_FOR_GRAPH
//! * [`maintain`]   online cell split / merge
//! * [`quantize`]   Hadamard rotation, 1-bit packing, POPCNT Hamming, asymmetric and FastScan
//!                  estimators
//! * [`persist`]    save / load
//! * [`concurrent`] ConcurrentIndex: immutable segments + a mutable head
//! * [`prof`]       opt-in phase timers

mod concurrent;
mod graph;
mod index;
mod maintain;
mod persist;
mod prof;
mod quantize;
mod tree;

use pyo3::prelude::*;

#[pymodule]
fn maivs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<index::FlatCellIndex>()?;
    m.add_class::<concurrent::ConcurrentIndex>()?;
    m.add_function(wrap_pyfunction!(prof::profile, m)?)?;
    m.add_function(wrap_pyfunction!(prof::profile_read, m)?)?;
    Ok(())
}
