//! persist: on-disk format for FlatCellIndex.
//!
//! A hand-written little-endian format instead of serde/bincode. The index is mostly big flat
//! Vec<primitive> arenas, so a file is a small header plus a few raw blocks; serde would add a
//! per-element code path for data that's already contiguous, and would need Serialize on a
//! #[pyclass].
//!
//! Derived state is rebuilt on load instead of stored:
//!   * id_to_slot  from ext_ids minus free_slots
//!   * cent_bits   from the centroids
//!   * rot_signs   from the stored seed
//!
//! Fixed little-endian IEEE-754, so a file written on one x86-64 machine loads on another. The
//! magic includes a format version; bump the last byte on any layout change so old files fail
//! loudly.
//!
//! These are plain functions, not #[pymethods]: pyo3 allows one #[pymethods] block per type without
//! the multiple-pymethods feature, so `index` has thin wrappers that call into here.

use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

use crate::index::{Cell, FlatCellIndex, ScanMode};
use crate::quantize as rabitq;

const MAGIC: &[u8; 8] = b"MAIVS2\x00\x01";

fn io_err(e: std::io::Error) -> PyErr {
    pyo3::exceptions::PyIOError::new_err(e.to_string())
}

fn wr_u64<W: Write>(w: &mut W, v: u64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn wr_f32<W: Write>(w: &mut W, v: f32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn rd_u64<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn rd_f32<R: Read>(r: &mut R) -> std::io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

/// Length-prefixed raw block of plain-old-data.
fn wr_pod<W: Write, T: Copy>(w: &mut W, v: &[T]) -> std::io::Result<()> {
    wr_u64(w, v.len() as u64)?;
    // Safety: every T used here is a primitive (u32/u64/i8/f32), with no padding, pointers or Drop,
    // so the slice is a valid byte range.
    let bytes = unsafe {
        std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v))
    };
    w.write_all(bytes)
}

fn rd_pod<R: Read, T: Copy + Default>(r: &mut R) -> std::io::Result<Vec<T>> {
    let n = rd_u64(r)? as usize;
    let mut v: Vec<T> = vec![T::default(); n];
    // Safety: as above; the Vec is already sized and owns its allocation.
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(
            v.as_mut_ptr() as *mut u8,
            n * std::mem::size_of::<T>(),
        )
    };
    r.read_exact(bytes)?;
    Ok(v)
}

/// Persist `ix` to `path`.
pub(crate) fn save_index(ix: &FlatCellIndex, path: &str) -> PyResult<()> {
    let f = File::create(path).map_err(io_err)?;
    let mut w = BufWriter::new(f);

    let mut go = || -> std::io::Result<()> {
        w.write_all(MAGIC)?;
        let hdr: [u64; 16] = [
            ix.dim as u64,
            match ix.scan_mode {
                ScanMode::F32 => 0,
                ScanMode::I8 => 1,
                ScanMode::Binary => 2,
            },
            ix.rerank_factor as u64,
            ix.dim_pad as u64,
            ix.n_words as u64,
            ix.seed,
            ix.n_live as u64,
            ix.max_assign as u64,
            ix.shortlist_mult as u64,
            ix.i8_rerank as u64,
            ix.keep_f32 as u64,
            ix.adaptive as u64,
            ix.two_stage as u64,
            // Routing settings can't be recomputed from the data; without them a reloaded index
            // falls back to the defaults and routes differently from the saved one.
            ix.tree_route as u64,
            ix.tree_min_cells as u64,
            ix.tree_fanout as u64,
        ];
        for v in hdr {
            wr_u64(&mut w, v)?;
        }
        wr_f32(&mut w, ix.quant_scale)?;
        wr_f32(&mut w, ix.spill_margin)?;
        wr_f32(&mut w, ix.at_margin)?;
        wr_f32(&mut w, ix.tree_beam_frac)?;

        // Flat codes even when only FastScan blocks are resident.
        wr_pod(&mut w, &ix.flat_bits())?;
        wr_pod(&mut w, &ix.scan_i8)?;
        wr_pod(&mut w, &ix.rr_f16)?;
        wr_pod(&mut w, &ix.sq_min)?;
        wr_pod(&mut w, &ix.sq_step)?;
        wr_pod(&mut w, &ix.vecs_f32)?;
        wr_pod(&mut w, &ix.ext_ids)?;
        wr_pod(&mut w, &ix.slot_cell)?;
        wr_pod(&mut w, &ix.free_slots)?;
        wr_pod(&mut w, &ix.slot_cells)?;
        // Filter tags travel with the index; without them a reloaded index
        // silently answers every filtered query as if unfiltered.
        wr_pod(&mut w, &ix.tags)?;

        wr_u64(&mut w, ix.cells.len() as u64)?;
        for c in &ix.cells {
            wr_pod(&mut w, &c.members)?;
            wr_pod(&mut w, &c.sum)?;
            wr_pod(&mut w, &c.centroid)?;
            wr_f32(&mut w, c.r_bound)?;
            wr_f32(&mut w, c.r_max)?;
        }
        w.flush()
    };
    go().map_err(io_err)
}

/// Load an index written by [`save_index`].
pub(crate) fn load_index(path: &str) -> PyResult<FlatCellIndex> {
    let f = File::open(path).map_err(io_err)?;
    let mut r = BufReader::new(f);

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic).map_err(io_err)?;
    if &magic != MAGIC {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "not a MAIVS v2 index file (bad magic or format version)",
        ));
    }

    let mut h = [0u64; 16];
    for slot in h.iter_mut() {
        *slot = rd_u64(&mut r).map_err(io_err)?;
    }
    let quant_scale = rd_f32(&mut r).map_err(io_err)?;
    let spill_margin = rd_f32(&mut r).map_err(io_err)?;
    let at_margin = rd_f32(&mut r).map_err(io_err)?;
    let tree_beam_frac = rd_f32(&mut r).map_err(io_err)?;

    let dim_pad = h[3] as usize;
    let seed = h[5];

    let bits: Vec<u64> = rd_pod(&mut r).map_err(io_err)?;
    let scan_i8: Vec<i8> = rd_pod(&mut r).map_err(io_err)?;
    let rr_f16: Vec<u16> = rd_pod(&mut r).map_err(io_err)?;
    let has_f16 = !rr_f16.is_empty();
    let sq_min: Vec<f32> = rd_pod(&mut r).map_err(io_err)?;
    let sq_step: Vec<f32> = rd_pod(&mut r).map_err(io_err)?;
    let has_sq = !sq_min.is_empty();
    let vecs_f32: Vec<f32> = rd_pod(&mut r).map_err(io_err)?;
    let ext_ids: Vec<u64> = rd_pod(&mut r).map_err(io_err)?;
    let slot_cell: Vec<u32> = rd_pod(&mut r).map_err(io_err)?;
    let free_slots: Vec<u32> = rd_pod(&mut r).map_err(io_err)?;
    let slot_cells: Vec<u32> = rd_pod(&mut r).map_err(io_err)?;
    let tags: Vec<u32> = rd_pod(&mut r).map_err(io_err)?;

    let ncell = rd_u64(&mut r).map_err(io_err)? as usize;
    let mut cells = Vec::with_capacity(ncell);
    for _ in 0..ncell {
        let members: Vec<u32> = rd_pod(&mut r).map_err(io_err)?;
        let sum: Vec<f32> = rd_pod(&mut r).map_err(io_err)?;
        let centroid: Vec<f32> = rd_pod(&mut r).map_err(io_err)?;
        let r_bound = rd_f32(&mut r).map_err(io_err)?;
        let r_max = rd_f32(&mut r).map_err(io_err)?;
        cells.push(Cell { members, sum, centroid, r_bound, r_max });
    }

    // Free slots still hold stale ids; excluding them keeps lookup consistent
    // with n_live and stops a deleted id resolving again after a reload.
    let free: HashSet<u32> = free_slots.iter().copied().collect();
    let mut id_to_slot = HashMap::with_capacity(ext_ids.len());
    for (slot, &id) in ext_ids.iter().enumerate() {
        if !free.contains(&(slot as u32)) {
            id_to_slot.insert(id, slot as u32);
        }
    }

    let mut ix = FlatCellIndex {
        dim: h[0] as usize,
        scan_mode: match h[1] {
            0 => ScanMode::F32,
            1 => ScanMode::I8,
            _ => ScanMode::Binary,
        },
        rerank_factor: h[2] as usize,
        dim_pad,
        n_words: h[4] as usize,
        seed,
        rot_signs: rabitq::make_signs(dim_pad, seed),
        bits,
        scan_i8,
        rr_f16,
        vecs_f32,
        ext_ids,
        slot_cell,
        free_slots,
        id_to_slot,
        cells,
        cent_bits: Vec::new(),
        cent_graph: None,
        graph_ids: Vec::new(),
        graph_route: true,
        graph_min_cells: crate::graph::MIN_CELLS_FOR_GRAPH,
        // derived state: rebuilt by finish_load(), not written to disk (the tree only depends on
        // the centroids)
        tags,
        tag_filter: None,
        cent_tree: None,
        tree_route: h[13] != 0,
        tree_min_cells: h[14] as usize,
        tree_fanout: h[15] as usize,
        tree_beam_frac,
        tree_cells: 0,
        par_scan: false,
        two_stage: h[12] != 0,
        // Derived, like the centroid codes and the tree: rebuilt by finish_load instead of stored,
        // so the file format doesn't change. The flag itself isn't saved because f32 and SQ8
        // re-order gave the same probe counts and scans on two corpora, so a loaded index returns
        // the same rows either way (which is what the save/load test checks).
        cent_sq8: Vec::new(),
        csq_min: Vec::new(),
        csq_step: Vec::new(),
        centroid_sq8: true,
        shortlist_mult: h[8] as usize,
        quant_scale,
        n_live: h[6] as usize,
        max_assign: h[7] as usize,
        spill_margin,
        // Not saved (keeps the file format the same). It only affects where future inserts get
        // their extra homes; the homes already in the file are restored exactly through slot_cells.
        spill_lambda: 0.0,
        // Saved member order is not guaranteed to be a compacted run.
        compacted: false,
        asym_cascade: 0,
        fastscan: false,
        fs_codes: Vec::new(),
        fs_off: Vec::new(),
        prim_len: Vec::new(),
        rep_off: Vec::new(),
        rep_bits: Vec::new(),
        slot_cells,
        i8_rerank: h[9] != 0,
        f16_rerank: has_f16,
        per_dim_sq: has_sq,
        sq_min,
        sq_step,
        keep_f32: h[10] != 0,
        adaptive: h[11] != 0,
        at_margin,
    };
    ix.finish_load();
    Ok(ix)
}
