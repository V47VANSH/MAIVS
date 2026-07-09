use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashSet, HashMap};
use parking_lot::RwLock;
use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, Ordering as AO};
use serde::{Serialize, Deserialize};

// --- profiling counters ---

static ROUTER_TOTAL_US: AtomicU64 = AtomicU64::new(0);
static CHILD_QUERY_TOTAL_US: AtomicU64 = AtomicU64::new(0);
static RERANK_TOTAL_US: AtomicU64 = AtomicU64::new(0);
static LEAVES_VISITED: AtomicU64 = AtomicU64::new(0);
static QUERY_COUNT: AtomicU64 = AtomicU64::new(0);
static SPLIT_COUNT: AtomicU64 = AtomicU64::new(0);
static SPLIT_DURATION_US: AtomicU64 = AtomicU64::new(0);
static VECTORS_REDISTRIBUTED: AtomicU64 = AtomicU64::new(0);
static INGEST_BATCH_US: AtomicU64 = AtomicU64::new(0);
static TREE_DEPTH_MAX: AtomicU64 = AtomicU64::new(0);

// --- profiling counters ---

#[derive(PartialEq, Clone, Copy)]
struct Score(f32);
impl Eq for Score {}
impl PartialOrd for Score {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { self.0.partial_cmp(&other.0) }
}
impl Ord for Score {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering { self.partial_cmp(other).unwrap_or(Ordering::Equal) }
}

#[derive(PartialEq, Clone, Copy)]
struct MinScore(f32);
impl Eq for MinScore {}
impl PartialOrd for MinScore {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { other.0.partial_cmp(&self.0) }
}
impl Ord for MinScore {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering { self.partial_cmp(other).unwrap_or(Ordering::Equal) }
}

// --- profiling counters ---

/// Metadata for a single node in the AIGVS tree (arena-allocated).
#[derive(Clone)]
struct NodeData {
    node_id: String,
    parent_idx: Option<usize>,
    depth: u32,
    centroid: Vec<f32>,
    kind: NodeKind,
}

#[derive(Clone)]
enum NodeKind {
    Trunk { router_idx: usize },
    Leaf  { graph_idx: usize, semantic_variance: f32 },
}

/// Hopfield router: maps a query to the best child node indices with a softmax beam search.
struct RouterData {
    /// Flattened row-major attractor matrix: row i = attractors[i*dim..(i+1)*dim].
    attractors: Vec<f32>,
    /// same order as the attractor rows: attractor index -> child node index
    child_node_indices: Vec<usize>,
    n_attractors: usize,
}

/// NSW graph for a single leaf.
struct GraphData {
    chunk_ids: Vec<String>,
    /// SQ8-quantized vectors, flattened row-major.
    vector_cache: Vec<i8>,
    local_graph: Vec<RwLock<Vec<usize>>>,
    deleted_indices: HashSet<usize>,
    entry_point_idx: Option<usize>,
    max_connections: usize,
    ef_construction: usize,
}

// --- serializable shadow types (RwLock can't derive Serialize) ---

#[derive(Serialize, Deserialize)]
struct EngineState {
    dim: usize,
    max_connections: usize,
    ef_construction: usize,
    polysemy_base: f32,
    polysemy_decay: f32,
    variance_split_thresh: f32,
    hard_cap: usize,
    next_id: u64,
    root_idx: Option<usize>,
    nodes: Vec<NodeDataSer>,
    routers: Vec<RouterDataSer>,
    graphs: Vec<GraphDataSer>,
}

#[derive(Serialize, Deserialize, Clone)]
struct NodeDataSer {
    node_id: String,
    parent_idx: Option<usize>,
    depth: u32,
    centroid: Vec<f32>,
    is_trunk: bool,
    router_or_graph_idx: usize,
    semantic_variance: f32,
}

#[derive(Serialize, Deserialize)]
struct RouterDataSer {
    attractors: Vec<f32>,
    child_node_indices: Vec<usize>,
    n_attractors: usize,
}

#[derive(Serialize, Deserialize)]
struct GraphDataSer {
    chunk_ids: Vec<String>,
    vector_cache: Vec<i8>,
    adjacency: Vec<Vec<usize>>,
    deleted_indices: Vec<usize>,
    entry_point_idx: Option<usize>,
    max_connections: usize,
    ef_construction: usize,
}

// --- profiling counters ---

#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let unrolled = len - (len % 4);
    let mut s = 0.0f32;
    let mut i = 0;
    while i < unrolled {
        unsafe {
            s += a.get_unchecked(i)     * b.get_unchecked(i);
            s += a.get_unchecked(i + 1) * b.get_unchecked(i + 1);
            s += a.get_unchecked(i + 2) * b.get_unchecked(i + 2);
            s += a.get_unchecked(i + 3) * b.get_unchecked(i + 3);
        }
        i += 4;
    }
    for j in unrolled..len {
        unsafe { s += a.get_unchecked(j) * b.get_unchecked(j); }
    }
    s
}

#[inline(always)]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    let len = a.len();
    let unrolled = len - (len % 8);
    let mut s = 0i32;
    let mut i = 0;
    while i < unrolled {
        unsafe {
            s += *a.get_unchecked(i) as i32 * *b.get_unchecked(i) as i32;
            s += *a.get_unchecked(i+1) as i32 * *b.get_unchecked(i+1) as i32;
            s += *a.get_unchecked(i+2) as i32 * *b.get_unchecked(i+2) as i32;
            s += *a.get_unchecked(i+3) as i32 * *b.get_unchecked(i+3) as i32;
            s += *a.get_unchecked(i+4) as i32 * *b.get_unchecked(i+4) as i32;
            s += *a.get_unchecked(i+5) as i32 * *b.get_unchecked(i+5) as i32;
            s += *a.get_unchecked(i+6) as i32 * *b.get_unchecked(i+6) as i32;
            s += *a.get_unchecked(i+7) as i32 * *b.get_unchecked(i+7) as i32;
        }
        i += 8;
    }
    for j in unrolled..len {
        unsafe { s += *a.get_unchecked(j) as i32 * *b.get_unchecked(j) as i32; }
    }
    s
}

#[inline(always)]
fn quantize_f32_to_i8(v: &[f32]) -> Vec<i8> {
    v.iter().map(|&x| (x * 127.0).clamp(-127.0, 127.0) as i8).collect()
}

#[inline(always)]
fn dequantize_i8_to_f32(v: &[i8]) -> Vec<f32> {
    v.iter().map(|&x| x as f32 / 127.0).collect()
}

fn normalize_f32(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() { *x /= norm; }
    }
}

// --- profiling counters ---

#[inline]
fn router_beam_search(
    attractors: &[f32],
    child_node_indices: &[usize],
    q: &[f32],
    dim: usize,
    n_attractors: usize,
    max_beam: usize,
    target_confidence: f32,
    beta: f32,
) -> Vec<usize> {
    if n_attractors == 0 { return vec![]; }

    let t0 = std::time::Instant::now();

    // Score all attractors
    let mut raw_scores: Vec<f32> = Vec::with_capacity(n_attractors);
    let mut max_s = f32::NEG_INFINITY;
    for row in attractors.chunks_exact(dim).take(n_attractors) {
        let s = dot(q, row);
        if s > max_s { max_s = s; }
        raw_scores.push(s);
    }

    // Dynamic beta from score variance if beta == 0
    let dynamic_beta = if beta == 0.0 && n_attractors > 1 {
        let mean: f32 = raw_scores.iter().sum::<f32>() / n_attractors as f32;
        let var: f32 = raw_scores.iter().map(|&s| (s - mean) * (s - mean)).sum::<f32>() / n_attractors as f32;
        if var > 1e-6 { 1.0 / var.sqrt() } else { 1.0 }
    } else { beta };

    // Numerically stable softmax
    let mut exp_sum = 0.0f32;
    let mut probs: Vec<(usize, f32)> = raw_scores.iter().enumerate().map(|(i, &s)| {
        let e = ((s - max_s) * dynamic_beta).exp();
        exp_sum += e;
        (i, e)
    }).collect();
    for (_, p) in probs.iter_mut() { *p /= exp_sum + 1e-9; }

    probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let mut selected = Vec::with_capacity(max_beam.min(n_attractors));
    let mut cumulative = 0.0f32;
    for (idx, prob) in probs.into_iter().take(max_beam) {
        if prob < 1e-9 { break; }
        selected.push(child_node_indices[idx]);
        cumulative += prob;
        if cumulative >= target_confidence { break; }
    }

    ROUTER_TOTAL_US.fetch_add(t0.elapsed().as_micros() as u64, AO::Relaxed);
    selected
}

// --- profiling counters ---

fn nsw_search_candidates(
    vector_cache: &[i8], local_graph: &[RwLock<Vec<usize>>],
    deleted_indices: &HashSet<usize>, q: &[i8], dim: usize,
    entry: usize, ef: usize, visited: &mut [bool],
) -> BinaryHeap<(Score, usize)> {
    let mut candidates: BinaryHeap<(Score, usize)> = BinaryHeap::new();
    let mut results: BinaryHeap<(MinScore, usize)> = BinaryHeap::new();

    let entry_score = dot_i8(q, &vector_cache[entry*dim..(entry+1)*dim]) as f32;
    candidates.push((Score(entry_score), entry));
    if !deleted_indices.contains(&entry) { results.push((MinScore(entry_score), entry)); }
    visited[entry] = true;

    while let Some((Score(cand_score), curr)) = candidates.pop() {
        if results.len() >= ef {
            if let Some(&(MinScore(worst), _)) = results.peek() {
                if cand_score < worst { break; }
            }
        }
        for &nb in &*local_graph[curr].read() {
            if !visited[nb] {
                visited[nb] = true;
                let nb_score = dot_i8(q, &vector_cache[nb*dim..(nb+1)*dim]) as f32;
                candidates.push((Score(nb_score), nb));
                if !deleted_indices.contains(&nb) {
                    if results.len() < ef {
                        results.push((MinScore(nb_score), nb));
                    } else if let Some(&(MinScore(worst), _)) = results.peek() {
                        if nb_score > worst {
                            results.pop();
                            results.push((MinScore(nb_score), nb));
                        }
                    }
                }
            }
        }
    }
    results.into_iter().map(|(MinScore(s), idx)| (Score(s), idx)).collect()
}

fn nsw_query(graph: &GraphData, q: &[i8], dim: usize, ef: usize) -> Vec<(String, f32)> {
    let ep = match graph.entry_point_idx { Some(e) => e, None => return vec![] };
    let mut visited = vec![false; graph.local_graph.len()];
    let heap = nsw_search_candidates(
        &graph.vector_cache, &graph.local_graph, &graph.deleted_indices,
        q, dim, ep, ef, &mut visited,
    );
    let mut v: Vec<(Score, usize)> = heap.into_sorted_vec();
    v.reverse();
    v.into_iter().map(|(Score(s), idx)| (graph.chunk_ids[idx].clone(), s)).collect()
}

fn heuristic_select(candidates: &[(usize, f32)], vector_cache: &[i8], dim: usize, m: usize) -> Vec<usize> {
    let mut result = Vec::with_capacity(m);
    for &(cand_idx, cand_sim) in candidates {
        if result.len() >= m { break; }
        let mut is_good = true;
        let cand_vec = &vector_cache[cand_idx*dim..(cand_idx+1)*dim];
        for &r_idx in &result {
            let sim_to_r = dot_i8(cand_vec, &vector_cache[r_idx*dim..(r_idx+1)*dim]) as f32;
            if sim_to_r > cand_sim { is_good = false; break; }
        }
        if is_good { result.push(cand_idx); }
    }
    if result.len() < m {
        for &(cand_idx, _) in candidates {
            if result.len() >= m { break; }
            if !result.contains(&cand_idx) { result.push(cand_idx); }
        }
    }
    result
}

fn prune_neighbours(neighbours: &mut Vec<usize>, vector_cache: &[i8], pivot: usize, dim: usize, max_conn: usize) {
    if neighbours.len() <= max_conn { return; }
    let pivot_vec = &vector_cache[pivot*dim..(pivot+1)*dim];
    let mut scored: Vec<(usize, f32)> = neighbours.iter()
        .map(|&nb| (nb, dot_i8(pivot_vec, &vector_cache[nb*dim..(nb+1)*dim]) as f32))
        .collect();
    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    *neighbours = heuristic_select(&scored, vector_cache, dim, max_conn);
}

/// Batch-insert vectors into a graph using parallel Rayon NSW construction.
fn batch_insert_to_graph(graph: &mut GraphData, chunk_ids: Vec<String>, vectors_i8: &[i8], dim: usize) {
    let m = graph.max_connections;
    let ef = graph.ef_construction;
    let n_new = chunk_ids.len();
    if n_new == 0 { return; }

    let start_idx = graph.chunk_ids.len();
    graph.chunk_ids.extend(chunk_ids.into_iter());
    graph.vector_cache.extend_from_slice(vectors_i8);
    for _ in 0..n_new {
        graph.local_graph.push(RwLock::new(Vec::with_capacity(m)));
    }

    let mut parallel_start = 0;
    if start_idx == 0 {
        graph.entry_point_idx = Some(0);
        parallel_start = 1;
    }

    let total_nodes = graph.local_graph.len();
    let vector_cache = &graph.vector_cache;
    let local_graph = &graph.local_graph;
    let deleted_indices = &graph.deleted_indices;
    let ep = graph.entry_point_idx.unwrap();

    let indices: Vec<usize> = (start_idx + parallel_start..start_idx + n_new).collect();
    indices.into_par_iter().for_each(|new_idx| {
        let mut visited = vec![false; total_nodes];
        let vec_data = &vector_cache[new_idx*dim..(new_idx+1)*dim];
        let candidates = nsw_search_candidates(
            vector_cache, local_graph, deleted_indices, vec_data, dim, ep, ef, &mut visited,
        );
        let cand_vec: Vec<(usize, f32)> = candidates.into_sorted_vec().into_iter()
            .rev().map(|(Score(s), idx)| (idx, s)).collect();
        let neighbours = heuristic_select(&cand_vec, vector_cache, dim, m);
        for nb_idx in neighbours {
            local_graph[new_idx].write().push(nb_idx);
            let mut nb_lock = local_graph[nb_idx].write();
            nb_lock.push(new_idx);
            if nb_lock.len() > m {
                let mut pruned = nb_lock.clone();
                prune_neighbours(&mut pruned, vector_cache, nb_idx, dim, m);
                *nb_lock = pruned;
            }
        }
    });

    // Update entry point
    let mut best_ep = graph.entry_point_idx.unwrap();
    let mut best_deg = graph.local_graph[best_ep].read().len();
    for i in start_idx..start_idx + n_new {
        let deg = graph.local_graph[i].read().len();
        if deg > best_deg { best_deg = deg; best_ep = i; }
    }
    graph.entry_point_idx = Some(best_ep);
}

/// Insert a single vector into a graph (sequential).
fn single_insert_to_graph(graph: &mut GraphData, chunk_id: String, vec_i8: &[i8], dim: usize) {
    let m = graph.max_connections;
    let ef = graph.ef_construction;
    let new_idx = graph.chunk_ids.len();

    graph.chunk_ids.push(chunk_id);
    graph.vector_cache.extend_from_slice(vec_i8);
    graph.local_graph.push(RwLock::new(Vec::with_capacity(m)));

    if new_idx == 0 {
        graph.entry_point_idx = Some(0);
        return;
    }

    let ep = graph.entry_point_idx.unwrap();
    let mut visited = vec![false; graph.local_graph.len()];
    let candidates = nsw_search_candidates(
        &graph.vector_cache, &graph.local_graph, &graph.deleted_indices,
        vec_i8, dim, ep, ef, &mut visited,
    );
    let cand_vec: Vec<(usize, f32)> = candidates.into_sorted_vec().into_iter()
        .rev().map(|(Score(s), idx)| (idx, s)).collect();
    let neighbours = heuristic_select(&cand_vec, &graph.vector_cache, dim, m);

    for nb_idx in neighbours {
        graph.local_graph[new_idx].write().push(nb_idx);
        let mut nb_lock = graph.local_graph[nb_idx].write();
        nb_lock.push(new_idx);
        if nb_lock.len() > m {
            let mut pruned = nb_lock.clone();
            prune_neighbours(&mut pruned, &graph.vector_cache, nb_idx, dim, m);
            *nb_lock = pruned;
        }
    }

    if graph.local_graph[new_idx].read().len() > graph.local_graph[ep].read().len() {
        graph.entry_point_idx = Some(new_idx);
    }
}

// --- profiling counters ---

/// Farthest-point sampling for maximally separated seed centroids.
fn farthest_point_seeds(vecs: &[f32], n: usize, dim: usize, k: usize) -> Vec<f32> {
    let avg: Vec<f32> = (0..dim).map(|d| {
        let mut s = 0.0f32;
        for i in 0..n { s += vecs[i * dim + d]; }
        s / n as f32
    }).collect();

    let mut avg_normed = avg.clone();
    normalize_f32(&mut avg_normed);

    // First seed: closest to average direction
    let mut best_i = 0;
    let mut best_s = f32::NEG_INFINITY;
    for i in 0..n {
        let s = dot(&vecs[i*dim..(i+1)*dim], &avg_normed);
        if s > best_s { best_s = s; best_i = i; }
    }

    let mut seeds: Vec<f32> = vecs[best_i*dim..(best_i+1)*dim].to_vec();

    for _ in 1..k {
        let num_seeds = seeds.len() / dim;
        let mut worst_max_sim = f32::INFINITY;
        let mut furthest = 0;
        for i in 0..n {
            let v = &vecs[i*dim..(i+1)*dim];
            let mut max_sim = f32::NEG_INFINITY;
            for s in 0..num_seeds {
                let sim = dot(v, &seeds[s*dim..(s+1)*dim]);
                if sim > max_sim { max_sim = sim; }
            }
            if max_sim < worst_max_sim {
                worst_max_sim = max_sim;
                furthest = i;
            }
        }
        seeds.extend_from_slice(&vecs[furthest*dim..(furthest+1)*dim]);
    }
    seeds
}

/// Variance-adaptive Hopfield convergence (the core AIGVS differentiator).
/// Ported from nodes.py L314-329.
fn hopfield_convergence(
    vecs: &[f32],          // flattened N*dim
    centroids: &mut [f32], // flattened K*dim (mutated in place)
    n: usize, k: usize, dim: usize,
    iterations: usize, lambda: f32,
) {
    for _iter in 0..iterations {
        let prev: Vec<f32> = centroids.to_vec();

        // scores[i][j] = dot(vecs[i], centroids[j])
        let mut scores = vec![0.0f32; n * k];
        for i in 0..n {
            let v = &vecs[i*dim..(i+1)*dim];
            for j in 0..k {
                scores[i*k + j] = dot(v, &centroids[j*dim..(j+1)*dim]);
            }
        }

        // Variance per cluster (column variance of scores)
        let mut means = vec![0.0f32; k];
        for i in 0..n { for j in 0..k { means[j] += scores[i*k + j]; } }
        for j in 0..k { means[j] /= n as f32; }

        let mut variances = vec![0.0f32; k];
        for i in 0..n {
            for j in 0..k {
                let d = scores[i*k + j] - means[j];
                variances[j] += d * d;
            }
        }
        for j in 0..k { variances[j] /= n as f32; }

        // betas = lambda / (variance + eps)
        let betas: Vec<f32> = variances.iter().map(|&v| lambda / (v + 1e-5)).collect();

        // Scale scores by per-cluster beta
        for i in 0..n { for j in 0..k { scores[i*k + j] *= betas[j]; } }

        // Softmax per row (numerically stable)
        for i in 0..n {
            let row = &mut scores[i*k..(i+1)*k];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() { *v = (*v - max_val).exp(); sum += *v; }
            for v in row.iter_mut() { *v /= sum + 1e-9; }
        }

        // centroids = normalize(W^T @ X)
        for j in 0..k {
            for d in 0..dim { centroids[j*dim + d] = 0.0; }
            for i in 0..n {
                let w = scores[i*k + j];
                for d in 0..dim {
                    centroids[j*dim + d] += w * vecs[i*dim + d];
                }
            }
            normalize_f32(&mut centroids[j*dim..(j+1)*dim]);
        }

        // Check convergence
        let mut diff = 0.0f32;
        for i in 0..centroids.len() { diff += (centroids[i] - prev[i]).abs(); }
        if diff < 1e-6 * (k * dim) as f32 { break; }
    }
}

/// High-cohesion check: if all seeds are very similar, the data is one cohesive cluster.
fn check_cohesion(centroids: &[f32], k: usize, dim: usize, threshold: f32) -> bool {
    if k <= 1 { return true; }
    for i in 0..k {
        for j in (i+1)..k {
            let sim = dot(&centroids[i*dim..(i+1)*dim], &centroids[j*dim..(j+1)*dim]);
            if sim < threshold { return false; }
        }
    }
    true
}

// --- profiling counters ---

#[pyclass]
struct AIGVSEngine {
    dim: usize,
    max_connections: usize,
    ef_construction: usize,

    // Arena-allocated tree
    nodes: Vec<NodeData>,
    node_name_to_idx: HashMap<String, usize>,
    root_idx: Option<usize>,

    // Separate arenas for routers and graphs
    routers: Vec<RouterData>,
    graphs: Vec<GraphData>,

    // Auto-incrementing node ID counter
    next_id: u64,

    // Split parameters
    polysemy_base: f32,
    polysemy_decay: f32,
    variance_split_thresh: f32,
    hard_cap: usize,
}

impl AIGVSEngine {
    // --- tree construction ---

    fn alloc_node(&mut self, data: NodeData) -> usize {
        let idx = self.nodes.len();
        self.node_name_to_idx.insert(data.node_id.clone(), idx);
        self.nodes.push(data);
        idx
    }

    fn alloc_router(&mut self) -> usize {
        let idx = self.routers.len();
        self.routers.push(RouterData {
            attractors: Vec::new(),
            child_node_indices: Vec::new(),
            n_attractors: 0,
        });
        idx
    }

    fn alloc_graph(&mut self) -> usize {
        let idx = self.graphs.len();
        self.graphs.push(GraphData {
            chunk_ids: Vec::new(),
            vector_cache: Vec::new(),
            local_graph: Vec::new(),
            deleted_indices: HashSet::new(),
            entry_point_idx: None,
            max_connections: self.max_connections,
            ef_construction: self.ef_construction,
        });
        idx
    }

    fn gen_id(&mut self, prefix: &str) -> String {
        self.next_id += 1;
        format!("{}_{}", prefix, self.next_id)
    }

    // --- tree construction ---

    /// Route a single vector down the tree to find the best leaf node index.
    fn route_to_leaf(&self, vec: &[f32]) -> Option<usize> {
        let mut current = self.root_idx?;
        loop {
            match &self.nodes[current].kind {
                NodeKind::Leaf { .. } => return Some(current),
                NodeKind::Trunk { router_idx } => {
                    let router = &self.routers[*router_idx];
                    if router.n_attractors == 0 { return Some(current); }
                    // Greedy routing: pick single best child
                    let mut best_score = f32::NEG_INFINITY;
                    let mut best_child = current;
                    for i in 0..router.n_attractors {
                        let s = dot(vec, &router.attractors[i*self.dim..(i+1)*self.dim]);
                        if s > best_score {
                            best_score = s;
                            best_child = router.child_node_indices[i];
                        }
                    }
                    current = best_child;
                }
            }
        }
    }

    // --- tree construction ---

    fn recursive_build_tree(&mut self, parent_idx: Option<usize>, depth: usize, ids: Vec<String>, vecs_f32: &[f32]) -> usize {
        let dim = self.dim;
        let n = ids.len();

        // Base case: if under hard_cap (and not at root), or too small to meaningfully cluster, make a Leaf
        if (n < self.hard_cap && depth > 0) || n < 5000 {
            let gidx = self.alloc_graph();
            let leaf_id = self.gen_id("leaf");
            
            // Calculate centroid
            let mut centroid = vec![0.0; dim];
            for i in 0..n {
                for d in 0..dim {
                    centroid[d] += vecs_f32[i*dim+d];
                }
            }
            if n > 0 {
                for d in 0..dim { centroid[d] /= n as f32; }
            }

            let leaf_idx = self.alloc_node(NodeData {
                node_id: leaf_id,
                parent_idx,
                depth: depth as u32,
                centroid,
                kind: NodeKind::Leaf { graph_idx: gidx, semantic_variance: 0.0 },
            });
            
            let vecs_i8 = quantize_f32_to_i8(vecs_f32);
            let t0 = std::time::Instant::now();
            batch_insert_to_graph(&mut self.graphs[gidx], ids, &vecs_i8, dim);
            println!("[AIGVS-Rust] Built leaf at depth {} ({} vectors) in {:.2}s", depth, n, t0.elapsed().as_secs_f64());
            
            TREE_DEPTH_MAX.fetch_max(depth as u64, AO::Relaxed);
            return leaf_idx;
        }

        // Recursive case: make a Trunk
        let router_idx = self.alloc_router();
        let trunk_id = self.gen_id("trunk");
        let trunk_idx = self.alloc_node(NodeData {
            node_id: trunk_id,
            parent_idx,
            depth: depth as u32,
            centroid: vec![0.0; dim],
            kind: NodeKind::Trunk { router_idx },
        });

        // Determine num_clusters
        let num_clusters = (n as f32 / 10000.0).sqrt().max(8.0).min(64.0) as usize;
        println!("[AIGVS-Rust] Partitioning {} vectors into {} macro-attractors at depth {}...", n, num_clusters, depth);
        
        let mut centroids = farthest_point_seeds(vecs_f32, n, dim, num_clusters);
        
        if check_cohesion(&centroids, num_clusters, dim, 0.85) && n < 500_000 {
            println!("[AIGVS-Rust] Data is highly cohesive - forcing leaf despite size.");
            // Could turn this trunk into a leaf by recursing with a huge hard_cap; simpler to keep
            // running Hopfield for macro discovery.
        }

        hopfield_convergence(vecs_f32, &mut centroids, n, num_clusters, dim, 15, 5.0);

        let polysemy_thresh = self.polysemy_base * self.polysemy_decay.powi(depth as i32);
        let mut cluster_assignments: Vec<Vec<usize>> = vec![Vec::new(); num_clusters];

        for i in 0..n {
            let v = &vecs_f32[i*dim..(i+1)*dim];
            let mut max_sim = f32::NEG_INFINITY;
            let mut sims = Vec::with_capacity(num_clusters);
            for j in 0..num_clusters {
                let s = dot(v, &centroids[j*dim..(j+1)*dim]);
                sims.push(s);
                if s > max_sim { max_sim = s; }
            }
            for j in 0..num_clusters {
                if sims[j] >= max_sim - polysemy_thresh {
                    cluster_assignments[j].push(i);
                }
            }
        }

        for c in 0..num_clusters {
            if cluster_assignments[c].is_empty() { continue; }
            
            let mut c_ids = Vec::with_capacity(cluster_assignments[c].len());
            let mut c_vecs_f32 = Vec::with_capacity(cluster_assignments[c].len() * dim);
            
            for &idx in &cluster_assignments[c] {
                c_ids.push(ids[idx].clone());
                c_vecs_f32.extend_from_slice(&vecs_f32[idx*dim..(idx+1)*dim]);
            }
            
            // Recurse!
            let child_idx = self.recursive_build_tree(Some(trunk_idx), depth + 1, c_ids, &c_vecs_f32);
            
            let router = &mut self.routers[router_idx];
            router.attractors.extend_from_slice(&centroids[c*dim..(c+1)*dim]);
            router.child_node_indices.push(child_idx);
            router.n_attractors += 1;
        }

        // Calculate trunk centroid from children (optional, but good practice)
        let router = &self.routers[router_idx];
        let mut trunk_centroid = vec![0.0; dim];
        if router.n_attractors > 0 {
            for j in 0..router.n_attractors {
                for d in 0..dim {
                    trunk_centroid[d] += router.attractors[j*dim+d];
                }
            }
            for d in 0..dim { trunk_centroid[d] /= router.n_attractors as f32; }
        }
        self.nodes[trunk_idx].centroid = trunk_centroid;

        trunk_idx
    }

    // --- dynamic splitting (Hopfield bifurcation) ---

    fn split_leaf(&mut self, leaf_node_idx: usize) {
        let t_start = std::time::Instant::now();
        let dim = self.dim;

        let (graph_idx, _old_var) = match &self.nodes[leaf_node_idx].kind {
            NodeKind::Leaf { graph_idx, semantic_variance } => (*graph_idx, *semantic_variance),
            _ => return,
        };

        let depth = self.nodes[leaf_node_idx].depth;
        let parent_idx = self.nodes[leaf_node_idx].parent_idx;

        // 1. Extract and dequantize vectors
        let chunk_ids: Vec<String> = self.graphs[graph_idx].chunk_ids.clone();
        let n = chunk_ids.len();
        if n < 2 { return; }

        let f32_vecs = dequantize_i8_to_f32(&self.graphs[graph_idx].vector_cache);

        // 2. Determine cluster count
        let num_clusters = (n as f32 / 10000.0).sqrt().max(3.0).min(16.0) as usize;

        // 3. Farthest-point seeding + Hopfield convergence
        let mut centroids = farthest_point_seeds(&f32_vecs, n, dim, num_clusters);

        // Cohesion check: if data is too tight, don't split
        if check_cohesion(&centroids, num_clusters, dim, 0.85) {
            println!("[AIGVS-Rust] Split aborted: leaf at depth {} is too cohesive.", depth);
            return;
        }

        hopfield_convergence(&f32_vecs, &mut centroids, n, num_clusters, dim, 15, 5.0);

        // 4. Depth-adaptive polysemy
        let polysemy_thresh = self.polysemy_base * self.polysemy_decay.powi(depth as i32);
        let mut cluster_assignments: Vec<Vec<usize>> = vec![Vec::new(); num_clusters];

        for i in 0..n {
            let v = &f32_vecs[i*dim..(i+1)*dim];
            let mut max_sim = f32::NEG_INFINITY;
            let mut sims = Vec::with_capacity(num_clusters);
            for j in 0..num_clusters {
                let s = dot(v, &centroids[j*dim..(j+1)*dim]);
                sims.push(s);
                if s > max_sim { max_sim = s; }
            }
            for j in 0..num_clusters {
                if sims[j] >= max_sim - polysemy_thresh {
                    cluster_assignments[j].push(i);
                }
            }
        }

        // 5. Convert old leaf into a trunk
        let new_router_idx = self.alloc_router();
        self.nodes[leaf_node_idx].kind = NodeKind::Trunk { router_idx: new_router_idx };

        // 6. Create new leaf children
        let new_depth = depth + 1;
        TREE_DEPTH_MAX.fetch_max(new_depth as u64, AO::Relaxed);

        for c in 0..num_clusters {
            if cluster_assignments[c].is_empty() { continue; }

            let new_gidx = self.alloc_graph();
            let leaf_id = self.gen_id("leaf");
            let leaf_idx = self.alloc_node(NodeData {
                node_id: leaf_id,
                parent_idx: Some(leaf_node_idx),
                depth: new_depth,
                centroid: centroids[c*dim..(c+1)*dim].to_vec(),
                kind: NodeKind::Leaf { graph_idx: new_gidx, semantic_variance: 0.0 },
            });

            let router = &mut self.routers[new_router_idx];
            router.attractors.extend_from_slice(&centroids[c*dim..(c+1)*dim]);
            router.child_node_indices.push(leaf_idx);
            router.n_attractors += 1;

            let c_ids: Vec<String> = cluster_assignments[c].iter().map(|&i| chunk_ids[i].clone()).collect();
            let c_vecs_i8: Vec<i8> = cluster_assignments[c].iter()
                .flat_map(|&i| self.graphs[graph_idx].vector_cache[i*dim..(i+1)*dim].to_vec())
                .collect();

            batch_insert_to_graph(&mut self.graphs[new_gidx], c_ids, &c_vecs_i8, dim);
        }

        // 7. Clear old graph memory
        self.graphs[graph_idx].chunk_ids.clear();
        self.graphs[graph_idx].vector_cache.clear();
        self.graphs[graph_idx].local_graph.clear();
        self.graphs[graph_idx].deleted_indices.clear();
        self.graphs[graph_idx].entry_point_idx = None;

        let redistributed = n;
        SPLIT_COUNT.fetch_add(1, AO::Relaxed);
        SPLIT_DURATION_US.fetch_add(t_start.elapsed().as_micros() as u64, AO::Relaxed);
        VECTORS_REDISTRIBUTED.fetch_add(redistributed as u64, AO::Relaxed);

        println!("[AIGVS-Rust] Split leaf at depth {} -> {} sub-leaves ({} vectors redistributed, {:.2}s)",
            depth, num_clusters, redistributed, t_start.elapsed().as_secs_f64());

        // 8. Recursive: check if any new leaves also need splitting
        if let NodeKind::Trunk { router_idx } = &self.nodes[leaf_node_idx].kind {
            let children: Vec<usize> = self.routers[*router_idx].child_node_indices.clone();
            for child_idx in children {
                if let NodeKind::Leaf { graph_idx, .. } = &self.nodes[child_idx].kind {
                    let count = self.graphs[*graph_idx].chunk_ids.len();
                    if count >= self.hard_cap {
                        self.split_leaf(child_idx);
                    }
                }
            }
        }
    }

    // --- tree construction ---

    fn ingest_batch_internal(&mut self, ids: Vec<String>, vecs_f32: &[f32]) {
        let dim = self.dim;
        let n = ids.len();
        if n == 0 { return; }

        let t0 = std::time::Instant::now();

        if self.root_idx.is_none() {
            // First batch: build the initial tree recursively
            let root_idx = self.recursive_build_tree(None, 0, ids, vecs_f32);
            self.root_idx = Some(root_idx);
        } else {
            // Subsequent batches: route into existing tree
            // Group vectors by destination leaf for efficient batch insert
            let mut leaf_batches: HashMap<usize, (Vec<String>, Vec<i8>)> = HashMap::new();

            for i in 0..n {
                let v = &vecs_f32[i*dim..(i+1)*dim];
                if let Some(leaf_node_idx) = self.route_to_leaf(v) {
                    if let NodeKind::Leaf { graph_idx, .. } = &self.nodes[leaf_node_idx].kind {
                        let entry = leaf_batches.entry(*graph_idx).or_insert_with(|| (Vec::new(), Vec::new()));
                        entry.0.push(ids[i].clone());
                        entry.1.extend(quantize_f32_to_i8(v));
                    }
                }
            }

            // Batch insert into each leaf
            for (gidx, (batch_ids, batch_vecs)) in leaf_batches {
                batch_insert_to_graph(&mut self.graphs[gidx], batch_ids, &batch_vecs, dim);
            }

            // Check split triggers on all leaves
            let all_leaf_info: Vec<(usize, usize)> = self.nodes.iter().enumerate()
                .filter_map(|(idx, node)| {
                    if let NodeKind::Leaf { graph_idx, .. } = &node.kind {
                        Some((idx, self.graphs[*graph_idx].chunk_ids.len()))
                    } else { None }
                })
                .collect();

            for (leaf_idx, count) in all_leaf_info {
                if count >= self.hard_cap {
                    self.split_leaf(leaf_idx);
                }
            }
        }

        INGEST_BATCH_US.fetch_add(t0.elapsed().as_micros() as u64, AO::Relaxed);
        println!("[AIGVS-Rust] Ingested batch of {} vectors in {:.2}s", n, t0.elapsed().as_secs_f64());
    }

    fn insert_single_internal(&mut self, id: String, vec_f32: &[f32]) {
        let dim = self.dim;

        if self.root_idx.is_none() {
            // Bootstrap: create root + single leaf
            let root = self.recursive_build_tree(None, 0, vec![id], vec_f32);
            self.root_idx = Some(root);
            return;
        }

        if let Some(leaf_node_idx) = self.route_to_leaf(vec_f32) {
            if let NodeKind::Leaf { graph_idx, .. } = &self.nodes[leaf_node_idx].kind {
                let gidx = *graph_idx;
                let vec_i8 = quantize_f32_to_i8(vec_f32);
                single_insert_to_graph(&mut self.graphs[gidx], id, &vec_i8, dim);

                // Update centroid (running mean)
                let count = self.graphs[gidx].chunk_ids.len();
                let centroid = &mut self.nodes[leaf_node_idx].centroid;
                for d in 0..dim {
                    centroid[d] = (centroid[d] * (count - 1) as f32 + vec_f32[d]) / count as f32;
                }
                normalize_f32(centroid);

                // Check split trigger
                if count >= self.hard_cap {
                    self.split_leaf(leaf_node_idx);
                }
            }
        }
    }

    // --- tree construction ---

    fn to_state(&self) -> EngineState {
        let nodes: Vec<NodeDataSer> = self.nodes.iter().map(|n| {
            let (is_trunk, idx, sv) = match &n.kind {
                NodeKind::Trunk { router_idx } => (true, *router_idx, 0.0),
                NodeKind::Leaf { graph_idx, semantic_variance } => (false, *graph_idx, *semantic_variance),
            };
            NodeDataSer {
                node_id: n.node_id.clone(), parent_idx: n.parent_idx,
                depth: n.depth, centroid: n.centroid.clone(),
                is_trunk, router_or_graph_idx: idx, semantic_variance: sv,
            }
        }).collect();

        let routers: Vec<RouterDataSer> = self.routers.iter().map(|r| RouterDataSer {
            attractors: r.attractors.clone(),
            child_node_indices: r.child_node_indices.clone(),
            n_attractors: r.n_attractors,
        }).collect();

        let graphs: Vec<GraphDataSer> = self.graphs.iter().map(|g| {
            let adjacency: Vec<Vec<usize>> = g.local_graph.iter().map(|lock| lock.read().clone()).collect();
            GraphDataSer {
                chunk_ids: g.chunk_ids.clone(), vector_cache: g.vector_cache.clone(),
                adjacency, deleted_indices: g.deleted_indices.iter().cloned().collect(),
                entry_point_idx: g.entry_point_idx,
                max_connections: g.max_connections, ef_construction: g.ef_construction,
            }
        }).collect();

        EngineState {
            dim: self.dim, max_connections: self.max_connections,
            ef_construction: self.ef_construction,
            polysemy_base: self.polysemy_base, polysemy_decay: self.polysemy_decay,
            variance_split_thresh: self.variance_split_thresh, hard_cap: self.hard_cap,
            next_id: self.next_id, root_idx: self.root_idx,
            nodes, routers, graphs,
        }
    }

    fn from_state(state: EngineState) -> Self {
        let nodes: Vec<NodeData> = state.nodes.iter().map(|n| {
            let kind = if n.is_trunk {
                NodeKind::Trunk { router_idx: n.router_or_graph_idx }
            } else {
                NodeKind::Leaf { graph_idx: n.router_or_graph_idx, semantic_variance: n.semantic_variance }
            };
            NodeData {
                node_id: n.node_id.clone(), parent_idx: n.parent_idx,
                depth: n.depth, centroid: n.centroid.clone(), kind,
            }
        }).collect();

        let mut node_name_to_idx = HashMap::new();
        for (i, n) in nodes.iter().enumerate() {
            node_name_to_idx.insert(n.node_id.clone(), i);
        }

        let routers: Vec<RouterData> = state.routers.into_iter().map(|r| RouterData {
            attractors: r.attractors, child_node_indices: r.child_node_indices,
            n_attractors: r.n_attractors,
        }).collect();

        let graphs: Vec<GraphData> = state.graphs.into_iter().map(|g| {
            let local_graph: Vec<RwLock<Vec<usize>>> = g.adjacency.into_iter()
                .map(|adj| RwLock::new(adj)).collect();
            GraphData {
                chunk_ids: g.chunk_ids, vector_cache: g.vector_cache,
                local_graph, deleted_indices: g.deleted_indices.into_iter().collect(),
                entry_point_idx: g.entry_point_idx,
                max_connections: g.max_connections, ef_construction: g.ef_construction,
            }
        }).collect();

        AIGVSEngine {
            dim: state.dim, max_connections: state.max_connections,
            ef_construction: state.ef_construction,
            nodes, node_name_to_idx, root_idx: state.root_idx,
            routers, graphs, next_id: state.next_id,
            polysemy_base: state.polysemy_base, polysemy_decay: state.polysemy_decay,
            variance_split_thresh: state.variance_split_thresh, hard_cap: state.hard_cap,
        }
    }
}

// --- profiling counters ---

#[pymethods]
impl AIGVSEngine {
    #[new]
    #[pyo3(signature = (embedding_dim, max_connections=12, ef_construction=80, polysemy_base=0.02,
                        polysemy_decay=0.5, hard_cap=150000))]
    fn new(
        embedding_dim: usize, max_connections: usize, ef_construction: usize,
        polysemy_base: f32, polysemy_decay: f32, hard_cap: usize,
    ) -> Self {
        AIGVSEngine {
            dim: embedding_dim, max_connections, ef_construction,
            nodes: Vec::new(), node_name_to_idx: HashMap::new(),
            root_idx: None, routers: Vec::new(), graphs: Vec::new(),
            next_id: 0, polysemy_base, polysemy_decay,
            variance_split_thresh: 0.5, hard_cap,
        }
    }

    /// Bulk ingestion; Rust builds or extends the tree.
    fn ingest_batch(&mut self, ids: Vec<String>, vectors: PyReadonlyArray2<f32>) {
        let arr = vectors.as_array();
        let n = ids.len();
        let dim = self.dim;

        // Normalize and flatten
        let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
        for i in 0..n {
            let row = arr.row(i);
            let mut v: Vec<f32> = row.iter().cloned().collect();
            normalize_f32(&mut v);
            flat.extend_from_slice(&v);
        }

        self.ingest_batch_internal(ids, &flat);
    }

    /// Single real-time insert.
    fn insert_single(&mut self, id: String, vector: PyReadonlyArray1<f32>) {
        let mut v: Vec<f32> = vector.as_array().iter().cloned().collect();
        normalize_f32(&mut v);
        self.insert_single_internal(id, &v);
    }

    /// Parallel multi-leaf search with Hopfield routing and NSW graph traversal.
    #[pyo3(signature = (query_vector, top_k, max_beam=10, target_confidence=0.95, beta=5.0, ef_search=100))]
    fn unified_search(
        &self, py: Python, query_vector: PyReadonlyArray1<f32>,
        top_k: usize, max_beam: usize, target_confidence: f32,
        beta: f32, ef_search: usize,
    ) -> Vec<String> {
        let root = match self.root_idx { Some(r) => r, None => return vec![] };
        let ef = ef_search.max(top_k);

        let mut q: Vec<f32> = query_vector.as_array().iter().cloned().collect();
        normalize_f32(&mut q);
        let q_i8 = quantize_f32_to_i8(&q);
        let dim = self.dim;

        let nodes = &self.nodes;
        let routers = &self.routers;
        let graphs = &self.graphs;

        py.allow_threads(|| {
            // Phase 1: sequential routing to collect leaf graph indices
            let mut stack: Vec<usize> = vec![root];
            let mut leaf_graph_indices: Vec<usize> = Vec::new();

            while let Some(curr_idx) = stack.pop() {
                match &nodes[curr_idx].kind {
                    NodeKind::Trunk { router_idx } => {
                        let router = &routers[*router_idx];
                        let selected = router_beam_search(
                            &router.attractors, &router.child_node_indices,
                            &q, dim, router.n_attractors,
                            max_beam, target_confidence, beta,
                        );
                        stack.extend(selected);
                    }
                    NodeKind::Leaf { graph_idx, .. } => {
                        leaf_graph_indices.push(*graph_idx);
                    }
                }
            }

            LEAVES_VISITED.fetch_add(leaf_graph_indices.len() as u64, AO::Relaxed);

            // Phase 2: parallel NSW search across all selected leaves
            let t_child = std::time::Instant::now();
            let all_candidates: Vec<(String, f32)> = leaf_graph_indices
                .par_iter()
                .flat_map(|&gidx| nsw_query(&graphs[gidx], &q_i8, dim, ef))
                .collect();
            CHILD_QUERY_TOTAL_US.fetch_add(t_child.elapsed().as_micros() as u64, AO::Relaxed);

            // Phase 3: global rerank, dedup, top-k
            let t_rerank = std::time::Instant::now();
            let mut sorted = all_candidates;
            sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

            let mut seen: HashSet<String> = HashSet::new();
            let top_results: Vec<String> = sorted.into_iter()
                .filter(|(id, _)| seen.insert(id.clone()))
                .take(top_k)
                .map(|(id, _)| id)
                .collect();
            RERANK_TOTAL_US.fetch_add(t_rerank.elapsed().as_micros() as u64, AO::Relaxed);
            QUERY_COUNT.fetch_add(1, AO::Relaxed);

            top_results
        })
    }

    /// Mark a chunk as deleted (tombstone).
    fn delete(&mut self, chunk_id: String) -> bool {
        for graph in &mut self.graphs {
            if let Some(pos) = graph.chunk_ids.iter().position(|x| *x == chunk_id) {
                graph.deleted_indices.insert(pos);
                return true;
            }
        }
        false
    }

    /// Save engine state to disk (binary).
    fn save(&self, path: String) -> PyResult<()> {
        let state = self.to_state();
        let encoded = bincode::serialize(&state)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Serialize failed: {}", e)))?;
        std::fs::write(&path, &encoded)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Write failed: {}", e)))?;
        println!("[AIGVS-Rust] Saved engine ({} bytes) to {}", encoded.len(), path);
        Ok(())
    }

    /// Load engine state from disk.
    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        let data = std::fs::read(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Read failed: {}", e)))?;
        let state: EngineState = bincode::deserialize(&data)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Deserialize failed: {}", e)))?;
        let engine = AIGVSEngine::from_state(state);
        println!("[AIGVS-Rust] Loaded engine from {} ({} nodes, {} graphs)",
            path, engine.nodes.len(), engine.graphs.len());
        Ok(engine)
    }

    /// Return tree statistics.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<pyo3::Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let n_trunks = self.nodes.iter().filter(|n| matches!(n.kind, NodeKind::Trunk { .. })).count();
        let n_leaves = self.nodes.iter().filter(|n| matches!(n.kind, NodeKind::Leaf { .. })).count();
        let total_vectors: usize = self.graphs.iter().map(|g| g.chunk_ids.len()).sum();
        let max_depth = self.nodes.iter().map(|n| n.depth).max().unwrap_or(0);

        dict.set_item("trunks", n_trunks)?;
        dict.set_item("leaves", n_leaves)?;
        dict.set_item("total_indexed_vectors", total_vectors)?;
        dict.set_item("max_depth", max_depth)?;
        dict.set_item("total_nodes", self.nodes.len())?;
        dict.set_item("routers", self.routers.len())?;
        dict.set_item("graphs", self.graphs.len())?;
        Ok(dict)
    }

    /// Print profiling statistics.
    #[pyo3(signature = ())]
    fn print_profile_stats(&self) {
        let count = QUERY_COUNT.load(AO::Relaxed).max(1);
        println!("\n===== AIGVS RUST PROFILE =====");
        println!("Queries: {}", count);
        println!("Router Total:       {:>8.2} us/query", ROUTER_TOTAL_US.load(AO::Relaxed) as f64 / count as f64);
        println!("Child Query Time:   {:>8.2} us/query", CHILD_QUERY_TOTAL_US.load(AO::Relaxed) as f64 / count as f64);
        println!("Rerank Time:        {:>8.2} us/query", RERANK_TOTAL_US.load(AO::Relaxed) as f64 / count as f64);
        println!("Leaves Visited:     {:>8.2} leaves/query", LEAVES_VISITED.load(AO::Relaxed) as f64 / count as f64);
        println!("--- Ingestion ---");
        println!("Splits:             {}", SPLIT_COUNT.load(AO::Relaxed));
        println!("Split Avg Duration: {:>8.2} ms", SPLIT_DURATION_US.load(AO::Relaxed) as f64 / SPLIT_COUNT.load(AO::Relaxed).max(1) as f64 / 1000.0);
        println!("Vectors Redistributed: {}", VECTORS_REDISTRIBUTED.load(AO::Relaxed));
        println!("Max Tree Depth:     {}", TREE_DEPTH_MAX.load(AO::Relaxed));
        println!("Total Ingest Time:  {:>8.2} ms", INGEST_BATCH_US.load(AO::Relaxed) as f64 / 1000.0);
        println!("==============================\n");
    }

    /// Reset all profiling counters.
    fn reset_profile(&self) {
        ROUTER_TOTAL_US.store(0, AO::Relaxed);
        CHILD_QUERY_TOTAL_US.store(0, AO::Relaxed);
        RERANK_TOTAL_US.store(0, AO::Relaxed);
        LEAVES_VISITED.store(0, AO::Relaxed);
        QUERY_COUNT.store(0, AO::Relaxed);
        SPLIT_COUNT.store(0, AO::Relaxed);
        SPLIT_DURATION_US.store(0, AO::Relaxed);
        VECTORS_REDISTRIBUTED.store(0, AO::Relaxed);
        INGEST_BATCH_US.store(0, AO::Relaxed);
        TREE_DEPTH_MAX.store(0, AO::Relaxed);
    }

    // --- legacy compatibility (kept for the old benchmarks during the migration) ---

    #[pyo3(signature = (node_id, parent_id=None, is_root=false))]
    fn register_branch(&mut self, node_id: String, parent_id: Option<String>, is_root: bool) {
        let router_idx = self.alloc_router();
        let parent_idx = parent_id.as_ref().and_then(|pid| self.node_name_to_idx.get(pid).copied());
        let node_idx = self.alloc_node(NodeData {
            node_id: node_id.clone(), parent_idx, depth: 0,
            centroid: vec![0.0; self.dim],
            kind: NodeKind::Trunk { router_idx },
        });
        if is_root { self.root_idx = Some(node_idx); }
        // Wire into parent's router
        if let Some(p_idx) = parent_idx {
            if let NodeKind::Trunk { router_idx: pr_idx } = &self.nodes[p_idx].kind {
                let router = &mut self.routers[*pr_idx];
                router.child_node_indices.push(node_idx);
                router.attractors.extend(vec![0.0f32; self.dim]);
                router.n_attractors += 1;
            }
        }
    }

    #[pyo3(signature = (node_id, parent_id, max_connections=16, ef_construction=100))]
    fn register_leaf(&mut self, node_id: String, parent_id: String, max_connections: usize, ef_construction: usize) {
        let graph_idx = self.graphs.len();
        self.graphs.push(GraphData {
            chunk_ids: Vec::new(), vector_cache: Vec::new(),
            local_graph: Vec::new(), deleted_indices: HashSet::new(),
            entry_point_idx: None, max_connections, ef_construction,
        });
        let parent_idx = self.node_name_to_idx.get(&parent_id).copied();
        let node_idx = self.alloc_node(NodeData {
            node_id: node_id.clone(), parent_idx, depth: 1,
            centroid: vec![0.0; self.dim],
            kind: NodeKind::Leaf { graph_idx, semantic_variance: 0.0 },
        });
        if let Some(p_idx) = parent_idx {
            if let NodeKind::Trunk { router_idx } = &self.nodes[p_idx].kind {
                let router = &mut self.routers[*router_idx];
                router.child_node_indices.push(node_idx);
                router.attractors.extend(vec![0.0f32; self.dim]);
                router.n_attractors += 1;
            }
        }
    }

    fn update_attractor(&mut self, router_id: String, child_id: String, centroid: PyReadonlyArray1<f32>) {
        let r_node_idx = match self.node_name_to_idx.get(&router_id) { Some(&i) => i, None => return };
        let router_idx = match &self.nodes[r_node_idx].kind {
            NodeKind::Trunk { router_idx } => *router_idx, _ => return,
        };
        let child_node_idx = match self.node_name_to_idx.get(&child_id) { Some(&i) => i, None => return };
        let vec: Vec<f32> = centroid.as_array().iter().copied().collect();
        let dim = self.dim;
        let router = &mut self.routers[router_idx];
        if let Some(pos) = router.child_node_indices.iter().position(|&x| x == child_node_idx) {
            let start = pos * dim;
            router.attractors[start..start+dim].copy_from_slice(&vec);
        }
    }

    fn remove_attractor(&mut self, router_id: String, child_id: String) {
        let r_node_idx = match self.node_name_to_idx.get(&router_id) { Some(&i) => i, None => return };
        let router_idx = match &self.nodes[r_node_idx].kind {
            NodeKind::Trunk { router_idx } => *router_idx, _ => return,
        };
        let child_node_idx = match self.node_name_to_idx.get(&child_id) { Some(&i) => i, None => return };
        let dim = self.dim;
        let router = &mut self.routers[router_idx];
        if let Some(pos) = router.child_node_indices.iter().position(|&x| x == child_node_idx) {
            router.child_node_indices.remove(pos);
            router.attractors.drain(pos*dim..(pos+1)*dim);
            router.n_attractors -= 1;
        }
    }

    fn native_insert_to_leaf(&mut self, leaf_id: String, chunk_id: String, vector: PyReadonlyArray1<f32>) {
        let l_idx = match self.node_name_to_idx.get(&leaf_id) { Some(&i) => i, None => return };
        let graph_idx = match &self.nodes[l_idx].kind {
            NodeKind::Leaf { graph_idx, .. } => *graph_idx, _ => return,
        };
        let v: Vec<f32> = vector.as_array().iter().copied().collect();
        let v_i8 = quantize_f32_to_i8(&v);
        single_insert_to_graph(&mut self.graphs[graph_idx], chunk_id, &v_i8, self.dim);
    }

    #[pyo3(signature = (leaf_id, chunk_ids, vectors))]
    fn native_batch_insert_to_leaf(&mut self, leaf_id: String, chunk_ids: Vec<String>, vectors: PyReadonlyArray2<f32>) {
        let l_idx = match self.node_name_to_idx.get(&leaf_id) { Some(&i) => i, None => return };
        let graph_idx = match &self.nodes[l_idx].kind {
            NodeKind::Leaf { graph_idx, .. } => *graph_idx, _ => return,
        };
        let arr = vectors.as_array();
        let n = chunk_ids.len();
        let dim = self.dim;
        let mut flat_i8: Vec<i8> = Vec::with_capacity(n * dim);
        if let Some(slc) = arr.as_slice() {
            flat_i8 = quantize_f32_to_i8(slc);
        } else {
            for i in 0..n {
                let row = arr.row(i);
                let v: Vec<f32> = row.iter().cloned().collect();
                flat_i8.extend(quantize_f32_to_i8(&v));
            }
        }
        batch_insert_to_graph(&mut self.graphs[graph_idx], chunk_ids, &flat_i8, dim);
    }

    fn native_mark_deleted(&mut self, leaf_id: String, chunk_id: String) {
        let l_idx = match self.node_name_to_idx.get(&leaf_id) { Some(&i) => i, None => return };
        let graph_idx = match &self.nodes[l_idx].kind {
            NodeKind::Leaf { graph_idx, .. } => *graph_idx, _ => return,
        };
        let graph = &mut self.graphs[graph_idx];
        if let Some(pos) = graph.chunk_ids.iter().position(|x| *x == chunk_id) {
            graph.deleted_indices.insert(pos);
        }
    }

    #[pyo3(signature = (node_id, query_vector, max_beam=1, target_confidence=1.0, beta=20.0))]
    fn route_query(&self, _py: Python, node_id: String, query_vector: PyReadonlyArray1<f32>,
        max_beam: usize, target_confidence: f32, beta: f32,
    ) -> Vec<String> {
        let node_idx = match self.node_name_to_idx.get(&node_id) { Some(&i) => i, None => return vec![] };
        let router_idx = match &self.nodes[node_idx].kind {
            NodeKind::Trunk { router_idx } => *router_idx, _ => return vec![],
        };
        let router = &self.routers[router_idx];
        if router.n_attractors == 0 { return vec![]; }

        let q: Vec<f32> = query_vector.as_array().iter().copied().collect();
        let dim = self.dim;

        let selected_node_indices = router_beam_search(
            &router.attractors, &router.child_node_indices,
            &q, dim, router.n_attractors, max_beam, target_confidence, beta,
        );

        selected_node_indices.iter()
            .map(|&ni| self.nodes[ni].node_id.clone())
            .collect()
    }
}

// --- profiling counters ---

#[pymodule]
fn aigvs_rust(m: &pyo3::Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AIGVSEngine>()?;
    Ok(())
}