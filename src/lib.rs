use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashSet, HashMap};
use parking_lot::RwLock;
use std::cmp::Ordering;
use std::sync::atomic::AtomicU64;

static ROUTER_SCORE_US: AtomicU64 = AtomicU64::new(0);
static ROUTER_SOFTMAX_US: AtomicU64 = AtomicU64::new(0);
static ROUTER_SORT_US: AtomicU64 = AtomicU64::new(0);
static ROUTER_TOTAL_US: AtomicU64 = AtomicU64::new(0);
static CHILD_QUERY_TOTAL_US: AtomicU64 = AtomicU64::new(0);
static RERANK_TOTAL_US: AtomicU64 = AtomicU64::new(0);
static CANDIDATE_VECTORS: AtomicU64 = AtomicU64::new(0);
static LEAVES_VISITED: AtomicU64 = AtomicU64::new(0);
static QUERY_COUNT: AtomicU64 = AtomicU64::new(0);

// --- float ordering wrappers for the heaps ---

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

// --- float ordering wrappers for the heaps ---

enum ExecutionNode {
    Branch {
        router_idx: usize,
        children_mapping: HashMap<String, usize>,
    },
    Leaf {
        leaf_idx: usize,
    },
}

#[pyclass]
struct AIGVSEngine {
    embedding_dim: usize,
    nodes: Vec<ExecutionNode>,
    node_id_to_index: HashMap<String, usize>,
    root_index: Option<usize>,
    routers: Vec<RustHopfieldRouterInternal>,
    leaves: Vec<RustFlatLeafInternal>,
}

struct RustHopfieldRouterInternal {
    id_lookup: Vec<String>,
    /// Flattened row-major attractor matrix: row i = x_matrix[i*dim .. (i+1)*dim].
    x_matrix: Vec<f32>,
}

struct RustFlatLeafInternal {
    chunk_ids: Vec<String>,
    /// Flattened row-major vector cache: row i = vector_cache[i*dim .. (i+1)*dim].
    vector_cache: Vec<i8>,
    deleted_indices: HashSet<usize>,
}

// --- dot product --- 4-way unrolled. For normalised vectors this is cosine similarity. With -C
// target-cpu=native the compiler vectorises it (AVX2 / AVX-512).

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
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { dot_i8_avx2(a, b) };
        }
    }
    dot_i8_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dot_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let unrolled = len - (len % 32);
    let mut sum = _mm256_setzero_si256();
    let mut i = 0;
    while i < unrolled {
        let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
        
        let va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
        let vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
        let va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
        let vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
        
        let dot_lo = _mm256_madd_epi16(va_lo, vb_lo);
        let dot_hi = _mm256_madd_epi16(va_hi, vb_hi);
        
        sum = _mm256_add_epi32(sum, _mm256_add_epi32(dot_lo, dot_hi));
        i += 32;
    }
    let mut arr = [0i32; 8];
    _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, sum);
    let mut total = arr.iter().sum();
    for j in i..len {
        total += *a.get_unchecked(j) as i32 * *b.get_unchecked(j) as i32;
    }
    total
}

#[inline(always)]
fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    let len = a.len();
    let unrolled = len - (len % 8);
    let mut s = 0i32;
    let mut i = 0;
    while i < unrolled {
        unsafe {
            s += *a.get_unchecked(i) as i32 * *b.get_unchecked(i) as i32;
            s += *a.get_unchecked(i + 1) as i32 * *b.get_unchecked(i + 1) as i32;
            s += *a.get_unchecked(i + 2) as i32 * *b.get_unchecked(i + 2) as i32;
            s += *a.get_unchecked(i + 3) as i32 * *b.get_unchecked(i + 3) as i32;
            s += *a.get_unchecked(i + 4) as i32 * *b.get_unchecked(i + 4) as i32;
            s += *a.get_unchecked(i + 5) as i32 * *b.get_unchecked(i + 5) as i32;
            s += *a.get_unchecked(i + 6) as i32 * *b.get_unchecked(i + 6) as i32;
            s += *a.get_unchecked(i + 7) as i32 * *b.get_unchecked(i + 7) as i32;
        }
        i += 8;
    }
    for j in unrolled..len {
        unsafe { s += *a.get_unchecked(j) as i32 * *b.get_unchecked(j) as i32; }
    }
    s
}

// --- router beam search --- Sequential, not rayon: spinning up rayon tasks (~5-20us) costs more
// than the dot products when a branch has fewer than ~200 children, which is the usual case. A
// tight loop over a contiguous f32 slice stays in L1.

#[inline]
fn router_beam_search(
    x_matrix: &[f32],
    id_lookup: &[String],
    children_mapping: &HashMap<String, usize>,
    q: &[f32],
    dim: usize,
    max_beam: usize,
    target_confidence: f32,
    beta: f32,
    activation_threshold: f32,
) -> Vec<usize> {
    if id_lookup.is_empty() { return vec![]; }

    let n = id_lookup.len();

    let t0 = std::time::Instant::now();
    // Score: tight sequential dot-product scan, fully in L1/L2 cache.
    let mut raw_scores: Vec<f32> = Vec::with_capacity(n);
    let mut max_s = f32::NEG_INFINITY;
    for row in x_matrix.chunks_exact(dim) {
        let s = dot(q, row);
        if s > max_s { max_s = s; }
        raw_scores.push(s);
    }
    let t1 = std::time::Instant::now();

    // Activation threshold: zero out children whose raw cosine is below the minimum (the OOD gate,
    // keeps queries out of unrelated branches).
    let mut any_above = false;
    for s in raw_scores.iter() {
        if *s >= activation_threshold { any_above = true; break; }
    }
    if !any_above { return vec![]; }

    let mut dynamic_beta = beta;
    if beta == 0.0 && n > 1 {
        let mut mean = 0.0;
        for &s in raw_scores.iter() { mean += s; }
        mean /= n as f32;
        let mut var = 0.0;
        for &s in raw_scores.iter() { var += (s - mean) * (s - mean); }
        var /= n as f32;
        if var > 1e-6 {
            dynamic_beta = 1.0 / var.sqrt();
        } else {
            dynamic_beta = 1.0;
        }
    }

    // Numerically stable softmax with temperature beta, only over activated children.
    let mut exp_sum = 0.0f32;
    let mut probs: Vec<(usize, f32)> = raw_scores
        .into_iter()
        .enumerate()
        .map(|(i, s)| {
            if s < activation_threshold {
                (i, 0.0f32)
            } else {
                let e = ((s - max_s) * dynamic_beta).exp();
                exp_sum += e;
                (i, e)
            }
        })
        .collect();
    for (_, p) in probs.iter_mut() {
        if *p > 0.0 { *p /= exp_sum + 1e-9; }
    }
    let t2 = std::time::Instant::now();

    probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let mut selected = Vec::with_capacity(max_beam.min(n));
    let mut cumulative = 0.0f32;
    for (idx, prob) in probs.into_iter().take(max_beam) {
        if prob == 0.0 { break; }  // All remaining are below threshold
        if let Some(&child_node_idx) = children_mapping.get(&id_lookup[idx]) {
            selected.push(child_node_idx);
        }
        cumulative += prob;
        if cumulative >= target_confidence { break; }
    }
    let t3 = std::time::Instant::now();

    ROUTER_SCORE_US.fetch_add(t1.duration_since(t0).as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    ROUTER_SOFTMAX_US.fetch_add(t2.duration_since(t1).as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    ROUTER_SORT_US.fetch_add(t3.duration_since(t2).as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    ROUTER_TOTAL_US.fetch_add(t3.duration_since(t0).as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    CANDIDATE_VECTORS.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);

    selected
}

// --- NSW insert helpers ---
fn flat_scan(
    leaf: &RustFlatLeafInternal,
    q: &[i8],
    dim: usize,
    ef: usize,
) -> Vec<(String, f32)> {
    let mut results: BinaryHeap<(MinScore, usize)> = BinaryHeap::with_capacity(ef);
    for (i, row) in leaf.vector_cache.chunks_exact(dim).enumerate() {
        if leaf.deleted_indices.contains(&i) { continue; }
        let s = dot_i8(q, row) as f32;
        if results.len() < ef {
            results.push((MinScore(s), i));
        } else if let Some(&(MinScore(worst), _)) = results.peek() {
            if s > worst {
                results.pop();
                results.push((MinScore(s), i));
            }
        }
    }
    let mut v: Vec<(f32, usize)> = results.into_iter().map(|(MinScore(s), idx)| (s, idx)).collect();
    v.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    v.into_iter().map(|(s, idx)| (leaf.chunk_ids[idx].clone(), s)).collect()
}

// --- NSW insert helpers ---

#[pymethods]
impl AIGVSEngine {
    #[new]
    fn new(embedding_dim: usize) -> Self {
        AIGVSEngine {
            embedding_dim,
            nodes: Vec::new(),
            node_id_to_index: HashMap::new(),
            root_index: None,
            routers: Vec::new(),
            leaves: Vec::new(),
        }
    }

    // --- tree construction ---

    #[pyo3(signature = (node_id, parent_id=None, is_root=false))]
    fn register_branch(&mut self, node_id: String, parent_id: Option<String>, is_root: bool) {
        let node_idx = self.nodes.len();
        let router_idx = self.routers.len();
        self.routers.push(RustHopfieldRouterInternal { id_lookup: Vec::new(), x_matrix: Vec::new() });
        self.nodes.push(ExecutionNode::Branch { router_idx, children_mapping: HashMap::new() });
        self.node_id_to_index.insert(node_id.clone(), node_idx);
        if is_root { self.root_index = Some(node_idx); }
        if let Some(pid) = parent_id {
            if let Some(&p_idx) = self.node_id_to_index.get(&pid) {
                if let ExecutionNode::Branch { ref mut children_mapping, .. } = self.nodes[p_idx] {
                    children_mapping.insert(node_id, node_idx);
                }
            }
        }
    }

    #[pyo3(signature = (node_id, parent_id))]
    fn register_leaf(&mut self, node_id: String, parent_id: String) {
        let node_idx = self.nodes.len();
        let leaf_idx = self.leaves.len();
        self.leaves.push(RustFlatLeafInternal {
            chunk_ids: Vec::new(),
            vector_cache: Vec::new(),
            deleted_indices: HashSet::new(),
        });
        self.nodes.push(ExecutionNode::Leaf { leaf_idx });
        self.node_id_to_index.insert(node_id.clone(), node_idx);
        if let Some(&p_idx) = self.node_id_to_index.get(&parent_id) {
            if let ExecutionNode::Branch { ref mut children_mapping, .. } = self.nodes[p_idx] {
                children_mapping.insert(node_id, node_idx);
            }
        }
    }

    // --- tree construction ---

    fn update_attractor(&mut self, router_id: String, child_id: String, centroid: PyReadonlyArray1<f32>) {
        let r_node_idx = match self.node_id_to_index.get(&router_id) { Some(&i) => i, None => return };
        let router_idx = match &self.nodes[r_node_idx] {
            ExecutionNode::Branch { router_idx, .. } => *router_idx, _ => return,
        };
        let vec: Vec<f32> = centroid.as_array().iter().copied().collect();
        let dim = self.embedding_dim;
        let router = &mut self.routers[router_idx];
        if let Some(pos) = router.id_lookup.iter().position(|x| *x == child_id) {
            let start = pos * dim;
            router.x_matrix[start..start + dim].copy_from_slice(&vec);
        } else {
            router.id_lookup.push(child_id);
            router.x_matrix.extend_from_slice(&vec);
        }
    }

    fn remove_attractor(&mut self, router_id: String, child_id: String) {
        let r_node_idx = match self.node_id_to_index.get(&router_id) { Some(&i) => i, None => return };
        let router_idx = match &self.nodes[r_node_idx] {
            ExecutionNode::Branch { router_idx, .. } => *router_idx, _ => return,
        };
        if let ExecutionNode::Branch { ref mut children_mapping, .. } = self.nodes[r_node_idx] {
            children_mapping.remove(&child_id);
        }
        let router = &mut self.routers[router_idx];
        if let Some(pos) = router.id_lookup.iter().position(|x| *x == child_id) {
            router.id_lookup.remove(pos);
            let dim = self.embedding_dim;
            let start = pos * dim;
            router.x_matrix.drain(start..start + dim);
        }
    }

    // --- tree construction ---

    fn native_insert_to_leaf(&mut self, leaf_id: String, chunk_id: String, vector: PyReadonlyArray1<f32>) {
        let l_idx = match self.node_id_to_index.get(&leaf_id) { Some(&i) => i, None => return };
        let leaf_idx = match &self.nodes[l_idx] { ExecutionNode::Leaf { leaf_idx } => *leaf_idx, _ => return };

        let leaf = &mut self.leaves[leaf_idx];
        let vec_data_f32: Vec<f32> = vector.as_array().iter().copied().collect();
        let vec_data: Vec<i8> = vec_data_f32.iter().map(|&x| (x * 127.0).clamp(-127.0, 127.0) as i8).collect();
        leaf.chunk_ids.push(chunk_id);
        leaf.vector_cache.extend_from_slice(&vec_data);
    }

    #[pyo3(signature = (leaf_id, chunk_ids, vectors))]
    fn native_batch_insert_to_leaf(&mut self, leaf_id: String, chunk_ids: Vec<String>, vectors: PyReadonlyArray2<f32>) {
        let l_idx = match self.node_id_to_index.get(&leaf_id) { Some(&i) => i, None => return };
        let leaf_idx = match &self.nodes[l_idx] { ExecutionNode::Leaf { leaf_idx } => *leaf_idx, _ => return };
        let leaf = &mut self.leaves[leaf_idx];
        let vectors_arr = vectors.as_array();
        let n_new = chunk_ids.len();
        if n_new == 0 { return; }
        
        leaf.chunk_ids.extend(chunk_ids.into_iter());
        if let Some(slc) = vectors_arr.as_slice() {
            let scaled: Vec<i8> = slc.iter().map(|&x| (x * 127.0).clamp(-127.0, 127.0) as i8).collect();
            leaf.vector_cache.extend_from_slice(&scaled);
        } else {
            for i in 0..n_new {
                let row = vectors_arr.row(i);
                let scaled: Vec<i8> = row.iter().map(|&x| (x * 127.0).clamp(-127.0, 127.0) as i8).collect();
                leaf.vector_cache.extend(scaled);
            }
        }
    }

    fn native_mark_deleted(&mut self, leaf_id: String, chunk_id: String) {
        let l_idx = match self.node_id_to_index.get(&leaf_id) { Some(&i) => i, None => return };
        let leaf_idx = match &self.nodes[l_idx] { ExecutionNode::Leaf { leaf_idx } => *leaf_idx, _ => return };
        let graph = &mut self.leaves[leaf_idx];
        if let Some(pos) = graph.chunk_ids.iter().position(|x| *x == chunk_id) {
            graph.deleted_indices.insert(pos);
        }
    }

    // --- search: parallel leaf traversal ---
    //   1. single-threaded walk to collect which leaves to search (routing is cheap, no rayon)
    //   2. rayon over the collected leaves; each leaf's NSW search is independent
    //   3. global rerank of all candidates, dedup, top-k
    // Keeps rayon on work units big enough to be worth it (a full NSW search over hundreds of
    // vectors per leaf).

    #[pyo3(signature = (query_vector, top_k, max_beam=5, target_confidence=0.90, beta=5.0, ef_search=100))]
    fn unified_search(
        &self,
        py: Python,
        query_vector: PyReadonlyArray1<f32>,
        top_k: usize,
        max_beam: usize,
        target_confidence: f32,
        beta: f32,
        ef_search: usize,
    ) -> Vec<String> {
        let root = match self.root_index { Some(r) => r, None => return vec![] };
        let ef = ef_search.max(top_k);

        let q: Vec<f32> = query_vector.as_array().iter().copied().collect();
        let q_i8: Vec<i8> = q.iter().map(|&x| (x * 127.0).clamp(-127.0, 127.0) as i8).collect();
        let dim = self.embedding_dim;
        let nodes  = &self.nodes;
        let routers = &self.routers;
        let leaves  = &self.leaves;

        py.allow_threads(|| {
            // phase 1: sequential routing to collect leaf indices (beam x children per level,
            // small)
            let mut stack: Vec<usize> = vec![root];
            let mut leaf_indices: Vec<usize> = Vec::new();

            while let Some(curr_idx) = stack.pop() {
                match &nodes[curr_idx] {
                    ExecutionNode::Branch { router_idx, children_mapping } => {
                        let router = &routers[*router_idx];
                        let selected = router_beam_search(
                            &router.x_matrix, &router.id_lookup, children_mapping,
                            &q, dim, max_beam, target_confidence, beta, f32::NEG_INFINITY,
                        );
                        stack.extend(selected);
                    }
                    ExecutionNode::Leaf { leaf_idx } => {
                        leaf_indices.push(*leaf_idx);
                    }
                }
            }

            LEAVES_VISITED.fetch_add(leaf_indices.len() as u64, std::sync::atomic::Ordering::Relaxed);

            // phase 2: parallel NSW search over the selected leaves; each task has its own result
            // Vec, merged afterwards
            let t_child = std::time::Instant::now();
            let all_candidates: Vec<(String, f32)> = leaf_indices
                .par_iter()
                .flat_map(|&lidx| {
                    flat_scan(&leaves[lidx], &q_i8, dim, ef)
                })
                .collect();
            CHILD_QUERY_TOTAL_US.fetch_add(t_child.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);

            // phase 3: global rerank, dedup, top-k
            let t_rerank = std::time::Instant::now();
            let mut sorted = all_candidates;
            sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

            let mut seen: HashSet<String> = HashSet::new();
            let top_results: Vec<String> = sorted.into_iter()
                .filter(|(id, _)| seen.insert(id.clone()))
                .take(top_k)
                .map(|(id, _)| id)
                .collect();
            RERANK_TOTAL_US.fetch_add(t_rerank.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            QUERY_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            
            top_results
        })
    }

    // --- scoped routing --- Finds the best child of one branch during insertion without starting
    // from the root. Sequential: inserts route one at a time and py.allow_threads isn't worth it
    // here.

    #[pyo3(signature = (node_id, query_vector, max_beam=1, target_confidence=1.0, beta=20.0))]
    fn route_query(
        &self,
        _py: Python,
        node_id: String,
        query_vector: PyReadonlyArray1<f32>,
        max_beam: usize,
        target_confidence: f32,
        beta: f32,
    ) -> Vec<String> {
        let node_idx = match self.node_id_to_index.get(&node_id) { Some(&i) => i, None => return vec![] };
        let (router_idx, children_mapping) = match &self.nodes[node_idx] {
            ExecutionNode::Branch { router_idx, children_mapping } => (*router_idx, children_mapping),
            ExecutionNode::Leaf { .. } => return vec![],
        };
        let router = &self.routers[router_idx];
        if router.id_lookup.is_empty() { return vec![]; }

        let q: Vec<f32> = query_vector.as_array().iter().copied().collect();
        let dim = self.embedding_dim;
        let n = router.id_lookup.len();

        // sequential scan; fast at small N and avoids rayon overhead
        let mut raw_scores: Vec<f32> = Vec::with_capacity(n);
        let mut max_s = f32::NEG_INFINITY;
        for row in router.x_matrix.chunks_exact(dim) {
            let s = dot(&q, row);
            if s > max_s { max_s = s; }
            raw_scores.push(s);
        }

        let mut exp_sum = 0.0f32;
        let mut probs: Vec<(usize, f32)> = raw_scores
            .into_iter()
            .enumerate()
            .map(|(i, s)| { let e = ((s - max_s) * beta).exp(); exp_sum += e; (i, e) })
            .collect();
        for (_, p) in probs.iter_mut() { *p /= exp_sum + 1e-9; }
        probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let mut result: Vec<String> = Vec::new();
        let mut cumulative = 0.0f32;
        for (idx, prob) in probs.into_iter().take(max_beam) {
            let child_str = &router.id_lookup[idx];
            if children_mapping.contains_key(child_str) {
                result.push(child_str.clone());
            }
            cumulative += prob;
            if cumulative >= target_confidence { break; }
        }
        result
    }

    #[pyo3(signature = ())]
    fn print_profile_stats(&self) {
        let count = QUERY_COUNT.load(std::sync::atomic::Ordering::Relaxed).max(1);
        println!("\n===== AIGVS RUST PROFILE =====");
        println!("Queries: {}", count);
        println!("Router Total:     {:>8.2} us/query", ROUTER_TOTAL_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / count as f64);
        println!("  - Score:        {:>8.2} us/query", ROUTER_SCORE_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / count as f64);
        println!("  - Softmax:      {:>8.2} us/query", ROUTER_SOFTMAX_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / count as f64);
        println!("  - Sort/Select:  {:>8.2} us/query", ROUTER_SORT_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / count as f64);
        println!("Child Query Time: {:>8.2} us/query", CHILD_QUERY_TOTAL_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / count as f64);
        println!("Rerank Time:      {:>8.2} us/query", RERANK_TOTAL_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / count as f64);
        println!("Candidate Vecs:   {:>8.2} vectors/query", CANDIDATE_VECTORS.load(std::sync::atomic::Ordering::Relaxed) as f64 / count as f64);
        println!("Leaves Visited:   {:>8.2} leaves/query", LEAVES_VISITED.load(std::sync::atomic::Ordering::Relaxed) as f64 / count as f64);
        println!("==============================\n");
    }
}

// --- NSW insert helpers ---

#[pymodule]
fn aigvs_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AIGVSEngine>()?;
    Ok(())
}