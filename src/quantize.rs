//! rabitq: randomised binary quantization for the cell scan.
//!
//! The cell scan is memory-bandwidth bound: throughput follows bytes per vector closely (f32 -> i8
//! is 4x fewer bytes and gave 3.4-3.9x). So 1 bit per dimension is the biggest cut available: at
//! d=384 a vector goes from 1536 B (f32) or 384 B (i8) to 64 B (512 bits after padding).
//!
//! Plain sign(x) is a bad quantizer because embedding dimensions aren't equally scaled or
//! independent; a few high-variance coordinates dominate. A random orthogonal rotation first
//! spreads the information across coordinates, which is what makes the sign bits useful and what
//! RaBitQ (Gao & Long, SIGMOD 2024) builds its error bound on. Rotation preserves inner products.
//!
//! The rotation is a fast Hadamard transform with random sign flips, (H*D) applied ROUNDS times,
//! instead of a dense random matrix. A dense d x d rotation is 147k FLOPs per query (~20-40 us),
//! which would cost more than the scan; the FHT is d*log2(d) ~ 4.6k ops (~1-2 us).
//!
//! For rotated unit vectors, Hamming distance between sign patterns is a monotone proxy for angle:
//! <q,o> ~ D - 2*hamming. That order is used to pick candidates, which are then reranked exactly,
//! so the quantizer only has to keep the true top-k inside the candidate pool.
//!
//! This is RaBitQ's rotation + sign codes with exact rerank, not its full unbiased estimator; the
//! estimator matters when you don't rerank, and we always do.
//!
//! Asymmetric estimators: symmetric Hamming throws away the query's magnitudes too. Keeping the
//! query at 4 bits per coordinate and the data at 1 bit ranks the same cells much better (R@10
//! after rerank 0.934 -> 0.994). Two versions:
//! * [`AsymQuery`]: 4 query bit-planes ANDed with the codes, four POPCNTs per word; only used as an
//!   optional second stage
//! * [`FsQuery`]: FastScan (Andre et al.): codes interleaved by nibble in 32-vector blocks, one
//!   16-entry u8 table per nibble position, one pshufb scores 32 vectors; the first-stage estimator
//!   with set_fastscan(True)

/// Number of (H*D) rounds. One round mixes poorly; two is enough for the sign bits to be close to
/// uniformly informative. A third costs ~1 us more for very little.
pub const ROUNDS: usize = 2;

/// Next power of two >= n (the FHT requires a power-of-two length).
#[inline]
pub fn pad_dim(n: usize) -> usize {
    n.next_power_of_two().max(64)
}

/// Round a requested code length down to whole 64-bit words.
///
/// The code doesn't need to be as long as the rotated vector. The rotation spreads information
/// evenly across coordinates, so any prefix of the rotated vector is an unbiased sample of the
/// whole. At d=384 the FHT pads to 512, so a full-length code spends a third of its bytes on
/// padding. Keeping 384 or 256 bits cuts the scan cost proportionally, and the exact rerank fixes
/// the coarser ordering.
#[inline]
pub fn code_words(code_bits: usize, dim_pad: usize) -> usize {
    let b = if code_bits == 0 { dim_pad } else { code_bits.min(dim_pad) };
    (b / 64).max(1)
}

/// Pack the sign pattern of the first `out.len()*64` rotated coordinates.
#[inline]
pub fn pack_signs_trunc(rotated: &[f32], out: &mut [u64]) {
    let bits = out.len() * 64;
    for w in out.iter_mut() {
        *w = 0;
    }
    for i in 0..bits.min(rotated.len()) {
        if rotated[i] < 0.0 {
            out[i >> 6] |= 1u64 << (i & 63);
        }
    }
}

/// In-place normalised fast Walsh-Hadamard transform. a.len() must be a power of two.
#[inline]
pub fn fht_inplace(a: &mut [f32]) {
    let n = a.len();
    // Levels h = 1, 2, 4 inside each 8-element block, in registers. Same butterflies in the same
    // order as the generic loop below, so the output is bit-identical; the generic loop's inner
    // trip count of 1-4 was what stopped it from vectorising.
    let mut h = 1;
    if n >= 8 && n % 8 == 0 {
        for blk in a.chunks_exact_mut(8) {
            let mut x = [blk[0], blk[1], blk[2], blk[3], blk[4], blk[5], blk[6], blk[7]];
            for &hh in &[1usize, 2, 4] {
                let mut i = 0;
                while i < 8 {
                    for j in i..i + hh {
                        let (u, v) = (x[j], x[j + hh]);
                        x[j] = u + v;
                        x[j + hh] = u - v;
                    }
                    i += hh << 1;
                }
            }
            blk.copy_from_slice(&x);
        }
        h = 8;
        // h >= 8: contiguous halves, which auto-vectorise.
        while h < n {
            for blk in a.chunks_exact_mut(h << 1) {
                let (lo, hi) = blk.split_at_mut(h);
                for (u, v) in lo.iter_mut().zip(hi.iter_mut()) {
                    let (x, y) = (*u, *v);
                    *u = x + y;
                    *v = x - y;
                }
            }
            h <<= 1;
        }
    }
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                unsafe {
                    let x = *a.get_unchecked(j);
                    let y = *a.get_unchecked(j + h);
                    *a.get_unchecked_mut(j) = x + y;
                    *a.get_unchecked_mut(j + h) = x - y;
                }
            }
            i += h << 1;
        }
        h <<= 1;
    }
    let s = 1.0 / (n as f32).sqrt();
    for v in a.iter_mut() {
        *v *= s;
    }
}

/// Deterministic +/-1 sign vectors for the rotation.
///
/// From a fixed seed with splitmix64, so the rotation is the same across processes and can be
/// rebuilt for a saved index; a query has to be rotated with exactly the operator the data was.
pub fn make_signs(dim_pad: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    };
    (0..ROUNDS)
        .map(|_| {
            (0..dim_pad)
                .map(|_| if next() & 1 == 0 { 1.0f32 } else { -1.0f32 })
                .collect()
        })
        .collect()
}

/// Apply the rotation to `v` (length <= dim_pad), writing into `scratch`.
#[inline]
pub fn rotate(v: &[f32], signs: &[Vec<f32>], scratch: &mut [f32]) {
    scratch[..v.len()].copy_from_slice(v);
    for s in scratch[v.len()..].iter_mut() {
        *s = 0.0;
    }
    for round in signs.iter() {
        for (x, &s) in scratch.iter_mut().zip(round.iter()) {
            *x *= s;
        }
        fht_inplace(scratch);
    }
}

/// Pack the sign pattern of a rotated vector into bit words (1 = negative).
#[inline]
pub fn pack_signs(rotated: &[f32], out: &mut [u64]) {
    for w in out.iter_mut() {
        *w = 0;
    }
    for (i, &x) in rotated.iter().enumerate() {
        if x < 0.0 {
            out[i >> 6] |= 1u64 << (i & 63);
        }
    }
}

/// Hamming distance between two packed sign vectors.
///
/// count_ones compiles to POPCNT with -C target-cpu=native. At d=384 that's 6-8 words, so 6-8 XORs
/// and POPCNTs per vector, vs 384 multiply-adds for int8.
#[inline(always)]
pub fn hamming(a: &[u64], b: &[u64]) -> u32 {
    // 512-bit codes (d = 257..512) are the shipped case: a fixed width lets the
    // compiler fully unroll and keep the query words in registers.
    if a.len() == 8 && b.len() >= 8 {
        let a: &[u64; 8] = a.try_into().unwrap();
        let b: &[u64; 8] = b[..8].try_into().unwrap();
        return (a[0] ^ b[0]).count_ones() + (a[1] ^ b[1]).count_ones()
            + (a[2] ^ b[2]).count_ones() + (a[3] ^ b[3]).count_ones()
            + (a[4] ^ b[4]).count_ones() + (a[5] ^ b[5]).count_ones()
            + (a[6] ^ b[6]).count_ones() + (a[7] ^ b[7]).count_ones();
    }
    let mut s = 0u32;
    for i in 0..a.len() {
        unsafe {
            s += (a.get_unchecked(i) ^ b.get_unchecked(i)).count_ones();
        }
    }
    s
}

/// Rank score from Hamming distance, higher is closer. dim_pad - 2*hamming is proportional to the
/// cosine of the angle between the rotated vectors, so it gives the same order as similarity.
#[inline(always)]
pub fn hamming_score(h: u32, dim_pad: usize) -> f32 {
    dim_pad as f32-2.0 * h as f32
}

// --- FastScan: the asymmetric estimate at Hamming speed ---
//
// Codes of 32 vectors are interleaved by 4-bit nibble. For each pair of nibble positions (2p, 2p+1)
// a block holds 32 bytes: bytes 0..16 are position 2p and bytes 16..32 position 2p+1, and byte i of
// a half has vector i's nibble in its low half and vector i+16's in its high half. The query builds
// one 16-entry u8 table per nibble position in the same layout, so one pshufb per half looks up 32
// vectors at two positions. Summing over all positions gives, per vector,
//     sum_j LUT_j[nib_j(x)] ~ (<Rq, sign(Rx)> - sum_j min_j) / delta
// which is the asymmetric inner-product estimate (real-valued query, 1-bit data). It ranks much
// better than symmetric Hamming for about the same cost per vector. A block is 32 vectors x 64 B,
// the same bytes as the codes.

pub const FS_BLOCK: usize = 32;

/// Bytes of one 32-vector block for `nw` code words.
#[inline]
pub fn fs_block_bytes(nw: usize) -> usize {
    nw * 64 * FS_BLOCK / 8 // = 32 vectors x nw*8 bytes
}

/// Nibble `j` of a packed code.
#[inline(always)]
fn nib(code: &[u64], j: usize) -> u8 {
    ((code[j >> 4] >> ((j & 15) * 4)) & 0xF) as u8
}

/// Interleave up to 32 codes (`codes[i]` = vector i's `nw` words; missing
/// lanes are zero) into one FastScan block.
pub fn fs_pack_block(codes: &[&[u64]], nw: usize, out: &mut [u8]) {
    let npos = nw * 16;
    for p in 0..npos / 2 {
        for half in 0..2 {
            let j = 2 * p + half;
            for i in 0..16 {
                let lo = codes.get(i).map(|c| nib(c, j)).unwrap_or(0);
                let hi = codes.get(i + 16).map(|c| nib(c, j)).unwrap_or(0);
                out[p * 32 + half * 16 + i] = lo | (hi << 4);
            }
        }
    }
}

/// Recover lane `lane`'s packed code from a FastScan block (inverse of fs_pack_block), so the flat
/// code arena can be dropped while the blocks exist.
pub fn fs_unpack(block: &[u8], lane: usize, nw: usize, out: &mut [u64]) {
    for w in out.iter_mut() {
        *w = 0;
    }
    let npos = nw * 16;
    let (i, hi) = (lane % 16, lane >= 16);
    for j in 0..npos {
        let byte = block[(j / 2) * 32 + (j % 2) * 16 + i];
        let n = if hi { byte >> 4 } else { byte & 0xF } as u64;
        out[j >> 4] |= n << ((j & 15) * 4);
    }
}

/// Per-query FastScan tables.
pub struct FsQuery {
    /// (npos/2) x 32 bytes, pair layout matching the blocks.
    pub lut: Vec<u8>,
    pub delta: f32,
    pub bias: f32,
    pub npos: usize,
    /// Largest possible table sum = 255 * npos.
    pub maxk: u32,
}

impl FsQuery {
    pub fn new(rq: &[f32], nw: usize) -> Self {
        let npos = nw * 16;
        let mut raw = vec![0f32; npos * 16];
        let mut mins = vec![0f32; npos];
        let mut range = 0f32;
        for j in 0..npos {
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for e in 0..16usize {
                let mut v = 0f32;
                for t in 0..4 {
                    let d = 4 * j + t;
                    let x = if d < rq.len() { rq[d] } else { 0.0 };
                    v += if (e >> t) & 1 == 1 { -x } else { x };
                }
                raw[j * 16 + e] = v;
                lo = lo.min(v);
                hi = hi.max(v);
            }
            mins[j] = lo;
            range = range.max(hi - lo);
        }
        let delta = (range / 255.0).max(1e-12);
        let mut lut = vec![0u8; npos * 16];
        for p in 0..npos / 2 {
            for half in 0..2 {
                let j = 2 * p + half;
                for e in 0..16 {
                    let q = ((raw[j * 16 + e] - mins[j]) / delta).round().clamp(0.0, 255.0);
                    lut[p * 32 + half * 16 + e] = q as u8;
                }
            }
        }
        FsQuery { lut, delta, bias: mins.iter().sum(), npos, maxk: 255 * npos as u32 }
    }

    /// Approximate <Rq, sign(Rx)> for a table sum.
    #[inline]
    pub fn estimate(&self, sum: u32) -> f32 {
        self.delta * sum as f32 + self.bias
    }

    /// Table sums for the 32 lanes of one block.
    #[inline]
    pub fn block_sums(&self, block: &[u8], out: &mut [u16; FS_BLOCK]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { fs_block_sums_avx2(block.as_ptr(), self.lut.as_ptr(), self.npos / 2, out) };
                return;
            }
        }
        fs_block_sums_scalar(block, &self.lut, self.npos / 2, out);
    }
}

fn fs_block_sums_scalar(block: &[u8], lut: &[u8], npairs: usize, out: &mut [u16; FS_BLOCK]) {
    for o in out.iter_mut() {
        *o = 0;
    }
    for p in 0..npairs {
        for half in 0..2 {
            let base = p * 32 + half * 16;
            for i in 0..16 {
                let b = block[base + i];
                out[i] += lut[base + (b & 0xF) as usize] as u16;
                out[i + 16] += lut[base + (b >> 4) as usize] as u16;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fs_block_sums_avx2(block: *const u8, lut: *const u8, npairs: usize,
                             out: &mut [u16; FS_BLOCK]) {
    use std::arch::x86_64::*;
    let mask = _mm256_set1_epi8(0x0F);
    let (mut a0l, mut a1l) = (_mm256_setzero_si256(), _mm256_setzero_si256());
    let (mut a0h, mut a1h) = (_mm256_setzero_si256(), _mm256_setzero_si256());
    for p in 0..npairs {
        let c = _mm256_loadu_si256(block.add(p * 32) as *const __m256i);
        let l = _mm256_loadu_si256(lut.add(p * 32) as *const __m256i);
        let lo = _mm256_and_si256(c, mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), mask);
        let rl = _mm256_shuffle_epi8(l, lo);
        let rh = _mm256_shuffle_epi8(l, hi);
        // u8 results into u16 lanes: a0 += (even + 256*odd), a1 += odd.
        a0l = _mm256_add_epi16(a0l, rl);
        a1l = _mm256_add_epi16(a1l, _mm256_srli_epi16(rl, 8));
        a0h = _mm256_add_epi16(a0h, rh);
        a1h = _mm256_add_epi16(a1h, _mm256_srli_epi16(rh, 8));
    }
    // even lanes = a0 - (a1 << 8); odd lanes = a1; then add the two 128-bit
    // halves (positions 2p and 2p+1) and interleave even/odd back to lanes.
    macro_rules! fold {
        ($a0:expr, $a1:expr, $dst:expr) => {{
            let even = _mm256_sub_epi16($a0, _mm256_slli_epi16($a1, 8));
            let e = _mm_add_epi16(_mm256_castsi256_si128(even), _mm256_extracti128_si256(even, 1));
            let o = _mm_add_epi16(_mm256_castsi256_si128($a1), _mm256_extracti128_si256($a1, 1));
            let dst: *mut u16 = $dst;
            _mm_storeu_si128(dst as *mut __m128i, _mm_unpacklo_epi16(e, o));
            _mm_storeu_si128(dst.add(8) as *mut __m128i, _mm_unpackhi_epi16(e, o));
        }};
    }
    fold!(a0l, a1l, out.as_mut_ptr());
    fold!(a0h, a1h, out.as_mut_ptr().add(16));
}

/// Asymmetric 1-bit estimator: the query stays real-valued (4-bit).
///
/// Symmetric Hamming binarises the query too and loses its magnitudes. Inside the same probed cells
/// on HotpotQA questions, R@10 after reranking the top 320 is 0.934 with Hamming and 0.994 with
/// this, same 64-byte codes and no extra memory.
///
/// With the rotated query rq quantised as lo + delta*u_i (u_i in 0..15) and code bit b_i = 1 where
/// the rotated vector is negative:
///     <rq, sign> = sum(rq) - 2 * sum_{b_i=1} rq_i
///     sum_{b_i=1} rq_i ~= lo*popcount(x) + delta * sum_p 2^p popcount(U_p & x)
/// where U_p is bit-plane p of u. Ranking by -key ranks by the asymmetric estimate, with 4
/// AND+POPCNT per code word.
pub struct AsymQuery {
    pub planes: Vec<u64>, // 4 * n_words
    pub lo: f32,
    pub delta: f32,
    pub nw: usize,
}

impl AsymQuery {
    pub fn new(rq: &[f32], nw: usize) -> Self {
        let bits = (nw * 64).min(rq.len());
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in rq[..bits].iter() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let delta = ((hi - lo) / 15.0).max(1e-12);
        let mut planes = vec![0u64; 4 * nw];
        for i in 0..bits {
            let u = (((rq[i] - lo) / delta).round().clamp(0.0, 15.0)) as u32;
            for p in 0..4 {
                if (u >> p) & 1 == 1 {
                    planes[p * nw + (i >> 6)] |= 1u64 << (i & 63);
                }
            }
        }
        AsymQuery { planes, lo, delta, nw }
    }

    /// Estimated inner product up to a per-query affine map (higher = closer).
    #[inline(always)]
    pub fn score(&self, x: &[u64]) -> f32 {
        let nw = self.nw;
        let (mut px, mut w) = (0u32, 0u32);
        for i in 0..nw {
            let xi = x[i];
            px += xi.count_ones();
            w += (self.planes[i] & xi).count_ones()
                + 2 * (self.planes[nw + i] & xi).count_ones()
                + 4 * (self.planes[2 * nw + i] & xi).count_ones()
                + 8 * (self.planes[3 * nw + i] & xi).count_ones();
        }
        -(self.lo * px as f32 + self.delta * w as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fht_is_orthonormal() {
        // A normalised Hadamard transform preserves the L2 norm.
        let mut a: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();
        let before: f32 = a.iter().map(|x| x * x).sum();
        fht_inplace(&mut a);
        let after: f32 = a.iter().map(|x| x * x).sum();
        assert!((before - after).abs() < 1e-3, "{before} vs {after}");
    }

    #[test]
    fn rotation_preserves_inner_product() {
        let dim = 64;
        let signs = make_signs(dim, 42);
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.11).cos()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.23).sin()).collect();
        let ip: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let mut ra = vec![0.0; dim];
        let mut rb = vec![0.0; dim];
        rotate(&a, &signs, &mut ra);
        rotate(&b, &signs, &mut rb);
        let ip_rot: f32 = ra.iter().zip(&rb).map(|(x, y)| x * y).sum();
        assert!((ip - ip_rot).abs() < 1e-3, "{ip} vs {ip_rot}");
    }

    #[test]
    fn hamming_orders_like_cosine() {
        // Closer vectors must have smaller Hamming distance after rotation.
        let dim = 256;
        let signs = make_signs(dim, 7);
        let q: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.05).sin()).collect();
        let near: Vec<f32> = q.iter().map(|x| x + 0.01).collect();
        let far: Vec<f32> = q.iter().map(|x| -x).collect();
        let (mut rq, mut rn, mut rf) = (vec![0.0; dim], vec![0.0; dim], vec![0.0; dim]);
        rotate(&q, &signs, &mut rq);
        rotate(&near, &signs, &mut rn);
        rotate(&far, &signs, &mut rf);
        let w = dim / 64;
        let (mut bq, mut bn, mut bf) = (vec![0u64; w], vec![0u64; w], vec![0u64; w]);
        pack_signs(&rq, &mut bq);
        pack_signs(&rn, &mut bn);
        pack_signs(&rf, &mut bf);
        assert!(hamming(&bq, &bn) < hamming(&bq, &bf));
    }
}
