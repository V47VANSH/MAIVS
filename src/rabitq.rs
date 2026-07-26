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
//! estimator matters when you don't rerank, and we always do. If it turns out to need an
//! impractical rerank_factor, switch to the full estimator.

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
    let mut h = 1;
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
