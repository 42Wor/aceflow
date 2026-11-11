//! Register-tiled matrix-vector multiplication.

use rayon::prelude::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2/FMA implementation retained from MallowRe.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_avx2(m: usize, k: usize, weights: &[f32], x: &[f32], y: &mut [f32]) {
    let mut row = 0;
    let k_chunks = (k / 8) * 8;

    while row + 4 <= m {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        let row0_ptr = weights.as_ptr().add(row * k);
        let row1_ptr = weights.as_ptr().add((row + 1) * k);
        let row2_ptr = weights.as_ptr().add((row + 2) * k);
        let row3_ptr = weights.as_ptr().add((row + 3) * k);
        let x_ptr = x.as_ptr();

        let mut j = 0;
        while j < k_chunks {
            let xv = _mm256_loadu_ps(x_ptr.add(j));
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(row0_ptr.add(j)), xv, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(row1_ptr.add(j)), xv, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(row2_ptr.add(j)), xv, acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(row3_ptr.add(j)), xv, acc3);
            j += 8;
        }

        let mut sum0 = hsum_avx2(acc0);
        let mut sum1 = hsum_avx2(acc1);
        let mut sum2 = hsum_avx2(acc2);
        let mut sum3 = hsum_avx2(acc3);
        while j < k {
            let xj = *x_ptr.add(j);
            sum0 += *row0_ptr.add(j) * xj;
            sum1 += *row1_ptr.add(j) * xj;
            sum2 += *row2_ptr.add(j) * xj;
            sum3 += *row3_ptr.add(j) * xj;
            j += 1;
        }

        y[row] = sum0;
        y[row + 1] = sum1;
        y[row + 2] = sum2;
        y[row + 3] = sum3;
        row += 4;
    }

    while row < m {
        let mut acc = _mm256_setzero_ps();
        let row_ptr = weights.as_ptr().add(row * k);
        let x_ptr = x.as_ptr();
        let mut j = 0;
        while j < k_chunks {
            let xv = _mm256_loadu_ps(x_ptr.add(j));
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(row_ptr.add(j)), xv, acc);
            j += 8;
        }
        let mut sum = hsum_avx2(acc);
        while j < k {
            sum += *row_ptr.add(j) * *x_ptr.add(j);
            j += 1;
        }
        y[row] = sum;
        row += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_avx2(value: __m256) -> f32 {
    let low = _mm256_castps256_ps128(value);
    let high = _mm256_extractf128_ps(value, 1);
    let sum128 = _mm_add_ps(low, high);
    let shuffled = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, shuffled);
    let shuffled2 = _mm_shuffle_ps(sum64, sum64, 1);
    _mm_cvtss_f32(_mm_add_ss(sum64, shuffled2))
}

/// AArch64 NEON implementation, using four output rows per input load.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gemv_neon(m: usize, k: usize, weights: &[f32], x: &[f32], y: &mut [f32]) {
    let mut row = 0;
    let k_chunks = (k / 4) * 4;
    while row + 4 <= m {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let row0 = weights.as_ptr().add(row * k);
        let row1 = weights.as_ptr().add((row + 1) * k);
        let row2 = weights.as_ptr().add((row + 2) * k);
        let row3 = weights.as_ptr().add((row + 3) * k);
        let mut j = 0;
        while j < k_chunks {
            let xv = vld1q_f32(x.as_ptr().add(j));
            acc0 = vfmaq_f32(acc0, vld1q_f32(row0.add(j)), xv);
            acc1 = vfmaq_f32(acc1, vld1q_f32(row1.add(j)), xv);
            acc2 = vfmaq_f32(acc2, vld1q_f32(row2.add(j)), xv);
            acc3 = vfmaq_f32(acc3, vld1q_f32(row3.add(j)), xv);
            j += 4;
        }
        let mut sum0 = vaddvq_f32(acc0);
        let mut sum1 = vaddvq_f32(acc1);
        let mut sum2 = vaddvq_f32(acc2);
        let mut sum3 = vaddvq_f32(acc3);
        while j < k {
            let xv = *x.get_unchecked(j);
            sum0 += *row0.add(j) * xv;
            sum1 += *row1.add(j) * xv;
            sum2 += *row2.add(j) * xv;
            sum3 += *row3.add(j) * xv;
            j += 1;
        }
        y[row] = sum0;
        y[row + 1] = sum1;
        y[row + 2] = sum2;
        y[row + 3] = sum3;
        row += 4;
    }

    while row < m {
        let row_ptr = weights.as_ptr().add(row * k);
        let mut acc = vdupq_n_f32(0.0);
        let mut j = 0;
        while j < k_chunks {
            acc = vfmaq_f32(acc, vld1q_f32(row_ptr.add(j)), vld1q_f32(x.as_ptr().add(j)));
            j += 4;
        }
        let mut sum = vaddvq_f32(acc);
        while j < k {
            sum += *row_ptr.add(j) * *x.get_unchecked(j);
            j += 1;
        }
        y[row] = sum;
        row += 1;
    }
}

pub fn gemv_scalar(m: usize, k: usize, weights: &[f32], x: &[f32], y: &mut [f32]) {
    for row in 0..m {
        let row_offset = row * k;
        let mut sum = 0.0f32;
        for j in 0..k {
            sum += weights[row_offset + j] * x[j];
        }
        y[row] = sum;
    }
}

#[inline]
pub fn gemv(m: usize, k: usize, weights: &[f32], x: &[f32], y: &mut [f32]) {
    assert_eq!(weights.len(), m * k, "weights length must be m * k");
    assert_eq!(x.len(), k, "x length must be k");
    assert_eq!(y.len(), m, "y length must be m");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && k >= 8 {
            unsafe { gemv_avx2(m, k, weights, x, y) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") && k >= 4 {
            unsafe { gemv_neon(m, k, weights, x, y) };
            return;
        }
    }
    gemv_scalar(m, k, weights, x, y);
}

/// Apply `y[row] = W @ x[row]` for every packed row. `W` is `(m, k)` row-major.
pub fn batched_gemv(m: usize, k: usize, weights: &[f32], x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len() % k, 0, "x length must be a multiple of k");
    assert_eq!(y.len() % m, 0, "y length must be a multiple of m");
    assert_eq!(x.len() / k, y.len() / m, "batched_gemv row count mismatch");
    y.par_chunks_mut(m)
        .zip(x.par_chunks(k))
        .for_each(|(dst, src)| gemv(m, k, weights, src, dst));
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot product length mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && a.len() >= 8 {
            return unsafe { dot_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") && a.len() >= 4 {
            return unsafe { dot_neon(a, b) };
        }
    }
    a.iter().zip(b).map(|(l, r)| l * r).sum()
}

#[inline]
pub fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len(), "axpy length mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && x.len() >= 8 {
            unsafe { axpy_avx2(alpha, x, y) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") && x.len() >= 4 {
            unsafe { axpy_neon(alpha, x, y) };
            return;
        }
    }
    for (dst, src) in y.iter_mut().zip(x) {
        *dst += alpha * *src;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let chunks = (n / 8) * 8;
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i < chunks {
        acc = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i)),
            _mm256_loadu_ps(b.as_ptr().add(i)),
            acc,
        );
        i += 8;
    }
    let mut sum = hsum_avx2(acc);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_avx2(alpha: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len();
    let chunks = (n / 8) * 8;
    let av = _mm256_set1_ps(alpha);
    let mut i = 0;
    while i < chunks {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        let yv = _mm256_loadu_ps(y.as_ptr().add(i));
        _mm256_storeu_ps(y.as_mut_ptr().add(i), _mm256_fmadd_ps(av, xv, yv));
        i += 8;
    }
    while i < n {
        y[i] += alpha * x[i];
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let chunks = (n / 4) * 4;
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0;
    while i < chunks {
        acc = vfmaq_f32(acc, vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
        i += 4;
    }
    let mut sum = vaddvq_f32(acc);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn axpy_neon(alpha: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len();
    let chunks = (n / 4) * 4;
    let av = vdupq_n_f32(alpha);
    let mut i = 0;
    while i < chunks {
        let yv = vfmaq_f32(vld1q_f32(y.as_ptr().add(i)), av, vld1q_f32(x.as_ptr().add(i)));
        vst1q_f32(y.as_mut_ptr().add(i), yv);
        i += 4;
    }
    while i < n {
        y[i] += alpha * x[i];
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_kernel_matches_scalar() {
        let weights: Vec<f32> = (0..40).map(|value| value as f32 * 0.125 - 2.0).collect();
        let x: Vec<f32> = (0..10).map(|value| value as f32 * 0.25).collect();
        let mut expected = vec![0.0; 4];
        let mut actual = vec![0.0; 4];
        gemv_scalar(4, 10, &weights, &x, &mut expected);
        gemv(4, 10, &weights, &x, &mut actual);
        for (left, right) in expected.iter().zip(&actual) {
            assert!((left - right).abs() < 1e-4);
        }
    }

    #[test]
    fn dot_and_axpy_match_scalar() {
        let a: Vec<f32> = (0..17).map(|value| value as f32 * 0.5 - 2.0).collect();
        let b: Vec<f32> = (0..17).map(|value| value as f32 * 0.25).collect();
        let expected: f32 = a.iter().zip(&b).map(|(l, r)| l * r).sum();
        assert!((dot(&a, &b) - expected).abs() < 1e-4);

        let mut y = vec![1.0f32; 17];
        let mut y_ref = y.clone();
        axpy(0.5, &a, &mut y);
        for (dst, src) in y_ref.iter_mut().zip(&a) {
            *dst += 0.5 * *src;
        }
        for (left, right) in y.iter().zip(&y_ref) {
            assert!((left - right).abs() < 1e-5);
        }
    }
}
