/// RMSNorm kernels retained from MallowRe.

pub const DEFAULT_EPS: f32 = 1e-5;

#[inline]
pub fn rmsnorm_core(x: &mut [f32], weights: &[f32], eps: f32) {
    let len = x.len();
    assert_eq!(weights.len(), len, "RMSNorm weight length mismatch");
    if len == 0 {
        return;
    }

    let chunks = x.chunks_exact(8);
    let remainder = chunks.remainder();
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut s4 = 0.0f32;
    let mut s5 = 0.0f32;
    let mut s6 = 0.0f32;
    let mut s7 = 0.0f32;
    for chunk in chunks {
        s0 += chunk[0] * chunk[0];
        s1 += chunk[1] * chunk[1];
        s2 += chunk[2] * chunk[2];
        s3 += chunk[3] * chunk[3];
        s4 += chunk[4] * chunk[4];
        s5 += chunk[5] * chunk[5];
        s6 += chunk[6] * chunk[6];
        s7 += chunk[7] * chunk[7];
    }
    let mut sum_sq = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
    for &value in remainder {
        sum_sq += value * value;
    }
    let inv_rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();

    let remainder_start = len - len % 8;
    for (values, scales) in x[..remainder_start]
        .chunks_exact_mut(8)
        .zip(weights[..remainder_start].chunks_exact(8))
    {
        values[0] *= inv_rms * scales[0];
        values[1] *= inv_rms * scales[1];
        values[2] *= inv_rms * scales[2];
        values[3] *= inv_rms * scales[3];
        values[4] *= inv_rms * scales[4];
        values[5] *= inv_rms * scales[5];
        values[6] *= inv_rms * scales[6];
        values[7] *= inv_rms * scales[7];
    }
    for (value, &scale) in x[remainder_start..]
        .iter_mut()
        .zip(&weights[remainder_start..])
    {
        *value *= inv_rms * scale;
    }
}

#[inline]
pub fn rmsnorm(x: &[f32], weights: &[f32], eps: f32) -> Vec<f32> {
    let mut output = x.to_vec();
    rmsnorm_core(&mut output, weights, eps);
    output
}

#[inline]
pub fn rmsnorm_inplace(x: &mut [f32], weights: &[f32], eps: f32) {
    rmsnorm_core(x, weights, eps);
}

#[inline]
pub fn rmsnorm_to(x: &[f32], weights: &[f32], eps: f32, output: &mut [f32]) {
    assert!(output.len() >= x.len(), "RMSNorm output buffer too small");
    output[..x.len()].copy_from_slice(x);
    rmsnorm_core(&mut output[..x.len()], weights, eps);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_matches_direct_formula() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![1.0; 4];
        let output = rmsnorm(&x, &weights, 1e-5);
        let inv = 1.0 / (7.5 + 1e-5f32).sqrt();
        for i in 0..4 {
            assert!((output[i] - x[i] * inv).abs() < 1e-6);
        }
    }
}
