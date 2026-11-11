//! CPU SIMD kernels and the reusable tensor arena.

use rayon::prelude::*;

use crate::device::Device;
use crate::tensor::Tensor;

pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1usize;
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride = stride.saturating_mul(shape[i]);
    }
    strides
}

/// A small reusable pool for owned contiguous CPU tensor buffers.
#[derive(Debug)]
pub struct TensorArena {
    buffers: Vec<Vec<f32>>,
    max_cached: usize,
}

impl Default for TensorArena {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorArena {
    pub fn new() -> Self {
        Self::with_max_cached(16)
    }

    pub fn with_max_cached(max_cached: usize) -> Self {
        Self {
            buffers: Vec::new(),
            max_cached,
        }
    }

    pub fn tensor(&mut self, shape: &[usize]) -> Tensor {
        let n = shape.iter().product();
        let candidate = self.buffers.iter().position(|buf| buf.capacity() >= n);
        let mut data = candidate
            .map(|index| self.buffers.swap_remove(index))
            .unwrap_or_else(|| Vec::with_capacity(n));
        data.resize(n, 0.0);
        data.fill(0.0);
        Tensor::from_vec_on(data, shape, Device::Cpu)
    }

    pub fn release(&mut self, mut tensor: Tensor) {
        tensor.assert_contiguous();
        if self.buffers.len() < self.max_cached {
            tensor.data.clear();
            self.buffers.push(tensor.data);
        }
    }

    pub fn clear(&mut self) {
        self.buffers.clear();
    }

    pub fn cached_buffers(&self) -> usize {
        self.buffers.len()
    }
}

/// `(M, K) @ (K, N)`.
pub fn gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "left matrix length mismatch");
    assert_eq!(b.len(), k * n, "right matrix length mismatch");
    let mut c = vec![0.0f32; m * n];
    c.par_chunks_mut(n).enumerate().for_each(|(i, crow)| {
        let arow = &a[i * k..(i + 1) * k];
        for p in 0..k {
            let aval = arow[p];
            if aval == 0.0 {
                continue;
            }
            let brow = &b[p * n..(p + 1) * n];
            for j in 0..n {
                crow[j] += aval * brow[j];
            }
        }
    });
    c
}

/// `(M, K) @ (N, K)^T`.
pub fn gemm_bt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "left matrix length mismatch");
    assert_eq!(b.len(), n * k, "right matrix length mismatch");
    let mut c = vec![0.0f32; m * n];
    c.par_chunks_mut(n).enumerate().for_each(|(i, crow)| {
        let arow = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let brow = &b[j * k..(j + 1) * k];
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += arow[p] * brow[p];
            }
            crow[j] = acc;
        }
    });
    c
}

/// Exact GELU form used by Hugging Face models (`erf` form).
pub fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + erf_approx(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// Abramowitz & Stegun 7.1.26, max error approximately 1.5e-7.
fn erf_approx(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-ax * ax).exp();
    sign * y
}
