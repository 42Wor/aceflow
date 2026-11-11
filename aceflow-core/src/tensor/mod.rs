//! Contiguous row-major tensors with WGPU-first device placement.
//!
//! Host `f32` values stay in [`Tensor::data`] so Whisper and CPU kernels keep
//! working. When the default device is WGPU, constructors allocate on GPU and
//! ops dispatch through the pipeline cache, falling back to CPU SIMD.

use rayon::prelude::*;

use crate::device::{default_device, promote_device, Device};

pub mod ops;
pub mod storage;

pub use ops::{contiguous_strides, gelu_erf, gemm, gemm_bt, TensorArena};
pub use storage::{CpuStorage, Storage, WgpuStorage};

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    /// Synchronized host mirror. GPU kernels write back here after dispatch.
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    storage: Storage,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        Self::zeros_on(shape, default_device())
    }

    pub fn zeros_on(shape: &[usize], device: Device) -> Self {
        let n = shape.iter().product();
        Self::from_vec_on(vec![0.0; n], shape, device)
    }

    pub fn ones(shape: &[usize]) -> Self {
        Self::full(shape, 1.0, default_device())
    }

    pub fn full(shape: &[usize], value: f32, device: Device) -> Self {
        let n = shape.iter().product();
        Self::from_vec_on(vec![value; n], shape, device)
    }

    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        Self::from_vec_on(data, shape, default_device())
    }

    pub fn from_slice(data: &[f32], shape: &[usize]) -> Self {
        Self::from_vec(data.to_vec(), shape)
    }

    pub fn from_vec_on(data: Vec<f32>, shape: &[usize], device: Device) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            n,
            "tensor size {} does not match shape {:?} ({})",
            data.len(),
            shape,
            n
        );
        Self {
            data,
            shape: shape.to_vec(),
            strides: contiguous_strides(shape),
            storage: Storage::from_device(device),
        }
    }

    pub fn device(&self) -> Device {
        self.storage.device()
    }

    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    pub fn to(&self, device: Device) -> Self {
        let mut out = self.clone();
        out.storage = Storage::from_device(device);
        out
    }

    pub fn cpu(&self) -> Self {
        self.to(Device::Cpu)
    }

    pub fn wgpu(&self) -> Self {
        self.to(Device::Wgpu(0))
    }

    fn finish(&self, data: Vec<f32>, shape: &[usize]) -> Self {
        Self::from_vec_on(data, shape, self.device())
    }

    fn finish_on(&self, data: Vec<f32>, shape: &[usize], device: Device) -> Self {
        Self::from_vec_on(data, shape, device)
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn rows(&self) -> usize {
        match self.shape.len() {
            0 => 0,
            1 => 1,
            _ => self.shape.iter().take(self.shape.len() - 1).product(),
        }
    }

    pub fn cols(&self) -> usize {
        *self.shape.last().unwrap_or(&1)
    }

    pub fn is_contiguous(&self) -> bool {
        if self.data.len() != self.shape.iter().product::<usize>()
            || self.strides.len() != self.shape.len()
        {
            return false;
        }
        let mut expected = 1usize;
        for (&dimension, &stride) in self.shape.iter().rev().zip(self.strides.iter().rev()) {
            if stride != expected {
                return false;
            }
            expected = expected.saturating_mul(dimension);
        }
        true
    }

    pub fn assert_contiguous(&self) {
        assert!(
            self.is_contiguous(),
            "tensor with shape {:?} and strides {:?} is not contiguous",
            self.shape,
            self.strides
        );
    }

    pub fn as_slice(&self) -> &[f32] {
        self.assert_contiguous();
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        self.assert_contiguous();
        &mut self.data
    }

    pub fn reshape(&self, shape: &[usize]) -> Self {
        self.assert_contiguous();
        self.finish(self.data.clone(), shape)
    }

    pub fn reshape_owned(self, shape: &[usize]) -> Self {
        self.assert_contiguous();
        let device = self.device();
        Self::from_vec_on(self.data, shape, device)
    }

    /// Transpose a rank-2 tensor.
    pub fn t(&self) -> Self {
        self.assert_contiguous();
        assert_eq!(self.shape.len(), 2);
        let (m, n) = (self.shape[0], self.shape[1]);
        let mut out = vec![0.0f32; self.data.len()];
        for i in 0..m {
            for j in 0..n {
                out[j * m + i] = self.data[i * n + j];
            }
        }
        self.finish(out, &[n, m])
    }

    /// Matrix product: `(M, K) @ (K, N) -> (M, N)`.
    pub fn matmul(&self, other: &Tensor) -> Self {
        self.assert_contiguous();
        other.assert_contiguous();
        assert_eq!(self.shape.len(), 2, "matmul lhs must be rank-2");
        assert_eq!(other.shape.len(), 2, "matmul rhs must be rank-2");
        let (m, k) = (self.shape[0], self.shape[1]);
        assert_eq!(other.shape[0], k, "matmul inner dim mismatch");
        let n = other.shape[1];
        let device = promote_device(self.device(), other.device());
        if let Some(data) = gpu_gemm(&self.data, &other.data, m, k, n, false, device) {
            return Self::from_vec_on(data, &[m, n], device);
        }
        let data = gemm(&self.data, &other.data, m, k, n);
        Self::from_vec_on(data, &[m, n], device)
    }

    /// `(M, K) @ (N, K)^T -> (M, N)` without materializing the transpose.
    pub fn matmul_bt(&self, other: &Tensor) -> Self {
        self.assert_contiguous();
        other.assert_contiguous();
        assert_eq!(self.shape.len(), 2);
        assert_eq!(other.shape.len(), 2);
        let (m, k) = (self.shape[0], self.shape[1]);
        assert_eq!(other.shape[1], k);
        let n = other.shape[0];
        let device = promote_device(self.device(), other.device());
        if let Some(data) = gpu_gemm(&self.data, &other.data, m, k, n, true, device) {
            return Self::from_vec_on(data, &[m, n], device);
        }
        let data = gemm_bt(&self.data, &other.data, m, k, n);
        Self::from_vec_on(data, &[m, n], device)
    }

    pub fn add(&self, other: &Tensor) -> Self {
        self.assert_contiguous();
        other.assert_contiguous();
        assert_eq!(self.shape, other.shape);
        let device = promote_device(self.device(), other.device());
        if let Some(data) = gpu_binary(&self.data, &other.data, 0, device) {
            return self.finish_on(data, &self.shape, device);
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        self.finish_on(data, &self.shape, device)
    }

    pub fn mul(&self, other: &Tensor) -> Self {
        self.assert_contiguous();
        other.assert_contiguous();
        assert_eq!(self.shape, other.shape);
        let device = promote_device(self.device(), other.device());
        if let Some(data) = gpu_binary(&self.data, &other.data, 1, device) {
            return self.finish_on(data, &self.shape, device);
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        self.finish_on(data, &self.shape, device)
    }

    pub fn add_inplace(&mut self, other: &Tensor) {
        self.assert_contiguous();
        other.assert_contiguous();
        assert_eq!(self.shape, other.shape);
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += *b;
        }
    }

    /// Broadcast-add a vector onto the last dimension.
    pub fn add_bias(&self, bias: &Tensor) -> Self {
        self.assert_contiguous();
        bias.assert_contiguous();
        let d = self.cols();
        assert_eq!(bias.numel(), d);
        let mut out = self.data.clone();
        out.par_chunks_mut(d).for_each(|row| {
            for (x, b) in row.iter_mut().zip(bias.data.iter()) {
                *x += *b;
            }
        });
        self.finish(out, &self.shape)
    }

    pub fn scale(&self, s: f32) -> Self {
        self.assert_contiguous();
        if let Some(data) = gpu_unary(&self.data, 2, s, self.device()) {
            return self.finish(data, &self.shape);
        }
        let data = self.data.iter().map(|x| x * s).collect();
        self.finish(data, &self.shape)
    }

    pub fn gelu(&self) -> Self {
        self.assert_contiguous();
        if let Some(data) = gpu_unary(&self.data, 0, 0.0, self.device()) {
            return self.finish(data, &self.shape);
        }
        let data = self.data.iter().copied().map(gelu_erf).collect();
        self.finish(data, &self.shape)
    }

    pub fn silu(&self) -> Self {
        self.assert_contiguous();
        if let Some(data) = gpu_unary(&self.data, 1, 0.0, self.device()) {
            return self.finish(data, &self.shape);
        }
        let data = self
            .data
            .iter()
            .map(|&x| x / (1.0 + (-x).exp()))
            .collect();
        self.finish(data, &self.shape)
    }

    /// LayerNorm over the last dimension (population variance).
    pub fn layer_norm(&self, weight: &Tensor, bias: &Tensor, eps: f32) -> Self {
        self.assert_contiguous();
        weight.assert_contiguous();
        bias.assert_contiguous();
        let d = self.cols();
        assert_eq!(weight.numel(), d);
        assert_eq!(bias.numel(), d);
        let rows = self.numel() / d.max(1);
        if let Some(out) = gpu_norm(
            &self.data,
            &weight.data,
            &bias.data,
            rows,
            d,
            eps,
            false,
            self.device(),
        ) {
            return self.finish(out, &self.shape);
        }
        let mut out = vec![0.0f32; self.data.len()];
        out.par_chunks_mut(d)
            .zip(self.data.par_chunks(d))
            .for_each(|(dst, src)| {
                let mean = src.iter().sum::<f32>() / d as f32;
                let var = src
                    .iter()
                    .map(|x| {
                        let c = x - mean;
                        c * c
                    })
                    .sum::<f32>()
                    / d as f32;
                let inv = 1.0 / (var + eps).sqrt();
                for j in 0..d {
                    dst[j] = (src[j] - mean) * inv * weight.data[j] + bias.data[j];
                }
            });
        self.finish(out, &self.shape)
    }

    /// RMSNorm over the last dimension.
    pub fn rms_norm(&self, weight: &Tensor, eps: f32) -> Self {
        self.assert_contiguous();
        weight.assert_contiguous();
        let d = self.cols();
        assert_eq!(weight.numel(), d);
        let rows = self.numel() / d.max(1);
        let dummy = vec![0.0f32; d];
        if let Some(out) = gpu_norm(
            &self.data,
            &weight.data,
            &dummy,
            rows,
            d,
            eps,
            true,
            self.device(),
        ) {
            return self.finish(out, &self.shape);
        }
        let mut out = vec![0.0f32; self.data.len()];
        out.par_chunks_mut(d)
            .zip(self.data.par_chunks(d))
            .for_each(|(dst, src)| {
                let mean_sq = src.iter().map(|x| x * x).sum::<f32>() / d as f32;
                let inv = 1.0 / (mean_sq + eps).sqrt();
                for j in 0..d {
                    dst[j] = src[j] * inv * weight.data[j];
                }
            });
        self.finish(out, &self.shape)
    }

    /// Softmax over the last dimension, numerically stable.
    pub fn softmax_last(&self) -> Self {
        self.assert_contiguous();
        let d = self.cols();
        let rows = self.numel() / d.max(1);
        if let Some(out) = gpu_softmax(&self.data, rows, d, self.device()) {
            return self.finish(out, &self.shape);
        }
        let mut out = vec![0.0f32; self.data.len()];
        out.par_chunks_mut(d)
            .zip(self.data.par_chunks(d))
            .for_each(|(dst, src)| {
                let max = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for j in 0..d {
                    let e = (src[j] - max).exp();
                    dst[j] = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                for x in dst.iter_mut() {
                    *x *= inv;
                }
            });
        self.finish(out, &self.shape)
    }

    pub fn slice_rows(&self, start: usize, len: usize) -> Self {
        self.assert_contiguous();
        let d = self.cols();
        let off = start * d;
        self.finish(self.data[off..off + len * d].to_vec(), &[len, d])
    }

    pub fn last_row(&self) -> Self {
        let n = self.rows();
        self.slice_rows(n - 1, 1)
    }

    pub fn concat_rows(&self, other: &Tensor) -> Self {
        self.assert_contiguous();
        other.assert_contiguous();
        let d = self.cols();
        assert_eq!(other.cols(), d);
        let mut data = self.data.clone();
        data.extend_from_slice(&other.data);
        self.finish(data, &[self.rows() + other.rows(), d])
    }

    /// Gather rows of a `(V, D)` embedding table.
    pub fn gather_rows(&self, ids: &[u32]) -> Self {
        self.assert_contiguous();
        assert_eq!(self.shape.len(), 2);
        let d = self.shape[1];
        let mut data = vec![0.0f32; ids.len() * d];
        for (i, &id) in ids.iter().enumerate() {
            let src = id as usize * d;
            data[i * d..(i + 1) * d].copy_from_slice(&self.data[src..src + d]);
        }
        self.finish(data, &[ids.len(), d])
    }

    pub fn argmax_last(&self) -> usize {
        self.assert_contiguous();
        let row = if self.shape.len() == 1 {
            &self.data
        } else {
            let d = self.cols();
            let start = (self.rows() - 1) * d;
            &self.data[start..]
        };
        row.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn apply_mask_neg_inf(&mut self, pred: impl Fn(usize) -> bool) {
        self.assert_contiguous();
        for (i, x) in self.data.iter_mut().enumerate() {
            if pred(i) {
                *x = f32::NEG_INFINITY;
            }
        }
    }
}

fn gpu_enabled(device: Device) -> bool {
    device.is_wgpu()
}

fn gpu_gemm(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    transpose_b: bool,
    device: Device,
) -> Option<Vec<f32>> {
    if !gpu_enabled(device) {
        return None;
    }
    #[cfg(feature = "wgpu-runtime")]
    {
        return crate::wgpu::ops::gemm(a, b, m, k, n, transpose_b);
    }
    #[cfg(not(feature = "wgpu-runtime"))]
    {
        let _ = (a, b, m, k, n, transpose_b);
        None
    }
}

fn gpu_binary(lhs: &[f32], rhs: &[f32], op: u32, device: Device) -> Option<Vec<f32>> {
    if !gpu_enabled(device) {
        return None;
    }
    #[cfg(feature = "wgpu-runtime")]
    {
        return crate::wgpu::ops::binary(lhs, rhs, op);
    }
    #[cfg(not(feature = "wgpu-runtime"))]
    {
        let _ = (lhs, rhs, op);
        None
    }
}

fn gpu_unary(values: &[f32], op: u32, scalar: f32, device: Device) -> Option<Vec<f32>> {
    if !gpu_enabled(device) {
        return None;
    }
    #[cfg(feature = "wgpu-runtime")]
    {
        return crate::wgpu::ops::unary(values, op, scalar);
    }
    #[cfg(not(feature = "wgpu-runtime"))]
    {
        let _ = (values, op, scalar);
        None
    }
}

fn gpu_softmax(values: &[f32], rows: usize, dim: usize, device: Device) -> Option<Vec<f32>> {
    if !gpu_enabled(device) {
        return None;
    }
    #[cfg(feature = "wgpu-runtime")]
    {
        return crate::wgpu::ops::softmax_rows(values, rows, dim);
    }
    #[cfg(not(feature = "wgpu-runtime"))]
    {
        let _ = (values, rows, dim);
        None
    }
}

fn gpu_norm(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    rows: usize,
    dim: usize,
    eps: f32,
    rms: bool,
    device: Device,
) -> Option<Vec<f32>> {
    if !gpu_enabled(device) {
        return None;
    }
    #[cfg(feature = "wgpu-runtime")]
    {
        return crate::wgpu::ops::normalize(x, weight, bias, rows, dim, eps, rms);
    }
    #[cfg(not(feature = "wgpu-runtime"))]
    {
        let _ = (x, weight, bias, rows, dim, eps, rms);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_identity_preserves_values() {
        let a = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], Device::Cpu);
        let identity = Tensor::from_vec_on(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], Device::Cpu);
        let c = a.matmul(&identity);
        assert_eq!(c.data, a.data);
        assert!(c.is_contiguous());
        assert_eq!(c.strides, vec![2, 1]);
    }

    #[test]
    fn gemm_bt_uses_reference_accumulation() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 1.0, -1.0, 3.0];
        assert_eq!(gemm_bt(&a, &b, 2, 2, 2), vec![4.0, 5.0, 10.0, 9.0]);
    }

    #[test]
    fn arena_reuses_a_buffer() {
        let mut arena = TensorArena::new();
        let tensor = arena.tensor(&[4, 4]);
        arena.release(tensor);
        assert_eq!(arena.cached_buffers(), 1);
        let tensor = arena.tensor(&[2, 2]);
        assert_eq!(tensor.data, vec![0.0; 4]);
        assert_eq!(arena.cached_buffers(), 0);
    }

    #[test]
    fn softmax_sums_to_one() {
        let tensor = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], &[1, 3], Device::Cpu);
        let sum: f32 = tensor.softmax_last().data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
