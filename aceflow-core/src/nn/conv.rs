//! 1D convolution. Dispatches to a WGSL kernel when `input` is on WGPU.

use crate::device::Device;
use crate::error::Result;
use crate::nn::Module;
use crate::tensor::Tensor;

#[derive(Clone, Debug)]
pub struct Conv1D {
    /// Weight shape `(out_channels, in_channels, kernel_size)`.
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: usize,
    pub padding: usize,
}

pub type Conv1d = Conv1D;

impl Conv1D {
    pub fn new(weight: Tensor, bias: Option<Tensor>, stride: usize, padding: usize) -> Self {
        weight.assert_contiguous();
        assert_eq!(weight.rank(), 3, "Conv1D weight must have rank 3");
        assert!(stride > 0, "Conv1D stride must be positive");
        if let Some(value) = &bias {
            value.assert_contiguous();
            assert_eq!(
                value.numel(),
                weight.shape[0],
                "Conv1D bias must match out_channels"
            );
        }
        Self {
            weight,
            bias,
            stride,
            padding,
        }
    }

    pub fn from_features(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
        device: Device,
    ) -> Self {
        let weight = Tensor::zeros_on(&[out_channels, in_channels, kernel_size], device);
        let bias = bias.then(|| Tensor::zeros_on(&[out_channels], device));
        Self::new(weight, bias, stride, padding)
    }

    /// Apply convolution to a contiguous `(in_channels, time)` tensor.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.assert_contiguous();
        assert_eq!(x.rank(), 2, "Conv1D input must have shape (channels, time)");
        let c_out = self.weight.shape[0];
        let c_in = self.weight.shape[1];
        let kernel = self.weight.shape[2];
        let t_in = x.shape[1];
        assert_eq!(x.shape[0], c_in, "Conv1D input channel mismatch");
        assert!(
            t_in + 2 * self.padding >= kernel,
            "Conv1D kernel exceeds padded input"
        );
        let t_out = (t_in + 2 * self.padding - kernel) / self.stride + 1;

        #[cfg(feature = "wgpu-runtime")]
        if x.device().is_wgpu() {
            if let Some(out) = crate::wgpu::ops::conv1d(
                &x.data,
                &self.weight.data,
                self.bias.as_ref().map(|bias| bias.data.as_slice()),
                c_out,
                c_in,
                kernel,
                t_in,
                t_out,
                self.stride,
                self.padding,
            ) {
                return Tensor::from_vec_on(out, &[c_out, t_out], x.device());
            }
        }

        let weight = &self.weight.data;
        let input = &x.data;
        let mut out = vec![0.0f32; c_out * t_out];
        use rayon::prelude::*;
        out.par_chunks_mut(t_out).enumerate().for_each(|(oc, dst)| {
            for t in 0..t_out {
                let mut acc = self.bias.as_ref().map_or(0.0, |bias| bias.data[oc]);
                for ic in 0..c_in {
                    for ki in 0..kernel {
                        let input_t = t * self.stride + ki;
                        let input_t = input_t as isize - self.padding as isize;
                        if input_t >= 0 && (input_t as usize) < t_in {
                            let w = weight[(oc * c_in + ic) * kernel + ki];
                            acc += w * input[ic * t_in + input_t as usize];
                        }
                    }
                }
                dst[t] = acc;
            }
        });
        Tensor::from_vec_on(out, &[c_out, t_out], x.device())
    }
}

impl Module for Conv1D {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(Conv1D::forward(self, input))
    }
}
