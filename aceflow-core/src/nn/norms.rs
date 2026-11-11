//! RMSNorm and LayerNorm. GPU reductions run when `input` is on WGPU.

use crate::device::Device;
use crate::error::Result;
use crate::nn::Module;
use crate::tensor::Tensor;

#[derive(Clone, Debug)]
pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(weight: Tensor, eps: f32) -> Self {
        weight.assert_contiguous();
        assert_eq!(weight.rank(), 1, "RMSNorm weight must be rank 1");
        Self { weight, eps }
    }

    pub fn from_dim(dim: usize, eps: f32, device: Device) -> Self {
        Self::new(Tensor::full(&[dim], 1.0, device), eps)
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.rms_norm(&self.weight, self.eps)
    }
}

impl Module for RMSNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(RMSNorm::forward(self, input))
    }
}

#[derive(Clone, Debug)]
pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f32) -> Self {
        weight.assert_contiguous();
        bias.assert_contiguous();
        assert_eq!(weight.rank(), 1, "LayerNorm weight must be rank 1");
        assert_eq!(bias.numel(), weight.numel(), "LayerNorm bias mismatch");
        Self { weight, bias, eps }
    }

    pub fn from_dim(dim: usize, eps: f32, device: Device) -> Self {
        Self::new(
            Tensor::full(&[dim], 1.0, device),
            Tensor::zeros_on(&[dim], device),
            eps,
        )
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.layer_norm(&self.weight, &self.bias, self.eps)
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(LayerNorm::forward(self, input))
    }
}
