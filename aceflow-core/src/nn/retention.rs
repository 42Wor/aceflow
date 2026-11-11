//! Multi-scale retention over `(time, d_model)` tensors.

use std::cell::RefCell;

use crate::cpu::recurrence::retention_step_core;
use crate::device::Device;
use crate::error::Result;
use crate::nn::linear::Linear;
use crate::nn::Module;
use crate::tensor::Tensor;

/// Recurrent multi-scale retention (RetNet-style) with a fused QKV projection.
#[derive(Clone, Debug)]
pub struct MultiScaleRetention {
    pub qkv: Linear,
    pub out_proj: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
    gamma: Vec<f32>,
    states: RefCell<Vec<f32>>,
}

impl MultiScaleRetention {
    pub fn new(qkv: Linear, out_proj: Linear, num_heads: usize) -> Self {
        let d_model = out_proj.out_features();
        assert_eq!(d_model % num_heads, 0);
        assert_eq!(qkv.out_features(), 3 * d_model);
        let head_dim = d_model / num_heads;
        let gamma = (0..num_heads)
            .map(|head| 1.0 - 2.0f32.powf(-5.0 - head as f32))
            .collect();
        Self {
            qkv,
            out_proj,
            num_heads,
            head_dim,
            gamma,
            states: RefCell::new(vec![0.0; num_heads * head_dim * head_dim]),
        }
    }

    pub fn from_features(d_model: usize, num_heads: usize, device: Device) -> Self {
        Self::new(
            Linear::from_features(d_model, 3 * d_model, false, device),
            Linear::from_features(d_model, d_model, false, device),
            num_heads,
        )
    }

    pub fn reset_state(&self) {
        self.states.borrow_mut().fill(0.0);
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.assert_contiguous();
        assert_eq!(x.rank(), 2, "retention input must be (time, d_model)");
        let time = x.shape[0];
        let d_model = self.head_dim * self.num_heads;
        let packed = self.qkv.forward(x);
        let mut output = vec![0.0f32; time * d_model];
        let mut step_out = vec![0.0f32; d_model];
        let mut states = self.states.borrow_mut();
        for t in 0..time {
            let qkv = &packed.data[t * 3 * d_model..(t + 1) * 3 * d_model];
            let (q, rest) = qkv.split_at(d_model);
            let (k, v) = rest.split_at(d_model);
            retention_step_core(
                &mut states,
                &self.gamma,
                0,
                self.num_heads,
                self.head_dim,
                q,
                k,
                v,
                &mut step_out,
            );
            output[t * d_model..(t + 1) * d_model].copy_from_slice(&step_out);
        }
        drop(states);
        let hidden = Tensor::from_vec_on(output, &[time, d_model], x.device());
        self.out_proj.forward(&hidden)
    }
}

impl Module for MultiScaleRetention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(MultiScaleRetention::forward(self, input))
    }
}
