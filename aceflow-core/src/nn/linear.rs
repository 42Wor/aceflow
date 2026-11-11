//! Linear projection. WGPU GEMM is used whenever `input` lives on GPU.

use crate::device::Device;
use crate::error::Result;
use crate::nn::Module;
use crate::tensor::Tensor;

#[derive(Clone, Debug)]
pub struct Linear {
    /// Pre-transposed weight with shape `(in_features, out_features)`.
    pub weight_t: Tensor,
    pub bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        weight.assert_contiguous();
        assert_eq!(weight.rank(), 2, "Linear weight must have rank 2");
        let out_features = weight.shape[0];
        let in_features = weight.shape[1];
        if let Some(value) = &bias {
            value.assert_contiguous();
            assert_eq!(
                value.numel(),
                out_features,
                "Linear bias must match out_features"
            );
        }
        Self {
            weight_t: weight.t(),
            bias,
            in_features,
            out_features,
        }
    }

    pub fn from_features(
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: Device,
    ) -> Self {
        let weight = Tensor::zeros_on(&[out_features, in_features], device);
        let bias = bias.then(|| Tensor::zeros_on(&[out_features], device));
        Self::new(weight, bias)
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn device(&self) -> Device {
        self.weight_t.device()
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.assert_contiguous();
        assert!(
            !x.shape.is_empty(),
            "Linear input must have at least one axis"
        );
        assert_eq!(
            x.cols(),
            self.in_features(),
            "Linear input feature mismatch"
        );
        let rows = x.rows();
        let flat = if x.rank() == 2 && x.shape[0] == rows {
            x.clone()
        } else {
            x.reshape(&[rows, self.in_features()])
        };
        let mut out = flat.matmul(&self.weight_t);
        if let Some(bias) = &self.bias {
            out = out.add_bias(bias);
        }
        let mut shape = x.shape.clone();
        *shape.last_mut().expect("checked non-empty shape") = self.out_features();
        out.reshape(&shape)
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(Linear::forward(self, input))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SwiGLU;

impl SwiGLU {
    pub fn forward(&self, gate: &Tensor, up: &Tensor) -> Tensor {
        gate.assert_contiguous();
        up.assert_contiguous();
        assert_eq!(gate.shape, up.shape, "SwiGLU tensor shape mismatch");
        let data = gate
            .data
            .iter()
            .zip(&up.data)
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        Tensor::from_vec_on(data, &gate.shape, gate.device())
    }

    /// Split the final axis in half as `(gate, up)` and apply SwiGLU.
    pub fn forward_packed(&self, packed: &Tensor) -> Tensor {
        packed.assert_contiguous();
        let packed_width = packed.cols();
        assert_eq!(
            packed_width % 2,
            0,
            "packed SwiGLU axis must have even length"
        );
        let width = packed_width / 2;
        let rows = packed.rows();
        let mut output = vec![0.0f32; rows * width];
        for row in 0..rows {
            let source = &packed.data[row * packed_width..(row + 1) * packed_width];
            let gate = &source[..width];
            let up = &source[width..];
            for i in 0..width {
                output[row * width + i] = (gate[i] / (1.0 + (-gate[i]).exp())) * up[i];
            }
        }
        let mut shape = packed.shape.clone();
        *shape.last_mut().expect("packed tensor has final axis") = width;
        Tensor::from_vec_on(output, &shape, packed.device())
    }
}
