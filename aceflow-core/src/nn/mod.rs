//! Modular neural-network blocks. Adding a layer means adding one file here.

use crate::error::Result;
use crate::tensor::Tensor;

mod attention;
mod conv;
mod linear;
mod norms;
mod retention;
mod rnn;

pub use attention::{
    cross_attention, scaled_dot_product_attention, sequence_downsample, sequence_upsample,
    AdditiveAttention, EncoderDecoderBlock, MultiHeadAttention,
};
pub use conv::{Conv1D, Conv1d};
pub use linear::{Linear, SwiGLU};
pub use norms::{LayerNorm, RMSNorm};
pub use retention::MultiScaleRetention;
pub use rnn::{rnn_forward, seq2seq_attention, AttentionKind, Rnn, RnnType};

/// Standard layer contract: inspect `input.device()` and dispatch WGPU / CPU.
pub trait Module {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    #[test]
    fn linear_uses_pytorch_weight_layout() {
        let weight = Tensor::from_vec_on(vec![1.0, 2.0, -1.0, 3.0], &[2, 2], Device::Cpu);
        let bias = Tensor::from_vec_on(vec![0.5, -0.5], &[2], Device::Cpu);
        let layer = Linear::new(weight, Some(bias));
        let input = Tensor::from_vec_on(vec![2.0, 4.0], &[1, 2], Device::Cpu);
        assert_eq!(layer.forward(&input).data, vec![10.5, 9.5]);
    }

    #[test]
    fn conv1d_matches_hand_calculation() {
        let weight = Tensor::from_vec_on(vec![1.0, 0.0, -1.0], &[1, 1, 3], Device::Cpu);
        let layer = Conv1D::new(weight, None, 1, 1);
        let input = Tensor::from_vec_on(vec![1.0, 2.0, 4.0], &[1, 3], Device::Cpu);
        assert_eq!(layer.forward(&input).data, vec![-2.0, -3.0, 2.0]);
    }

    #[test]
    fn swiglu_packed_splits_last_axis() {
        let packed = Tensor::from_vec_on(vec![0.0, 1.0, 2.0, 3.0], &[1, 4], Device::Cpu);
        let output = SwiGLU.forward_packed(&packed);
        assert_eq!(output.shape, vec![1, 2]);
        assert_eq!(output.data[0], 0.0);
        assert!((output.data[1] - 3.0 / (1.0 + (-1.0f32).exp())).abs() < 1e-6);
    }

    #[test]
    fn one_token_attention_returns_value() {
        let q = Tensor::from_vec_on(vec![1.0, 2.0], &[1, 2], Device::Cpu);
        let k = Tensor::from_vec_on(vec![3.0, 4.0], &[1, 2], Device::Cpu);
        let v = Tensor::from_vec_on(vec![5.0, 6.0], &[1, 2], Device::Cpu);
        assert_eq!(
            scaled_dot_product_attention(&q, &k, &v, 1, false).data,
            v.data
        );
    }

    #[test]
    fn downsample_mean_and_upsample_nearest() {
        let input = Tensor::from_vec_on(vec![1.0, 3.0, 5.0, 7.0], &[4, 1], Device::Cpu);
        let down = sequence_downsample(&input, 2, "mean");
        assert_eq!(down.shape, vec![2, 1]);
        assert_eq!(down.data, vec![2.0, 6.0]);
        let up = sequence_upsample(&down, 2);
        assert_eq!(up.shape, vec![4, 1]);
        assert_eq!(up.data, vec![2.0, 2.0, 6.0, 6.0]);
    }

    #[test]
    fn seq2seq_dot_attention_is_stable() {
        let query = Tensor::from_vec_on(vec![1.0, 0.0], &[1, 2], Device::Cpu);
        let keys = Tensor::from_vec_on(vec![1.0, 0.0, 0.0, 1.0], &[1, 2, 2], Device::Cpu);
        let (context, weights) =
            seq2seq_attention(&query, &keys, None, AttentionKind::Dot, None, None, None).unwrap();
        assert_eq!(context.shape, vec![1, 2]);
        assert_eq!(weights.shape, vec![1, 2]);
        assert!(weights.data[0] > weights.data[1]);
        let sum: f32 = weights.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
