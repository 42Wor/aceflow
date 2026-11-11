//! Multi-head, cross, and additive attention.

use rayon::prelude::*;

use crate::device::Device;
use crate::error::Result;
use crate::nn::linear::Linear;
use crate::nn::norms::LayerNorm;
use crate::nn::Module;
use crate::tensor::{gemm, gemm_bt, Tensor};

#[derive(Clone, Debug)]
pub struct MultiHeadAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub num_heads: usize,
}

impl MultiHeadAttention {
    pub fn new(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        out_proj: Linear,
        num_heads: usize,
    ) -> Self {
        assert!(num_heads > 0, "num_heads must be positive");
        let d_model = q_proj.out_features();
        assert_eq!(d_model % num_heads, 0, "d_model must divide num_heads");
        assert_eq!(k_proj.out_features(), d_model, "K projection mismatch");
        assert_eq!(v_proj.out_features(), d_model, "V projection mismatch");
        assert_eq!(
            out_proj.in_features(),
            d_model,
            "output projection mismatch"
        );
        Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
        }
    }

    pub fn from_features(d_model: usize, num_heads: usize, device: Device) -> Self {
        Self::new(
            Linear::from_features(d_model, d_model, false, device),
            Linear::from_features(d_model, d_model, false, device),
            Linear::from_features(d_model, d_model, false, device),
            Linear::from_features(d_model, d_model, false, device),
            num_heads,
        )
    }

    pub fn forward(&self, x: &Tensor, kv_source: Option<&Tensor>, causal: bool) -> Tensor {
        self.forward_with_kv(x, kv_source, causal).0
    }

    /// Return `(output, projected_k, projected_v)` for cache-friendly callers.
    pub fn forward_with_kv(
        self: &Self,
        x: &Tensor,
        kv_source: Option<&Tensor>,
        causal: bool,
    ) -> (Tensor, Tensor, Tensor) {
        let source = kv_source.unwrap_or(x);
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(source);
        let v = self.v_proj.forward(source);
        let context = scaled_dot_product_attention(&q, &k, &v, self.num_heads, causal);
        (self.out_proj.forward(&context), k, v)
    }

    pub fn forward_projected(&self, q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Tensor {
        let context = scaled_dot_product_attention(q, k, v, self.num_heads, causal);
        self.out_proj.forward(&context)
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(MultiHeadAttention::forward(self, input, None, false))
    }
}

/// Scaled dot-product attention for `(sequence, d_model)` tensors.
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    causal: bool,
) -> Tensor {
    q.assert_contiguous();
    k.assert_contiguous();
    v.assert_contiguous();
    assert_eq!(q.rank(), 2, "attention Q must be rank 2");
    assert_eq!(k.rank(), 2, "attention K must be rank 2");
    assert_eq!(v.rank(), 2, "attention V must be rank 2");
    assert_eq!(k.shape, v.shape, "attention K/V shape mismatch");
    let tq = q.shape[0];
    let tk = k.shape[0];
    let d_model = q.shape[1];
    assert_eq!(k.shape[1], d_model, "attention model dimension mismatch");
    assert!(num_heads > 0 && d_model % num_heads == 0);
    let head_dim = d_model / num_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let device = q.device();

    let head_outputs: Vec<Vec<f32>> = (0..num_heads)
        .into_par_iter()
        .map(|head| {
            let mut q_head = vec![0.0f32; tq * head_dim];
            let mut k_head = vec![0.0f32; tk * head_dim];
            let mut v_head = vec![0.0f32; tk * head_dim];
            for t in 0..tq {
                let src = t * d_model + head * head_dim;
                q_head[t * head_dim..(t + 1) * head_dim]
                    .copy_from_slice(&q.data[src..src + head_dim]);
            }
            for t in 0..tk {
                let src = t * d_model + head * head_dim;
                k_head[t * head_dim..(t + 1) * head_dim]
                    .copy_from_slice(&k.data[src..src + head_dim]);
                v_head[t * head_dim..(t + 1) * head_dim]
                    .copy_from_slice(&v.data[src..src + head_dim]);
            }
            for value in &mut q_head {
                *value *= scale;
            }

            let mut scores = gemm_bt(&q_head, &k_head, tq, head_dim, tk);
            if causal {
                for i in 0..tq {
                    let query_position = tk - tq + i;
                    for j in 0..tk {
                        if j > query_position {
                            scores[i * tk + j] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
            let scores = Tensor::from_vec_on(scores, &[tq, tk], device).softmax_last();
            gemm(&scores.data, &v_head, tq, tk, head_dim)
        })
        .collect();

    let mut output = vec![0.0f32; tq * d_model];
    for head in 0..num_heads {
        for t in 0..tq {
            let dst = t * d_model + head * head_dim;
            output[dst..dst + head_dim]
                .copy_from_slice(&head_outputs[head][t * head_dim..(t + 1) * head_dim]);
        }
    }
    Tensor::from_vec_on(output, &[tq, d_model], device)
}

/// Alias for encoder-decoder cross-attention (`Q` from decoder, `K/V` from encoder).
pub fn cross_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_heads: usize,
    causal: bool,
) -> Tensor {
    scaled_dot_product_attention(query, key, value, num_heads, causal)
}

/// Bahdanau-style additive attention over rank-2 `(time, dim)` tensors.
#[derive(Clone, Debug)]
pub struct AdditiveAttention {
    pub w_keys: Linear,
    pub w_query: Linear,
    pub v: Linear,
}

impl AdditiveAttention {
    pub fn new(w_keys: Linear, w_query: Linear, v: Linear) -> Self {
        Self {
            w_keys,
            w_query,
            v,
        }
    }

    /// Returns `(context [dim], weights [time])`.
    pub fn forward(&self, query: &Tensor, keys: &Tensor, values: &Tensor) -> (Tensor, Tensor) {
        let query = if query.rank() == 1 {
            query.reshape(&[1, query.numel()])
        } else {
            query.clone()
        };
        assert_eq!(query.rows(), 1, "additive attention query must be one vector");
        let key_proj = self.w_keys.forward(keys);
        let query_proj = self.w_query.forward(&query);
        let time = key_proj.rows();
        let hidden = key_proj.cols();
        assert_eq!(query_proj.cols(), hidden);
        let mut energy = vec![0.0f32; time * hidden];
        for t in 0..time {
            for h in 0..hidden {
                energy[t * hidden + h] =
                    (key_proj.data[t * hidden + h] + query_proj.data[h]).tanh();
            }
        }
        let device = keys.device();
        let scores = self
            .v
            .forward(&Tensor::from_vec_on(energy, &[time, hidden], device));
        let weights = Tensor::from_vec_on(scores.data.clone(), &[1, time], device).softmax_last();
        let context = gemm(&weights.data, &values.data, 1, time, values.cols());
        (
            Tensor::from_vec_on(context, &[values.cols()], device),
            Tensor::from_vec_on(weights.data.clone(), &[time], device),
        )
    }
}

/// Pool or stride-subsample the time axis of a rank-2 `(time, dim)` tensor.
pub fn sequence_downsample(x: &Tensor, factor: usize, mode: &str) -> Tensor {
    x.assert_contiguous();
    assert_eq!(x.rank(), 2, "sequence_downsample expects (time, dim)");
    assert!(factor > 0);
    let time = x.shape[0];
    let dim = x.shape[1];
    let out_time = (time + factor - 1) / factor;
    let mut out = vec![0.0f32; out_time * dim];
    match mode {
        "max" => {
            for t in 0..out_time {
                let start = t * factor;
                let end = (start + factor).min(time);
                for d in 0..dim {
                    let mut best = f32::NEG_INFINITY;
                    for src in start..end {
                        best = best.max(x.data[src * dim + d]);
                    }
                    out[t * dim + d] = best;
                }
            }
        }
        "stride" => {
            for t in 0..out_time {
                let src = (t * factor).min(time - 1);
                out[t * dim..(t + 1) * dim].copy_from_slice(&x.data[src * dim..(src + 1) * dim]);
            }
        }
        _ => {
            for t in 0..out_time {
                let start = t * factor;
                let end = (start + factor).min(time);
                let count = (end - start) as f32;
                for d in 0..dim {
                    let mut acc = 0.0;
                    for src in start..end {
                        acc += x.data[src * dim + d];
                    }
                    out[t * dim + d] = acc / count;
                }
            }
        }
    }
    Tensor::from_vec_on(out, &[out_time, dim], x.device())
}

/// Nearest-neighbor upsample along the time axis of a rank-2 tensor.
pub fn sequence_upsample(x: &Tensor, factor: usize) -> Tensor {
    x.assert_contiguous();
    assert_eq!(x.rank(), 2, "sequence_upsample expects (time, dim)");
    assert!(factor > 0);
    let time = x.shape[0];
    let dim = x.shape[1];
    let out_time = time * factor;
    let mut out = vec![0.0f32; out_time * dim];
    for t in 0..time {
        for k in 0..factor {
            let dst = (t * factor + k) * dim;
            out[dst..dst + dim].copy_from_slice(&x.data[t * dim..(t + 1) * dim]);
        }
    }
    Tensor::from_vec_on(out, &[out_time, dim], x.device())
}

/// Encoder self-attention + decoder self-attention + cross-attention scaffold.
#[derive(Clone, Debug)]
pub struct EncoderDecoderBlock {
    pub encoder_attn: MultiHeadAttention,
    pub decoder_attn: MultiHeadAttention,
    pub cross_attn: MultiHeadAttention,
    pub encoder_norm: Option<LayerNorm>,
    pub decoder_norm: Option<LayerNorm>,
}

impl EncoderDecoderBlock {
    pub fn new(
        encoder_attn: MultiHeadAttention,
        decoder_attn: MultiHeadAttention,
        cross_attn: MultiHeadAttention,
        encoder_norm: Option<LayerNorm>,
        decoder_norm: Option<LayerNorm>,
    ) -> Self {
        Self {
            encoder_attn,
            decoder_attn,
            cross_attn,
            encoder_norm,
            decoder_norm,
        }
    }

    pub fn encode(&self, src: &Tensor) -> Tensor {
        let encoded = self.encoder_attn.forward(src, None, false);
        match &self.encoder_norm {
            Some(norm) => norm.forward(&encoded),
            None => encoded,
        }
    }

    pub fn decode(&self, tgt: &Tensor, memory: &Tensor, causal: bool) -> Tensor {
        let decoded = self.decoder_attn.forward(tgt, None, causal);
        let mixed = self.cross_attn.forward(&decoded, Some(memory), false);
        match &self.decoder_norm {
            Some(norm) => norm.forward(&mixed),
            None => mixed,
        }
    }
}
