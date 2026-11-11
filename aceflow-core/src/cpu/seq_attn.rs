//! Batched seq2seq attention kernels that replace AceFlow `_attention_ops.c`.
//!
//! Scoring modes match the C extension (`dot`, `general`, `concat`) but use
//! PyTorch Linear weight layout `(out, in)` and SIMD GEMV/dot products.

use crate::cpu::gemv::{axpy, batched_gemv, dot, gemv};
use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum AttentionKind {
    Dot = 0,
    General = 1,
    Concat = 2,
}

impl AttentionKind {
    pub fn from_code(code: i32) -> Result<Self> {
        match code {
            0 => Ok(Self::Dot),
            1 => Ok(Self::General),
            2 => Ok(Self::Concat),
            _ => Err(Error(format!("unknown attention type code {code}"))),
        }
    }

    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "dot" => Ok(Self::Dot),
            "general" => Ok(Self::General),
            "concat" | "additive" | "bahdanau" => Ok(Self::Concat),
            _ => Err(Error(format!("unknown attention type '{name}'"))),
        }
    }
}

fn softmax_rows(scores: &mut [f32], batch: usize, seq_len: usize) {
    for row in scores.chunks_mut(seq_len).take(batch) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for value in row.iter_mut() {
            *value = (*value - max).exp();
            sum += *value;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for value in row.iter_mut() {
                *value *= inv;
            }
        }
    }
}

fn weight_as_square<'a>(weights: Option<&'a [f32]>, hidden: usize, what: &str) -> Result<&'a [f32]> {
    let weights = weights.ok_or_else(|| Error(format!("missing {what} weights")))?;
    if weights.len() != hidden * hidden {
        return Err(Error(format!(
            "{what} length {} must be hidden_size^2 ({})",
            weights.len(),
            hidden * hidden
        )));
    }
    Ok(weights)
}

fn flatten_v(v: Option<&[f32]>, hidden: usize) -> Result<Vec<f32>> {
    let v = v.ok_or_else(|| Error("missing v weights for concat attention".into()))?;
    if v.len() == hidden {
        Ok(v.to_vec())
    } else {
        Err(Error(format!(
            "v length {} must be hidden_size ({hidden})",
            v.len()
        )))
    }
}

/// Bahdanau / Luong attention over `[batch, hidden]` queries and
/// `[batch, seq, hidden]` encoder states.
///
/// Writes softmax weights `[batch, seq]` and context `[batch, hidden]`.
pub fn seq2seq_attention(
    decoder_hidden: &[f32],
    encoder_outputs: &[f32],
    scores: &mut [f32],
    context: &mut [f32],
    batch: usize,
    seq_len: usize,
    hidden: usize,
    kind: AttentionKind,
    w1: Option<&[f32]>,
    w2: Option<&[f32]>,
    v: Option<&[f32]>,
) -> Result<()> {
    if batch == 0 || seq_len == 0 || hidden == 0 {
        return Err(Error(
            "batch, seq_len, and hidden_size must be positive".into(),
        ));
    }
    if decoder_hidden.len() != batch * hidden {
        return Err(Error(format!(
            "decoder_hidden length {} does not match [{batch}, {hidden}]",
            decoder_hidden.len()
        )));
    }
    if encoder_outputs.len() != batch * seq_len * hidden {
        return Err(Error(format!(
            "encoder_outputs length {} does not match [{batch}, {seq_len}, {hidden}]",
            encoder_outputs.len()
        )));
    }
    if scores.len() != batch * seq_len || context.len() != batch * hidden {
        return Err(Error(
            "attention output buffers have the wrong length".into(),
        ));
    }

    scores.fill(0.0);
    context.fill(0.0);

    match kind {
        AttentionKind::Dot => {
            for b in 0..batch {
                let query = &decoder_hidden[b * hidden..(b + 1) * hidden];
                let keys = &encoder_outputs[b * seq_len * hidden..(b + 1) * seq_len * hidden];
                let dest = &mut scores[b * seq_len..(b + 1) * seq_len];
                gemv(seq_len, hidden, keys, query, dest);
            }
        }
        AttentionKind::General => {
            let w1 = weight_as_square(w1, hidden, "W1")?;
            let mut transformed = vec![0.0f32; seq_len * hidden];
            for b in 0..batch {
                let keys = &encoder_outputs[b * seq_len * hidden..(b + 1) * seq_len * hidden];
                batched_gemv(hidden, hidden, w1, keys, &mut transformed);
                let query = &decoder_hidden[b * hidden..(b + 1) * hidden];
                let dest = &mut scores[b * seq_len..(b + 1) * seq_len];
                gemv(seq_len, hidden, &transformed, query, dest);
            }
        }
        AttentionKind::Concat => {
            let w1 = weight_as_square(w1, hidden, "W1")?;
            let w2 = weight_as_square(w2, hidden, "W2")?;
            let v = flatten_v(v, hidden)?;
            let mut key_proj = vec![0.0f32; seq_len * hidden];
            let mut query_proj = vec![0.0f32; hidden];
            let mut activated = vec![0.0f32; hidden];
            for b in 0..batch {
                let keys = &encoder_outputs[b * seq_len * hidden..(b + 1) * seq_len * hidden];
                batched_gemv(hidden, hidden, w1, keys, &mut key_proj);
                gemv(
                    hidden,
                    hidden,
                    w2,
                    &decoder_hidden[b * hidden..(b + 1) * hidden],
                    &mut query_proj,
                );
                for t in 0..seq_len {
                    let key_row = &key_proj[t * hidden..(t + 1) * hidden];
                    for h in 0..hidden {
                        activated[h] = (key_row[h] + query_proj[h]).tanh();
                    }
                    scores[b * seq_len + t] = dot(&v, &activated);
                }
            }
        }
    }

    softmax_rows(scores, batch, seq_len);

    for b in 0..batch {
        let dest = &mut context[b * hidden..(b + 1) * hidden];
        dest.fill(0.0);
        for t in 0..seq_len {
            let weight = scores[b * seq_len + t];
            let src = &encoder_outputs[(b * seq_len + t) * hidden..][..hidden];
            axpy(weight, src, dest);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_attention_picks_matching_key() {
        let hidden = 2;
        let decoder = vec![1.0f32, 0.0, 0.0, 1.0];
        let encoder = vec![
            1.0, 0.0, 0.0, 1.0, // batch 0
            0.0, 1.0, 1.0, 0.0, // batch 1
        ];
        let mut scores = vec![0.0f32; 4];
        let mut context = vec![0.0f32; 4];
        seq2seq_attention(
            &decoder,
            &encoder,
            &mut scores,
            &mut context,
            2,
            2,
            hidden,
            AttentionKind::Dot,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(scores[0] > scores[1]);
        assert!(scores[2] > scores[3]);
        let sum0: f32 = scores[..2].iter().sum();
        assert!((sum0 - 1.0).abs() < 1e-5);
    }
}
