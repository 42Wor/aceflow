//! Packed stacked RNN / LSTM / GRU and seq2seq attention.

use crate::cpu::rnn::{self, RnnKind, RnnOutput};
use crate::cpu::seq_attn;
use crate::error::Result;
use crate::nn::Module;
use crate::tensor::Tensor;

pub use crate::cpu::rnn::RnnKind as RnnType;
pub use crate::cpu::seq_attn::AttentionKind;

/// Packed stacked RNN / LSTM / GRU. Layouts match PyTorch `batch_first=True`.
pub fn rnn_forward(
    input: &Tensor,
    hidden: &Tensor,
    cell: Option<&Tensor>,
    weights: &[f32],
    biases: &[f32],
    kind: RnnKind,
    bidirectional: bool,
) -> Result<(Tensor, Tensor, Tensor)> {
    input.assert_contiguous();
    hidden.assert_contiguous();
    assert_eq!(input.rank(), 3, "rnn input must be [batch, seq, features]");
    assert_eq!(hidden.rank(), 3, "rnn hidden must be [layers, batch, hidden]");
    let batch = input.shape[0];
    let seq_len = input.shape[1];
    let input_size = input.shape[2];
    let stacked = hidden.shape[0];
    assert_eq!(hidden.shape[1], batch, "rnn hidden batch mismatch");
    let hidden_size = hidden.shape[2];
    let dirs = if bidirectional { 2 } else { 1 };
    assert_eq!(
        stacked % dirs,
        0,
        "hidden layer axis must be num_layers * num_directions"
    );
    let num_layers = stacked / dirs;
    let cell_data = cell.map(|value| {
        value.assert_contiguous();
        value.data.as_slice()
    });
    let out = rnn::rnn_forward(
        &input.data,
        &hidden.data,
        cell_data.unwrap_or(&[]),
        weights,
        biases,
        batch,
        seq_len,
        input_size,
        hidden_size,
        num_layers,
        kind,
        bidirectional,
    )?;
    Ok(tensors_from_rnn(
        out,
        batch,
        seq_len,
        hidden_size,
        stacked,
        input.device(),
    ))
}

fn tensors_from_rnn(
    out: RnnOutput,
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    stacked: usize,
    device: crate::device::Device,
) -> (Tensor, Tensor, Tensor) {
    let dirs_hidden = out.output.len() / (batch * seq_len);
    (
        Tensor::from_vec_on(out.output, &[batch, seq_len, dirs_hidden], device),
        Tensor::from_vec_on(out.hidden, &[stacked, batch, hidden_size], device),
        Tensor::from_vec_on(out.cell, &[stacked, batch, hidden_size], device),
    )
}

/// Thin stacked-RNN module used by Python `mallow_core.nn`.
#[derive(Clone, Debug)]
pub struct Rnn {
    pub kind: RnnKind,
    pub bidirectional: bool,
}

impl Rnn {
    pub fn new(kind: RnnKind, bidirectional: bool) -> Self {
        Self {
            kind,
            bidirectional,
        }
    }

    pub fn forward_packed(
        &self,
        input: &Tensor,
        hidden: &Tensor,
        cell: Option<&Tensor>,
        weights: &[f32],
        biases: &[f32],
    ) -> Result<(Tensor, Tensor, Tensor)> {
        rnn_forward(
            input,
            hidden,
            cell,
            weights,
            biases,
            self.kind,
            self.bidirectional,
        )
    }
}

impl Module for Rnn {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let _ = input;
        Err("Rnn::forward requires packed weights; use forward_packed".into())
    }
}

/// Batched Bahdanau / Luong attention replacing AceFlow's C `_attention_ops`.
///
/// `query` is `[batch, hidden]`, `keys`/`values` are `[batch, seq, hidden]`.
/// Returns `(context [batch, hidden], weights [batch, seq])`.
pub fn seq2seq_attention(
    query: &Tensor,
    keys: &Tensor,
    values: Option<&Tensor>,
    kind: AttentionKind,
    w1: Option<&Tensor>,
    w2: Option<&Tensor>,
    v: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    query.assert_contiguous();
    keys.assert_contiguous();
    let values = values.unwrap_or(keys);
    values.assert_contiguous();
    let (batch, hidden) = match query.rank() {
        1 => (1usize, query.shape[0]),
        2 => (query.shape[0], query.shape[1]),
        _ => panic!("seq2seq query must be rank 1 or 2"),
    };
    assert_eq!(keys.rank(), 3, "seq2seq keys must be [batch, seq, hidden]");
    assert_eq!(keys.shape[0], batch);
    assert_eq!(keys.shape[2], hidden);
    let seq_len = keys.shape[1];
    assert_eq!(values.shape, keys.shape);
    let mut scores = vec![0.0f32; batch * seq_len];
    let mut context = vec![0.0f32; batch * hidden];
    seq_attn::seq2seq_attention(
        &query.data,
        &keys.data,
        &mut scores,
        &mut context,
        batch,
        seq_len,
        hidden,
        kind,
        w1.map(|tensor| tensor.data.as_slice()),
        w2.map(|tensor| tensor.data.as_slice()),
        v.map(|tensor| tensor.data.as_slice()),
    )?;
    Ok((
        Tensor::from_vec_on(context, &[batch, hidden], query.device()),
        Tensor::from_vec_on(scores, &[batch, seq_len], query.device()),
    ))
}
