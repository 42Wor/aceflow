//! PyTorch-compatible RNN / LSTM / GRU inference kernels.
//!
//! Replaces AceFlow's OpenMP C extension (`_rnn_ops.c`). Weights use the
//! PyTorch layout `(gate_size, in_features)` so a trained `nn.LSTM` can be
//! packed and run without a transpose. The inner projections are AVX2/FMA
//! (or NEON) GEMVs from [`crate::cpu::gemv`].

use crate::cpu::gemv::gemv;
use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum RnnKind {
    Rnn = 0,
    Lstm = 1,
    Gru = 2,
}

impl RnnKind {
    pub fn from_code(code: i32) -> Result<Self> {
        match code {
            0 => Ok(Self::Rnn),
            1 => Ok(Self::Lstm),
            2 => Ok(Self::Gru),
            _ => Err(Error(format!("unknown RNN type code {code}"))),
        }
    }

    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "rnn" | "birnn" => Ok(Self::Rnn),
            "lstm" | "bilstm" => Ok(Self::Lstm),
            "gru" | "bigru" => Ok(Self::Gru),
            _ => Err(Error(format!("unknown RNN type '{name}'"))),
        }
    }

    pub fn gate_mul(self) -> usize {
        match self {
            Self::Rnn => 1,
            Self::Lstm => 4,
            Self::Gru => 3,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RnnOutput {
    /// `[batch, seq, hidden * num_directions]`
    pub output: Vec<f32>,
    /// `[num_layers * num_directions, batch, hidden]`
    pub hidden: Vec<f32>,
    /// Same layout as `hidden`; zeros when the cell is not LSTM.
    pub cell: Vec<f32>,
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn bias_at(bias: &[f32], index: usize) -> f32 {
    if bias.is_empty() {
        0.0
    } else {
        bias[index]
    }
}

fn expected_param_lens(
    kind: RnnKind,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,
) -> (usize, usize) {
    let dirs = if bidirectional { 2 } else { 1 };
    let gmul = kind.gate_mul();
    let mut feat = input_size;
    let mut weights = 0usize;
    let mut biases = 0usize;
    for _ in 0..num_layers {
        for _ in 0..dirs {
            let gates = gmul * hidden_size;
            weights += gates * feat + gates * hidden_size;
            biases += 2 * gates;
        }
        feat = hidden_size * dirs;
    }
    (weights, biases)
}

fn project_step(
    x: &[f32],
    h: &[f32],
    w_ih: &[f32],
    w_hh: &[f32],
    b_ih: &[f32],
    b_hh: &[f32],
    gates: &mut [f32],
    hh: &mut [f32],
) {
    let gate_size = gates.len();
    gemv(gate_size, x.len(), w_ih, x, gates);
    gemv(gate_size, h.len(), w_hh, h, hh);
    for i in 0..gate_size {
        gates[i] += hh[i] + bias_at(b_ih, i) + bias_at(b_hh, i);
    }
}

fn lstm_cell(
    x: &[f32],
    h: &mut [f32],
    c: &mut [f32],
    w_ih: &[f32],
    w_hh: &[f32],
    b_ih: &[f32],
    b_hh: &[f32],
    gates: &mut [f32],
    hh: &mut [f32],
) {
    let hidden = h.len();
    project_step(x, h, w_ih, w_hh, b_ih, b_hh, gates, hh);
    for i in 0..hidden {
        let input_g = sigmoid(gates[i]);
        let forget = sigmoid(gates[hidden + i]);
        let cell_g = gates[2 * hidden + i].tanh();
        let output = sigmoid(gates[3 * hidden + i]);
        let cell = forget * c[i] + input_g * cell_g;
        c[i] = cell;
        h[i] = output * cell.tanh();
    }
}

fn gru_cell(
    x: &[f32],
    h: &mut [f32],
    w_ih: &[f32],
    w_hh: &[f32],
    b_ih: &[f32],
    b_hh: &[f32],
    gi: &mut [f32],
    gh: &mut [f32],
) {
    let hidden = h.len();
    let gates = 3 * hidden;
    gemv(gates, x.len(), w_ih, x, gi);
    gemv(gates, hidden, w_hh, h, gh);
    for i in 0..gates {
        gi[i] += bias_at(b_ih, i);
        gh[i] += bias_at(b_hh, i);
    }
    for i in 0..hidden {
        let reset = sigmoid(gi[i] + gh[i]);
        let update = sigmoid(gi[hidden + i] + gh[hidden + i]);
        let new = (gi[2 * hidden + i] + reset * gh[2 * hidden + i]).tanh();
        h[i] = (1.0 - update) * new + update * h[i];
    }
}

fn vanilla_cell(
    x: &[f32],
    h: &mut [f32],
    w_ih: &[f32],
    w_hh: &[f32],
    b_ih: &[f32],
    b_hh: &[f32],
    scratch: &mut [f32],
    hh: &mut [f32],
) {
    project_step(x, h, w_ih, w_hh, b_ih, b_hh, scratch, hh);
    for (slot, value) in h.iter_mut().zip(scratch.iter()) {
        *slot = value.tanh();
    }
}

/// Packed unidirectional or bidirectional stacked RNN.
///
/// * `input` is `[batch, seq, input_size]`
/// * `hidden0` / `cell0` are `[num_layers * dirs, batch, hidden]` (cell ignored unless LSTM)
/// * `weights` / `biases` are concatenated in PyTorch module order:
///   for each layer, for each direction (`fwd` then `rev`):
///   `W_ih`, `W_hh` then `b_ih`, `b_hh`.
pub fn rnn_forward(
    input: &[f32],
    hidden0: &[f32],
    cell0: &[f32],
    weights: &[f32],
    biases: &[f32],
    batch: usize,
    seq_len: usize,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    kind: RnnKind,
    bidirectional: bool,
) -> Result<RnnOutput> {
    if batch == 0 || seq_len == 0 || input_size == 0 || hidden_size == 0 || num_layers == 0 {
        return Err(Error(
            "batch, seq_len, input_size, hidden_size, and num_layers must be positive".into(),
        ));
    }
    let dirs = if bidirectional { 2 } else { 1 };
    if input.len() != batch * seq_len * input_size {
        return Err(Error(format!(
            "input length {} does not match [{batch}, {seq_len}, {input_size}]",
            input.len()
        )));
    }
    let stacked = num_layers * dirs;
    let state_len = stacked * batch * hidden_size;
    if !hidden0.is_empty() && hidden0.len() != state_len {
        return Err(Error(format!(
            "hidden length {} does not match [{stacked}, {batch}, {hidden_size}]",
            hidden0.len()
        )));
    }
    if kind == RnnKind::Lstm && !cell0.is_empty() && cell0.len() != state_len {
        return Err(Error(format!(
            "cell length {} does not match [{stacked}, {batch}, {hidden_size}]",
            cell0.len()
        )));
    }
    let (want_w, want_b) =
        expected_param_lens(kind, input_size, hidden_size, num_layers, bidirectional);
    if weights.len() != want_w {
        return Err(Error(format!(
            "weights length {} does not match expected {want_w}",
            weights.len()
        )));
    }
    if !biases.is_empty() && biases.len() != want_b {
        return Err(Error(format!(
            "biases length {} does not match expected {want_b}",
            biases.len()
        )));
    }

    let out_width = hidden_size * dirs;
    let mut layer_in = input.to_vec();
    let mut feat = input_size;
    let mut final_hidden = vec![0.0f32; state_len];
    let mut final_cell = vec![0.0f32; state_len];
    let mut layer_out = Vec::new();

    let mut w_off = 0usize;
    let mut b_off = 0usize;
    let gmul = kind.gate_mul();

    for layer in 0..num_layers {
        layer_out.resize(batch * seq_len * out_width, 0.0);
        layer_out.fill(0.0);
        for dir in 0..dirs {
            let gates = gmul * hidden_size;
            let w_ih = &weights[w_off..w_off + gates * feat];
            w_off += gates * feat;
            let w_hh = &weights[w_off..w_off + gates * hidden_size];
            w_off += gates * hidden_size;
            let (b_ih, b_hh) = if biases.is_empty() {
                (&[][..], &[][..])
            } else {
                let b_ih = &biases[b_off..b_off + gates];
                b_off += gates;
                let b_hh = &biases[b_off..b_off + gates];
                b_off += gates;
                (b_ih, b_hh)
            };

            let state_base = (layer * dirs + dir) * batch * hidden_size;
            let mut h = if hidden0.is_empty() {
                vec![0.0f32; batch * hidden_size]
            } else {
                hidden0[state_base..state_base + batch * hidden_size].to_vec()
            };
            let mut c = if kind != RnnKind::Lstm || cell0.is_empty() {
                vec![0.0f32; batch * hidden_size]
            } else {
                cell0[state_base..state_base + batch * hidden_size].to_vec()
            };

            let reverse = dir == 1;
            run_direction(
                kind,
                &layer_in,
                &mut h,
                &mut c,
                &mut layer_out,
                batch,
                seq_len,
                feat,
                hidden_size,
                out_width,
                dir * hidden_size,
                reverse,
                w_ih,
                w_hh,
                b_ih,
                b_hh,
            );
            final_hidden[state_base..state_base + batch * hidden_size].copy_from_slice(&h);
            if kind == RnnKind::Lstm {
                final_cell[state_base..state_base + batch * hidden_size].copy_from_slice(&c);
            }
        }
        layer_in.clone_from(&layer_out);
        feat = out_width;
    }

    Ok(RnnOutput {
        output: layer_in,
        hidden: final_hidden,
        cell: final_cell,
    })
}

fn run_direction(
    kind: RnnKind,
    input: &[f32],
    hidden: &mut [f32],
    cell: &mut [f32],
    output: &mut [f32],
    batch: usize,
    seq_len: usize,
    input_size: usize,
    hidden_size: usize,
    out_width: usize,
    dir_offset: usize,
    reverse: bool,
    w_ih: &[f32],
    w_hh: &[f32],
    b_ih: &[f32],
    b_hh: &[f32],
) {
    let gates = kind.gate_mul() * hidden_size;
    let mut gate_buf = vec![0.0f32; gates];
    let mut hh_buf = vec![0.0f32; gates];
    for step in 0..seq_len {
        let t = if reverse { seq_len - 1 - step } else { step };
        for b in 0..batch {
            let x = &input[(b * seq_len + t) * input_size..][..input_size];
            let h = &mut hidden[b * hidden_size..][..hidden_size];
            let c = &mut cell[b * hidden_size..][..hidden_size];
            match kind {
                RnnKind::Lstm => lstm_cell(x, h, c, w_ih, w_hh, b_ih, b_hh, &mut gate_buf, &mut hh_buf),
                RnnKind::Gru => gru_cell(x, h, w_ih, w_hh, b_ih, b_hh, &mut gate_buf, &mut hh_buf),
                RnnKind::Rnn => {
                    vanilla_cell(x, h, w_ih, w_hh, b_ih, b_hh, &mut gate_buf, &mut hh_buf)
                }
            }
            let dst = (b * seq_len + t) * out_width + dir_offset;
            output[dst..dst + hidden_size].copy_from_slice(h);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_gemv(m: usize, k: usize, w: &[f32], x: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0f32; m];
        for row in 0..m {
            let mut sum = 0.0;
            for j in 0..k {
                sum += w[row * k + j] * x[j];
            }
            y[row] = sum;
        }
        y
    }

    #[test]
    fn lstm_cell_matches_naive_reference() {
        let input_size = 3;
        let hidden = 2;
        let x = vec![0.5, -0.25, 0.75];
        let mut h = vec![0.1, -0.2];
        let mut c = vec![0.3, 0.05];
        let w_ih: Vec<f32> = (0..4 * hidden * input_size)
            .map(|i| (i as f32) * 0.05 - 0.4)
            .collect();
        let w_hh: Vec<f32> = (0..4 * hidden * hidden)
            .map(|i| (i as f32) * 0.07 - 0.2)
            .collect();
        let b_ih: Vec<f32> = (0..4 * hidden).map(|i| (i as f32) * 0.01).collect();
        let b_hh: Vec<f32> = (0..4 * hidden).map(|i| -0.02 * i as f32).collect();

        let gi = naive_gemv(4 * hidden, input_size, &w_ih, &x);
        let gh = naive_gemv(4 * hidden, hidden, &w_hh, &h);
        let mut gates = vec![0.0f32; 4 * hidden];
        for i in 0..4 * hidden {
            gates[i] = gi[i] + gh[i] + b_ih[i] + b_hh[i];
        }
        let mut href = h.clone();
        let mut cref = c.clone();
        for i in 0..hidden {
            let ig = sigmoid(gates[i]);
            let fg = sigmoid(gates[hidden + i]);
            let cg = gates[2 * hidden + i].tanh();
            let og = sigmoid(gates[3 * hidden + i]);
            cref[i] = fg * cref[i] + ig * cg;
            href[i] = og * cref[i].tanh();
        }

        let mut gbuf = vec![0.0; 4 * hidden];
        let mut hbuf = vec![0.0; 4 * hidden];
        lstm_cell(
            &x, &mut h, &mut c, &w_ih, &w_hh, &b_ih, &b_hh, &mut gbuf, &mut hbuf,
        );
        for (a, b) in h.iter().zip(&href) {
            assert!((a - b).abs() < 1e-5);
        }
        for (a, b) in c.iter().zip(&cref) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn stacked_lstm_writes_bt_layout() {
        let batch = 2;
        let seq = 3;
        let input_size = 2;
        let hidden = 2;
        let input: Vec<f32> = (0..batch * seq * input_size)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let w_ih = vec![0.1f32; 4 * hidden * input_size];
        let w_hh = vec![0.05f32; 4 * hidden * hidden];
        let mut weights = w_ih;
        weights.extend_from_slice(&w_hh);
        let biases = vec![0.0f32; 8 * hidden];
        let out = rnn_forward(
            &input,
            &[],
            &[],
            &weights,
            &biases,
            batch,
            seq,
            input_size,
            hidden,
            1,
            RnnKind::Lstm,
            false,
        )
        .unwrap();
        assert_eq!(out.output.len(), batch * seq * hidden);
        assert_eq!(out.hidden.len(), 1 * batch * hidden);
        assert!(out.output.iter().all(|v| v.is_finite()));
    }
}
