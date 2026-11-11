//! Fused SwiGLU activation retained from MallowRe.

#[inline]
pub fn swiglu_forward(ffn_dim: usize, gate_up: &[f32], hidden: &mut [f32]) {
    assert!(gate_up.len() >= 2 * ffn_dim, "gate_up buffer too small");
    assert!(hidden.len() >= ffn_dim, "hidden buffer too small");
    let gate = &gate_up[..ffn_dim];
    let up = &gate_up[ffn_dim..2 * ffn_dim];
    let chunks = (ffn_dim / 8) * 8;
    for i in 0..chunks {
        let g = gate[i];
        hidden[i] = (g / (1.0 + (-g).exp())) * up[i];
    }
    for i in chunks..ffn_dim {
        let g = gate[i];
        hidden[i] = (g / (1.0 + (-g).exp())) * up[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_gate_is_zero() {
        let mut output = [1.0];
        swiglu_forward(1, &[0.0, 4.0], &mut output);
        assert_eq!(output, [0.0]);
    }
}
