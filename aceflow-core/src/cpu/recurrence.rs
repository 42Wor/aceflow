/// CPU recurrent-retention state update and output projection.
#[inline(always)]
pub fn retention_step_core(
    states: &mut [f32],
    gamma: &[f32],
    layer_idx: usize,
    num_heads: usize,
    head_dim: usize,
    q_slice: &[f32],
    k_slice: &[f32],
    v_slice: &[f32],
    out_slice: &mut [f32],
) {
    let d = head_dim;
    let head_state_size = d * d;
    let layer_offset = layer_idx * num_heads * head_state_size;

    for h in 0..num_heads {
        let decay = gamma[h];
        let q_head = &q_slice[h * d..(h + 1) * d];
        let k_head = &k_slice[h * d..(h + 1) * d];
        let v_head = &v_slice[h * d..(h + 1) * d];
        let head_offset = layer_offset + h * head_state_size;
        let state_head = &mut states[head_offset..head_offset + head_state_size];
        let output_head = &mut out_slice[h * d..(h + 1) * d];
        output_head.fill(0.0);

        for i in 0..d {
            let k_value = k_head[i];
            let q_value = q_head[i];
            let state_row = &mut state_head[i * d..(i + 1) * d];
            for (state, (value, output)) in state_row
                .iter_mut()
                .zip(v_head.iter().zip(output_head.iter_mut()))
            {
                let updated = decay * *state + k_value * *value;
                *state = updated;
                *output += q_value * updated;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_head_step_matches_outer_product() {
        let mut state = vec![0.0; 4];
        let mut output = vec![0.0; 2];
        retention_step_core(
            &mut state,
            &[0.5],
            0,
            1,
            2,
            &[2.0, 3.0],
            &[4.0, 5.0],
            &[6.0, 7.0],
            &mut output,
        );
        assert_eq!(state, vec![24.0, 28.0, 30.0, 35.0]);
        assert_eq!(output, vec![138.0, 161.0]);
    }
}
