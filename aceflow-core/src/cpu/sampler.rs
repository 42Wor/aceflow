//! Native logit projection and MallowRe-compatible sampling.

use rayon::prelude::*;
use std::cell::Cell;
use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};

thread_local! {
    static RNG_STATE: Cell<(u64, u64)> = Cell::new({
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let s0 = nanos ^ 0x5851f42d4c957f2d;
        let s1 = nanos.rotate_left(17) ^ 0x14057b7ef767814f;
        (
            if s0 == 0 { 0x12345678 } else { s0 },
            if s1 == 0 { 0x87654321 } else { s1 },
        )
    });
}

#[inline]
pub fn random_uniform_f32() -> f32 {
    RNG_STATE.with(|cell| {
        let (mut s0, mut s1) = cell.get();
        let result = s0.wrapping_add(s1);
        s1 ^= s0;
        s0 = s0.rotate_left(24) ^ s1 ^ (s1 << 16);
        s1 = s1.rotate_left(37);
        cell.set((s0, s1));
        ((result >> 40) as f32) * (1.0 / 16_777_216.0)
    })
}

#[inline(always)]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let remainder_a = chunks_a.remainder();
    let remainder_b = chunks_b.remainder();
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let mut sum4 = 0.0f32;
    let mut sum5 = 0.0f32;
    let mut sum6 = 0.0f32;
    let mut sum7 = 0.0f32;
    for (left, right) in chunks_a.zip(chunks_b) {
        sum0 += left[0] * right[0];
        sum1 += left[1] * right[1];
        sum2 += left[2] * right[2];
        sum3 += left[3] * right[3];
        sum4 += left[4] * right[4];
        sum5 += left[5] * right[5];
        sum6 += left[6] * right[6];
        sum7 += left[7] * right[7];
    }
    let mut total = (sum0 + sum1) + (sum2 + sum3) + (sum4 + sum5) + (sum6 + sum7);
    for (left, right) in remainder_a.iter().zip(remainder_b) {
        total += left * right;
    }
    total
}

pub fn project_logits(x: &[f32], lm_head: &[f32], vocab_size: usize, d_model: usize) -> Vec<f32> {
    if vocab_size >= 4096 {
        lm_head
            .par_chunks_exact(d_model)
            .take(vocab_size)
            .map(|row| dot_product(row, x))
            .collect()
    } else {
        let mut logits = Vec::with_capacity(vocab_size);
        for row in lm_head.chunks_exact(d_model).take(vocab_size) {
            logits.push(dot_product(row, x));
        }
        logits
    }
}

pub fn apply_repetition_penalty(
    logits: &mut [f32],
    context_tokens: &[u32],
    repetition_penalty: f32,
) {
    if repetition_penalty == 1.0 || context_tokens.is_empty() {
        return;
    }
    let mut seen = HashSet::with_capacity(context_tokens.len());
    for &token in context_tokens {
        let index = token as usize;
        if index < logits.len() && seen.insert(index) {
            let value = logits[index];
            logits[index] = if value > 0.0 {
                value / repetition_penalty
            } else {
                value * repetition_penalty
            };
        }
    }
}

pub fn sample_token_from_logits(
    mut logits: Vec<f32>,
    context_tokens: &[u32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    apply_repetition_penalty(&mut logits, context_tokens, repetition_penalty);

    if temperature <= 1e-4 {
        let mut best_idx = 0;
        let mut best_value = logits[0];
        for (index, &value) in logits.iter().enumerate().skip(1) {
            if value > best_value {
                best_value = value;
                best_idx = index;
            }
        }
        return best_idx as u32;
    }

    let inv_temperature = 1.0 / temperature.max(1e-5);
    for value in &mut logits {
        *value *= inv_temperature;
    }
    let mut candidates: Vec<(usize, f32)> = logits.into_iter().enumerate().collect();
    if top_k > 0 && top_k < candidates.len() {
        candidates.select_nth_unstable_by(top_k, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(top_k);
    }
    candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let max_logit = candidates[0].1;
    let mut exp_sum = 0.0f32;
    let mut probabilities = Vec::with_capacity(candidates.len());
    for &(_, logit) in &candidates {
        let probability = (logit - max_logit).exp();
        exp_sum += probability;
        probabilities.push(probability);
    }
    let inv_exp_sum = 1.0 / exp_sum.max(1e-12);
    for probability in &mut probabilities {
        *probability *= inv_exp_sum;
    }

    let mut cutoff = candidates.len();
    if top_p < 1.0 {
        let mut cumulative = 0.0f32;
        for (index, &probability) in probabilities.iter().enumerate() {
            cumulative += probability;
            if cumulative > top_p {
                cutoff = index + 1;
                break;
            }
        }
        cutoff = cutoff.max(1).min(candidates.len());
    }
    let candidates = &candidates[..cutoff];
    let probabilities = &probabilities[..cutoff];
    let total_probability: f32 = probabilities.iter().sum();
    let sample = random_uniform_f32() * total_probability;
    let mut cumulative = 0.0f32;
    for (index, &probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if sample <= cumulative {
            return candidates[index].0 as u32;
        }
    }
    candidates[0].0 as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_sampling_and_penalty_match_legacy_rules() {
        assert_eq!(
            sample_token_from_logits(vec![1.0, 3.0, 2.0], &[], 0.0, 0, 1.0, 1.0),
            1
        );
        assert_eq!(
            sample_token_from_logits(vec![1.0, 3.0, 2.0], &[1], 0.0, 0, 1.0, 10.0),
            2
        );
    }
}
