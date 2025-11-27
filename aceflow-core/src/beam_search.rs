use ndarray::{Array2, Array3, Axis};
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::f32;

#[derive(Debug, Clone)]
pub struct BeamCandidate {
    pub sequence: Vec<usize>,
    pub score: f32,
    pub hidden_state: Option<Array2<f32>>,
    pub cell_state: Option<Array2<f32>>, // For LSTM
}

impl PartialEq for BeamCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.sequence == other.sequence && OrderedFloat(self.score) == OrderedFloat(other.score)
    }
}

impl Eq for BeamCandidate {}

impl PartialOrd for BeamCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BeamCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        OrderedFloat(self.score).cmp(&OrderedFloat(other.score))
    }
}

#[derive(Debug)]
pub struct BeamSearchResult {
    pub sequences: Vec<Vec<usize>>,
    pub scores: Vec<f32>,
}

pub struct BeamSearchConfig {
    pub beam_width: usize,
    pub max_length: usize,
    pub length_penalty: f32,
    pub early_stopping: bool,
    pub temperature: f32,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 5,
            max_length: 50,
            length_penalty: 0.6,
            early_stopping: true,
            temperature: 1.0,
        }
    }
}

pub struct BeamSearch {
    config: BeamSearchConfig,
}

impl BeamSearch {
    pub fn new(config: BeamSearchConfig) -> Self {
        Self { config }
    }

    /// Perform beam search for sequence generation
    pub fn search(
        &self,
        encoder_outputs: &Array2<f32>,
        initial_hidden: &Array2<f32>,
        initial_cell: Option<&Array2<f32>>,
        start_token: usize,
        end_token: usize,
        vocab_size: usize,
    ) -> BeamSearchResult {
        let batch_size = encoder_outputs.shape()[0];
        let mut all_results = Vec::with_capacity(batch_size);

        // Process each sequence in the batch in parallel
        all_results.par_extend((0..batch_size).into_par_iter().map(|batch_idx| {
            self.search_single_sequence(
                encoder_outputs.index_axis(Axis(0), batch_idx),
                initial_hidden.index_axis(Axis(0), batch_idx),
                initial_cell.map(|cell| cell.index_axis(Axis(0), batch_idx)),
                start_token,
                end_token,
                vocab_size,
            )
        }));

        // For now, return the first sequence result
        // In practice, you'd return all results
        all_results.into_iter().next().unwrap_or_else(|| BeamSearchResult {
            sequences: vec![],
            scores: vec![],
        })
    }

    fn search_single_sequence(
        &self,
        encoder_outputs: ndarray::ArrayView1<f32>,
        initial_hidden: ndarray::ArrayView1<f32>,
        initial_cell: Option<ndarray::ArrayView1<f32>>,
        start_token: usize,
        end_token: usize,
        vocab_size: usize,
    ) -> BeamSearchResult {
        let mut beams = BinaryHeap::new();
        
        // Initialize with start token
        let initial_candidate = BeamCandidate {
            sequence: vec![start_token],
            score: 0.0,
            hidden_state: Some(initial_hidden.to_owned().into_dimensionality().unwrap()),
            cell_state: initial_cell.map(|c| c.to_owned().into_dimensionality().unwrap()),
        };
        
        beams.push(initial_candidate);

        for step in 0..self.config.max_length {
            let mut next_beams = BinaryHeap::new();
            let mut completed_sequences = Vec::new();

            // Process each beam candidate
            while let Some(candidate) = beams.pop() {
                // Check if sequence is complete
                if candidate.sequence.last() == Some(&end_token) {
                    completed_sequences.push(candidate);
                    continue;
                }

                // Get next token probabilities (simulated - in practice, you'd run the model)
                let probabilities = self.simulate_next_token_probabilities(
                    &candidate,
                    &encoder_outputs,
                    vocab_size,
                );

                // Get top-k candidates
                let top_k_indices = self.get_top_k_indices(&probabilities, self.config.beam_width);

                for &token_idx in &top_k_indices {
                    let token_prob = probabilities[token_idx];
                    
                    if token_prob <= f32::MIN_POSITIVE {
                        continue;
                    }

                    let token_score = if self.config.temperature != 1.0 {
                        (token_prob.ln() / self.config.temperature).exp()
                    } else {
                        token_prob
                    };

                    let mut new_sequence = candidate.sequence.clone();
                    new_sequence.push(token_idx);

                    let length_penalty = if self.config.length_penalty != 0.0 {
                        ((1.0 + new_sequence.len() as f32) / 6.0).powf(self.config.length_penalty)
                    } else {
                        1.0
                    };

                    let new_score = candidate.score + token_score.ln() / length_penalty;

                    let new_candidate = BeamCandidate {
                        sequence: new_sequence,
                        score: new_score,
                        hidden_state: candidate.hidden_state.clone(),
                        cell_state: candidate.cell_state.clone(),
                    };

                    next_beams.push(new_candidate);
                }
            }

            // Keep only top beam_width candidates
            beams = self.truncate_beams(next_beams, self.config.beam_width);

            // Early stopping if all beams are complete
            if self.config.early_stopping && beams.is_empty() {
                break;
            }

            // Move completed sequences to results if we have enough good candidates
            if step == self.config.max_length - 1 {
                completed_sequences.extend(beams.into_vec());
                beams = BinaryHeap::from_vec(completed_sequences);
                break;
            }
        }

        // Convert final beams to result
        let mut final_beams: Vec<_> = beams.into_vec();
        final_beams.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        let sequences: Vec<Vec<usize>> = final_beams
            .iter()
            .map(|candidate| candidate.sequence.clone())
            .collect();

        let scores: Vec<f32> = final_beams
            .iter()
            .map(|candidate| candidate.score)
            .collect();

        BeamSearchResult { sequences, scores }
    }

    fn simulate_next_token_probabilities(
        &self,
        candidate: &BeamCandidate,
        encoder_outputs: &ndarray::ArrayView1<f32>,
        vocab_size: usize,
    ) -> Vec<f32> {
        // This is a simplified simulation
        // In practice, you would:
        // 1. Run the decoder with the current hidden state and last token
        // 2. Apply attention mechanism using encoder_outputs
        // 3. Get probability distribution over vocabulary
        
        let mut probabilities = vec![0.0; vocab_size];
        
        // Simulate some probability distribution
        // In real implementation, this would come from the model
        let last_token = *candidate.sequence.last().unwrap_or(&0);
        
        for i in 0..vocab_size {
            // Simple simulation: prefer tokens similar to the last one
            let base_prob = if i == last_token {
                0.3
            } else if i == last_token + 1 {
                0.2
            } else {
                0.5 / (vocab_size as f32 - 2.0)
            };
            
            // Add some noise based on encoder outputs (simulated attention)
            let attention_effect = if encoder_outputs.len() > 0 {
                encoder_outputs[i % encoder_outputs.len()].abs() * 0.1
            } else {
                0.0
            };
            
            probabilities[i] = base_prob + attention_effect;
        }
        
        // Normalize to probability distribution
        let sum: f32 = probabilities.iter().sum();
        if sum > 0.0 {
            for prob in probabilities.iter_mut() {
                *prob /= sum;
            }
        }
        
        probabilities
    }

    fn get_top_k_indices(&self, probabilities: &[f32], k: usize) -> Vec<usize> {
        let mut indexed_probs: Vec<(usize, f32)> = probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        
        // Sort by probability in descending order
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Take top k
        indexed_probs
            .into_iter()
            .take(k)
            .map(|(idx, _)| idx)
            .collect()
    }

    fn truncate_beams(&self, mut beams: BinaryHeap<BeamCandidate>, beam_width: usize) -> BinaryHeap<BeamCandidate> {
        let mut truncated = BinaryHeap::new();
        
        for _ in 0..beam_width {
            if let Some(candidate) = beams.pop() {
                truncated.push(candidate);
            } else {
                break;
            }
        }
        
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_beam_search_basic() {
        let config = BeamSearchConfig {
            beam_width: 3,
            max_length: 10,
            ..Default::default()
        };
        
        let beam_search = BeamSearch::new(config);
        
        // Mock data
        let encoder_outputs = Array::from_vec(vec![0.1, 0.2, 0.3]);
        let initial_hidden = Array::from_vec(vec![0.1, 0.1]);
        let start_token = 1;
        let end_token = 2;
        let vocab_size = 100;
        
        let result = beam_search.search_single_sequence(
            encoder_outputs.view(),
            initial_hidden.view(),
            None,
            start_token,
            end_token,
            vocab_size,
        );
        
        assert!(!result.sequences.is_empty());
        assert_eq!(result.sequences.len(), result.scores.len());
    }

    #[test]
    fn test_top_k_indices() {
        let beam_search = BeamSearch::new(BeamSearchConfig::default());
        let probs = vec![0.1, 0.5, 0.2, 0.15, 0.05];
        let top_3 = beam_search.get_top_k_indices(&probs, 3);
        
        assert_eq!(top_3, vec![1, 2, 3]); // Indices of highest probabilities
    }
}