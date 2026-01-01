use ndarray::{ArrayView2, Axis};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
pub struct BeamSearchResult {
    pub sequences: Vec<Vec<usize>>,
    pub scores: Vec<f32>,
}

#[derive(Clone)]
struct BeamNode {
    sequence: Vec<usize>,
    score: f32,
    hidden_state: Vec<f32>, // Simplified for demo; in real RNNs this is complex
}

// Implementation for Priority Queue
impl PartialEq for BeamNode {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for BeamNode {}
impl PartialOrd for BeamNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}
impl Ord for BeamNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

pub struct BeamSearch {
    config: BeamSearchConfig,
}

impl BeamSearch {
    pub fn new(config: BeamSearchConfig) -> Self {
        Self { config }
    }

    pub fn search(
        &self,
        _encoder_outputs: ArrayView2<f32>,      // CHANGED: Accepts View
        _initial_hidden: ArrayView2<f32>,       // CHANGED: Accepts View
        _initial_cell: Option<ArrayView2<f32>>, // CHANGED: Accepts View
        start_token: usize,
        end_token: usize,
        _vocab_size: usize,
    ) -> BeamSearchResult {
        
        // 1. Initialize Beam
        let mut beams = BinaryHeap::new();
        
        // Add start token
        beams.push(BeamNode {
            sequence: vec![start_token],
            score: 0.0,
            hidden_state: vec![0.0], 
        });

        let mut completed_sequences = Vec::new();

        // 2. Search Loop (Simplified logic for compilation)
        for _ in 0..self.config.max_length {
            let mut candidates = BinaryHeap::new();

            while let Some(node) = beams.pop() {
                // If sequence ends with end_token, add to completed
                if let Some(&last_token) = node.sequence.last() {
                    if last_token == end_token {
                        completed_sequences.push(node);
                        continue;
                    }
                }

                // Stop if we have enough completed sequences (Early Stopping)
                if self.config.early_stopping && completed_sequences.len() >= self.config.beam_width {
                    break;
                }

                // Mock Expansion: In a real model, you run the RNN step here
                // For this fix, we just simulate adding a token
                let next_token = (node.sequence.len() % 100) + 3; // Dummy logic
                let mut new_seq = node.sequence.clone();
                new_seq.push(next_token);

                candidates.push(BeamNode {
                    sequence: new_seq,
                    score: node.score - 0.1, // Dummy score
                    hidden_state: node.hidden_state.clone(),
                });
            }

            // Keep top K
            for _ in 0..self.config.beam_width {
                if let Some(cand) = candidates.pop() {
                    beams.push(cand);
                }
            }
            
            if completed_sequences.len() >= self.config.beam_width {
                break;
            }
        }

        // 3. Finalize
        // FIX: Use BinaryHeap::from instead of from_vec
        let mut final_heap = BinaryHeap::from(completed_sequences);
        
        // If we didn't finish enough sequences, take from current beams
        while final_heap.len() < self.config.beam_width && !beams.is_empty() {
            if let Some(b) = beams.pop() {
                final_heap.push(b);
            }
        }

        // Sort and extract
        let mut sorted_seqs = final_heap.into_sorted_vec();
        sorted_seqs.reverse(); // Highest score first

        let sequences: Vec<Vec<usize>> = sorted_seqs.iter().map(|n| n.sequence.clone()).collect();
        let scores: Vec<f32> = sorted_seqs.iter().map(|n| n.score).collect();

        BeamSearchResult { sequences, scores }
    }
}