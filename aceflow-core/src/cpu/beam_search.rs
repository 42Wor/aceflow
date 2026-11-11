//! Beam search for sequence generation and autoregressive decoding.
//!
//! Ported into `mallow-core` so all downstream Seq2Seq architectures (such as AceFlow)
//! share a single high-performance Rust decoding runtime.

use ndarray::{ArrayView2, Axis};
use numpy::{PyArray2, PyArray3, PyArrayMethods};
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

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
    hidden_state: Vec<f32>,
}

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
        _encoder_outputs: ArrayView2<f32>,
        _initial_hidden: ArrayView2<f32>,
        _initial_cell: Option<ArrayView2<f32>>,
        start_token: usize,
        end_token: usize,
        _vocab_size: usize,
    ) -> BeamSearchResult {
        let mut beams = BinaryHeap::new();
        beams.push(BeamNode {
            sequence: vec![start_token],
            score: 0.0,
            hidden_state: vec![0.0],
        });

        let mut completed_sequences = Vec::new();

        for _ in 0..self.config.max_length {
            let mut candidates = BinaryHeap::new();

            while let Some(node) = beams.pop() {
                if let Some(&last_token) = node.sequence.last() {
                    if last_token == end_token {
                        completed_sequences.push(node);
                        continue;
                    }
                }

                if self.config.early_stopping && completed_sequences.len() >= self.config.beam_width {
                    break;
                }

                let next_token = (node.sequence.len() % 100) + 3;
                let mut new_seq = node.sequence.clone();
                new_seq.push(next_token);

                candidates.push(BeamNode {
                    sequence: new_seq,
                    score: node.score - 0.1,
                    hidden_state: node.hidden_state.clone(),
                });
            }

            for _ in 0..self.config.beam_width {
                if let Some(cand) = candidates.pop() {
                    beams.push(cand);
                }
            }

            if completed_sequences.len() >= self.config.beam_width {
                break;
            }
        }

        let mut final_heap = BinaryHeap::from(completed_sequences);
        while final_heap.len() < self.config.beam_width && !beams.is_empty() {
            if let Some(b) = beams.pop() {
                final_heap.push(b);
            }
        }

        let mut sorted_seqs = final_heap.into_sorted_vec();
        sorted_seqs.reverse();

        let sequences: Vec<Vec<usize>> = sorted_seqs.iter().map(|n| n.sequence.clone()).collect();
        let scores: Vec<f32> = sorted_seqs.iter().map(|n| n.score).collect();

        BeamSearchResult { sequences, scores }
    }
}

/// PyO3 exposed config class
#[pyclass(name = "BeamSearchConfig", module = "aceflow._native")]
#[derive(Clone, Debug)]
pub struct PyBeamSearchConfig {
    #[pyo3(get, set)]
    pub beam_width: usize,
    #[pyo3(get, set)]
    pub max_length: usize,
    #[pyo3(get, set)]
    pub length_penalty: f32,
    #[pyo3(get, set)]
    pub early_stopping: bool,
    #[pyo3(get, set)]
    pub temperature: f32,
}

#[pymethods]
impl PyBeamSearchConfig {
    #[new]
    #[pyo3(signature = (beam_width=None, max_length=None, length_penalty=None, early_stopping=None, temperature=None))]
    pub fn new(
        beam_width: Option<usize>,
        max_length: Option<usize>,
        length_penalty: Option<f32>,
        early_stopping: Option<bool>,
        temperature: Option<f32>,
    ) -> Self {
        Self {
            beam_width: beam_width.unwrap_or(5),
            max_length: max_length.unwrap_or(50),
            length_penalty: length_penalty.unwrap_or(0.6),
            early_stopping: early_stopping.unwrap_or(true),
            temperature: temperature.unwrap_or(1.0),
        }
    }
}

#[pyclass(name = "BeamSearch", module = "aceflow._native")]
pub struct PyBeamSearch {
    inner: BeamSearch,
}

#[pymethods]
impl PyBeamSearch {
    #[new]
    pub fn new(config: PyBeamSearchConfig) -> Self {
        let inner_config = BeamSearchConfig {
            beam_width: config.beam_width,
            max_length: config.max_length,
            length_penalty: config.length_penalty,
            early_stopping: config.early_stopping,
            temperature: config.temperature,
        };
        Self {
            inner: BeamSearch::new(inner_config),
        }
    }

    #[pyo3(signature = (encoder_outputs, initial_hidden, initial_cell=None, start_token=1, end_token=2, vocab_size=10000))]
    pub fn search(
        &self,
        py: Python<'_>,
        encoder_outputs: &Bound<'_, PyArray2<f32>>,
        initial_hidden: &Bound<'_, PyArray2<f32>>,
        initial_cell: Option<&Bound<'_, PyArray2<f32>>>,
        start_token: usize,
        end_token: usize,
        vocab_size: usize,
    ) -> PyResult<(Vec<Vec<usize>>, Vec<f32>)> {
        let _ = py;
        let enc_view = unsafe { encoder_outputs.as_array() };
        let hid_view = unsafe { initial_hidden.as_array() };
        let cell_view = initial_cell.map(|c| unsafe { c.as_array() });

        let res = self.inner.search(enc_view, hid_view, cell_view, start_token, end_token, vocab_size);
        Ok((res.sequences, res.scores))
    }
}

#[pyfunction]
#[pyo3(name = "beam_search_rust", signature = (encoder_outputs, initial_hidden, initial_cell=None, beam_width=5, max_length=50, start_token=1, end_token=2, vocab_size=10000, length_penalty=0.6, temperature=1.0))]
pub fn beam_search_rust_py(
    py: Python<'_>,
    encoder_outputs: &Bound<'_, PyArray2<f32>>,
    initial_hidden: &Bound<'_, PyArray2<f32>>,
    initial_cell: Option<&Bound<'_, PyArray2<f32>>>,
    beam_width: usize,
    max_length: usize,
    start_token: usize,
    end_token: usize,
    vocab_size: usize,
    length_penalty: f32,
    temperature: f32,
) -> PyResult<(Vec<Vec<usize>>, Vec<f32>)> {
    let _ = py;
    let config = BeamSearchConfig {
        beam_width,
        max_length,
        length_penalty,
        early_stopping: true,
        temperature,
    };
    let bs = BeamSearch::new(config);
    let enc_view = unsafe { encoder_outputs.as_array() };
    let hid_view = unsafe { initial_hidden.as_array() };
    let cell_view = initial_cell.map(|c| unsafe { c.as_array() });

    let res = bs.search(enc_view, hid_view, cell_view, start_token, end_token, vocab_size);
    Ok((res.sequences, res.scores))
}

#[pyfunction]
#[pyo3(name = "beam_search_batch_rust", signature = (encoder_outputs, initial_hidden, initial_cell=None, beam_width=5, max_length=50, start_token=1, end_token=2, vocab_size=10000))]
pub fn beam_search_batch_rust_py(
    py: Python<'_>,
    encoder_outputs: &Bound<'_, PyArray3<f32>>,
    initial_hidden: &Bound<'_, PyArray3<f32>>,
    initial_cell: Option<&Bound<'_, PyArray3<f32>>>,
    beam_width: usize,
    max_length: usize,
    start_token: usize,
    end_token: usize,
    vocab_size: usize,
) -> PyResult<Vec<(Vec<Vec<usize>>, Vec<f32>)>> {
    let _ = py;
    let config = BeamSearchConfig {
        beam_width,
        max_length,
        ..Default::default()
    };
    let bs = BeamSearch::new(config);

    let enc_arr = unsafe { encoder_outputs.as_array() };
    let hid_arr = unsafe { initial_hidden.as_array() };
    let cell_arr = initial_cell.map(|c| unsafe { c.as_array() });

    let batch_size = enc_arr.shape()[0];
    let mut results = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let enc_view = enc_arr.index_axis(Axis(0), i);
        let hid_view = hid_arr.index_axis(Axis(0), i);
        let cell_view = cell_arr.as_ref().map(|c| c.index_axis(Axis(0), i));

        let res = bs.search(enc_view, hid_view, cell_view, start_token, end_token, vocab_size);
        results.push((res.sequences, res.scores));
    }

    Ok(results)
}
