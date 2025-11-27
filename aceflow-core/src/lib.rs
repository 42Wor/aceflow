use pyo3::prelude::*;
use numpy::{PyArray2, PyArray3, IntoPyArray};
use ndarray::{Array2, Array3};
use hashbrown::HashMap;

mod beam_search;
mod utils;

use beam_search::{BeamSearch, BeamSearchConfig, BeamSearchResult};

#[pyclass]
#[derive(Clone)]
struct PyBeamSearchConfig {
    #[pyo3(get, set)]
    beam_width: usize,
    #[pyo3(get, set)]
    max_length: usize,
    #[pyo3(get, set)]
    length_penalty: f32,
    #[pyo3(get, set)]
    early_stopping: bool,
    #[pyo3(get, set)]
    temperature: f32,
}

#[pymethods]
impl PyBeamSearchConfig {
    #[new]
    fn new(
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

#[pyclass]
struct PyBeamSearch {
    inner: BeamSearch,
}

#[pymethods]
impl PyBeamSearch {
    #[new]
    fn new(config: PyBeamSearchConfig) -> Self {
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
    fn search(
        &self,
        encoder_outputs: &PyArray2<f32>,
        initial_hidden: &PyArray2<f32>,
        initial_cell: Option<&PyArray2<f32>>,
        start_token: usize,
        end_token: usize,
        vocab_size: usize,
    ) -> PyResult<(Vec<Vec<usize>>, Vec<f32>)> {
        let encoder_array = unsafe { encoder_outputs.as_array() };
        let hidden_array = unsafe { initial_hidden.as_array() };
        
        let cell_array = initial_cell.map(|cell| unsafe { cell.as_array() });
        
        let result = self.inner.search(
            &encoder_array,
            &hidden_array,
            cell_array.as_ref(),
            start_token,
            end_token,
            vocab_size,
        );
        
        Ok((result.sequences, result.scores))
    }
}

/// High-performance beam search for sequence generation
#[pyfunction]
#[pyo3(signature = (encoder_outputs, initial_hidden, initial_cell=None, beam_width=5, max_length=50, start_token=1, end_token=2, vocab_size=10000, length_penalty=0.6, temperature=1.0))]
fn beam_search_rust(
    encoder_outputs: &PyArray2<f32>,
    initial_hidden: &PyArray2<f32>,
    initial_cell: Option<&PyArray2<f32>>,
    beam_width: usize,
    max_length: usize,
    start_token: usize,
    end_token: usize,
    vocab_size: usize,
    length_penalty: f32,
    temperature: f32,
) -> PyResult<(Vec<Vec<usize>>, Vec<f32>)> {
    let config = BeamSearchConfig {
        beam_width,
        max_length,
        length_penalty,
        early_stopping: true,
        temperature,
    };
    
    let beam_search = BeamSearch::new(config);
    
    let encoder_array = unsafe { encoder_outputs.as_array() };
    let hidden_array = unsafe { initial_hidden.as_array() };
    let cell_array = initial_cell.map(|cell| unsafe { cell.as_array() });
    
    let result = beam_search.search(
        &encoder_array,
        &hidden_array,
        cell_array.as_ref(),
        start_token,
        end_token,
        vocab_size,
    );
    
    Ok((result.sequences, result.scores))
}

/// Batch beam search for multiple sequences
#[pyfunction]
fn beam_search_batch_rust(
    encoder_outputs: &PyArray3<f32>,
    initial_hidden: &PyArray3<f32>,
    initial_cell: Option<&PyArray3<f32>>,
    beam_width: usize,
    max_length: usize,
    start_token: usize,
    end_token: usize,
    vocab_size: usize,
) -> PyResult<Vec<(Vec<Vec<usize>>, Vec<f32>)>> {
    let config = BeamSearchConfig {
        beam_width,
        max_length,
        ..Default::default()
    };
    
    let beam_search = BeamSearch::new(config);
    
    let encoder_array = unsafe { encoder_outputs.as_array() };
    let hidden_array = unsafe { initial_hidden.as_array() };
    let cell_array = initial_cell.map(|cell| unsafe { cell.as_array() });
    
    let batch_size = encoder_array.shape()[0];
    let mut results = Vec::with_capacity(batch_size);
    
    for i in 0..batch_size {
        let encoder_view = encoder_array.index_axis(ndarray::Axis(0), i);
        let hidden_view = hidden_array.index_axis(ndarray::Axis(0), i);
        let cell_view = cell_array.as_ref().map(|cell| cell.index_axis(ndarray::Axis(0), i));
        
        let result = beam_search.search(
            &encoder_view,
            &hidden_view,
            cell_view,
            start_token,
            end_token,
            vocab_size,
        );
        
        results.push((result.sequences, result.scores));
    }
    
    Ok(results)
}

#[pymodule]
fn aceflow_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(beam_search_rust, m)?)?;
    m.add_function(wrap_pyfunction!(beam_search_batch_rust, m)?)?;
    m.add_class::<PyBeamSearch>()?;
    m.add_class::<PyBeamSearchConfig>()?;
    
    Ok(())
}