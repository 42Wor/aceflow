use ndarray::{Array2, Array3};
use numpy::{PyArray2, PyArray3};
use ndarray::{ArrayView2, ArrayView3};
use pyo3::prelude::*;

/// Converts a Python 2D array to a Rust ndarray View
pub fn pyarray_to_array2<'a>(_py: Python<'a>, array: &'a PyArray2<f32>) -> ArrayView2<'a, f32> {
    // Safety: We assume the numpy array is C-contiguous or standard layout.
    // as_array() returns a View, which is cheap to copy.
    unsafe { array.as_array() }
}

/// Converts a Python 3D array to a Rust ndarray View
pub fn pyarray_to_array3<'a>(_py: Python<'a>, array: &'a PyArray3<f32>) -> ArrayView3<'a, f32> {
    unsafe { array.as_array() }
}
/// Utility functions for probability calculations
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exps: f32 = exps.iter().sum();
    exps.into_iter().map(|x| x / sum_exps).collect()
}

pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let log_sum_exps: f32 = exps.iter().sum::<f32>().ln();
    logits.iter().map(|&x| x - max_logit - log_sum_exps).collect()
}