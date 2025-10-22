use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use ndarray::{Array, Array2, Axis, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

// A simplified helper for the tanh activation function and its derivative
fn tanh(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(f32::tanh)
}

fn tanh_derivative(x: &Array2<f32>) -> Array2<f32> {
    1.0 - x.mapv(|v| v * v)
}

#[pyclass]
struct DenseLayer {
    // We are only using this struct to demonstrate the concept.
    // The actual parameters will be passed from Python for the training step.
}

#[pymethods]
impl DenseLayer {
    #[new]
    fn new() -> Self {
        DenseLayer {}
    }
}

/// This is the powerhouse function. It performs one full training step (forward pass,
/// loss calculation, and backward pass) for a simple GRU-based Seq2Seq model.
/// It takes all model parameters from Python, computes gradients, and returns them.
#[pyfunction]
#[pyo3(name = "run_training_step")]
fn run_training_step<'py>(
    py: Python<'py>,
    // --- Model Parameters ---
    w_embed: PyReadonlyArray2<'py, f32>, // Embedding weights
    w_uz_enc: PyReadonlyArray2<'py, f32>, // Encoder Update Gate Weights
    w_ur_enc: PyReadonlyArray2<'py, f32>, // Encoder Reset Gate Weights
    w_uh_enc: PyReadonlyArray2<'py, f32>, // Encoder Candidate Gate Weights
    w_uz_dec: PyReadonlyArray2<'py, f32>, // Decoder Update Gate Weights
    w_ur_dec: PyReadonlyArray2<'py, f32>, // Decoder Reset Gate Weights
    w_uh_dec: PyReadonlyArray2<'py, f32>, // Decoder Candidate Gate Weights
    w_out: PyReadonlyArray2<'py, f32>,    // Output Dense Layer Weights
    b_out: PyReadonlyArray2<'py, f32>,    // Output Dense Layer Bias
    // --- Data ---
    x_batch: PyReadonlyArray2<'py, i32>,  // Input batch (batch_size, seq_len)
    y_batch: PyReadonlyArray2<'py, f32>,  // Target batch (one-hot) (batch_size, seq_len, vocab_size)
    // --- Hyperparameters ---
    learning_rate: f32,
) -> PyResult<(f32, PyObject)> {
    // Convert all NumPy arrays from Python into ndarray Arrays
    let w_embed = w_embed.as_array();
    let w_uz_enc = w_uz_enc.as_array();
    let w_ur_enc = w_ur_enc.as_array();
    let w_uh_enc = w_uh_enc.as_array();
    let w_uz_dec = w_uz_dec.as_array();
    let w_ur_dec = w_ur_dec.as_array();
    let w_uh_dec = w_uh_dec.as_array();
    let w_out = w_out.as_array();
    let b_out = b_out.as_array();
    let x_batch = x_batch.as_array();
    let y_batch = y_batch.as_array();

    let (batch_size, seq_len) = x_batch.dim();
    let (hidden_size, vocab_size) = w_out.dim();

    // =================================================================
    // FORWARD PASS
    // =================================================================

    // --- Encoder ---
    let mut encoder_hidden_states = Vec::new();
    let mut h_enc = Array::zeros((batch_size, hidden_size));
    encoder_hidden_states.push(h_enc.clone());

    for t in 0..seq_len {
        let x_t = x_batch.slice(s![.., t]);
        let mut embed_t = Array::zeros((batch_size, w_embed.dim().1));
        for (i, &word_idx) in x_t.iter().enumerate() {
            embed_t.row_mut(i).assign(&w_embed.row(word_idx as usize));
        }
        
        let combined_enc = ndarray::concatenate(Axis(1), &[&embed_t.view(), &h_enc.view()]).unwrap();
        let z_enc = combined_enc.dot(&w_uz_enc).mapv(|v| 1.0 / (1.0 + (-v).exp())); // Sigmoid
        let r_enc = combined_enc.dot(&w_ur_enc).mapv(|v| 1.0 / (1.0 + (-v).exp())); // Sigmoid
        let h_tilde_enc_combined = ndarray::concatenate(Axis(1), &[&embed_t.view(), &(&r_enc * &h_enc).view()]).unwrap();
        let h_tilde_enc = tanh(&h_tilde_enc_combined.dot(&w_uh_enc));
        h_enc = (&(1.0 - &z_enc) * &h_enc) + (&z_enc * &h_tilde_enc);
        encoder_hidden_states.push(h_enc.clone());
    }

    // --- Decoder ---
    let mut h_dec = h_enc; // Use final encoder state as initial decoder state
    let mut loss = 0.0;
    let mut decoder_outputs = Vec::new();

    // For simplicity, we use the reversed input as the decoder input (teacher forcing)
    for t in (0..seq_len).rev() {
        let y_t_input_idx = x_batch.slice(s![.., t]);
        let mut embed_t = Array::zeros((batch_size, w_embed.dim().1));
        for (i, &word_idx) in y_t_input_idx.iter().enumerate() {
             embed_t.row_mut(i).assign(&w_embed.row(word_idx as usize));
        }

        let combined_dec = ndarray::concatenate(Axis(1), &[&embed_t.view(), &h_dec.view()]).unwrap();
        let z_dec = combined_dec.dot(&w_uz_dec).mapv(|v| 1.0 / (1.0 + (-v).exp())); // Sigmoid
        let r_dec = combined_dec.dot(&w_ur_dec).mapv(|v| 1.0 / (1.0 + (-v).exp())); // Sigmoid
        let h_tilde_dec_combined = ndarray::concatenate(Axis(1), &[&embed_t.view(), &(&r_dec * &h_dec).view()]).unwrap();
        let h_tilde_dec = tanh(&h_tilde_dec_combined.dot(&w_uh_dec));
        h_dec = (&(1.0 - &z_dec) * &h_dec) + (&z_dec * &h_tilde_dec);
        
        let output_logits = h_dec.dot(&w_out) + &b_out;
        let probs = output_logits.map_axis(Axis(1), |row| {
            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_row = row.mapv(|v| (v - max_val).exp());
            &exp_row / exp_row.sum()
        });

        // Calculate loss (simplified cross-entropy)
        let y_true_t = y_batch.slice(s![.., t, ..]);
        loss -= (&y_true_t * &probs.mapv(|v| v.ln())).sum() / batch_size as f32;
        decoder_outputs.push(probs);
    }
    loss /= seq_len as f32;

    // =================================================================
    // BACKWARD PASS (Simplified for clarity)
    // =================================================================
    // In a real library, this would be much more extensive. Here we just calculate
    // one gradient to prove the concept and update weights on the Python side.
    
    // We will calculate gradient for the output layer only.
    let y_true_final = y_batch.slice(s![.., 0, ..]); // Corresponds to last decoder step
    let probs_final = &decoder_outputs[0];
    let d_logits = probs_final - &y_true_final.to_owned();

    // The state `h_dec` here is from *before* the last step
    let last_h_dec = if seq_len > 1 {
        // This is a rough approximation for the example
        h_dec.clone()
    } else {
         Array::zeros((batch_size, hidden_size))
    };

    let dw_out = last_h_dec.t().dot(&d_logits);
    let db_out = d_logits.sum_axis(Axis(0)).insert_axis(Axis(0));

    // Return gradients to Python
    let gradients = PyDict::new(py);
    gradients.set_item("dw_out", dw_out.into_pyarray(py))?;
    gradients.set_item("db_out", db_out.into_pyarray(py))?;

    // NOTE: This is a *highly simplified* BPTT. A full implementation is
    // extremely complex and beyond a single file example. We are only updating the
    // output layer weights to demonstrate the round-trip of data and gradients.

    Ok((loss, gradients.into()))
}

#[pymodule]
fn aceflow_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<DenseLayer>()?;
    m.add_function(wrap_pyfunction!(run_training_step, m)?)?;
    Ok(())
}