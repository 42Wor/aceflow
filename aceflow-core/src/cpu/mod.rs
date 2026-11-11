pub mod beam_search;
pub mod gemv;
pub mod norm;
pub mod recurrence;
pub mod rnn;
pub mod sampler;
pub mod seq_attn;
pub mod swiglu;

pub use beam_search::{BeamSearch, BeamSearchConfig, BeamSearchResult};

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArrayDyn, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use recurrence::retention_step_core;

#[derive(Clone, Default)]
pub struct LayerWeights {
    pub w_qkv: Vec<f32>,
    pub w_o: Vec<f32>,
    pub w_gate_up: Vec<f32>,
    pub w_down: Vec<f32>,
    pub norm1_gamma: Vec<f32>,
    pub norm2_gamma: Vec<f32>,
    pub ret_norm_gamma: Vec<f32>,
    pub d_ffn: usize,
}

#[pyclass(module = "aceflow._native")]
pub struct AceEngine {
    num_layers: usize,
    d_model: usize,
    num_heads: usize,
    head_dim: usize,
    d_ffn: usize,
    states: Vec<f32>,
    gamma: Vec<f32>,
    vocab_size: usize,
    embedding_table: Vec<f32>,
    final_norm_weight: Option<Vec<f32>>,
    lm_head_weight: Option<Vec<f32>>,
    eps: f32,
    layers: Vec<LayerWeights>,
    qkv_buf: Vec<f32>,
    ret_out_buf: Vec<f32>,
    proj_out_buf: Vec<f32>,
    norm_buf: Vec<f32>,
    gate_up_buf: Vec<f32>,
    ffn_hidden_buf: Vec<f32>,
    ffn_out_buf: Vec<f32>,
}

#[pymethods]
impl AceEngine {
    #[new]
    #[pyo3(signature = (num_layers, d_model, num_heads, embedding_table=None))]
    pub fn new(
        num_layers: usize,
        d_model: usize,
        num_heads: usize,
        embedding_table: Option<PyReadonlyArrayDyn<'_, f32>>,
    ) -> PyResult<Self> {
        if num_heads == 0 || d_model % num_heads != 0 {
            return Err(PyValueError::new_err(format!(
                "d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )));
        }
        let head_dim = d_model / num_heads;
        let total_state_elements = num_layers * num_heads * head_dim * head_dim;
        let states = vec![0.0f32; total_state_elements];
        let gamma = (0..num_heads)
            .map(|head| 1.0 - 2.0f32.powf(-5.0 - head as f32))
            .collect();

        let (vocab_size, embedding_table) = if let Some(table) = embedding_table {
            let values = table.as_slice()?;
            if values.len() % d_model != 0 {
                return Err(PyValueError::new_err(format!(
                    "Embedding table length {} must be divisible by d_model {}",
                    values.len(),
                    d_model
                )));
            }
            (values.len() / d_model, values.to_vec())
        } else {
            (0, Vec::new())
        };

        let default_d_ffn = 4 * d_model;
        Ok(Self {
            num_layers,
            d_model,
            num_heads,
            head_dim,
            d_ffn: default_d_ffn,
            states,
            gamma,
            vocab_size,
            embedding_table,
            final_norm_weight: None,
            lm_head_weight: None,
            eps: 1e-5,
            layers: vec![LayerWeights::default(); num_layers],
            qkv_buf: vec![0.0; 3 * d_model],
            ret_out_buf: vec![0.0; d_model],
            proj_out_buf: vec![0.0; d_model],
            norm_buf: vec![0.0; d_model],
            gate_up_buf: vec![0.0; 2 * default_d_ffn],
            ffn_hidden_buf: vec![0.0; default_d_ffn],
            ffn_out_buf: vec![0.0; d_model],
        })
    }

    pub fn reset_memory(&mut self) {
        self.states.fill(0.0);
    }

    #[pyo3(signature = (embeddings, final_norm_gamma=None, lm_head=None))]
    pub fn load_weights(
        &mut self,
        embeddings: PyReadonlyArrayDyn<'_, f32>,
        final_norm_gamma: Option<PyReadonlyArrayDyn<'_, f32>>,
        lm_head: Option<PyReadonlyArrayDyn<'_, f32>>,
    ) -> PyResult<()> {
        let embeddings = embeddings.as_slice()?;
        if embeddings.len() % self.d_model != 0 {
            return Err(PyValueError::new_err(format!(
                "Embedding table length {} must be divisible by d_model {}",
                embeddings.len(),
                self.d_model
            )));
        }
        self.vocab_size = embeddings.len() / self.d_model;
        self.embedding_table = embeddings.to_vec();

        if let Some(gamma) = final_norm_gamma {
            let gamma = gamma.as_slice()?;
            if gamma.len() != self.d_model {
                return Err(PyValueError::new_err(format!(
                    "final_norm_gamma length {} must match d_model {}",
                    gamma.len(),
                    self.d_model
                )));
            }
            self.final_norm_weight = Some(gamma.to_vec());
        }

        if let Some(head) = lm_head {
            let head = head.as_slice()?;
            if head.len() != self.vocab_size * self.d_model {
                return Err(PyValueError::new_err(format!(
                    "lm_head length {} must match vocab_size * d_model ({} * {})",
                    head.len(),
                    self.vocab_size,
                    self.d_model
                )));
            }
            self.lm_head_weight = Some(head.to_vec());
        } else {
            self.lm_head_weight = None;
        }
        Ok(())
    }

    #[pyo3(signature = (
        layer_idx,
        w_qkv,
        w_o,
        w_gate_up,
        w_down,
        norm1_gamma,
        norm2_gamma,
        ret_norm_gamma=None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn load_layer_weights(
        &mut self,
        layer_idx: usize,
        w_qkv: PyReadonlyArrayDyn<'_, f32>,
        w_o: PyReadonlyArrayDyn<'_, f32>,
        w_gate_up: PyReadonlyArrayDyn<'_, f32>,
        w_down: PyReadonlyArrayDyn<'_, f32>,
        norm1_gamma: PyReadonlyArrayDyn<'_, f32>,
        norm2_gamma: PyReadonlyArrayDyn<'_, f32>,
        ret_norm_gamma: Option<PyReadonlyArrayDyn<'_, f32>>,
    ) -> PyResult<()> {
        if layer_idx >= self.num_layers {
            return Err(PyValueError::new_err(format!(
                "layer_idx {layer_idx} exceeds num_layers {}",
                self.num_layers
            )));
        }
        let qkv = w_qkv.as_slice()?;
        let output = w_o.as_slice()?;
        let gate_up = w_gate_up.as_slice()?;
        let down = w_down.as_slice()?;
        let norm1 = norm1_gamma.as_slice()?;
        let norm2 = norm2_gamma.as_slice()?;

        if qkv.len() != 3 * self.d_model * self.d_model {
            return Err(PyValueError::new_err(format!(
                "w_qkv size {} must match 3 * d_model^2 ({})",
                qkv.len(),
                3 * self.d_model * self.d_model
            )));
        }
        if output.len() != self.d_model * self.d_model {
            return Err(PyValueError::new_err(format!(
                "w_o size {} must match d_model^2 ({})",
                output.len(),
                self.d_model * self.d_model
            )));
        }
        if gate_up.len() % (2 * self.d_model) != 0 {
            return Err(PyValueError::new_err(format!(
                "w_gate_up size {} must be divisible by 2 * d_model ({})",
                gate_up.len(),
                2 * self.d_model
            )));
        }
        let d_ffn = gate_up.len() / (2 * self.d_model);
        if down.len() != self.d_model * d_ffn {
            return Err(PyValueError::new_err(format!(
                "w_down size {} must match d_model * d_ffn ({} * {})",
                down.len(),
                self.d_model,
                d_ffn
            )));
        }
        if norm1.len() != self.d_model || norm2.len() != self.d_model {
            return Err(PyValueError::new_err(format!(
                "norm1 ({}) and norm2 ({}) must match d_model ({})",
                norm1.len(),
                norm2.len(),
                self.d_model
            )));
        }
        let retention_norm = if let Some(gamma) = ret_norm_gamma {
            let gamma = gamma.as_slice()?;
            if gamma.len() != self.d_model {
                return Err(PyValueError::new_err(format!(
                    "ret_norm_gamma length {} must match d_model {}",
                    gamma.len(),
                    self.d_model
                )));
            }
            gamma.to_vec()
        } else {
            vec![1.0; self.d_model]
        };

        if self.gate_up_buf.len() < 2 * d_ffn {
            self.d_ffn = d_ffn;
            self.gate_up_buf = vec![0.0; 2 * d_ffn];
            self.ffn_hidden_buf = vec![0.0; d_ffn];
        }
        self.layers[layer_idx] = LayerWeights {
            w_qkv: qkv.to_vec(),
            w_o: output.to_vec(),
            w_gate_up: gate_up.to_vec(),
            w_down: down.to_vec(),
            norm1_gamma: norm1.to_vec(),
            norm2_gamma: norm2.to_vec(),
            ret_norm_gamma: retention_norm,
            d_ffn,
        };
        Ok(())
    }

    pub fn set_embedding_table(&mut self, table: PyReadonlyArrayDyn<'_, f32>) -> PyResult<()> {
        let values = table.as_slice()?;
        if values.len() % self.d_model != 0 {
            return Err(PyValueError::new_err(format!(
                "Embedding table length {} must be divisible by d_model {}",
                values.len(),
                self.d_model
            )));
        }
        self.vocab_size = values.len() / self.d_model;
        self.embedding_table = values.to_vec();
        Ok(())
    }

    pub fn get_embedding<'py>(
        &self,
        py: Python<'py>,
        token_id: usize,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        if self.embedding_table.is_empty() {
            return Err(PyValueError::new_err("Embedding table is not loaded."));
        }
        if token_id >= self.vocab_size {
            return Err(PyValueError::new_err(format!(
                "token_id {token_id} out of vocabulary bounds (vocab_size: {})",
                self.vocab_size
            )));
        }
        let start = token_id * self.d_model;
        Ok(PyArray1::from_slice_bound(
            py,
            &self.embedding_table[start..start + self.d_model],
        ))
    }

    #[pyo3(signature = (x, weights, eps=None))]
    pub fn rmsnorm<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArrayDyn<'py, f32>,
        weights: PyReadonlyArrayDyn<'py, f32>,
        eps: Option<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let x = x.as_slice()?;
        let weights = weights.as_slice()?;
        if x.len() != weights.len() {
            return Err(PyValueError::new_err(format!(
                "x length {} must match weights length {}",
                x.len(),
                weights.len()
            )));
        }
        Ok(PyArray1::from_vec_bound(
            py,
            norm::rmsnorm(x, weights, eps.unwrap_or(self.eps)),
        ))
    }

    #[pyo3(signature = (x, weights, eps=None))]
    pub fn rmsnorm_inplace(
        &self,
        mut x: PyReadwriteArray1<'_, f32>,
        weights: PyReadonlyArrayDyn<'_, f32>,
        eps: Option<f32>,
    ) -> PyResult<()> {
        let x = x.as_slice_mut()?;
        let weights = weights.as_slice()?;
        if x.len() != weights.len() {
            return Err(PyValueError::new_err(format!(
                "x length {} must match weights length {}",
                x.len(),
                weights.len()
            )));
        }
        norm::rmsnorm_inplace(x, weights, eps.unwrap_or(self.eps));
        Ok(())
    }

    #[pyo3(signature = (
        hidden_state,
        context_tokens,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.2
    ))]
    pub fn sample_next_token(
        &self,
        hidden_state: PyReadonlyArrayDyn<'_, f32>,
        context_tokens: Vec<u32>,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
    ) -> PyResult<u32> {
        let hidden = hidden_state.as_slice()?;
        if hidden.len() != self.d_model {
            return Err(PyValueError::new_err(format!(
                "hidden_state length {} must match d_model {}",
                hidden.len(),
                self.d_model
            )));
        }
        if self.embedding_table.is_empty() {
            return Err(PyValueError::new_err("Model weights not loaded"));
        }
        let normed = self
            .final_norm_weight
            .as_ref()
            .map_or_else(|| hidden.to_vec(), |w| norm::rmsnorm(hidden, w, self.eps));
        let head = self
            .lm_head_weight
            .as_ref()
            .unwrap_or(&self.embedding_table);
        let logits = sampler::project_logits(&normed, head, self.vocab_size, self.d_model);
        Ok(sampler::sample_token_from_logits(
            logits,
            &context_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
        ))
    }

    #[pyo3(signature = (
        logits,
        context_tokens,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.2
    ))]
    pub fn sample_token_from_logits(
        &self,
        logits: PyReadonlyArrayDyn<'_, f32>,
        context_tokens: Vec<u32>,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
    ) -> PyResult<u32> {
        Ok(sampler::sample_token_from_logits(
            logits.as_slice()?.to_vec(),
            &context_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
        ))
    }

    pub fn forward_layer(
        &mut self,
        layer_idx: usize,
        mut x: PyReadwriteArray1<'_, f32>,
    ) -> PyResult<()> {
        if layer_idx >= self.num_layers {
            return Err(PyValueError::new_err(format!(
                "layer_idx {layer_idx} exceeds num_layers {}",
                self.num_layers
            )));
        }
        let x = x.as_slice_mut()?;
        if x.len() != self.d_model {
            return Err(PyValueError::new_err(format!(
                "x length {} must match d_model {}",
                x.len(),
                self.d_model
            )));
        }
        self.forward_layer_internal(layer_idx, x);
        Ok(())
    }

    #[pyo3(signature = (
        prompt_tokens,
        max_new_tokens,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.2
    ))]
    pub fn generate_stream(
        &mut self,
        py: Python<'_>,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
    ) -> PyResult<Vec<u32>> {
        if self.embedding_table.is_empty() {
            return Err(PyValueError::new_err("Embedding table not loaded."));
        }
        for (index, layer) in self.layers.iter().enumerate() {
            if layer.w_qkv.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "Layer {index} weights are not loaded."
                )));
            }
        }
        Ok(py.allow_threads(|| {
            self.generate_stream_internal(
                &prompt_tokens,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
            )
        }))
    }

    pub fn retention_step<'py>(
        &mut self,
        py: Python<'py>,
        layer_idx: usize,
        q: PyReadonlyArray1<'py, f32>,
        k: PyReadonlyArray1<'py, f32>,
        v: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let q = q.as_slice()?;
        let k = k.as_slice()?;
        let v = v.as_slice()?;
        self.validate_recurrence_inputs(layer_idx, q, k, v)?;
        let output = PyArray1::<f32>::zeros_bound(py, [self.d_model], false);
        let output_slice = unsafe { output.as_slice_mut()? };
        retention_step_core(
            &mut self.states,
            &self.gamma,
            layer_idx,
            self.num_heads,
            self.head_dim,
            q,
            k,
            v,
            output_slice,
        );
        Ok(output)
    }

    pub fn retention_step_packed<'py>(
        &mut self,
        py: Python<'py>,
        layer_idx: usize,
        qkv: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let qkv = qkv.as_slice()?;
        let expected = self.d_model * 3;
        if qkv.len() != expected {
            return Err(PyValueError::new_err(format!(
                "Expected packed QKV vector of length {expected}, got {}",
                qkv.len()
            )));
        }
        self.validate_layer_index(layer_idx)?;
        let (q, rest) = qkv.split_at(self.d_model);
        let (k, v) = rest.split_at(self.d_model);
        let output = PyArray1::<f32>::zeros_bound(py, [self.d_model], false);
        let output_slice = unsafe { output.as_slice_mut()? };
        retention_step_core(
            &mut self.states,
            &self.gamma,
            layer_idx,
            self.num_heads,
            self.head_dim,
            q,
            k,
            v,
            output_slice,
        );
        Ok(output)
    }

    pub fn prefill_recurrent<'py>(
        &mut self,
        py: Python<'py>,
        layer_idx: usize,
        seq_len: usize,
        q_seq: PyReadonlyArray1<'py, f32>,
        k_seq: PyReadonlyArray1<'py, f32>,
        v_seq: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        self.validate_layer_index(layer_idx)?;
        let q = q_seq.as_slice()?;
        let k = k_seq.as_slice()?;
        let v = v_seq.as_slice()?;
        let expected = seq_len * self.d_model;
        if q.len() != expected || k.len() != expected || v.len() != expected {
            return Err(PyValueError::new_err(format!(
                "Expected sequence vectors of length {expected} (seq_len {seq_len} * d_model {}), got q={}, k={}, v={}",
                self.d_model,
                q.len(),
                k.len(),
                v.len()
            )));
        }
        let output = PyArray1::<f32>::zeros_bound(py, [self.d_model], false);
        let output_slice = unsafe { output.as_slice_mut()? };
        for step in 0..seq_len {
            let start = step * self.d_model;
            retention_step_core(
                &mut self.states,
                &self.gamma,
                layer_idx,
                self.num_heads,
                self.head_dim,
                &q[start..start + self.d_model],
                &k[start..start + self.d_model],
                &v[start..start + self.d_model],
                output_slice,
            );
        }
        Ok(output)
    }

    pub fn state_memory_kb(&self) -> usize {
        self.states.len() * std::mem::size_of::<f32>() / 1024
    }

    #[getter]
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    #[getter]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    #[getter]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    #[getter]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

impl AceEngine {
    fn validate_layer_index(&self, layer_idx: usize) -> PyResult<()> {
        if layer_idx >= self.num_layers {
            Err(PyValueError::new_err(format!(
                "layer_idx {layer_idx} out of range (total layers: {})",
                self.num_layers
            )))
        } else {
            Ok(())
        }
    }

    fn validate_recurrence_inputs(
        &self,
        layer_idx: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
    ) -> PyResult<()> {
        self.validate_layer_index(layer_idx)?;
        if q.len() != self.d_model || k.len() != self.d_model || v.len() != self.d_model {
            return Err(PyValueError::new_err(format!(
                "Expected input vectors of length {}, got q={}, k={}, v={}",
                self.d_model,
                q.len(),
                k.len(),
                v.len()
            )));
        }
        Ok(())
    }

    pub fn forward_layer_internal(&mut self, layer_idx: usize, x: &mut [f32]) {
        let d_model = self.d_model;
        let layer = &self.layers[layer_idx];
        let d_ffn = layer.d_ffn;
        norm::rmsnorm_to(x, &layer.norm1_gamma, self.eps, &mut self.norm_buf);
        gemv::gemv(
            3 * d_model,
            d_model,
            &layer.w_qkv,
            &self.norm_buf,
            &mut self.qkv_buf,
        );
        let (q, rest) = self.qkv_buf.split_at(d_model);
        let (k, v) = rest.split_at(d_model);
        retention_step_core(
            &mut self.states,
            &self.gamma,
            layer_idx,
            self.num_heads,
            self.head_dim,
            q,
            k,
            v,
            &mut self.ret_out_buf,
        );
        norm::rmsnorm_inplace(&mut self.ret_out_buf, &layer.ret_norm_gamma, self.eps);
        gemv::gemv(
            d_model,
            d_model,
            &layer.w_o,
            &self.ret_out_buf,
            &mut self.proj_out_buf,
        );
        for (value, residual) in x.iter_mut().zip(&self.proj_out_buf) {
            *value += residual;
        }

        norm::rmsnorm_to(x, &layer.norm2_gamma, self.eps, &mut self.norm_buf);
        gemv::gemv(
            2 * d_ffn,
            d_model,
            &layer.w_gate_up,
            &self.norm_buf,
            &mut self.gate_up_buf[..2 * d_ffn],
        );
        swiglu::swiglu_forward(
            d_ffn,
            &self.gate_up_buf[..2 * d_ffn],
            &mut self.ffn_hidden_buf[..d_ffn],
        );
        gemv::gemv(
            d_model,
            d_ffn,
            &layer.w_down,
            &self.ffn_hidden_buf[..d_ffn],
            &mut self.ffn_out_buf,
        );
        for (value, residual) in x.iter_mut().zip(&self.ffn_out_buf) {
            *value += residual;
        }
    }

    fn generate_stream_internal(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
    ) -> Vec<u32> {
        let mut current = vec![0.0f32; self.d_model];
        let mut context = prompt_tokens.to_vec();
        let mut generated = Vec::with_capacity(max_new_tokens);

        for &token in prompt_tokens {
            let start = token as usize * self.d_model;
            let end = start + self.d_model;
            if end <= self.embedding_table.len() {
                current.copy_from_slice(&self.embedding_table[start..end]);
            }
            for layer in 0..self.num_layers {
                self.forward_layer_internal(layer, &mut current);
            }
        }

        for _ in 0..max_new_tokens {
            let normed = self.final_norm_weight.as_ref().map_or_else(
                || current.clone(),
                |weight| norm::rmsnorm(&current, weight, self.eps),
            );
            let head = self
                .lm_head_weight
                .as_ref()
                .unwrap_or(&self.embedding_table);
            let logits = sampler::project_logits(&normed, head, self.vocab_size, self.d_model);
            let penalty_start = context.len().saturating_sub(60);
            let next = sampler::sample_token_from_logits(
                logits,
                &context[penalty_start..],
                temperature,
                top_k,
                top_p,
                repetition_penalty,
            );
            generated.push(next);
            context.push(next);
            if next == 2 {
                break;
            }

            let start = next as usize * self.d_model;
            let end = start + self.d_model;
            if end <= self.embedding_table.len() {
                current.copy_from_slice(&self.embedding_table[start..end]);
            } else {
                current.fill(0.0);
            }
            for layer in 0..self.num_layers {
                self.forward_layer_internal(layer, &mut current);
            }
        }
        generated
    }
}
