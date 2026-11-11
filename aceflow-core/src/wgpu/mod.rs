pub mod buffers;
pub mod context;
pub mod ops;
pub mod pipeline;

pub use buffers::{UniformConfig, WgpuBuffers, WgpuState};
pub use context::WgpuContext;
pub use pipeline::{PipelineCache, WgpuPipeline};

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArrayDyn, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::cpu::{gemv, norm, sampler, swiglu, LayerWeights};

#[pyclass(module = "aceflow._native")]
pub struct WgpuAceEngine {
    context: WgpuContext,
    pipeline: WgpuPipeline,
    buffers: WgpuBuffers,
    vocab_size: usize,
    embedding_table: Vec<f32>,
    final_norm_weight: Option<Vec<f32>>,
    lm_head_weight: Option<Vec<f32>>,
    eps: f32,
    d_ffn: usize,
    layers: Vec<LayerWeights>,
    qkv_buf: Vec<f32>,
    ret_out_buf: Vec<f32>,
    proj_out_buf: Vec<f32>,
    norm_buf: Vec<f32>,
    gate_up_buf: Vec<f32>,
    ffn_hidden_buf: Vec<f32>,
    ffn_out_buf: Vec<f32>,
}

impl WgpuAceEngine {
    fn dispatch_recurrent_pass(&self, layer_idx: usize) {
        let byte_size = (self.buffers.num_heads * self.buffers.head_dim * 4) as u64;
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline.pipeline);
            pass.set_bind_group(0, &self.buffers.bind_groups[layer_idx], &[]);
            let column_vectors = self.buffers.head_dim / 4;
            let workgroups_y = (column_vectors + 3) / 4;
            pass.dispatch_workgroups(1, workgroups_y, self.buffers.num_heads);
        }
        encoder.copy_buffer_to_buffer(
            &self.buffers.out_buf,
            0,
            &self.buffers.staging_buf,
            0,
            byte_size,
        );
        self.context.queue.submit(Some(encoder.finish()));
    }

    fn readback_staging_buffer(
        device: &wgpu::Device,
        staging: &wgpu::Buffer,
        output: &mut [f32],
    ) -> PyResult<()> {
        let slice = staging.slice(..);
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let done_callback = done.clone();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            if result.is_ok() {
                done_callback.store(true, std::sync::atomic::Ordering::Release);
            }
        });
        device.poll(wgpu::Maintain::Wait);
        if !done.load(std::sync::atomic::Ordering::Acquire) {
            return Err(PyValueError::new_err("GPU buffer map failed"));
        }
        let data = slice.get_mapped_range();
        output.copy_from_slice(bytemuck::cast_slice(&data));
        drop(data);
        staging.unmap();
        Ok(())
    }

    fn retention_step_internal(
        &self,
        layer_idx: usize,
        qkv: &[f32],
        output: &mut [f32],
    ) -> PyResult<()> {
        self.context
            .queue
            .write_buffer(&self.buffers.qkv_buf, 0, bytemuck::cast_slice(qkv));
        self.dispatch_recurrent_pass(layer_idx);
        Self::readback_staging_buffer(self.context.device.as_ref(), &self.buffers.staging_buf, output)
    }

    pub fn forward_layer_internal(&mut self, layer_idx: usize, x: &mut [f32]) -> PyResult<()> {
        let d_model = (self.buffers.num_heads * self.buffers.head_dim) as usize;
        let d_ffn = self.layers[layer_idx].d_ffn;
        norm::rmsnorm_to(
            x,
            &self.layers[layer_idx].norm1_gamma,
            self.eps,
            &mut self.norm_buf,
        );
        gemv::gemv(
            3 * d_model,
            d_model,
            &self.layers[layer_idx].w_qkv,
            &self.norm_buf,
            &mut self.qkv_buf,
        );
        self.context.queue.write_buffer(
            &self.buffers.qkv_buf,
            0,
            bytemuck::cast_slice(&self.qkv_buf),
        );
        self.dispatch_recurrent_pass(layer_idx);
        Self::readback_staging_buffer(
            self.context.device.as_ref(),
            &self.buffers.staging_buf,
            &mut self.ret_out_buf,
        )?;
        norm::rmsnorm_inplace(
            &mut self.ret_out_buf,
            &self.layers[layer_idx].ret_norm_gamma,
            self.eps,
        );
        gemv::gemv(
            d_model,
            d_model,
            &self.layers[layer_idx].w_o,
            &self.ret_out_buf,
            &mut self.proj_out_buf,
        );
        for (value, residual) in x.iter_mut().zip(&self.proj_out_buf) {
            *value += residual;
        }

        norm::rmsnorm_to(
            x,
            &self.layers[layer_idx].norm2_gamma,
            self.eps,
            &mut self.norm_buf,
        );
        gemv::gemv(
            2 * d_ffn,
            d_model,
            &self.layers[layer_idx].w_gate_up,
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
            &self.layers[layer_idx].w_down,
            &self.ffn_hidden_buf[..d_ffn],
            &mut self.ffn_out_buf,
        );
        for (value, residual) in x.iter_mut().zip(&self.ffn_out_buf) {
            *value += residual;
        }
        Ok(())
    }

    fn generate_stream_internal(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
    ) -> PyResult<Vec<u32>> {
        let d_model = (self.buffers.num_heads * self.buffers.head_dim) as usize;
        let mut current = vec![0.0f32; d_model];
        let mut context = prompt_tokens.to_vec();
        let mut generated = Vec::with_capacity(max_new_tokens);
        for &token in prompt_tokens {
            let start = token as usize * d_model;
            let end = start + d_model;
            if end <= self.embedding_table.len() {
                current.copy_from_slice(&self.embedding_table[start..end]);
            }
            for layer in 0..self.buffers.num_layers as usize {
                self.forward_layer_internal(layer, &mut current)?;
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
            let logits = sampler::project_logits(&normed, head, self.vocab_size, d_model);
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
            let start = next as usize * d_model;
            let end = start + d_model;
            if end <= self.embedding_table.len() {
                current.copy_from_slice(&self.embedding_table[start..end]);
            } else {
                current.fill(0.0);
            }
            for layer in 0..self.buffers.num_layers as usize {
                self.forward_layer_internal(layer, &mut current)?;
            }
        }
        Ok(generated)
    }
}

#[pymethods]
impl WgpuAceEngine {
    #[new]
    pub fn new(num_layers: usize, num_heads: usize, head_dim: usize) -> Self {
        let context = WgpuContext::new();
        let pipeline = WgpuPipeline::new(context.device.as_ref());
        let buffers = WgpuBuffers::new(
            context.device.as_ref(),
            &pipeline.bind_group_layout,
            num_heads,
            head_dim,
            num_layers,
        );
        let d_model = num_heads * head_dim;
        let default_d_ffn = 4 * d_model;
        Self {
            context,
            pipeline,
            buffers,
            vocab_size: 0,
            embedding_table: Vec::new(),
            final_norm_weight: None,
            lm_head_weight: None,
            eps: 1e-5,
            d_ffn: default_d_ffn,
            layers: vec![LayerWeights::default(); num_layers],
            qkv_buf: vec![0.0; 3 * d_model],
            ret_out_buf: vec![0.0; d_model],
            proj_out_buf: vec![0.0; d_model],
            norm_buf: vec![0.0; d_model],
            gate_up_buf: vec![0.0; 2 * default_d_ffn],
            ffn_hidden_buf: vec![0.0; default_d_ffn],
            ffn_out_buf: vec![0.0; d_model],
        }
    }

    #[pyo3(signature = (embeddings, final_norm_gamma=None, lm_head=None))]
    pub fn load_weights(
        &mut self,
        embeddings: PyReadonlyArrayDyn<'_, f32>,
        final_norm_gamma: Option<PyReadonlyArrayDyn<'_, f32>>,
        lm_head: Option<PyReadonlyArrayDyn<'_, f32>>,
    ) -> PyResult<()> {
        let d_model = (self.buffers.num_heads * self.buffers.head_dim) as usize;
        let embeddings = embeddings.as_slice()?;
        if embeddings.len() % d_model != 0 {
            return Err(PyValueError::new_err(format!(
                "Embedding table length {} must be divisible by d_model {d_model}",
                embeddings.len()
            )));
        }
        self.vocab_size = embeddings.len() / d_model;
        self.embedding_table = embeddings.to_vec();
        if let Some(gamma) = final_norm_gamma {
            let gamma = gamma.as_slice()?;
            if gamma.len() != d_model {
                return Err(PyValueError::new_err(format!(
                    "final_norm_gamma length {} must match d_model {d_model}",
                    gamma.len()
                )));
            }
            self.final_norm_weight = Some(gamma.to_vec());
        }
        if let Some(head) = lm_head {
            let head = head.as_slice()?;
            if head.len() != self.vocab_size * d_model {
                return Err(PyValueError::new_err(format!(
                    "lm_head length {} must match vocab_size * d_model ({} * {d_model})",
                    head.len(),
                    self.vocab_size
                )));
            }
            self.lm_head_weight = Some(head.to_vec());
        } else {
            self.lm_head_weight = None;
        }
        Ok(())
    }

    pub fn set_embedding_table(&mut self, table: PyReadonlyArrayDyn<'_, f32>) -> PyResult<()> {
        let d_model = (self.buffers.num_heads * self.buffers.head_dim) as usize;
        let values = table.as_slice()?;
        if values.len() % d_model != 0 {
            return Err(PyValueError::new_err(format!(
                "Embedding table length {} must be divisible by d_model {d_model}",
                values.len()
            )));
        }
        self.vocab_size = values.len() / d_model;
        self.embedding_table = values.to_vec();
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
        let d_model = (self.buffers.num_heads * self.buffers.head_dim) as usize;
        if layer_idx >= self.buffers.num_layers as usize {
            return Err(PyValueError::new_err(format!(
                "layer_idx {layer_idx} exceeds num_layers {}",
                self.buffers.num_layers
            )));
        }
        let qkv = w_qkv.as_slice()?;
        let output = w_o.as_slice()?;
        let gate_up = w_gate_up.as_slice()?;
        let down = w_down.as_slice()?;
        let norm1 = norm1_gamma.as_slice()?;
        let norm2 = norm2_gamma.as_slice()?;
        if qkv.len() != 3 * d_model * d_model {
            return Err(PyValueError::new_err("w_qkv dimension mismatch"));
        }
        if output.len() != d_model * d_model {
            return Err(PyValueError::new_err("w_o dimension mismatch"));
        }
        if gate_up.len() % (2 * d_model) != 0 {
            return Err(PyValueError::new_err("w_gate_up dimension mismatch"));
        }
        let d_ffn = gate_up.len() / (2 * d_model);
        if down.len() != d_model * d_ffn {
            return Err(PyValueError::new_err("w_down dimension mismatch"));
        }
        if norm1.len() != d_model || norm2.len() != d_model {
            return Err(PyValueError::new_err("norm dimension mismatch"));
        }
        let retention_norm = if let Some(gamma) = ret_norm_gamma {
            let values = gamma.as_slice()?;
            if values.len() != d_model {
                return Err(PyValueError::new_err("ret_norm_gamma dimension mismatch"));
            }
            values.to_vec()
        } else {
            vec![1.0; d_model]
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

    pub fn forward_layer(
        &mut self,
        layer_idx: usize,
        mut x: PyReadwriteArray1<'_, f32>,
    ) -> PyResult<()> {
        if layer_idx >= self.buffers.num_layers as usize {
            return Err(PyValueError::new_err(format!(
                "layer_idx {layer_idx} exceeds num_layers {}",
                self.buffers.num_layers
            )));
        }
        let d_model = (self.buffers.num_heads * self.buffers.head_dim) as usize;
        let x = x.as_slice_mut()?;
        if x.len() != d_model {
            return Err(PyValueError::new_err(format!(
                "x length {} must match d_model {d_model}",
                x.len()
            )));
        }
        self.forward_layer_internal(layer_idx, x)
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
        py.allow_threads(|| {
            self.generate_stream_internal(
                &prompt_tokens,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
            )
        })
    }

    pub fn get_embedding<'py>(
        &self,
        py: Python<'py>,
        token_id: usize,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let d_model = (self.buffers.num_heads * self.buffers.head_dim) as usize;
        if self.embedding_table.is_empty() {
            return Err(PyValueError::new_err("Embedding table is not loaded."));
        }
        if token_id >= self.vocab_size {
            return Err(PyValueError::new_err(format!(
                "token_id {token_id} out of vocabulary bounds (vocab_size: {})",
                self.vocab_size
            )));
        }
        let start = token_id * d_model;
        Ok(PyArray1::from_slice_bound(
            py,
            &self.embedding_table[start..start + d_model],
        ))
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
        let d_model = (self.buffers.num_heads * self.buffers.head_dim) as usize;
        let hidden = hidden_state.as_slice()?;
        if hidden.len() != d_model {
            return Err(PyValueError::new_err(format!(
                "hidden_state length {} must match d_model {d_model}",
                hidden.len()
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
        let logits = sampler::project_logits(&normed, head, self.vocab_size, d_model);
        Ok(sampler::sample_token_from_logits(
            logits,
            &context_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
        ))
    }

    pub fn reset_memory(&self) {
        let state_size = (self.buffers.num_layers
            * self.buffers.num_heads
            * self.buffers.head_dim
            * self.buffers.head_dim
            * std::mem::size_of::<f32>() as u32) as u64;
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.clear_buffer(&self.buffers.state_buffer, 0, Some(state_size));
        self.context.queue.submit(Some(encoder.finish()));
    }

    pub fn retention_step_packed<'py>(
        &mut self,
        py: Python<'py>,
        layer_idx: usize,
        qkv: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let qkv = qkv.as_slice()?;
        let d_model = (self.buffers.num_heads * self.buffers.head_dim) as usize;
        let expected = d_model * 3;
        if qkv.len() != expected {
            return Err(PyValueError::new_err(format!(
                "Expected packed QKV vector of length {expected}, got {}",
                qkv.len()
            )));
        }
        if layer_idx >= self.buffers.num_layers as usize {
            return Err(PyValueError::new_err(format!(
                "layer_idx {layer_idx} out of range (total layers: {})",
                self.buffers.num_layers
            )));
        }
        let output = PyArray1::<f32>::zeros_bound(py, [d_model], false);
        self.retention_step_internal(layer_idx, qkv, unsafe { output.as_slice_mut()? })?;
        Ok(output)
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
        let d_model = (self.buffers.num_heads * self.buffers.head_dim) as usize;
        if q.len() != d_model || k.len() != d_model || v.len() != d_model {
            return Err(PyValueError::new_err(format!(
                "Expected inputs of length {d_model}, got q={}, k={}, v={}",
                q.len(),
                k.len(),
                v.len()
            )));
        }
        if layer_idx >= self.buffers.num_layers as usize {
            return Err(PyValueError::new_err(format!(
                "layer_idx {layer_idx} out of range"
            )));
        }
        let mut qkv = Vec::with_capacity(3 * d_model);
        qkv.extend_from_slice(q);
        qkv.extend_from_slice(k);
        qkv.extend_from_slice(v);
        let output = PyArray1::<f32>::zeros_bound(py, [d_model], false);
        self.retention_step_internal(layer_idx, &qkv, unsafe { output.as_slice_mut()? })?;
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
        if layer_idx >= self.buffers.num_layers as usize {
            return Err(PyValueError::new_err(format!(
                "layer_idx {layer_idx} out of range (total layers: {})",
                self.buffers.num_layers
            )));
        }
        let d_model = (self.buffers.num_heads * self.buffers.head_dim) as usize;
        let expected = seq_len * d_model;
        let q = q_seq.as_slice()?;
        let k = k_seq.as_slice()?;
        let v = v_seq.as_slice()?;
        if q.len() != expected || k.len() != expected || v.len() != expected {
            return Err(PyValueError::new_err(format!(
                "Expected sequence vectors of length {expected} (seq_len {seq_len} * d_model {d_model}), got q={}, k={}, v={}",
                q.len(),
                k.len(),
                v.len()
            )));
        }
        let output = PyArray1::<f32>::zeros_bound(py, [d_model], false);
        let output_slice = unsafe { output.as_slice_mut()? };
        let mut packed = vec![0.0f32; 3 * d_model];
        for step in 0..seq_len {
            let start = step * d_model;
            packed[..d_model].copy_from_slice(&q[start..start + d_model]);
            packed[d_model..2 * d_model].copy_from_slice(&k[start..start + d_model]);
            packed[2 * d_model..].copy_from_slice(&v[start..start + d_model]);
            self.retention_step_internal(layer_idx, &packed, output_slice)?;
        }
        Ok(output)
    }

    pub fn state_memory_kb(&self) -> usize {
        self.buffers.num_layers as usize
            * self.buffers.num_heads as usize
            * self.buffers.head_dim as usize
            * self.buffers.head_dim as usize
            * std::mem::size_of::<f32>()
            / 1024
    }

    #[getter]
    pub fn d_model(&self) -> usize {
        (self.buffers.num_heads * self.buffers.head_dim) as usize
    }

    #[getter]
    pub fn num_heads(&self) -> usize {
        self.buffers.num_heads as usize
    }

    #[getter]
    pub fn head_dim(&self) -> usize {
        self.buffers.head_dim as usize
    }

    #[getter]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    #[getter]
    pub fn adapter_name(&self) -> &str {
        &self.context.adapter_name
    }
}
