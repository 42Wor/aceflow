use ndarray::{Array2, ArrayD, IxDyn};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::dsp;
use crate::cpu::rnn::{rnn_forward as rnn_forward_cpu, RnnKind};
use crate::cpu::seq_attn::{seq2seq_attention as seq2seq_attention_cpu, AttentionKind};
use crate::device::{self, Device};
use crate::nn::{
    AdditiveAttention as CoreAdditiveAttention, Conv1D as CoreConv1D,
    EncoderDecoderBlock as CoreEncoderDecoderBlock, LayerNorm as CoreLayerNorm,
    Linear as CoreLinear, MultiHeadAttention as CoreMultiHeadAttention,
    RMSNorm as CoreRMSNorm, SwiGLU as CoreSwiGLU, cross_attention, scaled_dot_product_attention,
    sequence_downsample, sequence_upsample,
};
use crate::tensor::Tensor;

fn parse_py_device(value: Option<&Bound<'_, PyAny>>) -> PyResult<Device> {
    match value {
        None => Ok(device::default_device()),
        Some(obj) => {
            if let Ok(dev) = obj.extract::<PyRef<'_, PyDevice>>() {
                return Ok(dev.inner);
            }
            if let Ok(text) = obj.extract::<String>() {
                return device::parse_device_str(&text).map_err(PyValueError::new_err);
            }
            Err(PyValueError::new_err(
                "device must be a Device, 'cpu', 'wgpu', or 'wgpu:<index>'",
            ))
        }
    }
}

fn array_from_py(data: &Bound<'_, PyAny>) -> PyResult<(Vec<f32>, Vec<usize>)> {
    if let Ok(array) = data.extract::<PyReadonlyArrayDyn<'_, f32>>() {
        return Ok((array.as_slice()?.to_vec(), array.shape().to_vec()));
    }
    let numpy = data.py().import_bound("numpy")?;
    let float32 = numpy.getattr("float32")?;
    let array = numpy.call_method1("ascontiguousarray", (data, float32))?;
    let array: PyReadonlyArrayDyn<'_, f32> = array.extract()?;
    Ok((array.as_slice()?.to_vec(), array.shape().to_vec()))
}

#[pyclass(name = "Device", module = "aceflow._native")]
#[derive(Clone, Debug)]
pub struct PyDevice {
    inner: Device,
}

impl PyDevice {
    fn from_inner(inner: Device) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyDevice {
    #[new]
    #[pyo3(signature = (kind, index=0))]
    fn new(kind: &str, index: usize) -> PyResult<Self> {
        let device = match kind.trim().to_ascii_lowercase().as_str() {
            "cpu" => Device::Cpu,
            "wgpu" | "gpu" => Device::Wgpu(index),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown device kind '{other}'"
                )))
            }
        };
        Ok(Self::from_inner(device.resolve()))
    }

    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind()
    }

    #[getter]
    fn index(&self) -> usize {
        match self.inner {
            Device::Wgpu(index) => index,
            Device::Cpu => 0,
        }
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("Device('{}')", self.inner)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        if let Ok(dev) = other.extract::<PyRef<'_, PyDevice>>() {
            return self.inner == dev.inner;
        }
        if let Ok(text) = other.extract::<String>() {
            if let Ok(parsed) = device::parse_device_str(&text) {
                return self.inner == parsed.resolve();
            }
        }
        false
    }
}

#[pyfunction]
fn default_device() -> PyDevice {
    PyDevice::from_inner(device::default_device())
}

#[pyfunction]
fn set_default_device(device: &Bound<'_, PyAny>) -> PyResult<()> {
    device::set_default_device(parse_py_device(Some(device))?);
    Ok(())
}

#[pyclass(name = "Tensor", module = "aceflow._native")]
#[derive(Clone)]
pub struct PyTensor {
    pub inner: Tensor,
}

impl PyTensor {
    pub fn from_inner(inner: Tensor) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (data, shape=None, device=None))]
    fn new(
        data: &Bound<'_, PyAny>,
        shape: Option<Vec<usize>>,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let (values, source_shape) = array_from_py(data)?;
        let shape = shape.unwrap_or(source_shape);
        let expected: usize = shape.iter().product();
        if values.len() != expected {
            return Err(PyValueError::new_err(format!(
                "data length {} does not match shape {:?} ({expected})",
                values.len(),
                shape
            )));
        }
        let device = parse_py_device(device)?;
        Ok(Self::from_inner(Tensor::from_vec_on(values, &shape, device)))
    }

    #[staticmethod]
    #[pyo3(signature = (shape, device=None))]
    fn zeros(shape: Vec<usize>, device: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(Self::from_inner(Tensor::zeros_on(
            &shape,
            parse_py_device(device)?,
        )))
    }

    #[staticmethod]
    #[pyo3(signature = (shape, device=None))]
    fn ones(shape: Vec<usize>, device: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(Self::from_inner(Tensor::full(
            &shape,
            1.0,
            parse_py_device(device)?,
        )))
    }

    #[getter]
    fn device(&self) -> PyDevice {
        PyDevice::from_inner(self.inner.device())
    }

    fn to(&self, device: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::from_inner(self.inner.to(parse_py_device(Some(device))?)))
    }

    fn cpu(&self) -> Self {
        Self::from_inner(self.inner.cpu())
    }

    fn wgpu(&self) -> Self {
        Self::from_inner(self.inner.wgpu())
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }

    #[getter]
    fn strides(&self) -> Vec<usize> {
        self.inner.strides.clone()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.rank()
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.numel()
    }

    fn numel(&self) -> usize {
        self.inner.numel()
    }

    fn rows(&self) -> usize {
        self.inner.rows()
    }

    fn cols(&self) -> usize {
        self.inner.cols()
    }

    fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let array = ArrayD::from_shape_vec(IxDyn(&self.inner.shape), self.inner.data.clone())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(array.into_pyarray_bound(py))
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        self.numpy(py)
    }

    #[pyo3(signature = (_dtype=None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        _dtype: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        self.numpy(py)
    }

    fn reshape(&self, shape: Vec<usize>) -> Self {
        Self::from_inner(self.inner.reshape(&shape))
    }

    fn transpose(&self) -> Self {
        Self::from_inner(self.inner.t())
    }

    fn t(&self) -> Self {
        Self::from_inner(self.inner.t())
    }

    fn matmul(&self, other: &PyTensor) -> Self {
        Self::from_inner(self.inner.matmul(&other.inner))
    }

    fn matmul_bt(&self, other: &PyTensor) -> Self {
        Self::from_inner(self.inner.matmul_bt(&other.inner))
    }

    fn add(&self, other: &PyTensor) -> Self {
        Self::from_inner(self.inner.add(&other.inner))
    }

    fn add_bias(&self, bias: &PyTensor) -> Self {
        Self::from_inner(self.inner.add_bias(&bias.inner))
    }

    fn scale(&self, value: f32) -> Self {
        Self::from_inner(self.inner.scale(value))
    }

    fn mul(&self, other: &PyTensor) -> Self {
        Self::from_inner(self.inner.mul(&other.inner))
    }

    fn gelu(&self) -> Self {
        Self::from_inner(self.inner.gelu())
    }

    fn layer_norm(&self, weight: &PyTensor, bias: &PyTensor, eps: f32) -> Self {
        Self::from_inner(self.inner.layer_norm(&weight.inner, &bias.inner, eps))
    }

    fn rms_norm(&self, weight: &PyTensor, eps: f32) -> Self {
        Self::from_inner(self.inner.rms_norm(&weight.inner, eps))
    }

    fn softmax_last(&self) -> Self {
        Self::from_inner(self.inner.softmax_last())
    }

    fn slice_rows(&self, start: usize, len: usize) -> Self {
        Self::from_inner(self.inner.slice_rows(start, len))
    }

    fn last_row(&self) -> Self {
        Self::from_inner(self.inner.last_row())
    }

    fn concat_rows(&self, other: &PyTensor) -> Self {
        Self::from_inner(self.inner.concat_rows(&other.inner))
    }

    fn gather_rows(&self, ids: Vec<u32>) -> Self {
        Self::from_inner(self.inner.gather_rows(&ids))
    }

    fn argmax_last(&self) -> usize {
        self.inner.argmax_last()
    }

    fn __matmul__(&self, other: &PyTensor) -> Self {
        self.matmul(other)
    }

    fn __add__(&self, other: &PyTensor) -> Self {
        self.add(other)
    }

    fn __mul__(&self, scalar: f32) -> Self {
        self.scale(scalar)
    }

    fn __len__(&self) -> usize {
        self.inner.shape.first().copied().unwrap_or(0)
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, device={}, dtype=float32)",
            self.inner.shape,
            self.inner.device()
        )
    }
}

#[pyclass(name = "Linear", module = "aceflow._native")]
#[derive(Clone)]
pub struct PyLinear {
    inner: CoreLinear,
}

#[pymethods]
impl PyLinear {
    #[new]
    #[pyo3(signature = (weight_or_in, bias_or_out=None, bias=true, device=None))]
    fn new(
        weight_or_in: &Bound<'_, PyAny>,
        bias_or_out: Option<&Bound<'_, PyAny>>,
        bias: bool,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if let Ok(weight) = weight_or_in.extract::<PyRef<'_, PyTensor>>() {
            let bias_tensor = match bias_or_out {
                Some(value) => Some(value.extract::<PyRef<'_, PyTensor>>()?.inner.clone()),
                None => None,
            };
            return Ok(Self {
                inner: CoreLinear::new(weight.inner.clone(), bias_tensor),
            });
        }
        let in_features: usize = weight_or_in.extract()?;
        let out_features: usize = bias_or_out
            .ok_or_else(|| PyValueError::new_err("Linear out_features is required"))?
            .extract()?;
        Ok(Self {
            inner: CoreLinear::from_features(
                in_features,
                out_features,
                bias,
                parse_py_device(device)?,
            ),
        })
    }

    #[getter]
    fn in_features(&self) -> usize {
        self.inner.in_features()
    }

    #[getter]
    fn out_features(&self) -> usize {
        self.inner.out_features()
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_inner(self.inner.forward(&x.inner))
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        self.forward(x)
    }

    #[staticmethod]
    #[pyo3(signature = (in_features, out_features, bias=true, device=None))]
    fn from_features(
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CoreLinear::from_features(
                in_features,
                out_features,
                bias,
                parse_py_device(device)?,
            ),
        })
    }

    #[getter]
    fn device(&self) -> PyDevice {
        PyDevice::from_inner(self.inner.device())
    }
}

#[pyclass(name = "Conv1D", module = "aceflow._native")]
#[derive(Clone)]
pub struct PyConv1D {
    inner: CoreConv1D,
}

#[pymethods]
impl PyConv1D {
    #[new]
    #[pyo3(signature = (weight, bias=None, stride=1, padding=0))]
    fn new(weight: &PyTensor, bias: Option<&PyTensor>, stride: usize, padding: usize) -> Self {
        Self {
            inner: CoreConv1D::new(
                weight.inner.clone(),
                bias.map(|value| value.inner.clone()),
                stride,
                padding,
            ),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_inner(self.inner.forward(&x.inner))
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        self.forward(x)
    }
}

#[pyclass(name = "RMSNorm", module = "aceflow._native")]
#[derive(Clone)]
pub struct PyRMSNorm {
    inner: CoreRMSNorm,
}

#[pymethods]
impl PyRMSNorm {
    #[new]
    #[pyo3(signature = (weight_or_dim, eps=1e-5, device=None))]
    fn new(
        weight_or_dim: &Bound<'_, PyAny>,
        eps: f32,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if let Ok(weight) = weight_or_dim.extract::<PyRef<'_, PyTensor>>() {
            return Ok(Self {
                inner: CoreRMSNorm::new(weight.inner.clone(), eps),
            });
        }
        let dim: usize = weight_or_dim.extract()?;
        Ok(Self {
            inner: CoreRMSNorm::from_dim(dim, eps, parse_py_device(device)?),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (dim, eps=1e-5, device=None))]
    fn from_dim(dim: usize, eps: f32, device: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(Self {
            inner: CoreRMSNorm::from_dim(dim, eps, parse_py_device(device)?),
        })
    }

    #[getter]
    fn device(&self) -> PyDevice {
        PyDevice::from_inner(self.inner.weight.device())
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_inner(self.inner.forward(&x.inner))
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        self.forward(x)
    }
}

#[pyclass(name = "LayerNorm", module = "aceflow._native")]
#[derive(Clone)]
pub struct PyLayerNorm {
    inner: CoreLayerNorm,
}

#[pymethods]
impl PyLayerNorm {
    #[new]
    #[pyo3(signature = (weight, bias, eps=1e-5))]
    fn new(weight: &PyTensor, bias: &PyTensor, eps: f32) -> Self {
        Self {
            inner: CoreLayerNorm::new(weight.inner.clone(), bias.inner.clone(), eps),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_inner(self.inner.forward(&x.inner))
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        self.forward(x)
    }
}

#[pyclass(name = "SwiGLU", module = "aceflow._native")]
#[derive(Clone, Default)]
pub struct PySwiGLU {
    inner: CoreSwiGLU,
}

#[pymethods]
impl PySwiGLU {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn forward(&self, gate: &PyTensor, up: &PyTensor) -> PyTensor {
        PyTensor::from_inner(self.inner.forward(&gate.inner, &up.inner))
    }

    fn forward_packed(&self, packed: &PyTensor) -> PyTensor {
        PyTensor::from_inner(self.inner.forward_packed(&packed.inner))
    }

    fn __call__(&self, gate: &PyTensor, up: &PyTensor) -> PyTensor {
        self.forward(gate, up)
    }
}

#[pyclass(name = "MultiHeadAttention", module = "aceflow._native")]
#[derive(Clone)]
pub struct PyMultiHeadAttention {
    inner: CoreMultiHeadAttention,
}

#[pymethods]
impl PyMultiHeadAttention {
    #[new]
    #[pyo3(signature = (q_proj_or_d, k_proj=None, v_proj=None, out_proj=None, num_heads=None, device=None))]
    fn new(
        q_proj_or_d: &Bound<'_, PyAny>,
        k_proj: Option<&Bound<'_, PyAny>>,
        v_proj: Option<&Bound<'_, PyAny>>,
        out_proj: Option<&Bound<'_, PyAny>>,
        num_heads: Option<usize>,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if let Ok(d_model) = q_proj_or_d.extract::<usize>() {
            let heads = match num_heads {
                Some(value) => value,
                None => k_proj
                    .ok_or_else(|| PyValueError::new_err("num_heads is required"))?
                    .extract::<usize>()?,
            };
            return Ok(Self {
                inner: CoreMultiHeadAttention::from_features(
                    d_model,
                    heads,
                    parse_py_device(device)?,
                ),
            });
        }
        let q_proj = q_proj_or_d.extract::<PyRef<'_, PyLinear>>()?;
        let k_proj = k_proj
            .ok_or_else(|| PyValueError::new_err("k_proj is required"))?
            .extract::<PyRef<'_, PyLinear>>()?;
        let v_proj = v_proj
            .ok_or_else(|| PyValueError::new_err("v_proj is required"))?
            .extract::<PyRef<'_, PyLinear>>()?;
        let out_proj = out_proj
            .ok_or_else(|| PyValueError::new_err("out_proj is required"))?
            .extract::<PyRef<'_, PyLinear>>()?;
        let num_heads =
            num_heads.ok_or_else(|| PyValueError::new_err("num_heads is required"))?;
        Ok(Self {
            inner: CoreMultiHeadAttention::new(
                q_proj.inner.clone(),
                k_proj.inner.clone(),
                v_proj.inner.clone(),
                out_proj.inner.clone(),
                num_heads,
            ),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (d_model, num_heads, device=None))]
    fn from_features(
        d_model: usize,
        num_heads: usize,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CoreMultiHeadAttention::from_features(
                d_model,
                num_heads,
                parse_py_device(device)?,
            ),
        })
    }

    #[pyo3(signature = (x, kv_source=None, causal=false))]
    fn forward(&self, x: &PyTensor, kv_source: Option<&PyTensor>, causal: bool) -> PyTensor {
        PyTensor::from_inner(self.inner.forward(
            &x.inner,
            kv_source.map(|value| &value.inner),
            causal,
        ))
    }

    #[pyo3(signature = (x, kv_source=None, causal=false))]
    fn __call__(&self, x: &PyTensor, kv_source: Option<&PyTensor>, causal: bool) -> PyTensor {
        self.forward(x, kv_source, causal)
    }
}

#[pyclass(name = "AdditiveAttention", module = "aceflow._native")]
#[derive(Clone)]
pub struct PyAdditiveAttention {
    inner: CoreAdditiveAttention,
}

#[pymethods]
impl PyAdditiveAttention {
    #[new]
    fn new(w_keys: &PyLinear, w_query: &PyLinear, v: &PyLinear) -> Self {
        Self {
            inner: CoreAdditiveAttention::new(
                w_keys.inner.clone(),
                w_query.inner.clone(),
                v.inner.clone(),
            ),
        }
    }

    fn forward(&self, query: &PyTensor, keys: &PyTensor, values: &PyTensor) -> (PyTensor, PyTensor) {
        let (context, weights) = self.inner.forward(&query.inner, &keys.inner, &values.inner);
        (PyTensor::from_inner(context), PyTensor::from_inner(weights))
    }

    fn __call__(&self, query: &PyTensor, keys: &PyTensor, values: &PyTensor) -> (PyTensor, PyTensor) {
        self.forward(query, keys, values)
    }
}

#[pyclass(name = "EncoderDecoderBlock", module = "aceflow._native")]
#[derive(Clone)]
pub struct PyEncoderDecoderBlock {
    inner: CoreEncoderDecoderBlock,
}

#[pymethods]
impl PyEncoderDecoderBlock {
    #[new]
    #[pyo3(signature = (encoder_attn, decoder_attn, cross_attn, encoder_norm=None, decoder_norm=None))]
    fn new(
        encoder_attn: &PyMultiHeadAttention,
        decoder_attn: &PyMultiHeadAttention,
        cross_attn: &PyMultiHeadAttention,
        encoder_norm: Option<&PyLayerNorm>,
        decoder_norm: Option<&PyLayerNorm>,
    ) -> Self {
        Self {
            inner: CoreEncoderDecoderBlock::new(
                encoder_attn.inner.clone(),
                decoder_attn.inner.clone(),
                cross_attn.inner.clone(),
                encoder_norm.map(|value| value.inner.clone()),
                decoder_norm.map(|value| value.inner.clone()),
            ),
        }
    }

    fn encode(&self, src: &PyTensor) -> PyTensor {
        PyTensor::from_inner(self.inner.encode(&src.inner))
    }

    #[pyo3(signature = (tgt, memory, causal=true))]
    fn decode(&self, tgt: &PyTensor, memory: &PyTensor, causal: bool) -> PyTensor {
        PyTensor::from_inner(self.inner.decode(&tgt.inner, &memory.inner, causal))
    }
}

#[pyfunction(name = "scaled_dot_product_attention")]
#[pyo3(signature = (q, k, v, num_heads=1, causal=false))]
fn scaled_dot_product_attention_py(
    q: &PyTensor,
    k: &PyTensor,
    v: &PyTensor,
    num_heads: usize,
    causal: bool,
) -> PyTensor {
    PyTensor::from_inner(scaled_dot_product_attention(
        &q.inner, &k.inner, &v.inner, num_heads, causal,
    ))
}

#[pyfunction(name = "cross_attention")]
#[pyo3(signature = (q, k, v, num_heads=1, causal=false))]
fn cross_attention_py(
    q: &PyTensor,
    k: &PyTensor,
    v: &PyTensor,
    num_heads: usize,
    causal: bool,
) -> PyTensor {
    PyTensor::from_inner(cross_attention(
        &q.inner, &k.inner, &v.inner, num_heads, causal,
    ))
}

#[pyfunction(name = "sequence_downsample")]
#[pyo3(signature = (x, factor, mode="mean"))]
fn sequence_downsample_py(x: &PyTensor, factor: usize, mode: &str) -> PyResult<PyTensor> {
    if factor == 0 {
        return Err(PyValueError::new_err("downsample factor must be positive"));
    }
    Ok(PyTensor::from_inner(sequence_downsample(
        &x.inner, factor, mode,
    )))
}

#[pyfunction(name = "sequence_upsample")]
fn sequence_upsample_py(x: &PyTensor, factor: usize) -> PyResult<PyTensor> {
    if factor == 0 {
        return Err(PyValueError::new_err("upsample factor must be positive"));
    }
    Ok(PyTensor::from_inner(sequence_upsample(&x.inner, factor)))
}

#[pyfunction(name = "resample_linear")]
fn resample_linear_py<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray1<'py, f32>,
    from_rate: u32,
    to_rate: u32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    if from_rate == 0 || to_rate == 0 {
        return Err(PyValueError::new_err("sample rates must be non-zero"));
    }
    Ok(dsp::resample_linear(samples.as_slice()?, from_rate, to_rate).into_pyarray_bound(py))
}

#[pyfunction(name = "hann_window")]
#[pyo3(signature = (n=dsp::N_FFT))]
fn hann_window_py<'py>(py: Python<'py>, n: usize) -> Bound<'py, PyArray1<f32>> {
    dsp::hann_window(n).into_pyarray_bound(py)
}

#[pyfunction(name = "mel_filters")]
#[pyo3(signature = (n_mels=dsp::N_MELS, n_fft=dsp::N_FFT, sample_rate=dsp::SAMPLE_RATE))]
fn mel_filters_py<'py>(
    py: Python<'py>,
    n_mels: usize,
    n_fft: usize,
    sample_rate: u32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let frequencies = n_fft / 2 + 1;
    let array = Array2::from_shape_vec(
        (n_mels, frequencies),
        dsp::mel_filters(n_mels, n_fft, sample_rate),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(array.into_pyarray_bound(py))
}

#[pyfunction(name = "stft_power")]
#[pyo3(signature = (audio, n_fft=dsp::N_FFT, hop_length=dsp::HOP_LENGTH))]
fn stft_power_py<'py>(
    py: Python<'py>,
    audio: PyReadonlyArray1<'py, f32>,
    n_fft: usize,
    hop_length: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if n_fft == 0 || hop_length == 0 {
        return Err(PyValueError::new_err(
            "n_fft and hop_length must be positive",
        ));
    }
    let (frames, values) = dsp::stft_power_with_frames(audio.as_slice()?, n_fft, hop_length);
    let array = Array2::from_shape_vec((n_fft / 2 + 1, frames), values)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(array.into_pyarray_bound(py))
}

#[pyfunction(name = "pad_or_trim")]
#[pyo3(signature = (audio, n_samples=None))]
fn pad_or_trim_py<'py>(
    py: Python<'py>,
    audio: PyReadonlyArray1<'py, f32>,
    n_samples: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    Ok(
        dsp::pad_or_trim_to(audio.as_slice()?, n_samples.unwrap_or(dsp::N_SAMPLES))
            .into_pyarray_bound(py),
    )
}

#[pyfunction(name = "load_wav")]
fn load_wav_py<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyArray1<f32>>> {
    dsp::load_wav(path)
        .map(|values| values.into_pyarray_bound(py))
        .map_err(|error| PyRuntimeError::new_err(error.0))
}

#[pyfunction(name = "log_mel_spectrogram")]
fn log_mel_spectrogram_py<'py>(
    py: Python<'py>,
    audio: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let values = dsp::log_mel_spectrogram(audio.as_slice()?);
    let array = Array2::from_shape_vec((dsp::N_MELS, dsp::N_FRAMES), values)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(array.into_pyarray_bound(py))
}

#[pyclass(name = "MelSpectrogram", module = "aceflow._native")]
#[derive(Clone)]
pub struct PyMelSpectrogram {
    inner: dsp::MelSpectrogram,
}

#[pymethods]
impl PyMelSpectrogram {
    #[new]
    #[pyo3(signature = (
        sample_rate=dsp::SAMPLE_RATE,
        n_fft=dsp::N_FFT,
        hop_length=dsp::HOP_LENGTH,
        n_mels=dsp::N_MELS,
        n_samples=dsp::N_SAMPLES
    ))]
    fn new(
        sample_rate: u32,
        n_fft: usize,
        hop_length: usize,
        n_mels: usize,
        n_samples: usize,
    ) -> PyResult<Self> {
        dsp::MelSpectrogram::new(sample_rate, n_fft, hop_length, n_mels, n_samples)
            .map(|inner| Self { inner })
            .map_err(|error| PyValueError::new_err(error.0))
    }

    #[staticmethod]
    fn whisper() -> Self {
        Self {
            inner: dsp::MelSpectrogram::whisper(),
        }
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        audio: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let values = self.inner.transform(audio.as_slice()?);
        let array = Array2::from_shape_vec((self.inner.n_mels, self.inner.n_frames), values)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(array.into_pyarray_bound(py))
    }

    fn __call__<'py>(
        &self,
        py: Python<'py>,
        audio: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.transform(py, audio)
    }

    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }

    #[getter]
    fn n_fft(&self) -> usize {
        self.inner.n_fft
    }

    #[getter]
    fn hop_length(&self) -> usize {
        self.inner.hop_length
    }

    #[getter]
    fn n_mels(&self) -> usize {
        self.inner.n_mels
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.inner.n_samples
    }

    #[getter]
    fn n_frames(&self) -> usize {
        self.inner.n_frames
    }
}

fn rnn_kind_from_py(value: &Bound<'_, PyAny>) -> PyResult<RnnKind> {
    if let Ok(code) = value.extract::<i32>() {
        return RnnKind::from_code(code).map_err(|error| PyValueError::new_err(error.0));
    }
    let name: String = value.extract()?;
    RnnKind::from_name(&name).map_err(|error| PyValueError::new_err(error.0))
}

fn attention_kind_from_py(value: &Bound<'_, PyAny>) -> PyResult<AttentionKind> {
    if let Ok(code) = value.extract::<i32>() {
        return AttentionKind::from_code(code).map_err(|error| PyValueError::new_err(error.0));
    }
    let name: String = value.extract()?;
    AttentionKind::from_name(&name).map_err(|error| PyValueError::new_err(error.0))
}

fn array_shape(array: &PyReadonlyArrayDyn<'_, f32>) -> Vec<usize> {
    array.shape().to_vec()
}

#[pyfunction(name = "rnn_forward")]
#[pyo3(signature = (input, hidden, weights, biases, rnn_type, bidirectional=false, cell=None))]
fn rnn_forward_py<'py>(
    py: Python<'py>,
    input: PyReadonlyArrayDyn<'py, f32>,
    hidden: PyReadonlyArrayDyn<'py, f32>,
    weights: PyReadonlyArrayDyn<'py, f32>,
    biases: PyReadonlyArrayDyn<'py, f32>,
    rnn_type: &Bound<'py, PyAny>,
    bidirectional: bool,
    cell: Option<PyReadonlyArrayDyn<'py, f32>>,
) -> PyResult<(
    Bound<'py, PyArrayDyn<f32>>,
    Bound<'py, PyArrayDyn<f32>>,
    Bound<'py, PyArrayDyn<f32>>,
)> {
    let kind = rnn_kind_from_py(rnn_type)?;
    let input_shape = array_shape(&input);
    if input_shape.len() != 3 {
        return Err(PyValueError::new_err(
            "rnn input must have shape [batch, seq, features]",
        ));
    }
    let hidden_shape = array_shape(&hidden);
    if hidden_shape.len() != 3 {
        return Err(PyValueError::new_err(
            "rnn hidden must have shape [num_layers * num_directions, batch, hidden]",
        ));
    }
    let batch = input_shape[0];
    let seq_len = input_shape[1];
    let input_size = input_shape[2];
    let stacked = hidden_shape[0];
    let hidden_size = hidden_shape[2];
    if hidden_shape[1] != batch {
        return Err(PyValueError::new_err("rnn hidden batch mismatch"));
    }
    let dirs = if bidirectional { 2 } else { 1 };
    if stacked % dirs != 0 {
        return Err(PyValueError::new_err(
            "hidden layer axis must be divisible by number of directions",
        ));
    }
    let num_layers = stacked / dirs;
    let cell_storage;
    let cell_slice: &[f32] = if let Some(cell) = cell {
        cell_storage = cell.as_slice()?.to_vec();
        &cell_storage
    } else {
        &[]
    };
    let input_data = input.as_slice()?.to_vec();
    let hidden_data = hidden.as_slice()?.to_vec();
    let weights = weights.as_slice()?.to_vec();
    let biases = biases.as_slice()?.to_vec();
    let output = py
        .allow_threads(|| {
            rnn_forward_cpu(
                &input_data,
                &hidden_data,
                cell_slice,
                &weights,
                &biases,
                batch,
                seq_len,
                input_size,
                hidden_size,
                num_layers,
                kind,
                bidirectional,
            )
        })
        .map_err(|error| PyValueError::new_err(error.0))?;
    let out_width = output.output.len() / (batch * seq_len);
    let output_arr = ArrayD::from_shape_vec(
        IxDyn(&[batch, seq_len, out_width]),
        output.output,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let hidden_arr = ArrayD::from_shape_vec(IxDyn(&[stacked, batch, hidden_size]), output.hidden)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let cell_arr = ArrayD::from_shape_vec(IxDyn(&[stacked, batch, hidden_size]), output.cell)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        output_arr.into_pyarray_bound(py),
        hidden_arr.into_pyarray_bound(py),
        cell_arr.into_pyarray_bound(py),
    ))
}

#[pyfunction(name = "bahdanau_attention_forward")]
#[pyo3(signature = (decoder_hidden, encoder_outputs, attention_type=None, w1=None, w2=None, v=None))]
fn bahdanau_attention_forward_py<'py>(
    py: Python<'py>,
    decoder_hidden: PyReadonlyArrayDyn<'py, f32>,
    encoder_outputs: PyReadonlyArrayDyn<'py, f32>,
    attention_type: Option<&Bound<'py, PyAny>>,
    w1: Option<PyReadonlyArrayDyn<'py, f32>>,
    w2: Option<PyReadonlyArrayDyn<'py, f32>>,
    v: Option<PyReadonlyArrayDyn<'py, f32>>,
) -> PyResult<(Bound<'py, PyArrayDyn<f32>>, Bound<'py, PyArrayDyn<f32>>)> {
    let kind = match attention_type {
        Some(value) => attention_kind_from_py(value)?,
        None => AttentionKind::Dot,
    };
    let query_shape = array_shape(&decoder_hidden);
    let key_shape = array_shape(&encoder_outputs);
    if query_shape.len() != 2 || key_shape.len() != 3 {
        return Err(PyValueError::new_err(
            "decoder_hidden must be [batch, hidden] and encoder_outputs [batch, seq, hidden]",
        ));
    }
    let batch = query_shape[0];
    let hidden = query_shape[1];
    if key_shape[0] != batch || key_shape[2] != hidden {
        return Err(PyValueError::new_err("attention batch/hidden mismatch"));
    }
    let seq_len = key_shape[1];
    let query = decoder_hidden.as_slice()?.to_vec();
    let keys = encoder_outputs.as_slice()?.to_vec();
    let w1 = w1.map(|value| value.as_slice().map(|slice| slice.to_vec())).transpose()?;
    let w2 = w2.map(|value| value.as_slice().map(|slice| slice.to_vec())).transpose()?;
    let v = v.map(|value| value.as_slice().map(|slice| slice.to_vec())).transpose()?;
    let mut scores = vec![0.0f32; batch * seq_len];
    let mut context = vec![0.0f32; batch * hidden];
    py.allow_threads(|| {
        seq2seq_attention_cpu(
            &query,
            &keys,
            &mut scores,
            &mut context,
            batch,
            seq_len,
            hidden,
            kind,
            w1.as_deref(),
            w2.as_deref(),
            v.as_deref(),
        )
    })
    .map_err(|error| PyValueError::new_err(error.0))?;
    let weights_arr = ArrayD::from_shape_vec(IxDyn(&[batch, seq_len]), scores)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let context_arr = ArrayD::from_shape_vec(IxDyn(&[batch, hidden]), context)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        weights_arr.into_pyarray_bound(py),
        context_arr.into_pyarray_bound(py),
    ))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDevice>()?;
    module.add_class::<PyTensor>()?;
    module.add_class::<PyLinear>()?;
    module.add_class::<PyConv1D>()?;
    module.add_class::<PyRMSNorm>()?;
    module.add_class::<PyLayerNorm>()?;
    module.add_class::<PySwiGLU>()?;
    module.add_class::<PyMultiHeadAttention>()?;
    module.add_class::<PyAdditiveAttention>()?;
    module.add_class::<PyEncoderDecoderBlock>()?;
    module.add_class::<PyMelSpectrogram>()?;
    module.add_function(wrap_pyfunction!(default_device, module)?)?;
    module.add_function(wrap_pyfunction!(set_default_device, module)?)?;
    module.add_function(wrap_pyfunction!(scaled_dot_product_attention_py, module)?)?;
    module.add_function(wrap_pyfunction!(cross_attention_py, module)?)?;
    module.add_function(wrap_pyfunction!(sequence_downsample_py, module)?)?;
    module.add_function(wrap_pyfunction!(sequence_upsample_py, module)?)?;
    module.add_function(wrap_pyfunction!(resample_linear_py, module)?)?;
    module.add_function(wrap_pyfunction!(hann_window_py, module)?)?;
    module.add_function(wrap_pyfunction!(mel_filters_py, module)?)?;
    module.add_function(wrap_pyfunction!(stft_power_py, module)?)?;
    module.add_function(wrap_pyfunction!(pad_or_trim_py, module)?)?;
    module.add_function(wrap_pyfunction!(load_wav_py, module)?)?;
    module.add_function(wrap_pyfunction!(log_mel_spectrogram_py, module)?)?;
    module.add_function(wrap_pyfunction!(rnn_forward_py, module)?)?;
    module.add_function(wrap_pyfunction!(bahdanau_attention_forward_py, module)?)?;
    module.add_class::<crate::cpu::beam_search::PyBeamSearch>()?;
    module.add_class::<crate::cpu::beam_search::PyBeamSearchConfig>()?;
    module.add_function(wrap_pyfunction!(crate::cpu::beam_search::beam_search_rust_py, module)?)?;
    module.add_function(wrap_pyfunction!(crate::cpu::beam_search::beam_search_batch_rust_py, module)?)?;
    Ok(())
}
