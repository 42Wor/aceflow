pub mod auto;
pub mod bpe;
pub mod hf_bpe;
pub mod word;

pub use auto::AutoTokenizer;
pub use bpe::{BpeTokenizer, BOS_ID, BYTE_OFFSET, EOS_ID, PAD_ID};

use numpy::PyArray1;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass(module = "aceflow._native")]
#[derive(Clone)]
pub struct FastTokenizer {
    pub inner: BpeTokenizer,
}

#[pymethods]
impl FastTokenizer {
    #[new]
    pub fn new(
        merges: HashMap<(usize, usize), usize>,
        merge_ranks: HashMap<(usize, usize), usize>,
    ) -> Self {
        Self {
            inner: BpeTokenizer::new(merges, merge_ranks),
        }
    }

    #[getter]
    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    pub fn __len__(&self) -> usize {
        self.inner.vocab_size()
    }

    #[pyo3(signature = (text, add_special_tokens=false))]
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<usize> {
        self.inner.encode(text, add_special_tokens)
    }

    #[pyo3(signature = (texts, add_special_tokens=false))]
    pub fn encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        add_special_tokens: bool,
    ) -> Vec<Vec<usize>> {
        py.allow_threads(|| self.inner.encode_batch_strings(&texts, add_special_tokens))
    }

    #[pyo3(signature = (texts, add_special_tokens=false))]
    pub fn encode_batch_flat<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
        add_special_tokens: bool,
    ) -> Bound<'py, PyArray1<i64>> {
        let flat = py.allow_threads(|| {
            self.inner
                .encode_batch_strings_flat(&texts, add_special_tokens)
        });
        PyArray1::from_vec_bound(py, flat)
    }

    pub fn decode(&self, tokens: Vec<usize>) -> String {
        self.inner.decode(&tokens)
    }

    pub fn save(&self, path_dir: &str) -> PyResult<()> {
        self.inner
            .save(path_dir)
            .map_err(|error| PyIOError::new_err(format!("Failed to save tokenizer: {error}")))
    }

    #[staticmethod]
    pub fn load(path_dir: &str) -> PyResult<Self> {
        let inner = BpeTokenizer::load(path_dir)
            .map_err(|error| PyIOError::new_err(format!("Failed to load tokenizer: {error}")))?;
        Ok(Self { inner })
    }
}

#[pyclass(name = "AutoTokenizer", module = "aceflow._native")]
#[derive(Clone)]
pub struct PyAutoTokenizer {
    pub inner: AutoTokenizer,
}

#[pymethods]
impl PyAutoTokenizer {
    #[staticmethod]
    pub fn from_pretrained(path: &str) -> PyResult<Self> {
        let inner = AutoTokenizer::from_pretrained(path)
            .map_err(|error| PyIOError::new_err(error.0))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    pub fn from_hf_json(text: &str) -> PyResult<Self> {
        let inner =
            AutoTokenizer::from_hf_json(text).map_err(|error| PyIOError::new_err(error.0))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (vocab, special_tokens=None, max_length=100, padding="post", truncation="post"))]
    pub fn from_word_level(
        vocab: HashMap<String, usize>,
        special_tokens: Option<HashMap<String, usize>>,
        max_length: usize,
        padding: &str,
        truncation: &str,
    ) -> Self {
        Self {
            inner: AutoTokenizer::from_word_level(
                vocab,
                special_tokens.unwrap_or_default(),
                max_length,
                padding,
                truncation,
            ),
        }
    }

    #[staticmethod]
    pub fn from_mallow(tokenizer: &FastTokenizer) -> Self {
        Self {
            inner: AutoTokenizer::from_mallow(tokenizer.inner.clone()),
        }
    }

    #[getter]
    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    pub fn __len__(&self) -> usize {
        self.inner.vocab_size()
    }

    #[getter]
    pub fn schema(&self) -> &'static str {
        self.inner.schema()
    }

    #[getter]
    pub fn pad_id(&self) -> usize {
        self.inner.pad_id()
    }

    #[getter]
    pub fn bos_id(&self) -> usize {
        self.inner.bos_id()
    }

    #[getter]
    pub fn eos_id(&self) -> usize {
        self.inner.eos_id()
    }

    #[getter]
    pub fn unk_id(&self) -> Option<usize> {
        self.inner.unk_id()
    }

    #[pyo3(signature = (text, add_special_tokens=true))]
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<usize> {
        self.inner.encode(text, add_special_tokens)
    }

    #[pyo3(signature = (texts, add_special_tokens=true))]
    pub fn encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        add_special_tokens: bool,
    ) -> Vec<Vec<usize>> {
        py.allow_threads(|| self.inner.encode_batch(&texts, add_special_tokens))
    }

    pub fn decode(&self, tokens: Vec<usize>) -> String {
        self.inner.decode(&tokens)
    }

    pub fn attention_mask(&self, tokens: Vec<usize>) -> Vec<u8> {
        self.inner.attention_mask(&tokens)
    }
}
