//! Schema-detecting tokenizer that covers Mallow BPE, AceFlow word-level, and HF JSON.

use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

use super::bpe::BpeTokenizer;
use super::hf_bpe::HfBpeTokenizer;
use super::word::{Side, WordLevelTokenizer};
use crate::error::{Error, Result};

#[derive(Clone, Debug)]
enum Backend {
    Mallow(BpeTokenizer),
    Word(WordLevelTokenizer),
    HuggingFace(HfBpeTokenizer),
}

#[derive(Clone, Debug)]
pub struct AutoTokenizer {
    backend: Backend,
    pad_to_max: bool,
}

impl AutoTokenizer {
    pub fn from_mallow(inner: BpeTokenizer) -> Self {
        Self {
            backend: Backend::Mallow(inner),
            pad_to_max: false,
        }
    }

    pub fn from_word_level(
        vocab: HashMap<String, usize>,
        specials: HashMap<String, usize>,
        max_length: usize,
        padding: &str,
        truncation: &str,
    ) -> Self {
        Self {
            backend: Backend::Word(WordLevelTokenizer::new(
                vocab,
                &specials,
                max_length,
                Side::parse(padding),
                Side::parse(truncation),
            )),
            pad_to_max: true,
        }
    }

    pub fn from_hf_json(text: &str) -> Result<Self> {
        Ok(Self {
            backend: Backend::HuggingFace(HfBpeTokenizer::from_json(text)?),
            pad_to_max: false,
        })
    }

    pub fn from_pretrained(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path.is_file() {
            let name = path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default();
            if name == "tokenizer.json" || name.ends_with(".json") {
                return Ok(Self {
                    backend: Backend::HuggingFace(HfBpeTokenizer::load(path)?),
                    pad_to_max: false,
                });
            }
        }

        let tokenizer_json = path.join("tokenizer.json");
        if tokenizer_json.is_file() {
            return Ok(Self {
                backend: Backend::HuggingFace(HfBpeTokenizer::load(&tokenizer_json)?),
                pad_to_max: false,
            });
        }

        let vocab_json = path.join("vocab.json");
        let ace_config = path.join("tokenizer_config.json");
        if ace_config.is_file() && vocab_json.is_file() {
            return load_aceflow_dir(path);
        }
        if vocab_json.is_file() && path.join("merges.txt").is_file() {
            return load_hf_vocab_merges(path);
        }
        if path.join("merges.txt").is_file() {
            let inner = BpeTokenizer::load(path)
                .map_err(|error| Error(format!("failed to load Mallow BPE: {error}")))?;
            return Ok(Self::from_mallow(inner));
        }
        Err(Error(format!(
            "unrecognized tokenizer directory: {}",
            path.display()
        )))
    }

    pub fn vocab_size(&self) -> usize {
        match &self.backend {
            Backend::Mallow(inner) => inner.vocab_size(),
            Backend::Word(inner) => inner.vocab_size(),
            Backend::HuggingFace(inner) => inner.vocab_size(),
        }
    }

    pub fn pad_id(&self) -> usize {
        match &self.backend {
            Backend::Mallow(_) => super::PAD_ID,
            Backend::Word(inner) => inner.pad_id,
            Backend::HuggingFace(inner) => inner.pad_id,
        }
    }

    pub fn bos_id(&self) -> usize {
        match &self.backend {
            Backend::Mallow(_) => super::BOS_ID,
            Backend::Word(inner) => inner.bos_id,
            Backend::HuggingFace(inner) => inner.bos_id,
        }
    }

    pub fn eos_id(&self) -> usize {
        match &self.backend {
            Backend::Mallow(_) => super::EOS_ID,
            Backend::Word(inner) => inner.eos_id,
            Backend::HuggingFace(inner) => inner.eos_id,
        }
    }

    pub fn unk_id(&self) -> Option<usize> {
        match &self.backend {
            Backend::Mallow(_) => None,
            Backend::Word(inner) => Some(inner.unk_id),
            Backend::HuggingFace(inner) => inner.unk_id,
        }
    }

    pub fn schema(&self) -> &'static str {
        match &self.backend {
            Backend::Mallow(_) => "mallow-bpe",
            Backend::Word(_) => "aceflow-word",
            Backend::HuggingFace(_) => "huggingface-bpe",
        }
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<usize> {
        match &self.backend {
            Backend::Mallow(inner) => inner.encode(text, add_special_tokens),
            Backend::Word(inner) => inner.encode(text, add_special_tokens),
            Backend::HuggingFace(inner) => inner.encode(text, add_special_tokens),
        }
    }

    pub fn encode_batch(&self, texts: &[String], add_special_tokens: bool) -> Vec<Vec<usize>> {
        match &self.backend {
            Backend::Word(inner) => inner.encode_batch(texts, add_special_tokens),
            _ => texts
                .par_iter()
                .map(|text| self.encode(text, add_special_tokens))
                .collect(),
        }
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        match &self.backend {
            Backend::Mallow(inner) => inner.decode(ids),
            Backend::Word(inner) => inner.decode(ids, true),
            Backend::HuggingFace(inner) => inner.decode(ids, true).trim().to_string(),
        }
    }

    pub fn attention_mask(&self, ids: &[usize]) -> Vec<u8> {
        let pad = self.pad_id();
        ids.iter()
            .map(|&id| if id == pad { 0 } else { 1 })
            .collect()
    }

    pub fn pad_to_max(&self) -> bool {
        self.pad_to_max
    }
}

fn load_aceflow_dir(path: &Path) -> Result<AutoTokenizer> {
    let config_text = std::fs::read_to_string(path.join("tokenizer_config.json"))
        .map_err(|error| Error(format!("tokenizer_config.json: {error}")))?;
    let config: serde_json::Value = serde_json::from_str(&config_text)
        .map_err(|error| Error(format!("tokenizer_config.json parse: {error}")))?;
    let max_length = config
        .get("max_length")
        .and_then(|value| value.as_u64())
        .unwrap_or(100) as usize;
    let padding = config
        .get("padding")
        .and_then(|value| value.as_str())
        .unwrap_or("post")
        .to_string();
    let truncation = config
        .get("truncation")
        .and_then(|value| value.as_str())
        .unwrap_or("post")
        .to_string();

    let vocab_text = std::fs::read_to_string(path.join("vocab.json"))
        .map_err(|error| Error(format!("vocab.json: {error}")))?;
    let raw: HashMap<String, usize> = serde_json::from_str(&vocab_text)
        .map_err(|error| Error(format!("vocab.json parse: {error}")))?;
    let mut specials = HashMap::new();
    for name in ["<pad>", "<start>", "<end>", "<unk>", "<mask>"] {
        if let Some(&id) = raw.get(name) {
            specials.insert(name.to_string(), id);
        }
    }
    Ok(AutoTokenizer::from_word_level(
        raw, specials, max_length, &padding, &truncation,
    ))
}

fn load_hf_vocab_merges(path: &Path) -> Result<AutoTokenizer> {
    let vocab_text = std::fs::read_to_string(path.join("vocab.json"))
        .map_err(|error| Error(format!("vocab.json: {error}")))?;
    let merges_text = std::fs::read_to_string(path.join("merges.txt"))
        .map_err(|error| Error(format!("merges.txt: {error}")))?;
    let mut merges = Vec::new();
    for line in merges_text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        merges.push(trimmed);
    }
    let payload = serde_json::json!({
        "model": {
            "vocab": serde_json::from_str::<serde_json::Value>(&vocab_text)
                .map_err(|error| Error(format!("vocab.json parse: {error}")))?,
            "merges": merges,
        }
    });
    AutoTokenizer::from_hf_json(&payload.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::bpe::BpeTokenizer;

    #[test]
    fn mallow_schema_keeps_pad_bos_eos() {
        let tokenizer = AutoTokenizer::from_mallow(BpeTokenizer::new(HashMap::new(), HashMap::new()));
        assert_eq!(tokenizer.schema(), "mallow-bpe");
        assert_eq!(tokenizer.encode("A", true), vec![1, 68, 2]);
        assert_eq!(tokenizer.decode(&[1, 68, 2]), "A");
    }
}
