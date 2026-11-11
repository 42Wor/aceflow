//! HuggingFace `tokenizer.json` / GPT-2 byte-level BPE.

use regex::Regex;
use rustc_hash::FxHashMap;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use crate::error::{Error, Result};

#[derive(Deserialize)]
struct TokenizerFile {
    #[serde(default)]
    model: ModelSection,
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
}

#[derive(Default, Deserialize)]
struct ModelSection {
    #[serde(default)]
    vocab: HashMap<String, u32>,
    #[serde(default)]
    merges: Merges,
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
}

#[derive(Default)]
struct Merges(Vec<(String, String)>);

impl<'de> Deserialize<'de> for Merges {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        let mut pairs = Vec::new();
        match value {
            serde_json::Value::Array(items) => {
                for item in items {
                    match item {
                        serde_json::Value::String(line) => {
                            let mut parts = line.split_whitespace();
                            if let (Some(left), Some(right)) = (parts.next(), parts.next()) {
                                pairs.push((left.to_string(), right.to_string()));
                            }
                        }
                        serde_json::Value::Array(pair) if pair.len() >= 2 => {
                            if let (Some(left), Some(right)) = (pair[0].as_str(), pair[1].as_str()) {
                                pairs.push((left.to_string(), right.to_string()));
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        Ok(Self(pairs))
    }
}

#[derive(Clone, Debug)]
pub struct HfBpeTokenizer {
    pub token_to_id: FxHashMap<String, usize>,
    pub id_to_token: FxHashMap<usize, String>,
    pub merge_ranks: FxHashMap<(String, String), usize>,
    pub special_ids: FxHashMap<String, usize>,
    pub byte_encoder: FxHashMap<u8, char>,
    pub byte_decoder: FxHashMap<char, u8>,
    pub pad_id: usize,
    pub bos_id: usize,
    pub eos_id: usize,
    pub unk_id: Option<usize>,
}

impl HfBpeTokenizer {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path.as_ref())
            .map_err(|error| Error(format!("failed to read tokenizer.json: {error}")))?;
        Self::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<Self> {
        let parsed: TokenizerFile = serde_json::from_str(text)
            .map_err(|error| Error(format!("tokenizer.json parse: {error}")))?;
        let mut token_to_id = FxHashMap::default();
        let mut id_to_token = FxHashMap::default();
        for (token, id) in parsed.model.vocab {
            token_to_id.insert(token.clone(), id as usize);
            id_to_token.insert(id as usize, token);
        }
        let mut special_ids = FxHashMap::default();
        for added in parsed.added_tokens {
            token_to_id.insert(added.content.clone(), added.id as usize);
            id_to_token.insert(added.id as usize, added.content.clone());
            if added.special || added.content.starts_with("<|") {
                special_ids.insert(added.content, added.id as usize);
            }
        }

        let mut merge_ranks = FxHashMap::default();
        for (rank, pair) in parsed.model.merges.0.into_iter().enumerate() {
            merge_ranks.insert(pair, rank);
        }

        let (byte_encoder, byte_decoder) = bytes_to_unicode();
        let lookup = |names: &[&str], fallback: usize| {
            for name in names {
                if let Some(id) = token_to_id.get(*name) {
                    return *id;
                }
            }
            fallback
        };

        Ok(Self {
            pad_id: lookup(&["<pad>", "[PAD]"], 0),
            bos_id: lookup(
                &[
                    "<|startoftranscript|>",
                    "<bos>",
                    "<s>",
                    "<start>",
                    "<|startoftext|>",
                ],
                1,
            ),
            eos_id: lookup(&["<|endoftext|>", "<eos>", "</s>", "<end>"], 2),
            unk_id: token_to_id.get("<unk>").copied().or_else(|| token_to_id.get("[UNK]").copied()),
            token_to_id,
            id_to_token,
            merge_ranks,
            special_ids,
            byte_encoder,
            byte_decoder,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len().max(self.token_to_id.len())
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<usize> {
        let mut ids = Vec::new();
        if add_special_tokens {
            ids.push(self.bos_id);
        }
        if text.is_empty() {
            if add_special_tokens {
                ids.push(self.eos_id);
            }
            return ids;
        }
        for caps in gpt2_pat().find_iter(text) {
            let piece = caps.as_str();
            let mut mapped = String::new();
            for byte in piece.as_bytes() {
                mapped.push(*self.byte_encoder.get(byte).unwrap_or(&char::from(*byte)));
            }
            for symbol in self.bpe(&mapped) {
                if let Some(&id) = self.token_to_id.get(&symbol) {
                    ids.push(id);
                } else if let Some(unk) = self.unk_id {
                    ids.push(unk);
                }
            }
        }
        if add_special_tokens {
            ids.push(self.eos_id);
        }
        ids
    }

    pub fn decode(&self, ids: &[usize], skip_special: bool) -> String {
        let mut pieces = String::new();
        for &id in ids {
            if skip_special && self.is_special(id) {
                continue;
            }
            if let Some(token) = self.id_to_token.get(&id) {
                pieces.push_str(token);
            }
        }
        let mut bytes = Vec::with_capacity(pieces.len());
        for ch in pieces.chars() {
            if let Some(byte) = self.byte_decoder.get(&ch) {
                bytes.push(*byte);
            } else {
                let mut buf = [0u8; 4];
                bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn is_special(&self, id: usize) -> bool {
        id == self.pad_id
            || id == self.bos_id
            || id == self.eos_id
            || self.special_ids.values().any(|&value| value == id)
    }

    fn bpe(&self, token: &str) -> Vec<String> {
        if token.is_empty() {
            return Vec::new();
        }
        let mut word: Vec<String> = token.chars().map(|ch| ch.to_string()).collect();
        if word.len() == 1 {
            return word;
        }
        loop {
            if word.len() < 2 {
                break;
            }
            let mut best = None;
            let mut best_rank = usize::MAX;
            for i in 0..word.len() - 1 {
                let pair = (word[i].clone(), word[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best = Some(i);
                    }
                }
            }
            let Some(i) = best else {
                break;
            };
            let merged = format!("{}{}", word[i], word[i + 1]);
            word.splice(i..=i + 1, std::iter::once(merged));
        }
        word
    }
}

fn bytes_to_unicode() -> (FxHashMap<u8, char>, FxHashMap<char, u8>) {
    let mut bs: Vec<u8> = (b'!'..=b'~').collect();
    bs.extend(0xA1u8..=0xACu8);
    bs.extend(0xAEu8..=0xFFu8);
    let mut cs: Vec<u32> = bs.iter().map(|b| *b as u32).collect();
    let mut n = 0u32;
    for b in 0u8..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut encoder = FxHashMap::default();
    let mut decoder = FxHashMap::default();
    for (b, c) in bs.into_iter().zip(cs.into_iter()) {
        let ch = char::from_u32(c).unwrap();
        encoder.insert(b, ch);
        decoder.insert(ch, b);
    }
    (encoder, decoder)
}

fn gpt2_pat() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        // GPT-2 pretokenizer without look-ahead (`regex` does not support `(?!...)`).
        // `\s+(?!\S)` is equivalent to trailing whitespace already covered by `\s+`.
        Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
            .expect("gpt2 regex")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_vocab_merges_encode_simple_ascii() {
        let json = r#"{
            "model": {
                "vocab": {"a": 10, "b": 11, "ab": 12, "<|endoftext|>": 2, "<|startoftranscript|>": 1, "<pad>": 0},
                "merges": ["a b"]
            },
            "added_tokens": [
                {"id": 0, "content": "<pad>", "special": true},
                {"id": 1, "content": "<|startoftranscript|>", "special": true},
                {"id": 2, "content": "<|endoftext|>", "special": true}
            ]
        }"#;
        let tokenizer = HfBpeTokenizer::from_json(json).unwrap();
        assert_eq!(tokenizer.encode("ab", false), vec![12]);
        assert_eq!(tokenizer.decode(&[12], true), "ab");
        assert_eq!(tokenizer.bos_id, 1);
        assert_eq!(tokenizer.eos_id, 2);
    }
}
