//! Whitespace word-level tokenizer used by AceFlow.

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Pre,
    Post,
}

impl Side {
    pub fn parse(value: &str) -> Self {
        if value.eq_ignore_ascii_case("pre") {
            Self::Pre
        } else {
            Self::Post
        }
    }
}

#[derive(Clone, Debug)]
pub struct WordLevelTokenizer {
    pub word2idx: FxHashMap<String, usize>,
    pub idx2word: FxHashMap<usize, String>,
    pub pad_id: usize,
    pub bos_id: usize,
    pub eos_id: usize,
    pub unk_id: usize,
    pub max_length: usize,
    pub padding: Side,
    pub truncation: Side,
}

impl WordLevelTokenizer {
    pub fn new(
        vocab: HashMap<String, usize>,
        specials: &HashMap<String, usize>,
        max_length: usize,
        padding: Side,
        truncation: Side,
    ) -> Self {
        let mut word2idx = FxHashMap::default();
        for (token, id) in vocab {
            word2idx.insert(token, id);
        }
        for (token, id) in specials {
            word2idx.insert(token.clone(), *id);
        }

        let lookup = |names: &[&str], fallback: usize| -> usize {
            for name in names {
                if let Some(id) = specials.get(*name).or_else(|| word2idx.get(*name)) {
                    return *id;
                }
            }
            fallback
        };

        let pad_id = lookup(&["<pad>", "[PAD]", "<|padding|>"], 0);
        let bos_id = lookup(
            &["<start>", "<bos>", "<s>", "<|startoftranscript|>"],
            1,
        );
        let eos_id = lookup(&["<end>", "<eos>", "</s>", "<|endoftext|>"], 2);
        let unk_id = lookup(&["<unk>", "[UNK]", "<|unk|>"], 3);

        word2idx.entry("<pad>".into()).or_insert(pad_id);
        word2idx.entry("<start>".into()).or_insert(bos_id);
        word2idx.entry("<end>".into()).or_insert(eos_id);
        word2idx.entry("<unk>".into()).or_insert(unk_id);

        let mut idx2word = FxHashMap::default();
        for (token, id) in &word2idx {
            idx2word.insert(*id, token.clone());
        }

        Self {
            word2idx,
            idx2word,
            pad_id,
            bos_id,
            eos_id,
            unk_id,
            max_length: max_length.max(1),
            padding,
            truncation,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.word2idx.len()
    }

    pub fn encode_word(&self, word: &str) -> usize {
        self.word2idx
            .get(word)
            .copied()
            .unwrap_or(self.unk_id)
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<usize> {
        let mut ids = Vec::new();
        if add_special_tokens {
            ids.push(self.bos_id);
        }
        if !text.is_empty() {
            for token in text.split_whitespace() {
                ids.push(self.encode_word(token));
            }
        }
        if add_special_tokens {
            ids.push(self.eos_id);
        }
        self.truncate_and_pad(ids)
    }

    pub fn encode_batch(&self, texts: &[String], add_special_tokens: bool) -> Vec<Vec<usize>> {
        texts
            .par_iter()
            .map(|text| self.encode(text, add_special_tokens))
            .collect()
    }

    pub fn decode(&self, ids: &[usize], skip_special: bool) -> String {
        let mut tokens = Vec::new();
        for &id in ids {
            if id == self.pad_id {
                continue;
            }
            if skip_special && (id == self.bos_id || id == self.eos_id) {
                continue;
            }
            if let Some(token) = self.idx2word.get(&id) {
                if skip_special && (token == "<pad>" || token == "<start>" || token == "<end>") {
                    continue;
                }
                tokens.push(token.as_str());
            }
        }
        tokens.join(" ")
    }

    fn truncate_and_pad(&self, mut ids: Vec<usize>) -> Vec<usize> {
        if ids.len() > self.max_length {
            ids = match self.truncation {
                Side::Post => ids[..self.max_length].to_vec(),
                Side::Pre => ids[ids.len() - self.max_length..].to_vec(),
            };
        }
        if ids.len() < self.max_length {
            let pad = vec![self.pad_id; self.max_length - ids.len()];
            ids = match self.padding {
                Side::Post => {
                    ids.extend(pad);
                    ids
                }
                Side::Pre => {
                    let mut out = pad;
                    out.extend(ids);
                    out
                }
            };
        }
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aceflow_specials_and_padding_match_legacy() {
        let mut vocab = HashMap::new();
        vocab.insert("<pad>".into(), 0);
        vocab.insert("<start>".into(), 1);
        vocab.insert("<end>".into(), 2);
        vocab.insert("<unk>".into(), 3);
        vocab.insert("hello".into(), 5);
        vocab.insert("world".into(), 6);
        let tokenizer = WordLevelTokenizer::new(vocab, &HashMap::new(), 6, Side::Post, Side::Post);
        let ids = tokenizer.encode("hello world", true);
        assert_eq!(ids, vec![1, 5, 6, 2, 0, 0]);
        assert_eq!(tokenizer.decode(&ids, true), "hello world");
        assert_eq!(tokenizer.encode_word("missing"), 3);
    }
}
