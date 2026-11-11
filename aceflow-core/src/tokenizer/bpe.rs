use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

pub const PAD_ID: usize = 0;
pub const BOS_ID: usize = 1;
pub const EOS_ID: usize = 2;
pub const BYTE_OFFSET: usize = 3;

#[inline(always)]
fn pair_to_key(p0: usize, p1: usize) -> u64 {
    ((p0 as u64) << 32) | ((p1 as u32) as u64)
}

#[inline(always)]
fn key_to_pair(key: u64) -> (usize, usize) {
    ((key >> 32) as usize, key as u32 as usize)
}

#[derive(Clone, Debug)]
pub struct BpeTokenizer {
    pub merges: FxHashMap<u64, usize>,
    pub merge_ranks: FxHashMap<u64, usize>,
    pub vocab: FxHashMap<usize, Vec<u8>>,
}

impl BpeTokenizer {
    pub fn new(
        merges: HashMap<(usize, usize), usize>,
        merge_ranks: HashMap<(usize, usize), usize>,
    ) -> Self {
        let mut fx_merges = FxHashMap::default();
        let mut fx_merge_ranks = FxHashMap::default();
        for (pair, id) in merges {
            fx_merges.insert(pair_to_key(pair.0, pair.1), id);
        }
        for (pair, rank) in merge_ranks {
            fx_merge_ranks.insert(pair_to_key(pair.0, pair.1), rank);
        }

        let mut vocab = FxHashMap::default();
        vocab.insert(PAD_ID, b"<pad>".to_vec());
        vocab.insert(BOS_ID, b"<bos>".to_vec());
        vocab.insert(EOS_ID, b"<eos>".to_vec());
        for byte in 0..256usize {
            vocab.insert(BYTE_OFFSET + byte, vec![byte as u8]);
        }

        let mut sorted_merges: Vec<((usize, usize), usize)> = fx_merge_ranks
            .iter()
            .map(|(&key, &rank)| (key_to_pair(key), rank))
            .collect();
        sorted_merges.sort_by_key(|&(_, rank)| rank);

        for (pair, _) in sorted_merges {
            if let (Some(left), Some(right)) = (vocab.get(&pair.0), vocab.get(&pair.1)) {
                let mut combined = left.clone();
                combined.extend_from_slice(right);
                let key = pair_to_key(pair.0, pair.1);
                if let Some(&new_id) = fx_merges.get(&key) {
                    vocab.insert(new_id, combined);
                }
            }
        }

        Self {
            merges: fx_merges,
            merge_ranks: fx_merge_ranks,
            vocab,
        }
    }

    #[inline(always)]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<usize> {
        let bytes = text.as_bytes();
        if bytes.is_empty() {
            return if add_special_tokens {
                vec![BOS_ID, EOS_ID]
            } else {
                Vec::new()
            };
        }

        let mut tokens: Vec<usize> = bytes
            .iter()
            .map(|&byte| byte as usize + BYTE_OFFSET)
            .collect();

        while tokens.len() >= 2 {
            let mut best_pair = None;
            let mut min_rank = usize::MAX;

            for i in 0..tokens.len() - 1 {
                let key = pair_to_key(tokens[i], tokens[i + 1]);
                if let Some(&rank) = self.merge_ranks.get(&key) {
                    if rank < min_rank {
                        min_rank = rank;
                        best_pair = Some((tokens[i], tokens[i + 1], key));
                    }
                }
            }

            if let Some((p0, p1, key)) = best_pair {
                let new_id = self.merges[&key];
                let mut write_idx = 0;
                let mut i = 0;
                let n = tokens.len();
                while i < n {
                    if i < n - 1 && tokens[i] == p0 && tokens[i + 1] == p1 {
                        tokens[write_idx] = new_id;
                        i += 2;
                    } else {
                        tokens[write_idx] = tokens[i];
                        i += 1;
                    }
                    write_idx += 1;
                }
                tokens.truncate(write_idx);
            } else {
                break;
            }
        }

        if add_special_tokens {
            let mut final_tokens = Vec::with_capacity(tokens.len() + 2);
            final_tokens.push(BOS_ID);
            final_tokens.extend(tokens);
            final_tokens.push(EOS_ID);
            final_tokens
        } else {
            tokens
        }
    }

    pub fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Vec<Vec<usize>> {
        texts
            .par_iter()
            .map(|text| self.encode(text, add_special_tokens))
            .collect()
    }

    pub fn encode_batch_strings(
        &self,
        texts: &[String],
        add_special_tokens: bool,
    ) -> Vec<Vec<usize>> {
        texts
            .par_iter()
            .map(|text| self.encode(text.as_str(), add_special_tokens))
            .collect()
    }

    pub fn encode_batch_strings_flat(
        &self,
        texts: &[String],
        add_special_tokens: bool,
    ) -> Vec<i64> {
        let nested = self.encode_batch_strings(texts, add_special_tokens);
        let total = nested.iter().map(Vec::len).sum();
        let mut flat = Vec::with_capacity(total);
        for sequence in nested {
            for token in sequence {
                flat.push(token as i64);
            }
        }
        flat
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        let mut raw_bytes = Vec::new();
        for &token in tokens {
            if token == PAD_ID || token == BOS_ID || token == EOS_ID {
                continue;
            }
            if let Some(bytes) = self.vocab.get(&token) {
                raw_bytes.extend_from_slice(bytes);
            }
        }
        String::from_utf8_lossy(&raw_bytes).into_owned()
    }

    pub fn save<P: AsRef<Path>>(&self, path_dir: P) -> std::io::Result<()> {
        let dir = path_dir.as_ref();
        std::fs::create_dir_all(dir)?;

        let mut file = File::create(dir.join("merges.txt"))?;
        writeln!(file, "# MallowTokenizer Merges")?;

        let mut sorted_merges: Vec<((usize, usize), usize)> = self
            .merge_ranks
            .iter()
            .map(|(&key, &rank)| (key_to_pair(key), rank))
            .collect();
        sorted_merges.sort_by_key(|&(_, rank)| rank);
        for (pair, _) in sorted_merges {
            let key = pair_to_key(pair.0, pair.1);
            if let Some(&new_id) = self.merges.get(&key) {
                writeln!(file, "{} {} {}", pair.0, pair.1, new_id)?;
            }
        }
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path_dir: P) -> std::io::Result<Self> {
        let file = File::open(path_dir.as_ref().join("merges.txt"))?;
        let reader = BufReader::new(file);
        let mut merges = HashMap::new();
        let mut merge_ranks = HashMap::new();
        let mut rank = 0;

        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() == 3 {
                if let (Ok(p0), Ok(p1), Ok(new_id)) = (
                    parts[0].parse::<usize>(),
                    parts[1].parse::<usize>(),
                    parts[2].parse::<usize>(),
                ) {
                    merges.insert((p0, p1), new_id);
                    merge_ranks.insert((p0, p1), rank);
                    rank += 1;
                }
            }
        }
        Ok(Self::new(merges, merge_ranks))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_ids_and_special_tokens_are_stable() {
        let tokenizer = BpeTokenizer::new(HashMap::new(), HashMap::new());
        assert_eq!(tokenizer.encode("A", false), vec![68]);
        assert_eq!(tokenizer.encode("A", true), vec![BOS_ID, 68, EOS_ID]);
        assert_eq!(tokenizer.decode(&[BOS_ID, 68, EOS_ID]), "A");
        assert_eq!(EOS_ID, 2);
    }

    #[test]
    fn ranked_merges_match_legacy_bpe() {
        let mut merges = HashMap::new();
        let mut ranks = HashMap::new();
        merges.insert((100, 101), 259);
        ranks.insert((100, 101), 0);
        let tokenizer = BpeTokenizer::new(merges, ranks);
        assert_eq!(tokenizer.encode("ab", false), vec![259]);
        assert_eq!(tokenizer.decode(&[259]), "ab");
    }
}
