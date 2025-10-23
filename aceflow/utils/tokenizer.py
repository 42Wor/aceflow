import numpy as np
from typing import List, Dict, Optional

class Tokenizer:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.vocab_size = 4
    
    def build_vocab(self, texts: List[str], min_freq: int = 1) -> None:
        """Build vocabulary from list of texts"""
        word_freq = {}
        for text in texts:
            for word in text.split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Add words meeting frequency threshold
        for word, freq in word_freq.items():
            if freq >= min_freq and word not in self.word2idx:
                idx = self.vocab_size
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                self.vocab_size += 1
    
    def encode(self, text: str, max_length: Optional[int] = None) -> np.ndarray:
        """Encode text to sequence of indices"""
        tokens = [self.word2idx.get(word, 3) for word in text.split()]
        tokens = [1] + tokens + [2]  # Add SOS and EOS
        
        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length-1] + [2]
            else:
                tokens = tokens + [0] * (max_length - len(tokens))
        
        return np.array(tokens, dtype=np.int32)
    
    def decode(self, indices: np.ndarray) -> str:
        """Decode indices to text"""
        tokens = []
        for idx in indices:
            if idx == 2:  # EOS
                break
            if idx not in [0, 1]:  # Skip PAD and SOS
                tokens.append(self.idx2word.get(idx, '<UNK>'))
        return ' '.join(tokens)