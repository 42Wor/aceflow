import json
import pickle
from collections import Counter
import os
from typing import List, Dict, Union, Optional

class Vocabulary:
    def __init__(self, name: str = "vocabulary"):
        self.name = name
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        self.special_tokens = {
            '<pad>': 0,
            '<start>': 1, 
            '<end>': 2,
            '<unk>': 3,
            '<mask>': 4
        }
        self._build_special_tokens()
        self.config = {
            'name': name,
            'vocab_size': len(self.special_tokens),
            'max_size': 50000,
            'min_freq': 2,
            'special_tokens': list(self.special_tokens.keys())
        }
    
    def _build_special_tokens(self):
        """Initialize special tokens"""
        for token, idx in self.special_tokens.items():
            self.word2idx[token] = idx
            self.idx2word[idx] = token
    
    def add_word(self, word: str, freq: int = 1):
        """Add a word to vocabulary"""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.word_freq[word] += freq
    
    def add_words(self, words: List[str]):
        """Add multiple words to vocabulary"""
        for word in words:
            self.add_word(word)
    
    def build_from_texts(self, texts: List[str], max_size: int = 50000, min_freq: int = 2):
        """Build vocabulary from list of texts"""
        self.config['max_size'] = max_size
        self.config['min_freq'] = min_freq
        
        # Count all words
        counter = Counter()
        for text in texts:
            if isinstance(text, str):
                tokens = text.split()
            else:
                tokens = text
            counter.update(tokens)
        
        # Filter by frequency and size
        valid_words = [word for word, count in counter.most_common(max_size) 
                      if count >= min_freq and word not in self.special_tokens]
        
        # Add valid words to vocabulary
        for word in valid_words:
            self.add_word(word, counter[word])
        
        self.config['vocab_size'] = len(self.word2idx)
        return self
    
    def encode_word(self, word: str) -> int:
        """Encode a single word to index"""
        return self.word2idx.get(word, self.special_tokens['<unk>'])
    
    def decode_idx(self, idx: int) -> str:
        """Decode a single index to word"""
        return self.idx2word.get(idx, '<unk>')
    
    def __len__(self) -> int:
        return len(self.word2idx)
    
    def __contains__(self, word: str) -> bool:
        return word in self.word2idx
    
    def get_frequency(self, word: str) -> int:
        """Get frequency of a word"""
        return self.word_freq.get(word, 0)
    
    def most_common(self, n: int = 10) -> List[str]:
        """Get n most common words"""
        return [word for word, _ in self.word_freq.most_common(n)]
    
    def save(self, folder_path: str):
        """Save vocabulary to folder"""
        os.makedirs(folder_path, exist_ok=True)
        
        # Save main data
        data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_freq': dict(self.word_freq),
            'config': self.config
        }
        
        with open(os.path.join(folder_path, 'vocabulary.pkl'), 'wb') as f:
            pickle.dump(data, f)
        
        # Save human-readable info
        info = {
            'name': self.config['name'],
            'vocab_size': len(self),
            'special_tokens': self.config['special_tokens'],
            'most_common_words': self.most_common(20)
        }
        
        with open(os.path.join(folder_path, 'vocabulary_info.json'), 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        print(f"Vocabulary saved to {folder_path}")
    
    @classmethod
    def load(cls, folder_path: str) -> 'Vocabulary':
        """Load vocabulary from folder"""
        with open(os.path.join(folder_path, 'vocabulary.pkl'), 'rb') as f:
            data = pickle.load(f)
        
        vocab = cls(name=data['config']['name'])
        vocab.word2idx = data['word2idx']
        vocab.idx2word = data['idx2word']
        vocab.word_freq = Counter(data['word_freq'])
        vocab.config = data['config']
        
        return vocab
    
    def info(self) -> Dict:
        """Get vocabulary information"""
        return {
            'name': self.config['name'],
            'total_words': len(self),
            'special_tokens': self.config['special_tokens'],
            'most_common_words': self.most_common(10),
            'config': self.config
        }
    
    def __str__(self) -> str:
        info = self.info()
        return (f"Vocabulary(name='{info['name']}', size={info['total_words']}, "
                f"special_tokens={info['special_tokens']})")
    
    def __repr__(self) -> str:
        return self.__str__()