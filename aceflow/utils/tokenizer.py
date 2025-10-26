import json
from collections import Counter
import pickle

class Tokenizer:
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>'}
        self.vocab_size = 4
        
    def fit(self, texts, max_vocab_size=10000):
        """Build vocabulary from texts"""
        counter = Counter()
        for text in texts:
            if isinstance(text, str):
                tokens = text.split()
            else:
                tokens = text
            counter.update(tokens)
        
        # Keep most common words
        most_common = counter.most_common(max_vocab_size - self.vocab_size)
        
        for word, _ in most_common:
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
    
    def encode(self, text, add_special_tokens=True):
        """Convert text to indices"""
        if isinstance(text, str):
            tokens = text.split()
        else:
            tokens = text
            
        indices = []
        if add_special_tokens:
            indices.append(self.word2idx['<start>'])
        
        for token in tokens:
            indices.append(self.word2idx.get(token, self.word2idx['<unk>']))
        
        if add_special_tokens:
            indices.append(self.word2idx['<end>'])
            
        return indices
    
    def decode(self, indices, remove_special_tokens=True):
        """Convert indices to text"""
        tokens = []
        for idx in indices:
            if remove_special_tokens and idx in [0, 1, 2]:  # Skip special tokens
                continue
            tokens.append(self.idx2word.get(idx, '<unk>'))
        return ' '.join(tokens)
    
    def save(self, filepath):
        """Save tokenizer to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'vocab_size': self.vocab_size
            }, f)
    
    @classmethod
    def load(cls, filepath):
        """Load tokenizer from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls()
        tokenizer.word2idx = data['word2idx']
        tokenizer.idx2word = data['idx2word']
        tokenizer.vocab_size = data['vocab_size']
        return tokenizer