import re
from collections import Counter

class Tokenizer:
    """Simple tokenizer for sequence processing"""
    
    def __init__(self, vocab_size=10000, oov_token="<UNK>", pad_token="<PAD>", 
                 start_token="<START>", end_token="<END>"):
        self.vocab_size = vocab_size
        self.oov_token = oov_token
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        
    def build_vocab(self, texts):
        """Build vocabulary from list of texts"""
        # Count words
        word_counts = Counter()
        for text in texts:
            words = self._tokenize_text(text)
            word_counts.update(words)
        
        # Most common words
        most_common = word_counts.most_common(self.vocab_size - 4)  # Reserve for special tokens
        
        # Build vocabulary
        self.word2idx = {
            self.pad_token: 0,
            self.start_token: 1,
            self.end_token: 2,
            self.oov_token: 3
        }
        
        # Add words to vocabulary
        for i, (word, count) in enumerate(most_common, start=4):
            self.word2idx[word] = i
        
        # Create reverse mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab = set(self.word2idx.keys())
    
    def _tokenize_text(self, text):
        """Tokenize text into words"""
        # Simple whitespace tokenization with lowercase
        return re.findall(r'\b\w+\b', text.lower())
    
    def encode(self, text, add_special_tokens=True):
        """Encode text to sequence of indices"""
        tokens = self._tokenize_text(text)
        
        indices = []
        if add_special_tokens:
            indices.append(self.word2idx[self.start_token])
        
        for token in tokens:
            indices.append(self.word2idx.get(token, self.word2idx[self.oov_token]))
        
        if add_special_tokens:
            indices.append(self.word2idx[self.end_token])
        
        return indices
    
    def decode(self, indices, remove_special_tokens=True):
        """Decode indices back to text"""
        tokens = []
        for idx in indices:
            if idx in self.idx2word:
                word = self.idx2word[idx]
                if remove_special_tokens and word in [self.pad_token, self.start_token, self.end_token]:
                    continue
                tokens.append(word)
        
        return ' '.join(tokens)
    
    def get_vocab_size(self):
        return len(self.word2idx)