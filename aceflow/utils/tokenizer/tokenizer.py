import os
import json
import pickle
from typing import List, Dict, Union, Optional, Tuple
from .vocabulary import Vocabulary
from .preprocessor import Preprocessor

class Tokenizer:
    def __init__(self, 
                 name: str = "tokenizer",
                 language: str = 'english',
                 max_length: int = 100,
                 padding: str = 'post',
                 truncation: str = 'post'):
        
        self.name = name
        self.vocab = Vocabulary(name=f"{name}_vocab")
        self.preprocessor = Preprocessor(language)
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        
        self.config = {
            'name': name,
            'language': language,
            'max_length': max_length,
            'padding': padding,
            'truncation': truncation,
            'vocab_size': len(self.vocab),
            'preprocessor_steps': self.preprocessor.config['steps']
        }
    
    def fit(self, 
            texts: List[str], 
            max_vocab_size: int = 50000,
            min_freq: int = 2,
            preprocess: bool = True) -> 'Tokenizer':
        """Fit tokenizer on texts"""
        
        # Preprocess texts if requested
        if preprocess:
            texts = self.preprocessor.process_batch(texts)
        
        # Build vocabulary
        self.vocab.build_from_texts(texts, max_vocab_size, min_freq)
        
        # Update config
        self.config.update({
            'vocab_size': len(self.vocab),
            'max_vocab_size': max_vocab_size,
            'min_freq': min_freq
        })
        
        return self
    
    def encode(self, 
               text: str, 
               add_special_tokens: bool = True,
               preprocess: bool = True,
               return_tensors: Optional[str] = None) -> Union[List[int], Dict]:
        """Encode text to indices"""
        
        # Preprocess
        if preprocess:
            text = self.preprocessor(text)
        
        # Tokenize
        tokens = text.split()
        
        # Convert to indices
        indices = []
        if add_special_tokens:
            indices.append(self.vocab.special_tokens['<start>'])
        
        for token in tokens:
            indices.append(self.vocab.encode_word(token))
        
        if add_special_tokens:
            indices.append(self.vocab.special_tokens['<end>'])
        
        # Apply truncation
        if len(indices) > self.max_length:
            if self.truncation == 'post':
                indices = indices[:self.max_length]
            else:  # pre
                indices = indices[-self.max_length:]
        
        # Apply padding
        if len(indices) < self.max_length:
            pad_id = self.vocab.special_tokens['<pad>']
            padding = [pad_id] * (self.max_length - len(indices))
            if self.padding == 'post':
                indices = indices + padding
            else:  # pre
                indices = padding + indices
        
        if return_tensors == 'list':
            return indices
        else:
            return {
                'input_ids': indices,
                'attention_mask': [1 if idx != self.vocab.special_tokens['<pad>'] else 0 for idx in indices],
                'token_count': len([idx for idx in indices if idx != self.vocab.special_tokens['<pad>']])
            }
    
    def encode_batch(self, 
                    texts: List[str],
                    add_special_tokens: bool = True,
                    preprocess: bool = True) -> List[Dict]:
        """Encode a batch of texts"""
        return [self.encode(text, add_special_tokens, preprocess) for text in texts]
    
    def decode(self, 
               indices: List[int],
               remove_special_tokens: bool = True,
               skip_padding: bool = True) -> str:
        """Decode indices to text"""
        
        tokens = []
        for idx in indices:
            if skip_padding and idx == self.vocab.special_tokens['<pad>']:
                continue
            if remove_special_tokens and idx in [self.vocab.special_tokens['<start>'], 
                                               self.vocab.special_tokens['<end>']]:
                continue
            token = self.vocab.decode_idx(idx)
            if token not in ['<pad>', '<start>', '<end>'] or not remove_special_tokens:
                tokens.append(token)
        
        return ' '.join(tokens)
    
    def decode_batch(self, 
                    batch_indices: List[List[int]],
                    remove_special_tokens: bool = True,
                    skip_padding: bool = True) -> List[str]:
        """Decode a batch of indices"""
        return [self.decode(indices, remove_special_tokens, skip_padding) 
                for indices in batch_indices]
    
    def tokenize(self, text: str, preprocess: bool = True) -> List[str]:
        """Tokenize text without converting to indices"""
        if preprocess:
            text = self.preprocessor(text)
        return text.split()
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping"""
        return self.vocab.word2idx.copy()
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)
    
    def add_special_tokens(self, tokens: Dict[str, int]):
        """Add special tokens"""
        # Note: This should be used carefully as it changes indices
        for token, idx in tokens.items():
            if token not in self.vocab.special_tokens:
                self.vocab.special_tokens[token] = idx
                self.vocab.word2idx[token] = idx
                self.vocab.idx2word[idx] = token
    
    def save(self, folder_path: str):
        """Save tokenizer to folder"""
        os.makedirs(folder_path, exist_ok=True)
        
        # Save tokenizer config
        with open(os.path.join(folder_path, 'tokenizer_config.json'), 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        # Save vocabulary
        vocab_folder = os.path.join(folder_path, 'vocabulary')
        self.vocab.save(vocab_folder)
        
        # Save preprocessor config
        with open(os.path.join(folder_path, 'preprocessor_config.json'), 'w', encoding='utf-8') as f:
            json.dump(self.preprocessor.config, f, indent=2, ensure_ascii=False)
        
        # Save tokenizer info
        info = self.info()
        with open(os.path.join(folder_path, 'tokenizer_info.json'), 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        print(f"Tokenizer saved to {folder_path}")
    
    @classmethod
    def load(cls, folder_path: str) -> 'Tokenizer':
        """Load tokenizer from folder"""
        # Load config
        with open(os.path.join(folder_path, 'tokenizer_config.json'), 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Create tokenizer instance
        tokenizer = cls(
            name=config['name'],
            language=config['language'],
            max_length=config['max_length'],
            padding=config['padding'],
            truncation=config['truncation']
        )
        
        # Load vocabulary
        vocab_folder = os.path.join(folder_path, 'vocabulary')
        tokenizer.vocab = Vocabulary.load(vocab_folder)
        
        # Load preprocessor config
        with open(os.path.join(folder_path, 'preprocessor_config.json'), 'r', encoding='utf-8') as f:
            preprocessor_config = json.load(f)
        
        tokenizer.preprocessor.config = preprocessor_config
        tokenizer.config = config
        
        return tokenizer
    
    def info(self) -> Dict:
        """Get tokenizer information"""
        vocab_info = self.vocab.info()
        preprocessor_info = self.preprocessor.info()
        
        return {
            'name': self.name,
            'vocabulary': vocab_info,
            'preprocessor': preprocessor_info,
            'config': self.config,
            'description': f"Tokenizer with {len(self.vocab)} tokens, {self.max_length} max length"
        }
    
    def __call__(self, text: str, **kwargs) -> Union[List[int], Dict]:
        """Make tokenizer callable"""
        return self.encode(text, **kwargs)
    
    def __len__(self) -> int:
        return len(self.vocab)
    
    def __str__(self) -> str:
        info = self.info()
        return (f"Tokenizer(name='{info['name']}', vocab_size={info['vocabulary']['total_words']}, "
                f"language='{info['preprocessor']['language']}')")
    
    def __repr__(self) -> str:
        return self.__str__()