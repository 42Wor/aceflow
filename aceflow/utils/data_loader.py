import numpy as np
from typing import Tuple, List, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from .tokenizer import Tokenizer

class DataLoader:
    def __init__(self, src_texts: List[str], tgt_texts: List[str], 
                 src_tokenizer: 'Tokenizer', tgt_tokenizer: 'Tokenizer',
                 batch_size: int = 32, max_length: int = 50):
        
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Encode all sequences
        self.src_sequences = [src_tokenizer.encode(text, max_length) for text in src_texts]
        self.tgt_sequences = [tgt_tokenizer.encode(text, max_length) for text in tgt_texts]
        
        self.num_samples = len(src_texts)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
    
    def __iter__(self):
        self.current_batch = 0
        return self
    
    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.current_batch >= self.num_batches:
            raise StopIteration
        
        start_idx = self.current_batch * self.batch_size
        end_idx = min((self.current_batch + 1) * self.batch_size, self.num_samples)
        
        src_batch = np.stack(self.src_sequences[start_idx:end_idx])
        tgt_batch = np.stack(self.tgt_sequences[start_idx:end_idx])
        
        self.current_batch += 1
        return src_batch, tgt_batch
    
    def __len__(self) -> int:
        return self.num_batches