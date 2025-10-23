import numpy as np
from tqdm import tqdm

class SequenceDataLoader:
    """Data loader for sequence pairs"""
    
    def __init__(self, source_sequences, target_sequences, batch_size=32, shuffle=True):
        self.source_sequences = source_sequences
        self.target_sequences = target_sequences
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(source_sequences)
        
        assert len(source_sequences) == len(target_sequences), "Source and target must have same length"
        
    def __iter__(self):
        """Create iterator over batches"""
        indices = np.arange(self.num_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_src = [self.source_sequences[i] for i in batch_indices]
            batch_tgt = [self.target_sequences[i] for i in batch_indices]
            
            yield batch_src, batch_tgt
    
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size