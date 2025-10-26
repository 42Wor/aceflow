import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, max_length=50):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        
        src_encoded = self.src_tokenizer.encode(src_text)
        tgt_encoded = self.tgt_tokenizer.encode(tgt_text)
        
        # Pad sequences
        src_padded = self.pad_sequence(src_encoded, self.max_length)
        tgt_padded = self.pad_sequence(tgt_encoded, self.max_length)
        
        return {
            'src': torch.tensor(src_padded, dtype=torch.long),
            'tgt': torch.tensor(tgt_padded, dtype=torch.long)
        }
    
    def pad_sequence(self, sequence, max_length):
        if len(sequence) < max_length:
            sequence = sequence + [0] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
        return sequence

def create_data_loader(src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, 
                      batch_size=32, max_length=50, shuffle=True):
    dataset = TranslationDataset(src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)