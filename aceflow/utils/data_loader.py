import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
from .tokenizer import Tokenizer

class TranslationDataset(Dataset):
    def __init__(self, 
                 src_texts: List[str], 
                 tgt_texts: List[str], 
                 src_tokenizer: Tokenizer,
                 tgt_tokenizer: Tokenizer,
                 max_length: int = 50):
        
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length
        
        # Set tokenizer max length
        self.src_tokenizer.max_length = max_length
        self.tgt_tokenizer.max_length = max_length
        
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        
        # Encode with new tokenizer
        src_encoded = self.src_tokenizer.encode(src_text, return_tensors='list')
        tgt_encoded = self.tgt_tokenizer.encode(tgt_text, return_tensors='list')
        
        return {
            'src': torch.tensor(src_encoded, dtype=torch.long),
            'tgt': torch.tensor(tgt_encoded, dtype=torch.long),
            'src_text': src_text,
            'tgt_text': tgt_text
        }

def create_data_loader(src_texts: List[str], 
                      tgt_texts: List[str], 
                      src_tokenizer: Tokenizer,
                      tgt_tokenizer: Tokenizer,
                      batch_size: int = 32, 
                      max_length: int = 50, 
                      shuffle: bool = True) -> DataLoader:
    
    dataset = TranslationDataset(src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)