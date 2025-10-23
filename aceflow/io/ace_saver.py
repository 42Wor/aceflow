import numpy as np
import pickle
import zlib
import os
from typing import Dict

class ACEFormat:
    @staticmethod
    def save_model(model, filepath: str) -> None:
        """Save model to .ace file"""
        model_data = {
            'src_vocab_size': model.src_vocab_size,
            'tgt_vocab_size': model.tgt_vocab_size,
            'embedding_dim': model.embedding_dim,
            'hidden_dim': model.hidden_dim,
            'num_layers': model.num_layers,
            'use_attention': model.use_attention,
            'params': model.get_params()
        }
        
        # Compress and save
        compressed_data = zlib.compress(pickle.dumps(model_data))
        
        with open(filepath, 'wb') as f:
            f.write(b'ACEFORMATv1.0')
            f.write(len(compressed_data).to_bytes(8, 'big'))
            f.write(compressed_data)
    
    @staticmethod
    def load_model(filepath: str) -> 'Seq2Seq':
        """Load model from .ace file"""
        with open(filepath, 'rb') as f:
            magic = f.read(12)  # Read magic bytes
            if magic != b'ACEFORMATv1.0':
                raise ValueError("Invalid ACE file format")
            
            data_len = int.from_bytes(f.read(8), 'big')
            compressed_data = f.read(data_len)
        
        # Decompress and load
        model_data = pickle.loads(zlib.decompress(compressed_data))
        
        # Create model
        from ..core.model import Seq2Seq
        model = Seq2Seq(
            src_vocab_size=model_data['src_vocab_size'],
            tgt_vocab_size=model_data['tgt_vocab_size'],
            embedding_dim=model_data['embedding_dim'],
            hidden_dim=model_data['hidden_dim'],
            num_layers=model_data['num_layers'],
            use_attention=model_data['use_attention']
        )
        
        # Set parameters
        model.set_params(model_data['params'])
        
        return model