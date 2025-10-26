import torch
import json
import h5py
import os
import zipfile
import tempfile

class AceModelSerializer:
    def __init__(self):
        self.metadata = {}
    
    def save_model(self, model, filepath):
        """Save model to .ace format (custom zip format with metadata)"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model weights
            model_path = os.path.join(temp_dir, 'model_weights.pt')
            torch.save(model.state_dict(), model_path)
            
            # Save metadata
            metadata = {
                'src_vocab_size': model.src_vocab_size,
                'tgt_vocab_size': model.tgt_vocab_size,
                'hidden_size': model.hidden_size,
                'num_layers': model.num_layers,
                'dropout': model.dropout,
                'rnn_type': model.rnn_type,
                'use_attention': model.use_attention,
                'teacher_forcing_ratio': model.teacher_forcing_ratio,
                'max_length': model.max_length
            }
            
            metadata_path = os.path.join(temp_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create .ace file (zip format)
            with zipfile.ZipFile(filepath, 'w') as zipf:
                zipf.write(model_path, 'model_weights.pt')
                zipf.write(metadata_path, 'metadata.json')
    
    def load_model(self, filepath):
        """Load model from .ace format"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract .ace file
            with zipfile.ZipFile(filepath, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # Load metadata
            metadata_path = os.path.join(temp_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Create model instance
            from ..core.model import Seq2SeqModel
            model = Seq2SeqModel(
                src_vocab_size=metadata['src_vocab_size'],
                tgt_vocab_size=metadata['tgt_vocab_size'],
                hidden_size=metadata['hidden_size'],
                num_layers=metadata['num_layers'],
                dropout=metadata['dropout'],
                rnn_type=metadata['rnn_type'],
                use_attention=metadata['use_attention'],
                teacher_forcing_ratio=metadata['teacher_forcing_ratio'],
                max_length=metadata['max_length']
            )
            
            # Load weights
            model_path = os.path.join(temp_dir, 'model_weights.pt')
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            
            return model