import pickle
import zlib
import json
import numpy as np
from datetime import datetime

def save_model(model, filepath, metadata=None):
    """
    Save Seq2Seq model to .ace format
    
    Args:
        model: Seq2Seq model instance
        filepath: Path to save the model
        metadata: Additional metadata to store
    """
    # Prepare model data
    model_data = {
        'parameters': model.get_parameters(),
        'training_history': model.training_history,
        'metadata': metadata or {}
    }
    
    # Add system metadata
    model_data['metadata']['saved_at'] = datetime.now().isoformat()
    model_data['metadata']['version'] = '1.0.0'
    model_data['metadata']['model_type'] = 'Seq2Seq'
    
    # Convert numpy arrays to lists for JSON serialization where needed
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    # Create ACE file structure
    ace_data = {
        'header': {
            'format': 'ACE',
            'version': '1.0',
            'created': datetime.now().isoformat()
        },
        'model_config': {
            'src_vocab_size': model.src_vocab_size,
            'tgt_vocab_size': model.tgt_vocab_size,
            'embedding_dim': model.embedding_dim,
            'hidden_dim': model.hidden_dim,
            'num_layers': model.num_layers
        },
        'parameters': convert_numpy(model_data['parameters']),
        'training_history': model_data['training_history'],
        'metadata': model_data['metadata']
    }
    
    # Serialize and compress
    serialized_data = pickle.dumps(ace_data)
    compressed_data = zlib.compress(serialized_data)
    
    # Write to file
    with open(filepath, 'wb') as f:
        # Write magic number
        f.write(b'ACE1')
        # Write compressed data
        f.write(compressed_data)
    
    print(f"Model saved successfully to {filepath}")

def load_model(filepath):
    """
    Load Seq2Seq model from .ace format
    
    Args:
        filepath: Path to .ace file
    
    Returns:
        Seq2Seq model instance
    """
    with open(filepath, 'rb') as f:
        # Check magic number
        magic = f.read(4)
        if magic != b'ACE1':
            raise ValueError("Invalid ACE file format")
        
        # Read and decompress data
        compressed_data = f.read()
        serialized_data = zlib.decompress(compressed_data)
        ace_data = pickle.loads(serialized_data)
    
    # Extract model configuration
    config = ace_data['model_config']
    
    # Create new model instance
    from ..core.model import Seq2Seq
    model = Seq2Seq(
        src_vocab_size=config['src_vocab_size'],
        tgt_vocab_size=config['tgt_vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers']
    )
    
    # Convert lists back to numpy arrays
    def restore_numpy(obj):
        if isinstance(obj, dict):
            if all(isinstance(k, str) and k.isdigit() for k in obj.keys()):
                # This was likely a numpy array
                return np.array([obj[str(i)] for i in range(len(obj))])
            else:
                return {k: restore_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            if all(isinstance(item, (int, float)) for item in obj):
                return np.array(obj)
            else:
                return [restore_numpy(item) for item in obj]
        else:
            return obj
    
    # Restore parameters
    restored_params = restore_numpy(ace_data['parameters'])
    model.set_parameters(restored_params)
    
    # Restore training history
    model.training_history = ace_data.get('training_history', {})
    
    print(f"Model loaded successfully from {filepath}")
    print(f"Metadata: {ace_data.get('metadata', {})}")
    
    return model