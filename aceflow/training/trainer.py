import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from ..core.model import Seq2Seq
from ..utils.data_loader import DataLoader
from .optimizer import AdamOptimizer

class Trainer:
    def __init__(self, model: Seq2Seq, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = AdamOptimizer(learning_rate)
    
    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute cross-entropy loss
        predictions: (batch_size, seq_len, vocab_size)
        targets: (batch_size, seq_len)
        """
        batch_size, seq_len, vocab_size = predictions.shape
        
        # Convert predictions to probabilities using softmax
        max_vals = np.max(predictions, axis=2, keepdims=True)
        exp_vals = np.exp(predictions - max_vals)
        probs = exp_vals / np.sum(exp_vals, axis=2, keepdims=True)
        
        # Compute cross-entropy loss
        loss = 0
        doutputs = np.zeros_like(predictions)
        
        for i in range(batch_size):
            for t in range(seq_len):
                target_idx = targets[i, t]
                if target_idx != 0:  # Ignore padding
                    loss += -np.log(probs[i, t, target_idx] + 1e-8)
                    doutputs[i, t, target_idx] = probs[i, t, target_idx] - 1
        
        loss /= (batch_size * seq_len)
        doutputs /= (batch_size * seq_len)
        
        return loss, doutputs
    
    def train_epoch(self, data_loader: DataLoader) -> float:
        total_loss = 0
        num_batches = len(data_loader)
        
        for src_batch, tgt_batch in tqdm(data_loader, desc="Training"):
            # Forward pass
            predictions = self.model.forward(src_batch, tgt_batch[:, :-1])
            
            # Compute loss and gradients
            loss, doutputs = self.compute_loss(predictions, tgt_batch[:, 1:])
            total_loss += loss
            
            # Backward pass
            self.model.backward(doutputs)
            
            # Update parameters
            self.update_parameters()
        
        return total_loss / num_batches
    
    def update_parameters(self):
        """Update all model parameters using Adam optimizer"""
        # Collect all parameters and gradients
        all_params = {}
        all_grads = {}
        
        # Encoder parameters
        all_params['encoder_embedding'] = self.model.encoder.embedding.params['W']
        all_grads['encoder_embedding'] = self.model.encoder.embedding.grads['W']
        
        for i, lstm in enumerate(self.model.encoder.lstm_layers):
            all_params[f'encoder_lstm_{i}_W'] = lstm.params['W']
            all_params[f'encoder_lstm_{i}_b'] = lstm.params['b']
            all_grads[f'encoder_lstm_{i}_W'] = lstm.grads['W']
            all_grads[f'encoder_lstm_{i}_b'] = lstm.grads['b']
        
        # Decoder parameters
        all_params['decoder_embedding'] = self.model.decoder.embedding.params['W']
        all_grads['decoder_embedding'] = self.model.decoder.embedding.grads['W']
        
        for i, lstm in enumerate(self.model.decoder.lstm_layers):
            all_params[f'decoder_lstm_{i}_W'] = lstm.params['W']
            all_params[f'decoder_lstm_{i}_b'] = lstm.params['b']
            all_grads[f'decoder_lstm_{i}_W'] = lstm.grads['W']
            all_grads[f'decoder_lstm_{i}_b'] = lstm.grads['b']
        
        if self.model.use_attention:
            all_params['attention_W'] = self.model.decoder.attention.W_a
            all_params['attention_v'] = self.model.decoder.attention.v_a
            all_grads['attention_W'] = self.model.decoder.attention.dW_a
            all_grads['attention_v'] = self.model.decoder.attention.dv_a
        
        all_params['output_W'] = self.model.decoder.output_layer.params['W']
        all_params['output_b'] = self.model.decoder.output_layer.params['b']
        all_grads['output_W'] = self.model.decoder.output_layer.grads['W']
        all_grads['output_b'] = self.model.decoder.output_layer.grads['b']
        
        # Update parameters
        self.optimizer.update(all_params, all_grads)