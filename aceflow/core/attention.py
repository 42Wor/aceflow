import numpy as np
from typing import Tuple

class Attention:
    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim
        self.W_a = np.random.randn(2 * hidden_dim, hidden_dim) * 0.01
        self.v_a = np.random.randn(hidden_dim, 1) * 0.01
    
    def forward(self, decoder_hidden: np.ndarray, encoder_outputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        decoder_hidden: (batch_size, hidden_dim)
        encoder_outputs: (batch_size, seq_len, hidden_dim)
        """
        self.decoder_hidden = decoder_hidden
        self.encoder_outputs = encoder_outputs
        
        batch_size, seq_len, hidden_dim = encoder_outputs.shape
        
        # Repeat decoder hidden state for each encoder time step
        decoder_hidden_repeated = np.repeat(decoder_hidden[:, np.newaxis, :], seq_len, axis=1)
        
        # Concatenate encoder outputs with decoder hidden state
        combined = np.concatenate([encoder_outputs, decoder_hidden_repeated], axis=2)
        
        # Compute attention scores
        scores = np.tanh(np.dot(combined, self.W_a))
        scores = np.dot(scores, self.v_a).squeeze(-1)  # (batch_size, seq_len)
        
        # Apply softmax to get attention weights
        max_scores = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Compute context vector
        context_vector = np.sum(encoder_outputs * attention_weights[:, :, np.newaxis], axis=1)
        
        self.cache = (combined, scores, attention_weights)
        return context_vector, attention_weights
    
    def backward(self, dcontext: np.ndarray, dattention_weights: np.ndarray) -> np.ndarray:
        combined, scores, attention_weights = self.cache
        batch_size, seq_len, hidden_dim = self.encoder_outputs.shape
        
        # Gradient through context vector computation
        d_encoder_weighted = dcontext[:, np.newaxis, :] * attention_weights[:, :, np.newaxis]
        d_attention_weights = np.sum(dcontext[:, np.newaxis, :] * self.encoder_outputs, axis=2)
        
        if dattention_weights is not None:
            d_attention_weights += dattention_weights
        
        # Gradient through softmax
        d_scores = attention_weights * (d_attention_weights - np.sum(attention_weights * d_attention_weights, 
                                                                   axis=1, keepdims=True))
        
        # Gradient through score computation
        d_combined_scores = np.dot(d_scores[:, :, np.newaxis], self.v_a.T)
        d_v_a = np.sum(np.dot(combined, self.W_a).transpose(0, 2, 1) @ d_scores[:, :, np.newaxis], axis=0)
        
        d_combined_tanh = d_combined_scores * (1 - np.tanh(np.dot(combined, self.W_a)) ** 2)
        
        # Gradient through linear transformation
        d_W_a = np.sum(combined.transpose(0, 2, 1) @ d_combined_tanh, axis=0)
        d_combined = np.dot(d_combined_tanh, self.W_a.T)
        
        # Split gradients
        d_encoder_outputs = d_combined[:, :, :hidden_dim] + d_encoder_weighted
        d_decoder_hidden = np.sum(d_combined[:, :, hidden_dim:], axis=1)
        
        # Store gradients
        self.dW_a = d_W_a
        self.dv_a = d_v_a
        
        return d_encoder_outputs, d_decoder_hidden