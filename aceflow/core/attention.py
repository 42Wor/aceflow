import math
import numpy as np

class Attention:
    """Bahdanau Attention Mechanism"""
    
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.W1 = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.V = np.random.randn(hidden_dim) * 0.01
        
    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden: (hidden_dim,)
        encoder_outputs: (seq_len, hidden_dim)
        """
        # Add batch dimension for broadcasting
        decoder_hidden = decoder_hidden[np.newaxis, :]  # (1, hidden_dim)
        
        # Score calculation
        scores = np.zeros(len(encoder_outputs))
        for i, encoder_output in enumerate(encoder_outputs):
            # encoder_output: (hidden_dim,)
            score = self.V @ np.tanh(self.W1 @ encoder_output + self.W2 @ decoder_hidden.T)
            scores[i] = score
            
        # Attention weights
        attention_weights = self._softmax(scores)
        
        # Context vector
        context_vector = np.zeros(self.hidden_dim)
        for i, weight in enumerate(attention_weights):
            context_vector += weight * encoder_outputs[i]
            
        return context_vector, attention_weights
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

class MultiHeadAttention:
    """Multi-Head Self Attention"""
    
    def __init__(self, hidden_dim, num_heads=8):
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Query, Key, Value projections
        self.W_q = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.W_k = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.W_v = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.W_o = np.random.randn(hidden_dim, hidden_dim) * 0.01
        
    def forward(self, x, mask=None):
        """
        x: (seq_len, hidden_dim)
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Linear projections
        Q = x @ self.W_q  # (seq_len, hidden_dim)
        K = x @ self.W_k  # (seq_len, hidden_dim)
        V = x @ self.W_v  # (seq_len, hidden_dim)
        
        # Reshape for multi-head
        Q = Q.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        
        # Scaled dot-product attention
        scores = Q @ K.transpose(0, 2, 1) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
            
        attention_weights = self._softmax(scores, axis=-1)
        output = attention_weights @ V
        
        # Concatenate heads
        output = output.transpose(1, 0, 2).reshape(seq_len, self.hidden_dim)
        
        # Final linear projection
        output = output @ self.W_o
        
        return output, attention_weights
    
    def _softmax(self, x, axis):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)