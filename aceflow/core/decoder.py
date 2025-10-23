import numpy as np
from .attention import Attention

class Decoder:
    """GRU Decoder with Attention Mechanism"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.1):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layer
        self.embedding = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # Attention mechanism
        self.attention = Attention(hidden_dim * 2)  # *2 for bidirectional encoder
        
        # GRU weights
        self.W_z = [np.random.randn(hidden_dim, hidden_dim + embedding_dim + hidden_dim * 2) * 0.01 
                   for _ in range(num_layers)]
        self.W_r = [np.random.randn(hidden_dim, hidden_dim + embedding_dim + hidden_dim * 2) * 0.01 
                   for _ in range(num_layers)]
        self.W_h = [np.random.randn(hidden_dim, hidden_dim + embedding_dim + hidden_dim * 2) * 0.01 
                   for _ in range(num_layers)]
        
        # Output projection
        self.W_out = np.random.randn(vocab_size, hidden_dim + hidden_dim * 2) * 0.01
        self.b_out = np.zeros(vocab_size)
        
        # Layer normalization
        self.ln = [{'gamma': np.ones(hidden_dim), 'beta': np.zeros(hidden_dim)} for _ in range(num_layers)]
        
    def gru_cell(self, x, h_prev, context, W_z, W_r, W_h, layer_norm):
        """GRU cell with context vector"""
        # Concatenate input, previous hidden state, and context
        x_h_context = np.concatenate([x, h_prev, context])
        
        # Update gate
        z = self._sigmoid(W_z @ x_h_context)
        
        # Reset gate
        r = self._sigmoid(W_r @ x_h_context)
        
        # Candidate hidden state
        x_rh_context = np.concatenate([x, r * h_prev, context])
        h_tilde = np.tanh(W_h @ x_rh_context)
        
        # New hidden state
        h_new = (1 - z) * h_prev + z * h_tilde
        
        # Layer normalization
        h_new = self._layer_norm(h_new, layer_norm['gamma'], layer_norm['beta'])
        
        return h_new
    
    def forward(self, input_token, hidden_states, encoder_outputs):
        """
        Single decoding step
        """
        # Embedding lookup
        embedded = self.embedding[input_token]
        
        # Initialize context for first layer
        context = np.zeros(self.hidden_dim * 2)
        
        # Update hidden states through layers
        new_hidden_states = []
        x = embedded
        
        for layer in range(self.num_layers):
            h_prev = hidden_states[layer]
            
            # Get context vector using attention
            if layer == 0:  # Only compute attention once
                context, attention_weights = self.attention.forward(h_prev, encoder_outputs)
            
            h_new = self.gru_cell(
                x, h_prev, context,
                self.W_z[layer], self.W_r[layer], self.W_h[layer],
                self.ln[layer]
            )
            
            new_hidden_states.append(h_new)
            x = h_new  # Output becomes input to next layer
        
        # Output projection
        output_input = np.concatenate([new_hidden_states[-1], context])
        logits = self.W_out @ output_input + self.b_out
        
        # Softmax for probabilities
        probs = self._softmax(logits)
        
        return probs, new_hidden_states, attention_weights, context
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def _layer_norm(self, x, gamma, beta, eps=1e-5):
        mean = np.mean(x)
        std = np.std(x)
        return gamma * (x - mean) / (std + eps) + beta