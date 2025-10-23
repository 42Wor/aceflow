import numpy as np

class Encoder:
    """Bidirectional GRU Encoder with Multi-Head Attention"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.1):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layer
        self.embedding = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # GRU weights (bidirectional)
        # Forward GRU
        self.W_z_f = [np.random.randn(hidden_dim, hidden_dim + embedding_dim) * 0.01 for _ in range(num_layers)]
        self.W_r_f = [np.random.randn(hidden_dim, hidden_dim + embedding_dim) * 0.01 for _ in range(num_layers)]
        self.W_h_f = [np.random.randn(hidden_dim, hidden_dim + embedding_dim) * 0.01 for _ in range(num_layers)]
        
        # Backward GRU
        self.W_z_b = [np.random.randn(hidden_dim, hidden_dim + embedding_dim) * 0.01 for _ in range(num_layers)]
        self.W_r_b = [np.random.randn(hidden_dim, hidden_dim + embedding_dim) * 0.01 for _ in range(num_layers)]
        self.W_h_b = [np.random.randn(hidden_dim, hidden_dim + embedding_dim) * 0.01 for _ in range(num_layers)]
        
        # Layer normalization
        self.ln_f = [{'gamma': np.ones(hidden_dim), 'beta': np.zeros(hidden_dim)} for _ in range(num_layers)]
        self.ln_b = [{'gamma': np.ones(hidden_dim), 'beta': np.zeros(hidden_dim)} for _ in range(num_layers)]
        
    def gru_cell(self, x, h_prev, W_z, W_r, W_h, layer_norm):
        """Single GRU cell with layer normalization"""
        # Concatenate input and previous hidden state
        x_h = np.concatenate([x, h_prev])
        
        # Update gate
        z = self._sigmoid(W_z @ x_h)
        
        # Reset gate
        r = self._sigmoid(W_r @ x_h)
        
        # Candidate hidden state
        x_rh = np.concatenate([x, r * h_prev])
        h_tilde = np.tanh(W_h @ x_rh)
        
        # New hidden state
        h_new = (1 - z) * h_prev + z * h_tilde
        
        # Layer normalization
        h_new = self._layer_norm(h_new, layer_norm['gamma'], layer_norm['beta'])
        
        return h_new
    
    def forward(self, input_sequence):
        """
        input_sequence: list of token indices
        Returns: encoder_outputs, hidden_states
        """
        seq_len = len(input_sequence)
        
        # Embedding lookup
        embedded = np.array([self.embedding[token] for token in input_sequence])
        
        # Initialize hidden states
        h_forward = [np.zeros(self.hidden_dim) for _ in range(self.num_layers)]
        h_backward = [np.zeros(self.hidden_dim) for _ in range(self.num_layers)]
        
        encoder_outputs_forward = []
        encoder_outputs_backward = []
        
        # Forward pass
        for t in range(seq_len):
            x = embedded[t]
            for layer in range(self.num_layers):
                h_forward[layer] = self.gru_cell(
                    x, h_forward[layer], 
                    self.W_z_f[layer], self.W_r_f[layer], self.W_h_f[layer],
                    self.ln_f[layer]
                )
                x = h_forward[layer]  # Output becomes input to next layer
            encoder_outputs_forward.append(h_forward[-1].copy())
        
        # Backward pass
        for t in range(seq_len-1, -1, -1):
            x = embedded[t]
            for layer in range(self.num_layers):
                h_backward[layer] = self.gru_cell(
                    x, h_backward[layer],
                    self.W_z_b[layer], self.W_r_b[layer], self.W_h_b[layer],
                    self.ln_b[layer]
                )
                x = h_backward[layer]
            encoder_outputs_backward.insert(0, h_backward[-1].copy())
        
        # Concatenate forward and backward outputs
        encoder_outputs = []
        for fwd, bwd in zip(encoder_outputs_forward, encoder_outputs_backward):
            combined = np.concatenate([fwd, bwd])
            encoder_outputs.append(combined)
        
        # Final hidden state (concatenated)
        final_hidden = np.concatenate([h_forward[-1], h_backward[-1]])
        
        return np.array(encoder_outputs), final_hidden
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))
    
    def _layer_norm(self, x, gamma, beta, eps=1e-5):
        mean = np.mean(x)
        std = np.std(x)
        return gamma * (x - mean) / (std + eps) + beta