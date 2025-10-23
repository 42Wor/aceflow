import numpy as np
from typing import Tuple, List
from .layers import LSTMCell, Embedding

class Encoder:
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int = 1):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm_layers = [LSTMCell(embedding_dim if i == 0 else hidden_dim, hidden_dim) 
                           for i in range(num_layers)]
    
    def forward(self, x: np.ndarray, h_prev: List[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        """
        x: (batch_size, seq_len)
        Returns: encoder_outputs, (h_final, c_final)
        """
        batch_size, seq_len = x.shape
        
        if h_prev is None:
            h_prev = [(np.zeros((batch_size, self.hidden_dim)), 
                      np.zeros((batch_size, self.hidden_dim))) 
                     for _ in range(self.num_layers)]
        
        # Embed input
        embedded = self.embedding.forward(x)  # (batch_size, seq_len, embedding_dim)
        
        encoder_outputs = []
        h_current = [ (h_prev[i][0].copy(), h_prev[i][1].copy()) for i in range(self.num_layers) ]
        
        # Process sequence
        for t in range(seq_len):
            x_t = embedded[:, t, :]  # (batch_size, embedding_dim)
            
            # Pass through LSTM layers
            for layer_idx in range(self.num_layers):
                h_current[layer_idx] = self.lstm_layers[layer_idx].forward(
                    x_t, h_current[layer_idx][0], h_current[layer_idx][1]
                )
                x_t = h_current[layer_idx][0]  # Output becomes input to next layer
            
            encoder_outputs.append(h_current[-1][0])
        
        encoder_outputs = np.stack(encoder_outputs, axis=1)  # (batch_size, seq_len, hidden_dim)
        final_states = [(h_current[i][0], h_current[i][1]) for i in range(self.num_layers)]
        
        self.cache = (embedded, seq_len)
        return encoder_outputs, final_states
    
    def backward(self, dencoder_outputs: np.ndarray) -> np.ndarray:
        embedded, seq_len = self.cache
        batch_size = embedded.shape[0]
        
        dembedded = np.zeros_like(embedded)
        dh_next = [np.zeros((batch_size, self.hidden_dim)) for _ in range(self.num_layers)]
        dc_next = [np.zeros((batch_size, self.hidden_dim)) for _ in range(self.num_layers)]
        
        # Backward through time
        for t in reversed(range(seq_len)):
            d_h_final = dencoder_outputs[:, t, :] + dh_next[-1]
            d_c_final = dc_next[-1]
            
            # Backward through LSTM layers in reverse order
            for layer_idx in reversed(range(self.num_layers)):
                if layer_idx == 0:
                    dx_t, dh_prev, dc_prev = self.lstm_layers[layer_idx].backward(
                        d_h_final, d_c_final
                    )
                    dembedded[:, t, :] += dx_t
                else:
                    dx_t, dh_prev, dc_prev = self.lstm_layers[layer_idx].backward(
                        d_h_final, d_c_final
                    )
                    dh_next[layer_idx-1] += dx_t
                    dc_next[layer_idx-1] += dc_prev
                
                d_h_final, d_c_final = dh_prev, dc_prev
        
        # Backward through embedding
        self.embedding.backward(dembedded.reshape(-1, self.embedding_dim))
        
        return None