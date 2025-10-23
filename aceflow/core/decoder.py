import numpy as np
from typing import Tuple, List, Optional
from .layers import LSTMCell, Embedding, Dense
from .attention import Attention

class Decoder:
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, 
                 num_layers: int = 1, use_attention: bool = True):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm_layers = [LSTMCell(embedding_dim if i == 0 else hidden_dim, hidden_dim) 
                           for i in range(num_layers)]
        
        if use_attention:
            self.attention = Attention(hidden_dim)
            self.output_layer = Dense(2 * hidden_dim, vocab_size)
        else:
            self.output_layer = Dense(hidden_dim, vocab_size)
    
    def forward(self, x: np.ndarray, encoder_outputs: np.ndarray, 
                h_prev: List[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        """
        x: (batch_size, target_seq_len)
        encoder_outputs: (batch_size, source_seq_len, hidden_dim)
        """
        batch_size, target_seq_len = x.shape
        
        if h_prev is None:
            h_prev = [(np.zeros((batch_size, self.hidden_dim)), 
                      np.zeros((batch_size, self.hidden_dim))) 
                     for _ in range(self.num_layers)]
        
        embedded = self.embedding.forward(x)
        
        outputs = []
        attention_weights_list = []
        h_current = [ (h_prev[i][0].copy(), h_prev[i][1].copy()) for i in range(self.num_layers) ]
        
        for t in range(target_seq_len):
            x_t = embedded[:, t, :]
            
            # Pass through LSTM layers
            for layer_idx in range(self.num_layers):
                h_current[layer_idx] = self.lstm_layers[layer_idx].forward(
                    x_t, h_current[layer_idx][0], h_current[layer_idx][1]
                )
                x_t = h_current[layer_idx][0]
            
            decoder_hidden = h_current[-1][0]
            
            if self.use_attention:
                # Apply attention
                context_vector, attention_weights = self.attention.forward(decoder_hidden, encoder_outputs)
                # Combine context with decoder hidden state
                combined = np.concatenate([context_vector, decoder_hidden], axis=1)
                output = self.output_layer.forward(combined)
                attention_weights_list.append(attention_weights)
            else:
                output = self.output_layer.forward(decoder_hidden)
            
            outputs.append(output)
        
        outputs = np.stack(outputs, axis=1)  # (batch_size, target_seq_len, vocab_size)
        final_states = [(h_current[i][0], h_current[i][1]) for i in range(self.num_layers)]
        
        self.cache = (embedded, encoder_outputs, attention_weights_list if self.use_attention else None)
        return outputs, final_states
    
    def backward(self, doutputs: np.ndarray) -> np.ndarray:
        embedded, encoder_outputs, attention_weights_list = self.cache
        batch_size, target_seq_len, vocab_size = doutputs.shape
        
        dembedded = np.zeros_like(embedded)
        dencoder_outputs = np.zeros_like(encoder_outputs) if self.use_attention else None
        dh_next = [np.zeros((batch_size, self.hidden_dim)) for _ in range(self.num_layers)]
        dc_next = [np.zeros((batch_size, self.hidden_dim)) for _ in range(self.num_layers)]
        
        for t in reversed(range(target_seq_len)):
            d_output_t = doutputs[:, t, :]
            
            if self.use_attention:
                # Backward through output layer
                d_combined = self.output_layer.backward(d_output_t)
                d_context = d_combined[:, :self.hidden_dim]
                d_decoder_hidden_1 = d_combined[:, self.hidden_dim:]
                
                # Backward through attention
                d_encoder_outputs_t, d_decoder_hidden_2 = self.attention.backward(d_context, None)
                dencoder_outputs += d_encoder_outputs_t
                d_decoder_hidden = d_decoder_hidden_1 + d_decoder_hidden_2 + dh_next[-1]
            else:
                d_decoder_hidden = self.output_layer.backward(d_output_t) + dh_next[-1]
            
            d_c_final = dc_next[-1]
            
            # Backward through LSTM layers
            for layer_idx in reversed(range(self.num_layers)):
                if layer_idx == 0:
                    dx_t, dh_prev, dc_prev = self.lstm_layers[layer_idx].backward(
                        d_decoder_hidden, d_c_final
                    )
                    dembedded[:, t, :] += dx_t
                else:
                    dx_t, dh_prev, dc_prev = self.lstm_layers[layer_idx].backward(
                        d_decoder_hidden, d_c_final
                    )
                    dh_next[layer_idx-1] += dx_t
                    dc_next[layer_idx-1] += dc_prev
                
                d_decoder_hidden, d_c_final = dh_prev, dc_prev
        
        # Backward through embedding
        self.embedding.backward(dembedded.reshape(-1, self.embedding_dim))
        
        return dencoder_outputs