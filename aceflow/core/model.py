import numpy as np
from typing import Tuple, List, Optional, Dict
from .encoder import Encoder
from .decoder import Decoder

class Seq2Seq:
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, 
                 embedding_dim: int = 256, hidden_dim: int = 512,
                 num_layers: int = 2, use_attention: bool = True,
                 dropout: float = 0.1):
        
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        self.encoder = Encoder(src_vocab_size, embedding_dim, hidden_dim, num_layers)
        self.decoder = Decoder(tgt_vocab_size, embedding_dim, hidden_dim, num_layers, use_attention)
        
        # Special tokens
        self.pad_token = 0
        self.sos_token = 1
        self.eos_token = 2
    
    def forward(self, src_seq: np.ndarray, tgt_seq: np.ndarray) -> np.ndarray:
        """
        src_seq: (batch_size, src_len)
        tgt_seq: (batch_size, tgt_len)
        """
        # Encode source sequence
        encoder_outputs, encoder_final_states = self.encoder.forward(src_seq)
        
        # Decode target sequence
        decoder_outputs, _ = self.decoder.forward(tgt_seq, encoder_outputs, encoder_final_states)
        
        self.cache = (encoder_outputs, encoder_final_states)
        return decoder_outputs
    
    def backward(self, doutputs: np.ndarray) -> None:
        encoder_outputs, encoder_final_states = self.cache
        
        # Backward through decoder
        dencoder_outputs = self.decoder.backward(doutputs)
        
        # Backward through encoder
        if dencoder_outputs is not None:
            self.encoder.backward(dencoder_outputs)
    
    def predict(self, src_seq: np.ndarray, max_length: int = 50) -> np.ndarray:
        """
        Generate prediction for source sequence
        """
        batch_size = src_seq.shape[0]
        
        # Encode source sequence
        encoder_outputs, encoder_final_states = self.encoder.forward(src_seq)
        
        # Start with SOS token
        current_tokens = np.full((batch_size, 1), self.sos_token, dtype=np.int32)
        predictions = []
        
        for _ in range(max_length):
            decoder_outputs, decoder_states = self.decoder.forward(
                current_tokens, encoder_outputs, encoder_final_states
            )
            
            # Get most probable next token
            next_tokens = np.argmax(decoder_outputs[:, -1, :], axis=1)
            predictions.append(next_tokens)
            
            # Stop if all sequences generated EOS
            if np.all(next_tokens == self.eos_token):
                break
            
            # Update input for next step
            current_tokens = next_tokens.reshape(-1, 1)
            encoder_final_states = decoder_states
        
        return np.stack(predictions, axis=1)
    
    def get_params(self) -> Dict:
        """Get all model parameters"""
        params = {}
        
        # Encoder parameters
        params['encoder_embedding'] = self.encoder.embedding.params['W']
        for i, lstm in enumerate(self.encoder.lstm_layers):
            params[f'encoder_lstm_{i}_W'] = lstm.params['W']
            params[f'encoder_lstm_{i}_b'] = lstm.params['b']
        
        # Decoder parameters
        params['decoder_embedding'] = self.decoder.embedding.params['W']
        for i, lstm in enumerate(self.decoder.lstm_layers):
            params[f'decoder_lstm_{i}_W'] = lstm.params['W']
            params[f'decoder_lstm_{i}_b'] = lstm.params['b']
        
        if self.use_attention:
            params['attention_W'] = self.decoder.attention.W_a
            params['attention_v'] = self.decoder.attention.v_a
        
        params['output_W'] = self.decoder.output_layer.params['W']
        params['output_b'] = self.decoder.output_layer.params['b']
        
        return params
    
    def set_params(self, params: Dict) -> None:
        """Set all model parameters"""
        # Encoder parameters
        self.encoder.embedding.params['W'] = params['encoder_embedding']
        for i, lstm in enumerate(self.encoder.lstm_layers):
            lstm.params['W'] = params[f'encoder_lstm_{i}_W']
            lstm.params['b'] = params[f'encoder_lstm_{i}_b']
        
        # Decoder parameters
        self.decoder.embedding.params['W'] = params['decoder_embedding']
        for i, lstm in enumerate(self.decoder.lstm_layers):
            lstm.params['W'] = params[f'decoder_lstm_{i}_W']
            lstm.params['b'] = params[f'decoder_lstm_{i}_b']
        
        if self.use_attention:
            self.decoder.attention.W_a = params['attention_W']
            self.decoder.attention.v_a = params['attention_v']
        
        self.decoder.output_layer.params['W'] = params['output_W']
        self.decoder.output_layer.params['b'] = params['output_b']