import numpy as np
import pickle
from tqdm import tqdm
from ..core.encoder import Encoder
from ..core.decoder import Decoder

class Seq2Seq:
    """Complete Seq2Seq Model with Training Utilities"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim=256, 
                 hidden_dim=512, num_layers=2, dropout=0.1):
        
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initialize encoder and decoder
        self.encoder = Encoder(src_vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'bleu_scores': []
        }
    
    def forward(self, src_sequence, tgt_sequence=None, teacher_forcing_ratio=0.5, max_length=50):
        """
        Forward pass for training or inference
        """
        # Encode source sequence
        encoder_outputs, encoder_hidden = self.encoder.forward(src_sequence)
        
        # Initialize decoder hidden state
        decoder_hidden = self._init_decoder_hidden(encoder_hidden)
        
        # Start token (assuming 0 is start token, 1 is end token)
        decoder_input = 0
        decoded_tokens = []
        attention_weights = []
        log_probs = []
        
        # Determine maximum length
        max_len = len(tgt_sequence) if tgt_sequence is not None else max_length
        
        for t in range(max_len):
            # Decoder forward step
            probs, decoder_hidden, attn_weights, context = self.decoder.forward(
                decoder_input, decoder_hidden, encoder_outputs
            )
            
            # Store attention weights
            attention_weights.append(attn_weights)
            
            # Get next token
            if tgt_sequence is not None and np.random.random() < teacher_forcing_ratio:
                # Teacher forcing
                decoder_input = tgt_sequence[t]
            else:
                # Greedy decoding
                decoder_input = np.argmax(probs)
            
            decoded_tokens.append(decoder_input)
            log_probs.append(np.log(probs[decoder_input] + 1e-8))
            
            # Stop if end token is generated
            if decoder_input == 1:  # Assuming 1 is end token
                break
        
        return decoded_tokens, log_probs, attention_weights
    
    def train_step(self, src_sequence, tgt_sequence, optimizer, teacher_forcing_ratio=0.5):
        """Single training step"""
        # Forward pass
        decoded_tokens, log_probs, _ = self.forward(
            src_sequence, tgt_sequence, teacher_forcing_ratio
        )
        
        # Calculate loss (negative log likelihood)
        loss = -np.sum(log_probs[:len(tgt_sequence)]) / len(tgt_sequence)
        
        # Backward pass (simplified)
        # In a full implementation, you'd compute gradients here
        
        return loss, decoded_tokens
    
    def predict(self, src_sequence, max_length=50):
        """Generate prediction for source sequence"""
        decoded_tokens, _, attention_weights = self.forward(
            src_sequence, None, teacher_forcing_ratio=0.0, max_length=max_length
        )
        return decoded_tokens, attention_weights
    
    def _init_decoder_hidden(self, encoder_hidden):
        """Initialize decoder hidden state from encoder final state"""
        # Split bidirectional encoder hidden and use as initial decoder state
        hidden_dim = self.hidden_dim
        forward_hidden = encoder_hidden[:hidden_dim]
        
        # Initialize all decoder layers with the forward hidden state
        return [forward_hidden.copy() for _ in range(self.num_layers)]
    
    def get_parameters(self):
        """Get all model parameters for saving"""
        params = {
            'encoder': {
                'embedding': self.encoder.embedding,
                'W_z_f': self.encoder.W_z_f,
                'W_r_f': self.encoder.W_r_f,
                'W_h_f': self.encoder.W_h_f,
                'W_z_b': self.encoder.W_z_b,
                'W_r_b': self.encoder.W_r_b,
                'W_h_b': self.encoder.W_h_b,
                'ln_f': self.encoder.ln_f,
                'ln_b': self.encoder.ln_b,
            },
            'decoder': {
                'embedding': self.decoder.embedding,
                'W_z': self.decoder.W_z,
                'W_r': self.decoder.W_r,
                'W_h': self.decoder.W_h,
                'W_out': self.decoder.W_out,
                'b_out': self.decoder.b_out,
                'ln': self.decoder.ln,
                'attention': {
                    'W1': self.decoder.attention.W1,
                    'W2': self.decoder.attention.W2,
                    'V': self.decoder.attention.V
                }
            },
            'config': {
                'src_vocab_size': self.src_vocab_size,
                'tgt_vocab_size': self.tgt_vocab_size,
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers
            }
        }
        return params
    
    def set_parameters(self, params):
        """Set model parameters from loaded data"""
        # Set encoder parameters
        self.encoder.embedding = params['encoder']['embedding']
        self.encoder.W_z_f = params['encoder']['W_z_f']
        self.encoder.W_r_f = params['encoder']['W_r_f']
        self.encoder.W_h_f = params['encoder']['W_h_f']
        self.encoder.W_z_b = params['encoder']['W_z_b']
        self.encoder.W_r_b = params['encoder']['W_r_b']
        self.encoder.W_h_b = params['encoder']['W_h_b']
        self.encoder.ln_f = params['encoder']['ln_f']
        self.encoder.ln_b = params['encoder']['ln_b']
        
        # Set decoder parameters
        self.decoder.embedding = params['decoder']['embedding']
        self.decoder.W_z = params['decoder']['W_z']
        self.decoder.W_r = params['decoder']['W_r']
        self.decoder.W_h = params['decoder']['W_h']
        self.decoder.W_out = params['decoder']['W_out']
        self.decoder.b_out = params['decoder']['b_out']
        self.decoder.ln = params['decoder']['ln']
        self.decoder.attention.W1 = params['decoder']['attention']['W1']
        self.decoder.attention.W2 = params['decoder']['attention']['W2']
        self.decoder.attention.V = params['decoder']['attention']['V']