import torch
import torch.nn as nn
import json
import os
from typing import List, Dict, Union, Optional, Tuple
import numpy as np

from ..utils.serialization import AceModelSerializer
from .layers import Encoder, Decoder
from .attention import AttentionalDecoder

try:
    from aceflow_core import beam_search_rust, beam_search_batch_rust
    HAS_RUST_EXTENSION = True
except ImportError:
    HAS_RUST_EXTENSION = False
    print("Warning: Rust extension not available, using Python beam search")

class Seq2SeqModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, hidden_size=256, 
                 num_layers=2, dropout=0.1, rnn_type='lstm', use_attention=True,
                 teacher_forcing_ratio=0.5, max_length=50, bidirectional=False,
                 attention_method='concat', embedding_dim=None):
        super(Seq2SeqModel, self).__init__()
        
        # Store all parameters
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.use_attention = use_attention
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.max_length = max_length
        self.bidirectional = bidirectional
        self.attention_method = attention_method
        self.embedding_dim = embedding_dim or hidden_size
        
        # Validate RNN type
        valid_rnn_types = ['rnn', 'lstm', 'gru', 'birnn', 'bilstm', 'bigru']
        if self.rnn_type not in valid_rnn_types:
            raise ValueError(f"Invalid RNN type: {rnn_type}. Choose from {valid_rnn_types}")
        
        # Build encoder
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            embedding_dim=self.embedding_dim
        )
        
        # Build decoder
        if use_attention:
            self.decoder = AttentionalDecoder(
                vocab_size=tgt_vocab_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                rnn_type=rnn_type,
                attention_method=attention_method,
                encoder_bidirectional=bidirectional
            )
        else:
            from .layers import Decoder
            self.decoder = Decoder(
                vocab_size=tgt_vocab_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                rnn_type=rnn_type,
                encoder_bidirectional=bidirectional
            )
        
        # Store configuration
        self.config = {
            'src_vocab_size': src_vocab_size,
            'tgt_vocab_size': tgt_vocab_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'rnn_type': rnn_type,
            'use_attention': use_attention,
            'teacher_forcing_ratio': teacher_forcing_ratio,
            'max_length': max_length,
            'bidirectional': bidirectional,
            'attention_method': attention_method,
            'embedding_dim': self.embedding_dim
        }
    
    def forward(self, src, tgt=None, teacher_forcing_ratio=None):
        batch_size = src.size(0)
        
        # Forward pass through encoder
        encoder_outputs, encoder_hidden = self.encoder(src)
        
        # Initialize decoder
        decoder_hidden = self._init_decoder_hidden(encoder_hidden)
        decoder_input = torch.tensor([[1]] * batch_size, device=src.device)  # Start token
        
        # Store outputs
        decoder_outputs = []
        attention_weights = []
        
        # Use provided teacher_forcing_ratio or default
        tf_ratio = teacher_forcing_ratio if teacher_forcing_ratio is not None else self.teacher_forcing_ratio
        
        max_len = tgt.size(1) if tgt is not None else self.max_length
        
        for t in range(max_len):
            if self.use_attention:
                decoder_output, decoder_hidden, attn_weights = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                attention_weights.append(attn_weights)
            else:
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            
            decoder_outputs.append(decoder_output)
            
            # Teacher forcing
            if tgt is not None and torch.rand(1).item() < tf_ratio:
                decoder_input = tgt[:, t].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()
        
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        
        if self.use_attention:
            attention_weights = torch.stack(attention_weights, dim=1)
            return decoder_outputs, attention_weights
        else:
            return decoder_outputs
    
    def _init_decoder_hidden(self, encoder_hidden):
        """Initialize decoder hidden state from encoder hidden state"""
        if self.rnn_type in ['lstm', 'bilstm']:
            # For LSTM: (hidden, cell)
            if self.bidirectional:
                # Sum bidirectional layers
                hidden = encoder_hidden[0][::2] + encoder_hidden[0][1::2]  # Even and odd
                cell = encoder_hidden[1][::2] + encoder_hidden[1][1::2]
                return (hidden, cell)
            else:
                return encoder_hidden
        else:
            # For RNN/GRU
            if self.bidirectional:
                # Sum bidirectional layers
                hidden = encoder_hidden[::2] + encoder_hidden[1::2]
                return hidden
            else:
                return encoder_hidden
    
    def encode(self, src):
        """Encode source sequences"""
        encoder_outputs, encoder_hidden = self.encoder(src)
        return encoder_outputs, encoder_hidden
    
    def decode(self, decoder_input, decoder_hidden, encoder_outputs):
        """Decode with current hidden state"""
        if self.use_attention:
            return self.decoder(decoder_input, decoder_hidden, encoder_outputs)
        else:
            return self.decoder(decoder_input, decoder_hidden)
    
    def beam_search(self, src, beam_width=5, max_length=50):
        """Beam search for inference"""
        if HAS_RUST_EXTENSION:
            return self._beam_search_rust(src, beam_width, max_length)
        else:
            return self._beam_search_python(src, beam_width, max_length)
    
    def _beam_search_rust(self, src, beam_width=5, max_length=50):
        """Rust-accelerated beam search"""
        self.eval()
        
        with torch.no_grad():
            # Encode source sequence
            encoder_outputs, encoder_hidden = self.encode(src)
            
            batch_size = src.size(0)
            
            # Prepare inputs for Rust
            encoder_outputs_np = encoder_outputs.cpu().numpy().astype(np.float32)
            
            if isinstance(encoder_hidden, tuple):  # LSTM
                hidden_np = encoder_hidden[0].cpu().numpy().astype(np.float32)
                cell_np = encoder_hidden[1].cpu().numpy().astype(np.float32)
            else:  # GRU/RNN
                hidden_np = encoder_hidden.cpu().numpy().astype(np.float32)
                cell_np = None
            
            # Call Rust implementation
            if batch_size == 1:
                # Single sequence
                sequences, scores = beam_search_rust(
                    encoder_outputs_np[0],  # Remove batch dimension
                    hidden_np[0],           # Remove batch dimension  
                    cell_np[0] if cell_np is not None else None,
                    beam_width=beam_width,
                    max_length=max_length,
                    start_token=1,  # <start> token
                    end_token=2,    # <end> token
                    vocab_size=self.tgt_vocab_size,
                    length_penalty=0.6,
                    temperature=1.0
                )
                
                # Return best sequence
                best_sequence = sequences[0] if sequences else []
                return best_sequence
                
            else:
                # Batch processing
                batch_results = beam_search_batch_rust(
                    encoder_outputs_np,
                    hidden_np,
                    cell_np,
                    beam_width=beam_width,
                    max_length=max_length,
                    start_token=1,
                    end_token=2,
                    vocab_size=self.tgt_vocab_size
                )
                
                # Return best sequences for each batch item
                best_sequences = [result[0][0] if result[0] else [] for result in batch_results]
                return best_sequences
    
    def _beam_search_python(self, src, beam_width=5, max_length=50):
        """Fallback Python beam search implementation"""
        self.eval()
        
        with torch.no_grad():
            # Encode source sequence
            encoder_outputs, encoder_hidden = self.encode(src)
            batch_size, src_len, hidden_size = encoder_outputs.size()
            
            # Initialize beams
            start_token = 1
            end_token = 2
            
            # For each sequence in batch
            all_sequences = []
            
            for batch_idx in range(batch_size):
                # Initialize beams for this sequence
                beams = [{
                    'sequence': [start_token],
                    'score': 0.0,
                    'hidden': self._get_batch_item(encoder_hidden, batch_idx),
                    'attention_weights': []
                }]
                
                for step in range(max_length):
                    new_beams = []
                    
                    for beam in beams:
                        # Check if beam is complete
                        if beam['sequence'][-1] == end_token:
                            new_beams.append(beam)
                            continue
                        
                        # Prepare decoder input
                        last_token = beam['sequence'][-1]
                        decoder_input = torch.tensor([[last_token]], device=src.device)
                        
                        # Get encoder outputs for this batch item
                        encoder_outputs_single = encoder_outputs[batch_idx:batch_idx+1]
                        
                        # Decode
                        if self.use_attention:
                            decoder_output, new_hidden, attn_weights = self.decode(
                                decoder_input, beam['hidden'], encoder_outputs_single
                            )
                        else:
                            decoder_output, new_hidden = self.decode(
                                decoder_input, beam['hidden'], encoder_outputs_single
                            )
                        
                        # Get top k candidates
                        decoder_output = decoder_output.squeeze(1)  # Remove sequence dimension
                        log_probs = torch.log_softmax(decoder_output, dim=-1)
                        topk_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)
                        
                        for i in range(beam_width):
                            new_token = topk_indices[0, i].item()
                            new_score = beam['score'] + topk_probs[0, i].item()
                            
                            new_sequence = beam['sequence'] + [new_token]
                            new_beam = {
                                'sequence': new_sequence,
                                'score': new_score,
                                'hidden': new_hidden,
                                'attention_weights': beam['attention_weights'] + [attn_weights] if self.use_attention else []
                            }
                            new_beams.append(new_beam)
                    
                    # Keep top beam_width beams
                    new_beams.sort(key=lambda x: x['score'], reverse=True)
                    beams = new_beams[:beam_width]
                    
                    # Early stopping if all beams are complete
                    if all(beam['sequence'][-1] == end_token for beam in beams):
                        break
                
                # Add best sequence for this batch item
                best_sequence = beams[0]['sequence'] if beams else []
                all_sequences.append(best_sequence)
            
            return all_sequences[0] if batch_size == 1 else all_sequences
    
    def _get_batch_item(self, hidden, batch_idx):
        """Extract hidden state for specific batch item"""
        if isinstance(hidden, tuple):  # LSTM
            return (
                hidden[0][:, batch_idx:batch_idx+1, :],
                hidden[1][:, batch_idx:batch_idx+1, :]
            )
        else:  # RNN/GRU
            return hidden[:, batch_idx:batch_idx+1, :]
    
    def get_rnn_info(self):
        """Get information about the RNN configuration"""
        return {
            'rnn_type': self.rnn_type,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'has_attention': self.use_attention,
            'attention_method': self.attention_method if self.use_attention else None,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }
    
    def resize_token_embeddings(self, new_src_vocab_size=None, new_tgt_vocab_size=None):
        """Resize token embeddings for transfer learning"""
        if new_src_vocab_size is not None and new_src_vocab_size != self.src_vocab_size:
            # Resize encoder embeddings
            old_embedding = self.encoder.embedding
            new_embedding = nn.Embedding(new_src_vocab_size, old_embedding.embedding_dim)
            
            # Copy old weights
            with torch.no_grad():
                min_size = min(new_src_vocab_size, self.src_vocab_size)
                new_embedding.weight[:min_size] = old_embedding.weight[:min_size]
            
            self.encoder.embedding = new_embedding
            self.src_vocab_size = new_src_vocab_size
            self.config['src_vocab_size'] = new_src_vocab_size
        
        if new_tgt_vocab_size is not None and new_tgt_vocab_size != self.tgt_vocab_size:
            # Resize decoder embeddings and output layer
            old_embedding = self.decoder.embedding
            new_embedding = nn.Embedding(new_tgt_vocab_size, old_embedding.embedding_dim)
            
            # Copy old weights
            with torch.no_grad():
                min_size = min(new_tgt_vocab_size, self.tgt_vocab_size)
                new_embedding.weight[:min_size] = old_embedding.weight[:min_size]
            
            self.decoder.embedding = new_embedding
            
            # Resize output layer
            if hasattr(self.decoder, 'out'):
                old_out = self.decoder.out
                new_out = nn.Linear(old_out.in_features, new_tgt_vocab_size)
                
                with torch.no_grad():
                    min_size = min(new_tgt_vocab_size, self.tgt_vocab_size)
                    new_out.weight[:min_size] = old_out.weight[:min_size]
                    if old_out.bias is not None:
                        new_out.bias[:min_size] = old_out.bias[:min_size]
                
                self.decoder.out = new_out
            
            self.tgt_vocab_size = new_tgt_vocab_size
            self.config['tgt_vocab_size'] = new_tgt_vocab_size
    
    def save(self, filepath):
        """Save model to .ace format"""
        serializer = AceModelSerializer()
        serializer.save_model(self, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load model from .ace format"""
        serializer = AceModelSerializer()
        return serializer.load_model(filepath)
    
    def __repr__(self):
        return (f"Seq2SeqModel(src_vocab_size={self.src_vocab_size}, "
                f"tgt_vocab_size={self.tgt_vocab_size}, "
                f"hidden_size={self.hidden_size}, "
                f"rnn_type='{self.rnn_type}', "
                f"use_attention={self.use_attention})")

# Factory function for easier model creation
def create_seq2seq_model(src_vocab_size, tgt_vocab_size, **kwargs):
    """
    Factory function to create Seq2SeqModel with proper parameters
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        **kwargs: Additional model parameters
    
    Returns:
        Seq2SeqModel instance
    """
    return Seq2SeqModel(src_vocab_size, tgt_vocab_size, **kwargs)