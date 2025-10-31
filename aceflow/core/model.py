import torch
import torch.nn as nn
import json
import os
from ..utils.serialization import AceModelSerializer
from .layers import Encoder, Decoder
from .attention import AttentionalDecoder

class Seq2SeqModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, hidden_size=256, 
                 num_layers=2, dropout=0.1, rnn_type='lstm', use_attention=True,
                 teacher_forcing_ratio=0.5, max_length=50, bidirectional=False,
                 attention_method='concat', embedding_dim=None):
        super(Seq2SeqModel, self).__init__()
        
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
            embedding_dim=embedding_dim
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
            'embedding_dim': embedding_dim
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
        encoder_outputs, encoder_hidden = self.encoder(src)
        return encoder_outputs, encoder_hidden
    
    def decode(self, decoder_input, decoder_hidden, encoder_outputs):
        if self.use_attention:
            return self.decoder(decoder_input, decoder_hidden, encoder_outputs)
        else:
            return self.decoder(decoder_input, decoder_hidden)
    
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
    
    def save(self, filepath):
        """Save model to .ace format"""
        serializer = AceModelSerializer()
        serializer.save_model(self, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load model from .ace format"""
        serializer = AceModelSerializer()
        return serializer.load_model(filepath)
    
    def beam_search(self, src, beam_width=5, max_length=50):
        """Beam search for inference"""
        self.eval()
        with torch.no_grad():
            # Encode source
            encoder_outputs, encoder_hidden = self.encode(src)
            
            # Initialize beams
            start_token = 1
            beams = [([start_token], 0, encoder_hidden)]
            
            for _ in range(max_length):
                new_beams = []
                
                for seq, score, hidden in beams:
                    # Check if sequence ended
                    if seq[-1] == 2:  # End token
                        new_beams.append((seq, score, hidden))
                        continue
                    
                    # Prepare decoder input
                    decoder_input = torch.tensor([[seq[-1]]], device=src.device)
                    
                    # Decode
                    if self.use_attention:
                        decoder_output, new_hidden, _ = self.decode(decoder_input, hidden, encoder_outputs)
                    else:
                        decoder_output, new_hidden = self.decode(decoder_input, hidden)
                    
                    # Get top k candidates
                    log_probs = torch.log_softmax(decoder_output.squeeze(), dim=0)
                    topk_probs, topk_indices = torch.topk(log_probs, beam_width)
                    
                    for i in range(beam_width):
                        new_seq = seq + [topk_indices[i].item()]
                        new_score = score + topk_probs[i].item()
                        new_beams.append((new_seq, new_score, new_hidden))
                
                # Keep top beam_width beams
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
                
                # Check if all beams ended
                if all(seq[-1] == 2 for seq, _, _ in beams):
                    break
            
            # Return best sequence
            best_sequence = beams[0][0]
            return best_sequence