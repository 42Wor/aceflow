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
                 teacher_forcing_ratio=0.5, max_length=50):
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
        
        # Build encoder and decoder
        self.encoder = Encoder(src_vocab_size, hidden_size, num_layers, dropout, rnn_type)
        
        if use_attention:
            self.decoder = AttentionalDecoder(tgt_vocab_size, hidden_size, num_layers, dropout, rnn_type)
        else:
            self.decoder = Decoder(tgt_vocab_size, hidden_size, num_layers, dropout, rnn_type)
        
        self.use_attention = use_attention
        
    def forward(self, src, tgt=None, teacher_forcing_ratio=None):
        batch_size = src.size(0)
        
        # Forward pass through encoder
        encoder_outputs, encoder_hidden = self.encoder(src)
        
        # Initialize decoder
        decoder_hidden = encoder_hidden
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
    
    def encode(self, src):
        encoder_outputs, encoder_hidden = self.encoder(src)
        return encoder_outputs, encoder_hidden
    
    def decode(self, decoder_input, decoder_hidden, encoder_outputs):
        if self.use_attention:
            return self.decoder(decoder_input, decoder_hidden, encoder_outputs)
        else:
            return self.decoder(decoder_input, decoder_hidden)
    
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