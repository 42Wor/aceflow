import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import RNNLayer 
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, method='concat'):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.method = method
        
        if method == 'concat':
            self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.V = nn.Linear(hidden_size, 1, bias=False)
        elif method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, decoder_hidden, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        if self.method == 'concat':
            # Bahdanau style (additive)
            decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
            energy = torch.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden))
            attention_scores = self.V(energy).squeeze(-1)
        elif self.method == 'general':
            # Luong style (general)
            decoder_hidden = decoder_hidden.unsqueeze(1)
            energy = self.attn(encoder_outputs)
            attention_scores = torch.bmm(decoder_hidden, energy.transpose(1, 2)).squeeze(1)
        elif self.method == 'dot':
            # Luong style (dot)
            decoder_hidden = decoder_hidden.unsqueeze(1)
            attention_scores = torch.bmm(decoder_hidden, encoder_outputs.transpose(1, 2)).squeeze(1)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Calculate context vector
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context_vector = context_vector.squeeze(1)
        
        return context_vector, attention_weights

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention for Transformer-like models"""
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        
        output = self.output(context)
        return output, attention_weights

class AttentionalDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=2, dropout=0.1, 
                 rnn_type='lstm', attention_method='concat', encoder_bidirectional=False):
        super(AttentionalDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        # Adjust for bidirectional encoder
        encoder_factor = 2 if encoder_bidirectional else 1
        self.encoder_output_size = hidden_size * encoder_factor
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Attention mechanism
        self.attention = BahdanauAttention(self.encoder_output_size, method=attention_method)
        
        # RNN layer
        self.rnn = RNNLayer(
            input_size=hidden_size + self.encoder_output_size,  # [embedding; context]
            hidden_size=self.encoder_output_size,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            bidirectional=False
        )
        
        # Output projection
        self.out = nn.Linear(self.encoder_output_size * 2, vocab_size)
        
    def forward(self, x, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(x))
        
        # Get attention context
        if isinstance(hidden, tuple):  # LSTM
            decoder_hidden = hidden[0][-1]  # Take last layer hidden state
        else:  # GRU or RNN
            decoder_hidden = hidden[-1]
            
        context, attention_weights = self.attention(decoder_hidden, encoder_outputs)
        
        # Combine embedded input and context
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        
        # RNN forward pass
        output, hidden = self.rnn(rnn_input, hidden)
        
        # Combine output and context for final prediction
        output = torch.cat([output, context.unsqueeze(1)], dim=2)
        output = self.out(output)
        
        return output, hidden, attention_weights