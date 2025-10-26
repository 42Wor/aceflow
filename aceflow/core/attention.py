import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, seq_len, hidden_size]
        
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # Repeat decoder hidden state for each encoder time step
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Calculate attention scores
        energy = torch.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden))
        attention_scores = self.V(energy).squeeze(-1)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Calculate context vector
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context_vector = context_vector.squeeze(1)
        
        return context_vector, attention_weights

class AttentionalDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=2, dropout=0.1, rnn_type='lstm'):
        super(AttentionalDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(hidden_size)
        
        rnn_class = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_class(
            hidden_size * 2, hidden_size, num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.out = nn.Linear(hidden_size * 2, vocab_size)
        
    def forward(self, x, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(x))
        
        # Get attention context
        if isinstance(hidden, tuple):  # LSTM
            decoder_hidden = hidden[0][-1]  # Take last layer hidden state
        else:  # GRU
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